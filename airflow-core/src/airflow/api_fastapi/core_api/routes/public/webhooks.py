# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import hashlib
import hmac
from typing import Any

import structlog
from fastapi import Header, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select

from airflow.api_fastapi.common.dagbag import DagBagDep, get_latest_version_of_dag
from airflow.api_fastapi.common.db.common import SessionDep
from airflow.api_fastapi.common.router import AirflowRouter
from airflow.api_fastapi.core_api.datamodels.dag_run import TriggerDAGRunPostBody
from airflow.api_fastapi.core_api.openapi.exceptions import create_openapi_http_exception_doc
from airflow.configuration import conf
from airflow.models.dag import DagModel, DagTag
from airflow.utils import timezone
from airflow.utils.state import DagRunState
from airflow.utils.types import DagRunTriggeredByType, DagRunType

logger = structlog.get_logger(__name__)

webhooks_router = AirflowRouter(tags=["Webhook"], prefix="/webhooks")

# DAGs opt in to GitHub webhook triggers by carrying a tag of the form
# "github:{owner}/{repo}" (e.g. "github:myorg/myrepo").
_GITHUB_TAG_PREFIX = "github:"


class _TriggeredRun(BaseModel):
    dag_id: str
    run_id: str


class _TriggerError(BaseModel):
    dag_id: str
    error: str


class GithubWebhookResponse(BaseModel):
    message: str = ""
    triggered: list[_TriggeredRun] = []
    errors: list[_TriggerError] = []


def _verify_github_signature(payload: bytes, signature_header: str | None, secret: str) -> None:
    if not signature_header:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Missing X-Hub-Signature-256 header")
    if not signature_header.startswith("sha256="):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid signature format")
    mac = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256)
    expected = "sha256=" + mac.hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid webhook signature")


@webhooks_router.post(
    "/github",
    responses=create_openapi_http_exception_doc(
        [
            status.HTTP_403_FORBIDDEN,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ]
    ),
)
async def github_push_webhook(
    request: Request,
    session: SessionDep,
    dag_bag: DagBagDep,
    x_hub_signature_256: str | None = Header(default=None),
    x_github_event: str | None = Header(default=None),
) -> GithubWebhookResponse:
    """Accept GitHub push events and trigger DAGs tagged with ``github:{owner}/{repo}``.

    Signature verification uses HMAC-SHA256 against the shared secret stored in
    ``[api] github_webhook_secret``.  Only ``push`` events trigger DAG runs; all
    other event types are acknowledged and ignored.
    """
    secret = conf.get("api", "github_webhook_secret", fallback=None)
    if not secret:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Webhook is not configured: set [api] github_webhook_secret in airflow.cfg",
        )

    payload_bytes = await request.body()
    _verify_github_signature(payload_bytes, x_hub_signature_256, secret)

    if x_github_event != "push":
        return GithubWebhookResponse(message=f"Event type '{x_github_event}' is not handled")

    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid JSON payload")

    repo_full_name: str = payload.get("repository", {}).get("full_name", "")
    if not repo_full_name:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "Missing repository.full_name in payload"
        )

    ref: str = payload.get("ref", "")
    head_sha: str = payload.get("head_commit", {}).get("id", "") or payload.get("after", "")
    pusher: str = payload.get("pusher", {}).get("name", "github-webhook")

    tag_value = f"{_GITHUB_TAG_PREFIX}{repo_full_name}"
    dag_ids = session.scalars(
        select(DagModel.dag_id)
        .join(DagModel.tags)
        .where(
            DagTag.name == tag_value,
            DagModel.is_active.is_(True),
            DagModel.is_paused.is_(False),
        )
    ).all()

    if not dag_ids:
        logger.info("No active DAGs tagged for repository", repo=repo_full_name, tag=tag_value)
        return GithubWebhookResponse(message="No matching DAGs found")

    now = timezone.utcnow()
    sha_prefix = head_sha[:8] if head_sha else "unknown"
    run_id_base = f"github_push__{sha_prefix}__{int(now.timestamp())}"
    push_conf = {
        "github_ref": ref,
        "github_sha": head_sha,
        "github_repository": repo_full_name,
        "github_pusher": pusher,
    }

    triggered: list[_TriggeredRun] = []
    errors: list[_TriggerError] = []

    for dag_id in dag_ids:
        try:
            dag = get_latest_version_of_dag(dag_bag, dag_id, session)
            body = TriggerDAGRunPostBody(dag_run_id=run_id_base, conf=push_conf)
            params = body.validate_context(dag)
            dag_run = dag.create_dagrun(
                run_id=params["run_id"],
                logical_date=params["logical_date"],
                data_interval=params["data_interval"],
                run_after=params["run_after"],
                conf=params["conf"],
                run_type=DagRunType.MANUAL,
                triggered_by=DagRunTriggeredByType.REST_API,
                triggering_user_name=pusher,
                state=DagRunState.QUEUED,
                partition_key=params["partition_key"],
                session=session,
            )
            triggered.append(_TriggeredRun(dag_id=dag_id, run_id=dag_run.run_id))
            logger.info("Triggered DAG run from GitHub push", dag_id=dag_id, run_id=dag_run.run_id)
        except Exception as exc:
            logger.warning("Failed to trigger DAG from GitHub push", dag_id=dag_id, error=str(exc))
            errors.append(_TriggerError(dag_id=dag_id, error=str(exc)))

    return GithubWebhookResponse(triggered=triggered, errors=errors)
