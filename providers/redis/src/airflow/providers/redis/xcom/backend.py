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

import collections
from functools import cache
from typing import TYPE_CHECKING, Any

from airflow.providers.common.compat.sdk import conf
from airflow.providers.redis.hooks.redis import RedisHook
from airflow.providers.redis.version_compat import AIRFLOW_V_3_0_PLUS

if TYPE_CHECKING:
    from airflow.sdk.execution_time.comms import XComResult

if AIRFLOW_V_3_0_PLUS:
    from airflow.sdk.bases.xcom import BaseXCom
else:
    from airflow.models.xcom import BaseXCom  # type: ignore[no-redef]

# Prefix that distinguishes Redis-backed DB references from inline values.
_KEY_PREFIX = "xcom"

# Minimal wrapper so BaseXCom.deserialize_value can be called on a raw payload
# string fetched from Redis.  BaseXCom.deserialize_value expects result.value.
# The private _XComValueWrapper from task-sdk is not part of the public API, so
# we define an equivalent here.
_Wrapper = collections.namedtuple("_Wrapper", "value")


@cache
def _conn_id() -> str:
    """Return the Redis connection ID to use for XCom storage."""
    return conf.get("redis", "xcom_conn_id", fallback=RedisHook.default_conn_name)


@cache
def _hook() -> RedisHook:
    """Return a cached RedisHook; the underlying redis.Redis client is also lazy-cached by the hook."""
    return RedisHook(redis_conn_id=_conn_id())


def _make_redis_key(
    dag_id: str | None,
    run_id: str | None,
    task_id: str | None,
    key: str | None,
    map_index: int | None,
) -> str:
    """
    Build the Redis storage key from XCom identity fields.

    Format: ``xcom:{dag_id}:{run_id}:{task_id}:{key}:{map_index}``

    Components are joined with colons for human readability.  Because the key is
    never parsed back into its parts (it is only used for exact-match lookups and
    deletes), colons within component values (e.g. scheduled run_ids) do not
    cause ambiguity.
    """
    return ":".join(
        [
            _KEY_PREFIX,
            dag_id or "",
            run_id or "",
            task_id or "",
            key or "",
            str(map_index if map_index is not None else -1),
        ]
    )


class RedisXComBackend(BaseXCom):
    """
    XCom backend that stores serialized payloads in Redis.

    The metadata database holds only a compact key reference; the actual value
    bytes live in Redis under a composite key built from ``dag_id``, ``run_id``,
    ``task_id``, ``key``, and ``map_index``.  Arbitrary Python objects are
    supported because serialization delegates to ``BaseXCom.serialize_value``
    (Airflow's serde layer) before writing to Redis, and
    ``BaseXCom.deserialize_value`` after reading back.

    **Configuration**

    Set the connection used for XCom storage with::

        [redis]
        xcom_conn_id = redis_default   # default

    **Enabling**

    Point Airflow at this class::

        [core]
        xcom_backend = airflow.providers.redis.xcom.backend.RedisXComBackend
    """

    @staticmethod
    def serialize_value(
        value: Any,
        *,
        key: str | None = None,
        task_id: str | None = None,
        dag_id: str | None = None,
        run_id: str | None = None,
        map_index: int | None = None,
    ) -> str:
        """
        Serialize *value* into Redis and return a DB reference string.

        1. Serialize the Python object with ``BaseXCom.serialize_value`` (JSON via
           Airflow's serde) so arbitrary types round-trip correctly.
        2. Write the resulting payload to Redis under the composite key.
        3. Return ``BaseXCom.serialize_value(redis_key)`` — a JSON-encoded string
           of the Redis key — which the metadata DB stores as the XCom value.
        """
        redis_key = _make_redis_key(dag_id, run_id, task_id, key, map_index)

        payload: str = BaseXCom.serialize_value(
            value=value,
            key=key,
            task_id=task_id,
            dag_id=dag_id,
            run_id=run_id,
            map_index=map_index,
        )
        _hook().get_conn().set(redis_key, payload)

        # The DB row holds only the key reference so the metadata DB stays small.
        return BaseXCom.serialize_value(redis_key)

    @staticmethod
    def deserialize_value(result: Any) -> Any:
        """
        Fetch the payload from Redis and deserialize it.

        ``result.value`` (the DB string) is first unpacked with
        ``BaseXCom.deserialize_value`` to recover the Redis key.  If the
        recovered value is not a recognized Redis key reference (e.g. an inline
        XCom written before this backend was enabled), it is returned as-is.
        """
        redis_key = BaseXCom.deserialize_value(result)

        if not isinstance(redis_key, str) or not redis_key.startswith(_KEY_PREFIX + ":"):
            # Inline value — not stored in Redis.  Return what base deserialization gave us.
            return redis_key

        raw = _hook().get_conn().get(redis_key)
        if raw is None:
            return None

        payload = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return BaseXCom.deserialize_value(_Wrapper(payload))

    @staticmethod
    def purge(xcom: XComResult, session: Any = None) -> None:
        """
        Delete the Redis entry for *xcom*.

        Called by ``BaseXCom.delete`` after the DB row has been identified but
        before the ``DeleteXCom`` message is sent to the supervisor.
        Failures are silently swallowed so a missing Redis key never blocks row
        deletion.
        """
        try:
            redis_key = BaseXCom.deserialize_value(xcom)
        except Exception:
            return

        if isinstance(redis_key, str) and redis_key.startswith(_KEY_PREFIX + ":"):
            _hook().get_conn().delete(redis_key)
