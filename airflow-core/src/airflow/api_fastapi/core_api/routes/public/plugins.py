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

import os

import structlog
from fastapi import Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import ValidationError

from airflow import plugins_manager, settings
from airflow.api_fastapi.auth.managers.models.resource_details import AccessView
from airflow.api_fastapi.common.parameters import QueryLimit, QueryOffset
from airflow.api_fastapi.common.router import AirflowRouter
from airflow.api_fastapi.core_api.datamodels.plugins import (
    PluginCollectionResponse,
    PluginImportErrorCollectionResponse,
    PluginResponse,
)
from airflow.api_fastapi.core_api.openapi.exceptions import create_openapi_http_exception_doc
from airflow.api_fastapi.core_api.security import requires_access_view

logger = structlog.get_logger(__name__)

plugins_router = AirflowRouter(tags=["Plugin"], prefix="/plugins")


@plugins_router.get(
    "",
    dependencies=[Depends(requires_access_view(AccessView.PLUGINS))],
)
def get_plugins(
    limit: QueryLimit,
    offset: QueryOffset,
) -> PluginCollectionResponse:
    plugins_info = sorted(plugins_manager.get_plugin_info(), key=lambda x: x["name"])
    valid_plugins: list[PluginResponse] = []
    for plugin_dict in plugins_info:
        try:
            # Validate each plugin individually
            plugin = PluginResponse.model_validate(plugin_dict)
            valid_plugins.append(plugin)
        except ValidationError as e:
            logger.warning(
                "Skipping invalid plugin due to error",
                plugin_name=plugin_dict.get("name", "<unknown>"),
                error=str(e),
            )
            continue

    offset_value = offset.value or 0
    limit_value = limit.value if limit.value is not None else len(valid_plugins)

    paginated_plugins = valid_plugins[offset_value : offset_value + limit_value]
    return PluginCollectionResponse(
        plugins=paginated_plugins,
        total_entries=len(valid_plugins),
    )


@plugins_router.get(
    "/importErrors",
    dependencies=[Depends(requires_access_view(AccessView.PLUGINS))],
)
def import_errors() -> PluginImportErrorCollectionResponse:
    import_errors = plugins_manager.get_import_errors()
    return PluginImportErrorCollectionResponse.model_validate(
        {
            "import_errors": [{"source": source, "error": error} for source, error in import_errors.items()],
            "total_entries": len(import_errors),
        }
    )


@plugins_router.get(
    "/{plugin_name}/files",
    responses=create_openapi_http_exception_doc(
        [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
        ]
    ),
    dependencies=[Depends(requires_access_view(AccessView.PLUGINS))],
)
def get_plugin_file(
    plugin_name: str,
    file_path: str = Query(..., description="Relative path to the file within the plugin's directory."),
) -> FileResponse:
    """Serve a static file from a plugin's directory.

    Files are resolved relative to a per-plugin subdirectory inside the global
    plugins folder (``[core] plugins_folder``).  Plugin authors place bundled
    assets — reference docs, JSON schemas, etc. — in that subdirectory and
    expose them through this endpoint.
    """
    known_names = {p["name"] for p in plugins_manager.get_plugin_info()}
    if plugin_name not in known_names:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"Plugin '{plugin_name}' not found",
        )

    plugins_root = os.path.realpath(settings.PLUGINS_FOLDER)
    plugin_dir = os.path.realpath(os.path.join(plugins_root, plugin_name))

    # Reject plugin_name values that resolve outside the plugins root (e.g. "../other").
    if not plugin_dir.startswith(plugins_root + os.sep):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Invalid plugin name",
        )

    target = os.path.realpath(os.path.join(plugin_dir, file_path))

    # Reject file_path values that escape the plugin's directory (e.g. "../../etc/passwd").
    if not target.startswith(plugin_dir + os.sep):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Invalid file_path: must stay within the plugin's directory",
        )

    if not os.path.isfile(target):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"File '{file_path}' not found in plugin '{plugin_name}'",
        )

    return FileResponse(target)
