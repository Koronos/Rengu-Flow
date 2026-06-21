"""HTTP + WebSocket routes for the Toolbox section. Thin layer over ``toolbox.py``."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from rengu_flow.config import local_config
from rengu_flow_ui import toolbox
from rengu_flow_ui._http_util import http_errors
from rengu_flow_ui.settings import ui_token

API_PREFIX = "/api/v1"


class ToolBody(BaseModel):
    name: str = "Untitled tool"
    description: str = ""
    entrypoint: str = "run"
    requirements: list[str] = Field(default_factory=list)
    script: str = ""
    inputs: list[dict] = Field(default_factory=list)


class ToolUpdateBody(BaseModel):
    name: str | None = None
    description: str | None = None
    entrypoint: str | None = None
    requirements: list[str] | None = None
    script: str | None = None
    inputs: list[dict] | None = None


class RunBody(BaseModel):
    values: dict = Field(default_factory=dict)


@contextmanager
def _http_errors():
    try:
        with http_errors("Tool not found"):
            yield
    except toolbox.ExecutionDisabledError as e:
        raise HTTPException(409, str(e))
    except toolbox.RunActiveError as e:
        raise HTTPException(409, str(e))
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))


def register_toolbox_routes(app: FastAPI) -> None:
    @app.get(f"{API_PREFIX}/toolbox/enabled")
    def toolbox_enabled_route() -> dict[str, bool]:
        return {"enabled": local_config.toolbox_enabled()}

    @app.get(f"{API_PREFIX}/toolbox/tools")
    def list_toolbox_tools() -> list[dict]:
        return toolbox.list_tools()

    @app.post(f"{API_PREFIX}/toolbox/tools")
    def create_toolbox_tool(body: ToolBody) -> dict:
        with _http_errors():
            return toolbox.create_tool(
                name=body.name,
                description=body.description,
                entrypoint=body.entrypoint,
                requirements=body.requirements,
                script=body.script,
                inputs=body.inputs,
            )

    @app.get(f"{API_PREFIX}/toolbox/tools/{{tool_id}}")
    def get_toolbox_tool(tool_id: str) -> dict:
        with _http_errors():
            return toolbox.get_tool(tool_id)

    @app.put(f"{API_PREFIX}/toolbox/tools/{{tool_id}}")
    def update_toolbox_tool(tool_id: str, body: ToolUpdateBody) -> dict:
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        with _http_errors():
            return toolbox.update_tool(tool_id, **fields)

    @app.delete(f"{API_PREFIX}/toolbox/tools/{{tool_id}}")
    def delete_toolbox_tool(tool_id: str) -> dict:
        with _http_errors():
            toolbox.delete_tool(tool_id)
        return {"ok": True}

    @app.post(f"{API_PREFIX}/toolbox/tools/{{tool_id}}/run")
    def run_toolbox_tool(tool_id: str, body: RunBody) -> dict:
        with _http_errors():
            return toolbox.run_tool(tool_id, body.values)

    @app.get(f"{API_PREFIX}/toolbox/tools/{{tool_id}}/run")
    def toolbox_run_status(tool_id: str) -> dict:
        with _http_errors():
            return toolbox.run_status(tool_id)

    @app.get(f"{API_PREFIX}/toolbox/tools/{{tool_id}}/log")
    def toolbox_log(tool_id: str, offset: int = 0) -> dict:
        chunk, new_offset = toolbox.read_log(tool_id, offset)
        status = toolbox.run_status(tool_id).get("status", "idle")
        return {"chunk": chunk, "offset": new_offset, "status": status}

    @app.post(f"{API_PREFIX}/toolbox/tools/{{tool_id}}/run/cancel")
    def toolbox_cancel(tool_id: str) -> dict:
        with _http_errors():
            toolbox.cancel_run(tool_id)
        return {"ok": True}

    @app.websocket(f"{API_PREFIX}/toolbox/tools/{{tool_id}}/log/ws")
    async def toolbox_log_ws(websocket: WebSocket, tool_id: str) -> None:
        token = ui_token()
        if token:
            qs_token = websocket.query_params.get("token", "")
            if qs_token != token:
                await websocket.close(code=4401, reason="Invalid token")
                return
        await websocket.accept()
        offset = 0
        try:
            while True:
                chunk, offset = await asyncio.to_thread(toolbox.read_log, tool_id, offset)
                if chunk:
                    await websocket.send_text(chunk)
                status_dict = await asyncio.to_thread(toolbox.run_status, tool_id)
                status = status_dict.get("status", "idle")
                if status != "running":
                    chunk, offset = await asyncio.to_thread(toolbox.read_log, tool_id, offset)
                    if chunk:
                        await websocket.send_text(chunk)
                    break
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass
