"""FastAPI application for Renga Flow UI control plane."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from renga_flow_ui import configs_store, datasets_store, db, jobs, live_stream, metrics_tb, runs_scanner, signals
from renga_flow_ui.dataset_form import form_to_toml as dataset_form_to_toml
from renga_flow_ui.dataset_form import parse_toml_to_form
from renga_flow_ui.dataset_image_preview import (
    list_dataset_preview_images,
    resolve_image_token,
)
from renga_flow_ui.dataset_scan import scan_folder
from renga_flow_ui.fs_stat import stat_path
from renga_flow_ui.dataset_schema import get_dataset_schema
from renga_flow_ui.config_form import form_to_toml, toml_to_form
from renga_flow_ui.config_schema import get_schema
from renga_flow_ui.docs_reader import DocNotFoundError, DocPathError, read_doc
from renga_flow_ui import tensorboard_server
from renga_flow_ui.paths import resolve_repo_path
from renga_flow_ui.system_stats import collect_system_stats
from renga_flow_ui.settings import (
    ensure_data_dirs,
    logs_dir,
    repo_root,
    ui_host,
    ui_port,
    ui_token,
    web_dist_dir,
)

API_PREFIX = "/api/v1"


class ConfigCreate(BaseModel):
    content: str


class ConfigUpdate(BaseModel):
    content: str


class ConfigExportBody(BaseModel):
    content: str
    name: str | None = None


class ValidateBody(BaseModel):
    content: str | None = None
    config_id: str | None = None


class RunStart(BaseModel):
    config_id: int | str | None = None
    content: str | None = None
    num_gpus: int = Field(default=1, ge=1)
    resume_from: str | None = None
    output_dir: str | None = None
    extra_args: str = ""
    reset_dataloader: bool = False
    reset_optimizer: bool = False
    enqueue: bool = True
    start_immediately: bool = False
    source_run_dir: str | None = None


class JobUpdate(BaseModel):
    config_id: int | str | None = None
    num_gpus: int | None = Field(default=None, ge=1)
    resume_from: str | None = None
    output_dir: str | None = None
    extra_args: str | None = None
    reset_dataloader: bool | None = None
    reset_optimizer: bool | None = None


class JobImportPreviewBody(BaseModel):
    run_path: str


class JobImportBody(BaseModel):
    run_path: str
    import_config: bool = True
    config_id: str | None = None
    import_dataset: bool = True
    dataset_id: str | None = None
    allow_duplicate: bool = False


class ContinueRunBody(BaseModel):
    """Resume a run folder with an edited TOML (e.g. more epochs)."""

    run_path: str
    content: str
    config_id: str | None = None
    save_to_library: bool = False
    num_gpus: int = Field(default=1, ge=1)
    extra_args: str = ""
    reset_dataloader: bool = False
    reset_optimizer: bool = False
    enqueue: bool = True
    start_immediately: bool = False


class SignalBody(BaseModel):
    type: str


class TensorboardStartBody(BaseModel):
    output_dir: str = "output"
    port: int | None = Field(default=None, ge=1, le=65535)
    host: str | None = None


class TomlParseBody(BaseModel):
    content: str


class TomlRenderBody(BaseModel):
    form: dict[str, Any]
    base_content: str | None = None


class RegistryProbeBody(BaseModel):
    optimizer: str | None = None
    scheduler: str | None = None


class DatasetCreate(BaseModel):
    content: str
    name: str | None = None


class DatasetUpdate(BaseModel):
    content: str
    name: str | None = None


class DatasetComposeBody(BaseModel):
    source_ids: list[int]


class DatasetScanBody(BaseModel):
    path: str


class FsStatBody(BaseModel):
    path: str
    expect: str | None = Field(default=None, description="file | dir")


class DatasetPreviewImagesBody(BaseModel):
    content: str
    directory_index: int | None = None
    limit: int = Field(default=24, ge=1, le=48)
    offset: int = Field(default=0, ge=0)


def create_app() -> FastAPI:
    ensure_data_dirs()
    db.init_db()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        tensorboard_server.stop_tensorboard()

    app = FastAPI(title="Renga Flow UI", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://{ui_host()}:{ui_port()}",
            f"http://127.0.0.1:{ui_port()}",
            f"http://localhost:{ui_port()}",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def check_token(request: Request, call_next):
        token = ui_token()
        if token and request.url.path.startswith(API_PREFIX):
            auth = request.headers.get("X-Renga-Flow-Token") or request.headers.get(
                "Authorization", ""
            ).removeprefix("Bearer ").strip()
            if auth != token:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
        return await call_next(request)

    # --- Configs ---
    @app.get(f"{API_PREFIX}/configs")
    def list_configs(
        q: str | None = None,
        page: int | None = Query(None, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        sort: str = Query("id", description="id | name | created_at | updated_at"),
        order: str = Query("desc", description="asc | desc"),
    ) -> dict[str, Any]:
        if page is not None:
            return configs_store.search_configs_page(
                q or "", page=page, page_size=page_size, sort=sort, order=order
            )
        return {"configs": configs_store.list_configs_summary(sort=sort, order=order)}

    @app.get(f"{API_PREFIX}/configs/{{config_id}}")
    def get_config(config_id: str) -> dict[str, str]:
        try:
            return {"id": config_id, "content": configs_store.read_config_text(config_id)}
        except FileNotFoundError:
            raise HTTPException(404, "Config not found")

    def _training_export_response(content: str, bundle_stem: str) -> Response:
        from renga_flow_ui.training_export import build_training_export_zip

        try:
            zip_bytes, filename = build_training_export_zip(content, bundle_stem=bundle_stem)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get(f"{API_PREFIX}/configs/{{config_id}}/export")
    def export_config(config_id: str) -> Response:
        try:
            content = configs_store.read_config_text(config_id)
        except FileNotFoundError:
            raise HTTPException(404, "Config not found")
        return _training_export_response(content, bundle_stem=str(config_id))

    @app.post(f"{API_PREFIX}/configs/export-bundle")
    def export_config_bundle(body: ConfigExportBody) -> Response:
        stem = (body.name or "training_export").strip() or "training_export"
        return _training_export_response(body.content, bundle_stem=stem)

    @app.post(f"{API_PREFIX}/configs")
    def post_config(body: ConfigCreate) -> dict[str, int]:
        cid = configs_store.create_config(body.content)
        return {"id": cid}

    @app.post(f"{API_PREFIX}/configs/import")
    def import_config(body: ConfigCreate) -> dict[str, int]:
        cid = configs_store.create_config(body.content)
        return {"id": cid}

    @app.put(f"{API_PREFIX}/configs/{{config_id}}")
    def put_config(config_id: str, body: ConfigUpdate) -> dict[str, int]:
        if not configs_store.config_exists(config_id):
            raise HTTPException(404, "Config not found")
        cid = configs_store.update_config_text(config_id, body.content)
        return {"id": cid}

    @app.delete(f"{API_PREFIX}/configs/{{config_id}}")
    def delete_config(config_id: str) -> dict[str, bool]:
        try:
            configs_store.delete_config(config_id)
        except FileNotFoundError:
            raise HTTPException(404, "Config not found")
        return {"ok": True}

    @app.post(f"{API_PREFIX}/configs/{{config_id}}/duplicate")
    def duplicate_config_route(config_id: str) -> dict[str, int]:
        try:
            nid = configs_store.duplicate_config(config_id)
        except FileNotFoundError:
            raise HTTPException(404, "Config not found")
        return {"id": nid}

    @app.post(f"{API_PREFIX}/configs/{{config_id}}/validate")
    def validate_config_id(config_id: str) -> dict[str, Any]:
        try:
            text = configs_store.read_config_text(config_id)
        except FileNotFoundError:
            raise HTTPException(404, "Config not found")
        return configs_store.validate_toml_text(text)

    @app.post(f"{API_PREFIX}/validate")
    def validate_inline(body: ValidateBody) -> dict[str, Any]:
        if body.content is not None:
            return configs_store.validate_toml_text(body.content)
        if body.config_id:
            return validate_config_id(body.config_id)
        raise HTTPException(400, "Provide content or config_id")

    @app.post(f"{API_PREFIX}/validate-only")
    def validate_only_run(body: ValidateBody) -> dict[str, Any]:
        path: Path | None = None
        if body.config_id:
            path = configs_store.write_config_temp_for_validate(body.config_id)
        elif body.content:
            import tempfile

            tmp = Path(tempfile.mkdtemp(dir=configs_store.staging_dir()))
            path = tmp / "validate.toml"
            path.write_text(body.content, encoding="utf-8")
        else:
            raise HTTPException(400, "Provide content or config_id")
        cmd = [sys.executable, "-m", "renga_flow.main", "--config", str(path), "--validate-only"]
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    # --- Datasets library ---
    @app.get(f"{API_PREFIX}/datasets")
    def list_datasets(
        q: str | None = None,
        page: int | None = Query(None, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        sort: str = Query("id", description="id | name | created_at | updated_at"),
        order: str = Query("desc", description="asc | desc"),
    ) -> dict[str, Any]:
        if page is not None:
            result = datasets_store.search_datasets_page(
                q or "", page=page, page_size=page_size, sort=sort, order=order
            )
            for row in result["items"]:
                row["dataset_ref"] = datasets_store.dataset_library_ref(row["id"])
            return result
        items = []
        for row in datasets_store.list_datasets_summary():
            did = row["id"]
            try:
                text = datasets_store.read_dataset_text(did)
                val = datasets_store.validate_dataset_text(text)
                preview = val.get("preview") if val.get("ok") else None
            except Exception:
                preview = None
            items.append(
                {
                    **row,
                    "dataset_ref": datasets_store.dataset_library_ref(did),
                    "preview": preview,
                }
            )
        return {"datasets": items, "picker": datasets_store.list_for_training_picker()}

    @app.get(f"{API_PREFIX}/datasets/schema")
    def dataset_schema() -> dict[str, Any]:
        return get_dataset_schema()

    @app.get(f"{API_PREFIX}/augmentations")
    def augmentation_catalog() -> dict[str, Any]:
        from renga_flow.data.augmentation.ui_schema import get_augmentation_catalog

        return get_augmentation_catalog()

    @app.get(f"{API_PREFIX}/datasets/folder-suggestions")
    def dataset_folder_suggestions(
        exclude: str | None = Query(None, description="Dataset id to omit when collecting paths"),
    ) -> dict[str, Any]:
        from renga_flow_ui.dataset_folder_suggestions import collect_folder_suggestions

        return collect_folder_suggestions(exclude_dataset_id=exclude or None)

    @app.get(f"{API_PREFIX}/datasets/preview-image")
    def dataset_preview_image(t: str = Query(..., description="Signed preview token")):
        try:
            path = resolve_image_token(t)
        except ValueError as e:
            raise HTTPException(400, str(e))
        media = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".jpe": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media)

    @app.get(f"{API_PREFIX}/datasets/{{dataset_id}}")
    def get_dataset(dataset_id: str) -> dict[str, Any]:
        try:
            return datasets_store.read_dataset_for_ui(dataset_id)
        except FileNotFoundError:
            raise HTTPException(404, "Dataset not found")

    @app.get(f"{API_PREFIX}/datasets/{{dataset_id}}/export")
    def export_dataset(dataset_id: str) -> dict[str, str]:
        from renga_flow_ui.training_export import export_dataset_toml_text

        try:
            content = datasets_store.read_dataset_text(dataset_id)
        except FileNotFoundError:
            raise HTTPException(404, "Dataset not found")
        return {
            "id": dataset_id,
            "content": export_dataset_toml_text(content),
            "filename": f"dataset_{dataset_id}.toml",
        }

    @app.post(f"{API_PREFIX}/datasets")
    def post_dataset(body: DatasetCreate) -> dict[str, Any]:
        cid = datasets_store.create_dataset(body.content, name=body.name)
        row = datasets_store.read_dataset_for_ui(cid)
        return {
            "id": cid,
            "name": row["name"],
            "dataset_ref": datasets_store.dataset_library_ref(cid),
        }

    @app.post(f"{API_PREFIX}/datasets/import")
    def import_dataset(body: DatasetCreate) -> dict[str, Any]:
        cid = datasets_store.create_dataset(body.content, name=body.name)
        row = datasets_store.read_dataset_for_ui(cid)
        return {
            "id": cid,
            "name": row["name"],
            "dataset_ref": datasets_store.dataset_library_ref(cid),
        }

    @app.put(f"{API_PREFIX}/datasets/{{dataset_id}}")
    def put_dataset(dataset_id: str, body: DatasetUpdate) -> dict[str, Any]:
        if not datasets_store.dataset_exists(dataset_id):
            raise HTTPException(404, "Dataset not found")
        did = datasets_store.update_dataset_text(
            dataset_id, body.content, name=body.name
        )
        row = datasets_store.read_dataset_for_ui(did)
        return {"id": did, "name": row["name"]}

    @app.delete(f"{API_PREFIX}/datasets/{{dataset_id}}")
    def delete_dataset(dataset_id: str) -> dict[str, bool]:
        try:
            datasets_store.delete_dataset(dataset_id)
        except FileNotFoundError:
            raise HTTPException(404, "Dataset not found")
        return {"ok": True}

    @app.post(f"{API_PREFIX}/datasets/{{dataset_id}}/duplicate")
    def duplicate_dataset_route(dataset_id: str) -> dict[str, int]:
        try:
            nid = datasets_store.duplicate_dataset(dataset_id)
        except FileNotFoundError:
            raise HTTPException(404, "Dataset not found")
        return {"id": nid}

    @app.post(f"{API_PREFIX}/datasets/{{dataset_id}}/validate")
    def validate_dataset_id(dataset_id: str) -> dict[str, Any]:
        try:
            text = datasets_store.read_dataset_text(dataset_id)
        except FileNotFoundError:
            raise HTTPException(404, "Dataset not found")
        return datasets_store.validate_dataset_text(text)

    @app.post(f"{API_PREFIX}/datasets/validate")
    def validate_dataset_inline(body: ValidateBody) -> dict[str, Any]:
        if body.content is None:
            raise HTTPException(400, "Provide content")
        return datasets_store.validate_dataset_text(body.content)

    @app.post(f"{API_PREFIX}/datasets/parse-toml")
    def dataset_parse(body: TomlParseBody) -> dict[str, Any]:
        from renga_flow_ui.dataset_schema import get_dataset_schema

        schema = get_dataset_schema()
        form, ui_notes = parse_toml_to_form(body.content, schema)
        return {"ok": True, "form": form, "ui_notes": ui_notes}

    @app.post(f"{API_PREFIX}/datasets/render-toml")
    def dataset_render(body: TomlRenderBody) -> dict[str, Any]:
        try:
            content = dataset_form_to_toml(body.form)
            return {"ok": True, "content": content}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post(f"{API_PREFIX}/datasets/preview")
    def dataset_preview(body: ValidateBody) -> dict[str, Any]:
        if body.content is None:
            raise HTTPException(400, "Provide content")
        val = datasets_store.validate_dataset_text(body.content)
        if not val.get("ok"):
            return val
        return {"ok": True, "preview": val["preview"]}

    @app.post(f"{API_PREFIX}/fs/stat")
    def fs_stat_post(body: FsStatBody) -> dict[str, Any]:
        expect = body.expect if body.expect in ("file", "dir") else None
        return stat_path(body.path, expect=expect)

    @app.get(f"{API_PREFIX}/fs/stat")
    def fs_stat_get(
        path: str = Query(...),
        expect: str | None = Query(None, description="file | dir"),
    ) -> dict[str, Any]:
        kind = expect if expect in ("file", "dir") else None
        return stat_path(path, expect=kind)

    @app.post(f"{API_PREFIX}/datasets/scan-path")
    def dataset_scan_path(body: DatasetScanBody) -> dict[str, Any]:
        from pathlib import Path

        from renga_flow_ui.dataset_image_preview import issue_image_token

        from renga_flow_ui.dataset_scan import IMAGE_EXTENSIONS

        result = scan_folder(body.path)
        if result.get("ok") and result.get("sample_files"):
            root = Path(result["path"])
            tokens: list[str] = []
            for name in result["sample_files"]:
                if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                tokens.append(issue_image_token(0, name, root))
                if len(tokens) >= 4:
                    break
            if tokens:
                result["preview_tokens"] = tokens
                result["preview_token"] = tokens[0]
        return result

    @app.post(f"{API_PREFIX}/datasets/preview-images")
    def dataset_preview_images(body: DatasetPreviewImagesBody) -> dict[str, Any]:
        return list_dataset_preview_images(
            body.content,
            directory_index=body.directory_index,
            limit=body.limit,
            offset=body.offset,
        )

    @app.post(f"{API_PREFIX}/datasets/compose")
    def dataset_compose(body: DatasetComposeBody) -> dict[str, Any]:
        try:
            tid = datasets_store.compose_datasets(body.source_ids)
            return {
                "id": tid,
                "dataset_ref": datasets_store.dataset_library_ref(tid),
                "preview": datasets_store.validate_dataset_text(
                    datasets_store.read_dataset_text(tid)
                ).get("preview"),
            }
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))

    @app.post(f"{API_PREFIX}/datasets/import-example")
    def import_dataset_example(
        path: str = Query(..., description="Example path under repo"),
    ) -> dict[str, Any]:
        src = repo_root() / path
        if not src.is_file():
            raise HTTPException(404, "Example file not found")
        cid = datasets_store.import_example(src)
        return {"id": cid, "dataset_ref": datasets_store.dataset_library_ref(cid)}

    @app.post(f"{API_PREFIX}/configs/import-example")
    def import_config_example(
        path: str = Query(...),
    ) -> dict[str, int]:
        src = Path(path)
        if not src.is_file():
            src = repo_root() / path
        if not src.is_file():
            raise HTTPException(404, "Example file not found")
        cid = configs_store.import_example(src)
        return {"id": cid}

    # --- Train hub (unified jobs + disk runs) ---
    @app.get(f"{API_PREFIX}/train/runs")
    def list_train_runs(
        q: str | None = None,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        include_disk: bool = Query(True),
        output_dir: str = Query("output"),
        state: str | None = Query(None, description="active | queued | finished | disk"),
    ) -> dict[str, Any]:
        from renga_flow_ui import training_hub

        return training_hub.list_training_runs(
            q=q or "",
            page=page,
            page_size=page_size,
            include_disk=include_disk,
            output_dir=output_dir,
            state_filter=state,
        )

    @app.get(f"{API_PREFIX}/train/active")
    def train_active() -> dict[str, Any]:
        from renga_flow_ui import training_hub

        active = training_hub.get_active_training_run()
        return {"active": active}

    @app.get(f"{API_PREFIX}/train/preview-image")
    def train_preview_image(
        run_dir: str = Query(...),
        name: str = Query(...),
    ):
        from renga_flow_ui import training_hub

        try:
            path = training_hub.resolve_preview_image(run_dir, name)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        media = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media)

    # --- Jobs / runs ---
    @app.get(f"{API_PREFIX}/jobs")
    def list_jobs() -> dict[str, Any]:
        from renga_flow_ui import job_queue

        job_list = job_queue.list_jobs_sorted()
        running = sum(1 for j in job_list if j.state in ("running", "stopping"))
        pending = sum(1 for j in job_list if j.state == "pending")
        return {
            "jobs": [_job_dict(j) for j in job_list],
            "stats": {"running": running, "pending": pending},
        }

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            j = jobs.poll_job(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found")
        return _job_dict(j)

    @app.get(f"{API_PREFIX}/jobs/import/candidates")
    def list_import_candidates(output_dir: str = "output") -> dict[str, Any]:
        root = resolve_repo_path(output_dir)
        runs = runs_scanner.scan_output_runs(root)
        imported_dirs = {
            j.run_dir
            for j in db.list_jobs(limit=500)
            if j.run_dir
        }
        for r in runs:
            r["already_imported"] = r["path"] in imported_dirs
        return {"output_dir": str(root.resolve()), "runs": runs}

    @app.post(f"{API_PREFIX}/jobs/import/preview")
    def preview_job_import(body: JobImportPreviewBody) -> dict[str, Any]:
        from renga_flow_ui import job_import

        try:
            return job_import.preview_import(body.run_path)
        except job_import.JobImportError as e:
            raise HTTPException(400, str(e))

    @app.post(f"{API_PREFIX}/jobs/import")
    def import_job_from_run(body: JobImportBody) -> dict[str, Any]:
        from renga_flow_ui import job_import

        try:
            job = job_import.import_run(
                body.run_path,
                import_config=body.import_config,
                config_id=body.config_id,
                import_dataset=body.import_dataset,
                dataset_id=body.dataset_id,
                allow_duplicate=body.allow_duplicate,
            )
        except job_import.JobImportError as e:
            raise HTTPException(400, str(e))
        return _job_dict(job)

    @app.get(f"{API_PREFIX}/runs/config")
    def get_run_config(run_path: str = Query(..., description="Run folder path")) -> dict[str, Any]:
        from renga_flow_ui.run_config import RunConfigError, describe_run_config

        try:
            return describe_run_config(run_path)
        except RunConfigError as e:
            raise HTTPException(400, str(e))

    @app.post(f"{API_PREFIX}/jobs/continue-run")
    def continue_run_job(body: ContinueRunBody) -> dict[str, Any]:
        from renga_flow_ui import job_queue
        from renga_flow_ui.run_config import RunConfigError

        try:
            job = job_queue.enqueue_continue_run(
                body.run_path,
                body.content,
                config_id=body.config_id,
                save_to_library=body.save_to_library,
                num_gpus=body.num_gpus,
                extra_args=body.extra_args,
                reset_dataloader=body.reset_dataloader,
                reset_optimizer=body.reset_optimizer,
                enqueue=body.enqueue,
                start_immediately=body.start_immediately,
            )
        except RunConfigError as e:
            raise HTTPException(400, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return _job_dict(job)

    @app.post(f"{API_PREFIX}/jobs")
    def start_job(body: RunStart) -> dict[str, Any]:
        from renga_flow_ui import job_queue

        if not body.config_id and not body.content and not body.source_run_dir:
            raise HTTPException(400, "Provide config_id, content, or source_run_dir")
        if body.config_id:
            try:
                configs_store.read_config_text(body.config_id)
            except FileNotFoundError:
                raise HTTPException(404, "Config not found")

        kwargs = dict(
            config_id=body.config_id,
            content=body.content,
            num_gpus=body.num_gpus,
            resume_from=body.resume_from,
            output_dir=body.output_dir,
            extra_args=body.extra_args,
            reset_dataloader=body.reset_dataloader,
            reset_optimizer=body.reset_optimizer,
            source_run_dir=body.source_run_dir,
        )
        try:
            if body.start_immediately or not body.enqueue:
                job = job_queue.start_job_immediately(**kwargs)
            else:
                job = job_queue.enqueue_job(**kwargs)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return _job_dict(job)

    @app.patch(f"{API_PREFIX}/jobs/{{job_id}}")
    def patch_job(job_id: str, body: JobUpdate) -> dict[str, Any]:
        from renga_flow_ui import job_queue

        try:
            job = job_queue.update_pending_job(
                job_id,
                config_id=body.config_id,
                num_gpus=body.num_gpus,
                resume_from=body.resume_from,
                output_dir=body.output_dir,
                extra_args=body.extra_args,
                reset_dataloader=body.reset_dataloader,
                reset_optimizer=body.reset_optimizer,
            )
        except KeyError:
            raise HTTPException(404, "Job not found")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return _job_dict(job)

    @app.delete(f"{API_PREFIX}/jobs/{{job_id}}")
    def delete_job(job_id: str) -> dict[str, str]:
        from renga_flow_ui import job_queue

        try:
            job_queue.delete_pending_job(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": "deleted"}

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/queue/move")
    def move_job_in_queue(job_id: str, direction: str = Query(..., pattern="^(up|down)$")) -> dict[str, Any]:
        from renga_flow_ui import job_queue

        try:
            job = job_queue.move_queue(job_id, direction)
        except KeyError:
            raise HTTPException(404, "Job not found")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return _job_dict(job)

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/queue/start-now")
    def bump_job_to_run(job_id: str) -> dict[str, Any]:
        from renga_flow_ui import job_queue

        try:
            job = db.get_job(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found")
        if job.state != "pending":
            raise HTTPException(400, "Only pending jobs can be moved to front")
        job_queue.bump_pending_after(job_id)
        started = job_queue.try_start_next()
        return _job_dict(started or db.get_job(job_id))

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/stop")
    def stop_job(job_id: str) -> dict[str, Any]:
        try:
            jobs.stop_job(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found")
        return _job_dict(jobs.poll_job(job_id))

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}/logs")
    def get_logs(job_id: str, offset: int = 0) -> dict[str, Any]:
        try:
            chunk, new_offset = jobs.tail_log(job_id, offset)
        except KeyError:
            raise HTTPException(404, "Job not found")
        return {"chunk": chunk, "offset": new_offset}

    @app.websocket(f"{API_PREFIX}/jobs/{{job_id}}/logs/ws")
    async def logs_ws(websocket: WebSocket, job_id: str):
        await websocket.accept()
        offset = 0
        try:
            while True:
                try:
                    chunk, offset = jobs.tail_log(job_id, offset)
                    if chunk:
                        await websocket.send_text(chunk)
                except KeyError:
                    await websocket.send_text("[error] job not found\n")
                    break
                jobs.poll_job(job_id)
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass

    @app.websocket(f"{API_PREFIX}/jobs/{{job_id}}/live/ws")
    async def job_live_ws(websocket: WebSocket, job_id: str):
        await websocket.accept()

        async def send_json(payload: dict[str, Any]) -> None:
            await websocket.send_text(json.dumps(payload, default=str))

        try:
            await live_stream.run_job_live_ws(send_json, job_id)
        except WebSocketDisconnect:
            pass

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/signals")
    def job_signal(job_id: str, body: SignalBody) -> dict[str, str]:
        job = db.get_job(job_id)
        run_dir = job.run_dir
        if not run_dir and job.output_dir:
            scanned = runs_scanner.scan_output_runs(job.output_dir)
            if scanned:
                run_dir = scanned[-1]["path"]
                db.update_job(job_id, run_dir=run_dir)
        if not run_dir:
            raise HTTPException(400, "Run directory unknown; wait for training to create output folder")
        try:
            path = signals.send_signal(run_dir, body.type)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        return {"path": path}

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}/metrics")
    def job_metrics(job_id: str) -> dict[str, Any]:
        job = db.get_job(job_id)
        run_dir = job.run_dir
        if not run_dir and job.output_dir:
            scanned = runs_scanner.scan_output_runs(job.output_dir)
            if scanned:
                run_dir = scanned[-1]["path"]
        if not run_dir:
            return {"scalars": {}}
        return {"scalars": metrics_tb.read_scalars(run_dir)}

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}/artifacts")
    def job_artifacts(job_id: str) -> dict[str, Any]:
        job = db.get_job(job_id)
        if not job.run_dir:
            if job.output_dir:
                scanned = runs_scanner.scan_output_runs(job.output_dir)
                if scanned:
                    return {"artifacts": scanned[-1]["artifacts"]}
            return {"artifacts": []}
        return {"artifacts": runs_scanner.describe_run_dir(Path(job.run_dir))["artifacts"]}

    # --- Filesystem runs ---
    @app.get(f"{API_PREFIX}/runs")
    def list_fs_runs(output_dir: str = "output") -> dict[str, Any]:
        return {"runs": runs_scanner.scan_output_runs(resolve_repo_path(output_dir))}

    @app.get(f"{API_PREFIX}/runs/discover")
    def discover_run(output_dir: str = "output") -> dict[str, Any]:
        runs = runs_scanner.scan_output_runs(resolve_repo_path(output_dir))
        if not runs:
            return {"run": None}
        return {"run": runs[-1]}

    @app.get(f"{API_PREFIX}/runs/{{run_name}}")
    def get_fs_run(run_name: str, output_dir: str = "output") -> dict[str, Any]:
        path = resolve_repo_path(output_dir) / run_name
        if not path.is_dir():
            raise HTTPException(404, "Run not found")
        return runs_scanner.describe_run_dir(path)

    @app.post(f"{API_PREFIX}/runs/{{run_name}}/signals")
    def fs_run_signal(run_name: str, body: SignalBody, output_dir: str = "output") -> dict[str, str]:
        run_dir = resolve_repo_path(output_dir) / run_name
        try:
            path = signals.send_signal(run_dir, body.type)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        return {"path": path}

    @app.get(f"{API_PREFIX}/runs/{{run_name}}/metrics")
    def fs_run_metrics(run_name: str, output_dir: str = "output") -> dict[str, Any]:
        run_dir = resolve_repo_path(output_dir) / run_name
        if not run_dir.is_dir():
            raise HTTPException(404, "Run not found")
        return {"scalars": metrics_tb.read_scalars(run_dir)}

    @app.get(f"{API_PREFIX}/tensorboard/status")
    def tensorboard_status() -> dict[str, Any]:
        return tensorboard_server.tensorboard_status()

    @app.post(f"{API_PREFIX}/tensorboard/start")
    async def tensorboard_start(body: TensorboardStartBody) -> dict[str, Any]:
        try:
            return await run_in_threadpool(
                lambda: tensorboard_server.start_tensorboard(
                    body.output_dir,
                    port=body.port,
                    host=body.host,
                )
            )
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except RuntimeError as e:
            raise HTTPException(500, str(e))

    @app.post(f"{API_PREFIX}/tensorboard/stop")
    def tensorboard_stop() -> dict[str, Any]:
        return tensorboard_server.stop_tensorboard()

    @app.get(f"{API_PREFIX}/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(f"{API_PREFIX}/system/stats")
    def system_stats() -> dict[str, Any]:
        return collect_system_stats()

    @app.get(f"{API_PREFIX}/schema")
    def config_schema() -> dict[str, Any]:
        return get_schema()

    @app.post(f"{API_PREFIX}/registry/probe")
    def registry_probe(body: RegistryProbeBody) -> dict[str, Any]:
        from renga_flow_ui.registry_probe import probe_optimizer, probe_scheduler

        out: dict[str, Any] = {}
        if body.optimizer is not None:
            out["optimizer"] = probe_optimizer(body.optimizer)
        if body.scheduler is not None:
            out["scheduler"] = probe_scheduler(body.scheduler)
        if not out:
            raise HTTPException(400, "Provide optimizer and/or scheduler")
        return out

    @app.get(f"{API_PREFIX}/docs/index")
    def get_docs_index() -> dict[str, list[dict[str, str]]]:
        from renga_flow_ui.docs_reader import list_docs_index

        return {"items": list_docs_index()}

    @app.get(f"{API_PREFIX}/docs")
    def get_doc(path: str = Query(..., description="Path under docs/, e.g. docs/user/web-ui.md")) -> dict[str, str]:
        try:
            return read_doc(path)
        except DocNotFoundError:
            raise HTTPException(404, "Documentation file not found")
        except DocPathError as e:
            raise HTTPException(400, str(e))

    @app.post(f"{API_PREFIX}/configs/parse-toml")
    def parse_toml_endpoint(payload: TomlParseBody) -> dict[str, Any]:
        try:
            from renga_flow_ui.config_form import form_values_for_ui
            from renga_flow_ui.config_schema import get_schema

            form = form_values_for_ui(toml_to_form(payload.content), get_schema())
            return {"ok": True, "form": form}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post(f"{API_PREFIX}/configs/render-toml")
    def render_toml_endpoint(payload: TomlRenderBody) -> dict[str, Any]:
        try:
            content = form_to_toml(payload.form, base_content=payload.base_content)
            return {"ok": True, "content": content}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    dist = web_dist_dir()
    if dist.is_dir() and (dist / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            if full_path.startswith("api"):
                raise HTTPException(404)
            target = dist / full_path
            if full_path and target.is_file():
                return FileResponse(target)
            return FileResponse(dist / "index.html")

    return app


def _job_dict(job: db.JobRecord) -> dict[str, Any]:
    return {
        "id": job.id,
        "config_id": job.config_id,
        "config_path": job.config_path,
        "state": job.state,
        "pid": job.pid,
        "run_dir": job.run_dir,
        "output_dir": job.output_dir,
        "num_gpus": job.num_gpus,
        "resume_from": job.resume_from,
        "log_path": job.log_path,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "exit_code": job.exit_code,
        "extra_args": job.extra_args,
        "queue_position": job.queue_position,
        "source_run_dir": job.source_run_dir,
    }
