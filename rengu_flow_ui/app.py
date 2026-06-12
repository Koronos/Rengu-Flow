"""FastAPI application for Rengu Flow UI control plane."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rengu_flow.version import package_version, version_info
from rengu_flow_ui import datasets_store, db, jobs, live_stream, metrics_tb, run_staging, runs_scanner, signals
from rengu_flow_ui.dataset_form import form_to_toml as dataset_form_to_toml
from rengu_flow_ui.dataset_form import parse_toml_to_form
from rengu_flow_ui.dataset_image_preview import (
    list_dataset_preview_images,
    resolve_image_token,
)
from rengu_flow_ui.dataset_scan import scan_folder
from rengu_flow_ui.fs_stat import stat_path
from rengu_flow_ui.dataset_schema import get_dataset_schema
from rengu_flow_ui.config_form import coerce_preview_prompts_for_toml, form_to_toml, toml_to_form
from rengu_flow_ui.config_schema import get_schema
from rengu_flow_ui.docs_reader import DocNotFoundError, DocPathError, read_doc
from rengu_flow_ui import tensorboard_server
from rengu_flow_ui.paths import PathError, resolve_example_path, resolve_repo_path
from rengu_track.system_stats import collect_system_stats
from rengu_flow_ui.settings import (
    ensure_data_dirs,
    repo_root,
    ui_host,
    ui_port,
    ui_token,
    web_dist_dir,
)

API_PREFIX = "/api/v1"

_logger = logging.getLogger("rengu_flow_ui.app")

# Suffix -> media type for served image files (dataset previews and training previews).
_IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".jpe": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


@contextmanager
def _job_http_errors():
    """Translate job-operation errors to HTTP status: missing job -> 404, bad request -> 400."""
    try:
        yield
    except KeyError:
        raise HTTPException(404, "Job not found")
    except ValueError as e:
        raise HTTPException(400, str(e))


class ConfigExportBody(BaseModel):
    content: str
    name: str | None = None


class ValidateBody(BaseModel):
    content: str | None = None


class RunStart(BaseModel):
    content: str | None = None
    num_gpus: int = Field(default=1, ge=1)
    resume_from: str | None = None
    output_dir: str | None = None
    extra_args: str = ""
    reset_dataloader: bool = False
    reset_optimizer: bool = False
    cache_only: bool = False
    trust_cache: bool = False
    regenerate_cache: bool = False
    enqueue: bool = True
    start_immediately: bool = False
    save_for_later: bool = False
    source_run_dir: str | None = None


class JobUpdate(BaseModel):
    content: str | None = None
    num_gpus: int | None = Field(default=None, ge=1)
    resume_from: str | None = None
    output_dir: str | None = None
    extra_args: str | None = None
    reset_dataloader: bool | None = None
    reset_optimizer: bool | None = None
    cache_only: bool | None = None
    trust_cache: bool | None = None
    regenerate_cache: bool | None = None


class MaintenanceResetBody(BaseModel):
    confirmation: str | None = None
    confirm: bool = False


class DepsInstallBody(BaseModel):
    profile: str
    execute: bool = False
    confirm: bool = False


class JobImportPreviewBody(BaseModel):
    run_path: str


class JobImportBody(BaseModel):
    run_path: str
    import_dataset: bool = True
    dataset_id: str | None = None
    allow_duplicate: bool = False


class ContinueRunBody(BaseModel):
    """Resume a run folder with an edited TOML (e.g. more epochs).

    When ``job_id`` is given, the existing run record is reused (edited in place and re-queued);
    otherwise a record is created for a filesystem-only run that has none yet.
    """

    run_path: str = ""
    job_id: str | None = None
    content: str
    num_gpus: int = Field(default=1, ge=1)
    extra_args: str = ""
    reset_dataloader: bool = False
    reset_optimizer: bool = False
    resume_from: str | None = None
    from_scratch: bool = False
    enqueue: bool = True
    start_immediately: bool = False


class QueueReorderBody(BaseModel):
    ids: list[int]


class SignalBody(BaseModel):
    type: str


class PreviewConfigBody(BaseModel):
    # Full replacement for the run's [preview] table (prompts, cadence, enabled, sampling).
    preview: dict[str, Any]
    # When true, also drop a `preview` signal so a sample renders immediately.
    preview_now: bool = False


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

    app = FastAPI(title="Rengu Flow UI", version=package_version(), lifespan=lifespan)

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
            auth = request.headers.get("X-Rengu-Flow-Token") or request.headers.get(
                "Authorization", ""
            ).removeprefix("Bearer ").strip()
            if auth != token:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})
        return await call_next(request)

    @app.exception_handler(Exception)
    async def _unhandled_exception(_request: Request, exc: Exception) -> JSONResponse:
        # Surface the real reason instead of a bare "Internal Server Error" so the UI can show
        # (and the user can copy/paste) an actionable message. This is a local single-user tool,
        # so echoing the exception text is acceptable. HTTPException keeps its own handler and is
        # not routed here. The full traceback still lands in the server log.
        _logger.exception("Unhandled error on %s", getattr(_request, "url", "?"))
        return JSONResponse(
            status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"}
        )

    # --- Training config TOML: validation + export (no standalone library) ---
    def _training_export_response(content: str, bundle_stem: str) -> Response:
        from rengu_flow_ui.training_export import build_training_export_zip

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

    @app.post(f"{API_PREFIX}/configs/export-bundle")
    def export_config_bundle(body: ConfigExportBody) -> Response:
        stem = (body.name or "training_export").strip() or "training_export"
        return _training_export_response(body.content, bundle_stem=stem)

    @app.post(f"{API_PREFIX}/validate")
    def validate_inline(body: ValidateBody) -> dict[str, Any]:
        if body.content is not None:
            return run_staging.validate_toml_text(body.content)
        raise HTTPException(400, "Provide content")

    @app.post(f"{API_PREFIX}/validate-only")
    def validate_only_run(body: ValidateBody) -> dict[str, Any]:
        if not body.content:
            raise HTTPException(400, "Provide content")
        import shutil
        from uuid import uuid4

        temp_id = f"validate-{uuid4().hex}"
        # Materialize staging first so dataset library refs resolve like training does.
        try:
            path = run_staging.materialize_staging(body.content, temp_id)
        except Exception as e:
            return {"ok": False, "error": str(e)}

        try:
            cmd = [
                sys.executable,
                "-m",
                "rengu_flow.main",
                "--config",
                str(path),
                "--validate-only",
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root()),
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Validation timed out after 120s"}
        finally:
            shutil.rmtree(run_staging.staging_dir() / temp_id, ignore_errors=True)

        result: dict[str, Any] = {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if proc.returncode != 0:
            result["error"] = _parse_validate_error(proc.stderr, proc.stdout)
        return result

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
        items = [
            {**row, "dataset_ref": datasets_store.dataset_library_ref(row["id"])}
            for row in datasets_store.list_datasets_summary()
        ]
        return {"datasets": items, "picker": datasets_store.list_for_training_picker()}

    @app.get(f"{API_PREFIX}/datasets/schema")
    def dataset_schema() -> dict[str, Any]:
        return get_dataset_schema()

    @app.get(f"{API_PREFIX}/augmentations")
    def augmentation_catalog() -> dict[str, Any]:
        from rengu_flow.data.augmentation.ui_schema import get_augmentation_catalog

        return get_augmentation_catalog()

    @app.get(f"{API_PREFIX}/datasets/folder-suggestions")
    def dataset_folder_suggestions(
        exclude: str | None = Query(None, description="Dataset id to omit when collecting paths"),
    ) -> dict[str, Any]:
        from rengu_flow_ui.dataset_folder_suggestions import collect_folder_suggestions

        return collect_folder_suggestions(exclude_dataset_id=exclude or None)

    @app.get(f"{API_PREFIX}/datasets/preview-image")
    def dataset_preview_image(t: str = Query(..., description="Signed preview token")):
        try:
            path = resolve_image_token(t)
        except ValueError as e:
            raise HTTPException(400, str(e))
        media = _IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media)

    @app.get(f"{API_PREFIX}/datasets/{{dataset_id}}")
    def get_dataset(dataset_id: str) -> dict[str, Any]:
        try:
            return datasets_store.read_dataset_for_ui(dataset_id)
        except FileNotFoundError:
            raise HTTPException(404, "Dataset not found")

    @app.get(f"{API_PREFIX}/datasets/{{dataset_id}}/export")
    def export_dataset(dataset_id: str) -> dict[str, str]:
        from rengu_flow_ui.training_export import export_dataset_toml_text

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
        return _created_dataset_response(body.content, body.name)

    @app.post(f"{API_PREFIX}/datasets/import")
    def import_dataset(body: DatasetCreate) -> dict[str, Any]:
        return _created_dataset_response(body.content, body.name)

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
        from rengu_flow_ui.dataset_schema import get_dataset_schema

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

        from rengu_flow_ui.dataset_image_preview import issue_image_token

        from rengu_flow_ui.dataset_scan import IMAGE_EXTENSIONS

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
        path: str = Query(..., description="Example path under repo examples/"),
    ) -> dict[str, Any]:
        try:
            src = resolve_example_path(path)
        except (PathError, FileNotFoundError):
            raise HTTPException(404, "Example file not found")
        cid = datasets_store.import_example(src)
        return {"id": cid, "dataset_ref": datasets_store.dataset_library_ref(cid)}

    # --- Train hub (unified jobs + disk runs) ---
    @app.get(f"{API_PREFIX}/train/runs")
    def list_train_runs(
        q: str | None = None,
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        state: str | None = Query(None, description="active | queued | finished"),
    ) -> dict[str, Any]:
        from rengu_flow_ui import training_hub

        return training_hub.list_training_runs(
            q=q or "",
            page=page,
            page_size=page_size,
            state_filter=state,
        )

    @app.get(f"{API_PREFIX}/train/active")
    def train_active() -> dict[str, Any]:
        from rengu_flow_ui import training_hub

        active = training_hub.get_active_training_run()
        return {"active": active}

    @app.get(f"{API_PREFIX}/train/preview-image")
    def train_preview_image(
        run_dir: str = Query(...),
        name: str = Query(...),
    ):
        from rengu_flow_ui import training_hub

        try:
            path = training_hub.resolve_preview_image(run_dir, name)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        media = _IMAGE_MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media)

    # --- Jobs / runs ---
    @app.get(f"{API_PREFIX}/jobs")
    def list_jobs() -> dict[str, Any]:
        from rengu_flow_ui import job_queue

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
        from rengu_flow_ui import training_hub

        d = _job_dict(j)
        # Live progress (step/percent/loss/ETA, and the caching phase) for the detail
        # view, derived from the latest @@RFPROG@@ marker in the log — surfaces even
        # before the run folder exists (caching).
        run_dir = training_hub.resolve_job_run_dir(j)
        marker = live_stream._latest_marker(job_id)
        d["progress"] = training_hub.compute_run_progress(run_dir, marker=marker)
        return d

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
        from rengu_flow_ui import job_import

        try:
            return job_import.preview_import(body.run_path)
        except job_import.JobImportError as e:
            raise HTTPException(400, str(e))

    @app.post(f"{API_PREFIX}/jobs/import")
    def import_job_from_run(body: JobImportBody) -> dict[str, Any]:
        from rengu_flow_ui import job_import

        try:
            job = job_import.import_run(
                body.run_path,
                import_dataset=body.import_dataset,
                dataset_id=body.dataset_id,
                allow_duplicate=body.allow_duplicate,
            )
        except job_import.JobImportError as e:
            raise HTTPException(400, str(e))
        return _job_dict(job)

    @app.get(f"{API_PREFIX}/runs/config")
    def get_run_config(run_path: str = Query(..., description="Run folder path")) -> dict[str, Any]:
        from rengu_flow_ui.run_config import RunConfigError, describe_run_config

        try:
            return describe_run_config(run_path)
        except RunConfigError as e:
            raise HTTPException(400, str(e))

    @app.post(f"{API_PREFIX}/jobs/continue-run")
    def continue_run_job(body: ContinueRunBody) -> dict[str, Any]:
        from rengu_flow_ui import job_queue
        from rengu_flow_ui.run_config import RunConfigError

        try:
            if body.job_id:
                # Reuse the existing record: edit in place and re-queue (one folder, one record).
                job = job_queue.continue_existing(
                    body.job_id,
                    content=body.content,
                    num_gpus=body.num_gpus,
                    extra_args=body.extra_args,
                    reset_dataloader=body.reset_dataloader,
                    reset_optimizer=body.reset_optimizer,
                    resume_from=body.resume_from,
                    from_scratch=body.from_scratch,
                    enqueue=body.enqueue,
                )
            else:
                # Filesystem-only run with no record yet: create one that resumes the folder.
                job = job_queue.enqueue_continue_run(
                    body.run_path,
                    body.content,
                    num_gpus=body.num_gpus,
                    extra_args=body.extra_args,
                    reset_dataloader=body.reset_dataloader,
                    reset_optimizer=body.reset_optimizer,
                    resume_from=body.resume_from,
                    from_scratch=body.from_scratch,
                    enqueue=body.enqueue,
                    start_immediately=body.start_immediately,
                )
        except KeyError:
            raise HTTPException(404, "Job not found")
        except RunConfigError as e:
            raise HTTPException(400, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        return _job_dict(job)

    @app.post(f"{API_PREFIX}/jobs")
    def start_job(body: RunStart) -> dict[str, Any]:
        from rengu_flow_ui import job_queue

        if not body.content and not body.source_run_dir:
            raise HTTPException(400, "Provide content or source_run_dir")

        kwargs = dict(
            content=body.content,
            num_gpus=body.num_gpus,
            resume_from=body.resume_from,
            output_dir=body.output_dir,
            extra_args=body.extra_args,
            reset_dataloader=body.reset_dataloader,
            reset_optimizer=body.reset_optimizer,
            cache_only=body.cache_only,
            trust_cache=body.trust_cache,
            regenerate_cache=body.regenerate_cache,
            source_run_dir=body.source_run_dir,
        )
        try:
            if body.save_for_later:
                job = job_queue.save_draft(**kwargs)
            elif body.start_immediately or not body.enqueue:
                job = job_queue.start_job_immediately(**kwargs)
            else:
                job = job_queue.enqueue_job(**kwargs)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return _job_dict(job)

    @app.patch(f"{API_PREFIX}/jobs/{{job_id}}")
    def patch_job(job_id: str, body: JobUpdate) -> dict[str, Any]:
        from rengu_flow_ui import job_queue

        with _job_http_errors():
            job = job_queue.update_pending_job(
                job_id,
                content=body.content,
                num_gpus=body.num_gpus,
                resume_from=body.resume_from,
                output_dir=body.output_dir,
                extra_args=body.extra_args,
                reset_dataloader=body.reset_dataloader,
                reset_optimizer=body.reset_optimizer,
                cache_only=body.cache_only,
                trust_cache=body.trust_cache,
                regenerate_cache=body.regenerate_cache,
            )
        return _job_dict(job)

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/enqueue")
    def enqueue_saved_job(job_id: str) -> dict[str, Any]:
        """Promote a saved (new) run into the pending queue."""
        from rengu_flow_ui import job_queue

        with _job_http_errors():
            job = job_queue.enqueue_existing(job_id)
        return _job_dict(job)

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/dequeue")
    def dequeue_job_endpoint(job_id: str) -> dict[str, Any]:
        """Remove a queued (pending) run from the queue, keeping it as a saved draft."""
        from rengu_flow_ui import job_queue

        with _job_http_errors():
            job = job_queue.dequeue_job(job_id)
        return _job_dict(job)

    @app.post(f"{API_PREFIX}/jobs/queue/reorder")
    def reorder_queue(body: QueueReorderBody) -> dict[str, Any]:
        from rengu_flow_ui import job_queue

        pending = job_queue.reorder_queue(body.ids)
        return {"queue": [_job_dict(j) for j in pending]}

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}/seed")
    def seed_job_config(job_id: str) -> dict[str, str]:
        """Config TOML for a new run cloned from this one (run_name gets a _N suffix)."""
        import toml

        try:
            job = db.get_job(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found")
        content = job.config_content or ""
        if not content.strip():
            raise HTTPException(400, "This run has no config content to copy")
        # Keep the original dataset reference: revert any per-job staging path that may
        # have leaked into config_content (e.g. from a prior continue/import).
        from rengu_flow_ui.job_import import unstage_config_dataset_refs

        content = unstage_config_dataset_refs(content, run_dir=job.run_dir)
        try:
            cfg = toml.loads(content)
            changed = False
            # A fresh run must not inherit a resume pointer from the source config, or it
            # would resume into the SOURCE run's folder instead of starting a new one.
            if cfg.pop("resume_from_checkpoint", None) is not None:
                changed = True
            base = cfg.get("run_name")
            if isinstance(base, str) and base.strip():
                cfg["run_name"] = run_staging.next_run_name(base)
                changed = True
            if changed:
                content = toml.dumps(cfg)
        except Exception:
            pass
        return {"content": content}

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}/checkpoints")
    def job_checkpoints(job_id: str) -> dict[str, Any]:
        from rengu_flow_ui import training_hub
        from rengu_flow_ui.run_config import list_checkpoints

        try:
            job = db.get_job(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found")
        run_dir = training_hub.resolve_job_run_dir(job)
        if run_dir is None:
            return {"checkpoints": [], "run_dir": None}
        return {"checkpoints": list_checkpoints(run_dir), "run_dir": str(run_dir)}

    @app.get(f"{API_PREFIX}/runs/checkpoints")
    def run_checkpoints(run_dir: str = Query(..., description="Run folder path")) -> dict[str, Any]:
        from rengu_flow_ui.run_config import list_checkpoints

        try:
            return {"checkpoints": list_checkpoints(run_dir), "run_dir": run_dir}
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.delete(f"{API_PREFIX}/jobs/{{job_id}}")
    def delete_job(job_id: str) -> dict[str, str]:
        from rengu_flow_ui import job_queue

        with _job_http_errors():
            job_queue.delete_job_record(job_id)
        return {"ok": "deleted"}

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/queue/move")
    def move_job_in_queue(job_id: str, direction: str = Query(..., pattern="^(up|down)$")) -> dict[str, Any]:
        from rengu_flow_ui import job_queue

        with _job_http_errors():
            job = job_queue.move_queue(job_id, direction)
        return _job_dict(job)

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/queue/start-now")
    def bump_job_to_run(job_id: str) -> dict[str, Any]:
        from rengu_flow_ui import job_queue

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
        # Force stop: terminate the process tree without writing a save_quit signal. A killed
        # process never consumes a signal, so writing one would only leave a stale file that
        # makes the next run reusing the folder quit on its first step. The graceful path is the
        # separate /signals {save_quit} endpoint, which lets the run checkpoint and exit itself.
        try:
            jobs.stop_job(job_id, graceful_signal=False)
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
        token = ui_token()
        if token:
            # Check token from query param (?token=...) since WS headers are unreliable
            # across browsers. Falls back to Sec-WebSocket-Protocol for clients that use it.
            qs_token = websocket.query_params.get("token", "")
            if qs_token != token:
                await websocket.close(code=4401, reason="Invalid token")
                return
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
                job = jobs.poll_job(job_id)
                if job.state not in ("running", "stopping"):
                    # Job reached a terminal state — flush any trailing output and close,
                    # rather than tailing a dead job forever until the client disconnects.
                    try:
                        chunk, offset = jobs.tail_log(job_id, offset)
                        if chunk:
                            await websocket.send_text(chunk)
                    except KeyError:
                        pass
                    break
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass

    @app.websocket(f"{API_PREFIX}/jobs/{{job_id}}/live/ws")
    async def job_live_ws(websocket: WebSocket, job_id: str):
        token = ui_token()
        if token:
            qs_token = websocket.query_params.get("token", "")
            if qs_token != token:
                await websocket.close(code=4401, reason="Invalid token")
                return
        await websocket.accept()

        async def send_json(payload: dict[str, Any]) -> None:
            await websocket.send_text(json.dumps(payload, default=str))

        try:
            await live_stream.run_job_live_ws(send_json, job_id)
        except WebSocketDisconnect:
            pass

    @app.websocket(f"{API_PREFIX}/system/stats/ws")
    async def system_stats_ws(websocket: WebSocket):
        # Global host stats (CPU/RAM/GPU) pushed on a dedicated socket so a single connection
        # replaces the per-client 2s HTTP polling of GET /system/stats.
        await websocket.accept()

        async def send_json(payload: dict[str, Any]) -> None:
            await websocket.send_text(json.dumps(payload, default=str))

        try:
            await live_stream.run_system_stats_ws(send_json)
        except WebSocketDisconnect:
            pass

    @app.get(f"{API_PREFIX}/signals")
    def list_signals() -> dict[str, Any]:
        return signals.list_signal_definitions()

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/signals")
    def job_signal(job_id: str, body: SignalBody) -> dict[str, str]:
        job = db.get_job(job_id)
        if job.state not in signals.ACTIVE_JOB_STATES:
            raise HTTPException(
                409,
                f"Job is not active (state={job.state!r}); signals only apply to running training",
            )
        from rengu_flow_ui import training_hub

        run_dir = training_hub.resolve_job_run_dir(job)
        if not run_dir:
            raise HTTPException(400, "Run directory unknown; wait for training to create output folder")
        try:
            path = signals.send_signal(run_dir, body.type)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        # A user quit (any *_quit) stops the run AND the queue: flip it to "stopping" so it
        # lands in "stopped" (not "finished"), which the queue treats as a deliberate halt
        # and does not auto-advance past. Plain save/export/preview/reload keep it running.
        if body.type in ("save_quit", "quit"):
            db.update_job(job_id, state="stopping")
        return {"path": path}

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}/preview-config")
    def get_preview_config_for_job(job_id: str) -> dict[str, Any]:
        """Current [preview] table for a job's live config (for the live editor)."""
        import toml as _toml

        from rengu_flow_ui import training_hub

        job = db.get_job(job_id)
        candidates: list[Path] = []
        if job.config_path:
            candidates.append(Path(job.config_path))
        run_dir = training_hub.resolve_job_run_dir(job)
        if run_dir:
            rc = runs_scanner.pick_main_config_path(Path(run_dir))
            if rc:
                candidates.append(rc)
        preview: dict[str, Any] = {}
        model_type = ""
        for path in candidates:
            try:
                if not path.is_file():
                    continue
                cfg = _toml.load(str(path))
                if not isinstance(cfg, dict):
                    continue
                model = cfg.get("model")
                if not model_type and isinstance(model, dict) and isinstance(model.get("type"), str):
                    model_type = model["type"]
                if isinstance(cfg.get("preview"), dict):
                    preview = cfg["preview"]
                    break
            except Exception:
                continue
        # model_type lets the live editor reuse the config form's per-prompt fields (it drives
        # which model-specific override fields are shown), so previews keep their names/overrides.
        return {
            "preview": preview,
            "model_type": model_type,
            "active": job.state in signals.ACTIVE_JOB_STATES,
        }

    @app.post(f"{API_PREFIX}/jobs/{{job_id}}/preview-config")
    def update_preview_config(job_id: str, body: PreviewConfigBody) -> dict[str, Any]:
        """Edit a running job's [preview] live: write the new table into the run's config
        files, then signal the trainer to hot-reload it. Persists for resume/continue."""
        import toml as _toml

        job = db.get_job(job_id)
        if job.state not in signals.ACTIVE_JOB_STATES:
            raise HTTPException(
                409,
                f"Job is not active (state={job.state!r}); live preview edits need a running run",
            )
        from rengu_flow_ui import training_hub

        run_dir = training_hub.resolve_job_run_dir(job)
        if not run_dir:
            raise HTTPException(400, "Run directory unknown; wait for training to create output folder")

        def _write_preview(path: Path) -> None:
            try:
                cfg = _toml.load(str(path)) if path.is_file() else {}
            except Exception:
                cfg = {}
            if not isinstance(cfg, dict):
                cfg = {}
            cfg["preview"] = body.preview
            # The live editor now sends named prompts (tables) alongside plain-string prompts;
            # coerce the mixed list to all-tables so toml.dumps can encode it (same helper the
            # config form uses on save).
            coerce_preview_prompts_for_toml(cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_toml.dumps(cfg), encoding="utf-8")

        # The live process reads --config = job.config_path (its staged TOML); the run
        # folder's config is what resume/continue reads. Update both so the change applies
        # now and survives a resume.
        targets: list[Path] = []
        if job.config_path:
            targets.append(Path(job.config_path))
        run_cfg = runs_scanner.pick_main_config_path(Path(run_dir)) or (Path(run_dir) / "train.toml")
        targets.append(run_cfg)
        seen: set[str] = set()
        for t in targets:
            key = str(t.resolve()) if t else ""
            if not key or key in seen:
                continue
            seen.add(key)
            _write_preview(t)

        try:
            signals.send_signal(run_dir, "reload_config")
            if body.preview_now:
                signals.send_signal(run_dir, "preview")
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        return {"ok": True, "run_dir": str(run_dir), "preview_now": body.preview_now}

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}/metrics")
    def job_metrics(job_id: str) -> dict[str, Any]:
        from rengu_flow_ui import training_hub

        job = db.get_job(job_id)
        run_dir = training_hub.resolve_job_run_dir(job)
        if not run_dir:
            return {"scalars": {}, "preview_images": []}
        return _run_metrics_payload(run_dir)

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}/artifacts")
    def job_artifacts(job_id: str) -> dict[str, Any]:
        from rengu_flow_ui import training_hub

        job = db.get_job(job_id)
        run_dir = training_hub.resolve_job_run_dir(job)
        if not run_dir:
            return {"artifacts": []}
        return {"artifacts": runs_scanner.describe_run_dir(Path(run_dir))["artifacts"]}

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

    # Registered before /runs/{run_name} so the literal path wins over the {run_name} param.
    @app.get(f"{API_PREFIX}/runs/compare")
    def compare_fs_runs(runs: str = "", output_dir: str = "output") -> dict[str, Any]:
        """Cross-run comparison: manifest rows + hparam columns + scalar series + timelines.

        ``runs`` is a comma-separated list of run folder names; empty selects all tracked runs.
        """
        from rengu_track import reader

        root = resolve_repo_path(output_dir)
        names = [n.strip() for n in runs.split(",") if n.strip()]
        if names:
            run_dirs: list[Path] = [root / n for n in names]
        else:
            # Discover ALL run folders (by config/event files), not just manifest ones, so runs
            # trained before tracking still show up.
            run_dirs = [Path(r["path"]) for r in runs_scanner.scan_output_runs(root)]
        return reader.compare_runs(run_dirs, config_fallback=_compare_config_row)

    # On-demand per-metric series for the comparison view (lazy-loaded as each chart scrolls in).
    @app.get(f"{API_PREFIX}/runs/series")
    def fs_runs_series(
        runs: str = "",
        tag: str = "",
        max_points: int = 500,
        output_dir: str = "output",
    ) -> dict[str, Any]:
        from rengu_track import reader

        if not tag:
            raise HTTPException(400, "tag is required")
        root = resolve_repo_path(output_dir)
        names = [n.strip() for n in runs.split(",") if n.strip()]
        run_dirs = [root / n for n in names] if names else reader.list_run_dirs(root)
        cap = max_points if max_points and max_points > 0 else None
        return {"tag": tag, "series": reader.series_for(run_dirs, tag, max_points=cap)}

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
        return _run_metrics_payload(run_dir)

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

    @app.get(f"{API_PREFIX}/version")
    def version() -> dict[str, str | None]:
        """renga version + git commit + installed kaon, for display in the UI."""
        return version_info()

    def _require_maintenance() -> None:
        from rengu_flow_ui import maintenance

        if not maintenance.maintenance_enabled():
            raise HTTPException(
                403,
                "Maintenance API disabled. Set RENGUFLOW_MAINTENANCE=1 to enable it.",
            )

    @app.get(f"{API_PREFIX}/maintenance/enabled")
    def maintenance_enabled_route() -> dict[str, bool]:
        from rengu_flow_ui import maintenance

        return {"enabled": maintenance.maintenance_enabled()}

    @app.get(f"{API_PREFIX}/maintenance/status")
    def maintenance_status() -> dict[str, Any]:
        from rengu_flow_ui import maintenance

        _require_maintenance()
        return maintenance.get_status()

    @app.post(f"{API_PREFIX}/maintenance/database/reset")
    def maintenance_database_reset(
        body: MaintenanceResetBody | None = None,
    ) -> dict[str, Any]:
        from rengu_flow_ui import maintenance

        _require_maintenance()
        body = body or MaintenanceResetBody()
        confirm = body.confirm or body.confirmation == maintenance.DB_RESET_CONFIRM_TOKEN
        try:
            return maintenance.reset_database(confirm=confirm)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post(f"{API_PREFIX}/maintenance/submodules/update")
    def maintenance_submodules_update() -> dict[str, Any]:
        from rengu_flow_ui import maintenance

        _require_maintenance()
        return maintenance.submodule_update()

    @app.post(f"{API_PREFIX}/maintenance/deps/install")
    def maintenance_deps_install(body: DepsInstallBody) -> dict[str, Any]:
        from rengu_flow_ui import maintenance

        _require_maintenance()
        try:
            return maintenance.deps_install(
                body.profile, execute=body.execute, confirm=body.confirm
            )
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get(f"{API_PREFIX}/system/stats")
    def system_stats() -> dict[str, Any]:
        return collect_system_stats()

    @app.get(f"{API_PREFIX}/schema")
    def config_schema() -> dict[str, Any]:
        return get_schema()

    @app.post(f"{API_PREFIX}/registry/probe")
    def registry_probe(body: RegistryProbeBody) -> dict[str, Any]:
        from rengu_flow_ui.registry_probe import probe_optimizer, probe_scheduler

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
        from rengu_flow_ui.docs_reader import list_docs_index

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
            from rengu_flow_ui.config_form import form_values_for_ui
            from rengu_flow_ui.config_schema import get_schema

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


def _parse_validate_error(stderr: str, stdout: str) -> str:
    """Extract a concise message from a failed `--validate-only` subprocess.

    Prefer the line containing ``Config validation failed:`` from stderr; fall back to the
    last non-empty stderr line, else the last non-empty stdout line.
    """
    stderr_lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    for ln in stderr_lines:
        if "Config validation failed:" in ln:
            return ln
    if stderr_lines:
        return stderr_lines[-1]
    stdout_lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    if stdout_lines:
        return stdout_lines[-1]
    return "Config validation failed."


def _run_metrics_payload(run_dir) -> dict[str, Any]:
    """Scalars + preview images for a run folder — the shared body of the /metrics endpoints."""
    from rengu_flow_ui import training_hub

    return {
        "scalars": metrics_tb.read_scalars(run_dir, max_points=1000),
        "preview_images": training_hub.list_run_preview_images(run_dir),
    }


def _compare_config_row(run_dir) -> dict[str, Any] | None:
    """Build a comparison row for a run that has no run.json (trained before tracking).

    Hyperparameters come from the run's config TOML; summary/lineage are empty (the trainer never
    recorded them). Returns None for a folder that is neither a config-bearing nor TB-bearing run.
    """
    from rengu_track.run import flatten_hparams

    rd = Path(run_dir)
    cfg_path = runs_scanner.pick_main_config_path(rd)
    has_tb = any(rd.glob("events.out.tfevents.*"))
    if cfg_path is None and not has_tb:
        return None
    config: dict[str, Any] = {}
    if cfg_path is not None:
        try:
            import tomlkit

            config = tomlkit.parse(cfg_path.read_text(encoding="utf-8")).unwrap()
        except Exception:
            config = {}
    return {
        "run_id": rd.name,
        "name": config.get("run_name") or rd.name,
        "status": "imported",
        "created_at": "",
        "updated_at": "",
        "hparams": flatten_hparams(config),
        "summary": {},
        "system_summary": {},
        "lineage": {},
        "hardware": {},
        "tags": [],
        "last_scalars": {},
    }


def _created_dataset_response(content: str, name: str | None) -> dict[str, Any]:
    """Create a dataset from TOML and return the UI's create/import response shape."""
    cid = datasets_store.create_dataset(content, name=name)
    row = datasets_store.read_dataset_for_ui(cid)
    return {
        "id": cid,
        "name": row["name"],
        "dataset_ref": datasets_store.dataset_library_ref(cid),
    }


def _job_run_name(job: db.JobRecord) -> str | None:
    """The run's own `run_name` from its config snapshot, when set (else None)."""
    if not job.config_content:
        return None
    try:
        import toml

        name = toml.loads(job.config_content).get("run_name")
    except Exception:
        return None
    return name.strip() if isinstance(name, str) and name.strip() else None


def _job_dict(job: db.JobRecord) -> dict[str, Any]:
    return {
        "id": job.id,
        "run_name": _job_run_name(job),
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
        "config_content": job.config_content,
        "cache_only": job.cache_only,
        "trust_cache": job.trust_cache,
        "regenerate_cache": job.regenerate_cache,
    }
