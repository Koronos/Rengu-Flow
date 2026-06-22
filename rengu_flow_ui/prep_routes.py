"""API routes for the dataset-prep section (tag editor sessions, backups, quarantine)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rengu_flow.prep.caption_store import CaptionStore
from rengu_flow_ui._http_util import http_errors
from rengu_flow_ui.dataset_image_preview import issue_image_token
from rengu_flow_ui.tag_sessions import TagSessionStore

API_PREFIX = "/api/v1"

tag_sessions = TagSessionStore()


@contextmanager
def _prep_http_errors():
    try:
        with http_errors("Session not found"):
            yield
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


class OpenSessionBody(BaseModel):
    path: str
    format: str = "sidecar"
    ext: str = ".txt"


class StageOpsBody(BaseModel):
    ops: list[dict] = Field(default_factory=list)


class QueryBody(BaseModel):
    filter: dict = Field(default_factory=dict)
    scope: str = "tag_lines"


class SizeQueryBody(BaseModel):
    below: int | None = None  # short side < below
    above: int | None = None  # long side > above


class RestoreBackupBody(BaseModel):
    path: str
    backup: str


class RestoreQuarantineBody(BaseModel):
    path: str
    batch: str


class CreatePrepJobBody(BaseModel):
    stage: str
    config: dict = Field(default_factory=dict)
    start_now: bool = False


class DownloadModelBody(BaseModel):
    stage: str
    model_id: str


class RequeueBody(BaseModel):
    start_now: bool = False


class PromptPreviewBody(BaseModel):
    caption: dict = Field(default_factory=dict)
    # Example grounding tags so the preview shows the tags block in place.
    sample_tags: list[str] = Field(
        default_factory=lambda: ["1girl", "long hair", "smile"]
    )


def register_prep_routes(app: FastAPI) -> None:
    @app.post(f"{API_PREFIX}/prep/jobs")
    def create_prep_job(body: CreatePrepJobBody):
        import toml

        from rengu_flow.prep.config import parse_prep_config
        from rengu_flow_ui.app import _job_dict
        from rengu_flow_ui.prep_jobs import enqueue_prep_job

        with _prep_http_errors():
            config = parse_prep_config(body.config)
            config.validate_for_stage(body.stage)
            job = enqueue_prep_job(
                body.stage, toml.dumps(body.config), start_now=body.start_now
            )
            return _job_dict(job)

    @app.post(f"{API_PREFIX}/prep/jobs/{{job_id}}/requeue")
    def requeue_prep_job_route(job_id: str, body: RequeueBody | None = None):
        from rengu_flow_ui.app import _job_dict
        from rengu_flow_ui.prep_jobs import requeue_prep_job

        with _prep_http_errors():
            try:
                job = requeue_prep_job(job_id, start_now=bool(body and body.start_now))
            except KeyError:
                raise HTTPException(404, "Job not found")
            return _job_dict(job)

    @app.get(f"{API_PREFIX}/prep/jobs/{{job_id}}/report")
    def prep_job_report(job_id: str):
        import json

        from rengu_flow_ui import db

        try:
            job = db.get_job(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found")
        report = Path(job.run_dir or "") / "report.json"
        if not report.is_file():
            return {"report": None}
        return {"report": json.loads(report.read_text(encoding="utf-8"))}

    @app.get(f"{API_PREFIX}/prep/jobs/{{job_id}}/config")
    def prep_job_config(job_id: str):
        """Parsed config of a prep job, so the form can seed a new job from it."""
        import toml

        from rengu_flow_ui import db

        try:
            job = db.get_job(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found")
        content = job.config_content or ""
        config = toml.loads(content) if content.strip() else {}
        return {"config": config}

    @app.get(f"{API_PREFIX}/prep/models")
    def prep_models(stage: str = Query(...)):
        from rengu_flow.prep.models import list_models

        with _prep_http_errors():
            return {"models": list_models(stage)}

    @app.get(f"{API_PREFIX}/prep/caption-prompts")
    def prep_caption_prompts():
        from rengu_flow.prep.captioner import list_prompt_options

        return list_prompt_options()

    @app.post(f"{API_PREFIX}/prep/caption-prompts/preview")
    def prep_caption_prompt_preview(body: PromptPreviewBody):
        """Render the EXACT prompt a caption job would send for these settings.

        Single source of truth for the form's prompt preview: ToriiGate composes its
        native trained format, JoyCaption the instruction prompt — the UI never has
        to replicate that logic.
        """
        from rengu_flow.prep.captioner import build_prompt, captioner_config_from_stage
        from rengu_flow.prep.config import CaptionStageConfig, _fill_dataclass

        with _prep_http_errors():
            stage = _fill_dataclass(CaptionStageConfig(), body.caption, context="caption")
            config = captioner_config_from_stage(stage)
            prompt = build_prompt(
                config, tags=body.sample_tags or None, image_key="preview"
            )
            return {
                "prompt": prompt,
                "native_format": (
                    config.model == "toriigate-0.5" and not (config.prompt or "").strip()
                ),
            }

    @app.post(f"{API_PREFIX}/prep/models/download")
    def prep_model_download(body: DownloadModelBody):
        from rengu_flow.prep.models import ensure_model

        with _prep_http_errors():
            path = ensure_model(body.model_id, body.stage)
            return {"ok": True, "path": str(path)}

    @app.post(f"{API_PREFIX}/prep/tags/sessions")
    def open_tag_session(body: OpenSessionBody):
        with _prep_http_errors():
            session = tag_sessions.open(body.path, fmt=body.format, ext=body.ext)
            return session.summary()

    @app.get(f"{API_PREFIX}/prep/tags/sessions/{{session_id}}")
    def tag_session_summary(session_id: str):
        with _prep_http_errors():
            return tag_sessions.get(session_id).summary()

    @app.get(f"{API_PREFIX}/prep/tags/sessions/{{session_id}}/stats")
    def tag_session_stats(session_id: str, scope: str = Query("line1")):
        with _prep_http_errors():
            return tag_sessions.stats(session_id, scope=scope)

    @app.post(f"{API_PREFIX}/prep/tags/sessions/{{session_id}}/query")
    def tag_session_query(session_id: str, body: QueryBody):
        with _prep_http_errors():
            result = tag_sessions.query(session_id, body.filter, scope=body.scope)
            folder = Path(tag_sessions.get(session_id).captions.folder).resolve()
            result["previews"] = {
                key: issue_image_token(0, key, folder) for key in result["keys"]
            }
            return result

    @app.post(f"{API_PREFIX}/prep/tags/sessions/{{session_id}}/size-query")
    def tag_session_size_query(session_id: str, body: SizeQueryBody):
        with _prep_http_errors():
            if body.below is None and body.above is None:
                raise ValueError("Provide 'below' and/or 'above'")
            result = tag_sessions.size_query(
                session_id, below=body.below, above=body.above
            )
            folder = Path(tag_sessions.get(session_id).captions.folder).resolve()
            result["previews"] = {
                key: issue_image_token(0, key, folder) for key in result["keys"]
            }
            return result

    @app.post(f"{API_PREFIX}/prep/tags/sessions/{{session_id}}/ops")
    def tag_session_stage_ops(session_id: str, body: StageOpsBody):
        with _prep_http_errors():
            if not body.ops:
                raise ValueError("No ops provided")
            return tag_sessions.stage_ops(session_id, body.ops)

    @app.post(f"{API_PREFIX}/prep/tags/sessions/{{session_id}}/undo")
    def tag_session_undo(session_id: str):
        with _prep_http_errors():
            return tag_sessions.undo(session_id)

    @app.get(f"{API_PREFIX}/prep/tags/sessions/{{session_id}}/diff")
    def tag_session_diff(session_id: str, limit: int | None = Query(None, ge=1)):
        with _prep_http_errors():
            return tag_sessions.diff(session_id, limit=limit)

    @app.post(f"{API_PREFIX}/prep/tags/sessions/{{session_id}}/commit")
    def tag_session_commit(session_id: str):
        with _prep_http_errors():
            return tag_sessions.commit(session_id)

    @app.delete(f"{API_PREFIX}/prep/tags/sessions/{{session_id}}")
    def tag_session_close(session_id: str):
        tag_sessions.close(session_id)
        return {"ok": True}

    @app.get(f"{API_PREFIX}/prep/tags/backups")
    def tag_backups(path: str = Query(...)):
        with _prep_http_errors():
            return {"backups": CaptionStore.list_backups(path)}

    @app.post(f"{API_PREFIX}/prep/tags/restore")
    def tag_restore(body: RestoreBackupBody):
        with _prep_http_errors():
            restored = CaptionStore.restore_snapshot(body.path, body.backup)
            return {"restored": restored}

    @app.get(f"{API_PREFIX}/prep/tags/quarantine")
    def tag_quarantine_list(path: str = Query(...)):
        with _prep_http_errors():
            return {"batches": CaptionStore.list_quarantine(path)}

    @app.post(f"{API_PREFIX}/prep/tags/quarantine/restore")
    def tag_quarantine_restore(body: RestoreQuarantineBody):
        with _prep_http_errors():
            restored = CaptionStore.restore_quarantine(body.path, body.batch)
            return {"restored": restored}
