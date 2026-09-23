"""edit_caption prep stage: GGUF multi-model backend, control pairing in the CaptionStore,
prompt layout + reply cleanup, the stage runner and its config/CLI/route plumbing.

No GPU, binary or network: llama-server, downloads and requests are mocked.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

import rengu_flow.prep.gguf_captioner as gg
from rengu_flow.cli import prep_cmd
from rengu_flow.prep.caption_store import CaptionStore
from rengu_flow.prep.config import EditCaptionStageConfig, parse_prep_config
from rengu_flow.prep.edit_captioner import (
    DEFAULT_EDIT_PROMPT,
    build_edit_content,
    clean_instruction,
    edit_caption_folder,
    render_edit_prompt,
)
from rengu_flow.prep.runner import run_stage

pytestmark = pytest.mark.no_ui_db


def _img(path: Path, color=(200, 30, 30), size=(64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def edit_set(tmp_path):
    """targets/{a,b,m,orphan,gap}.jpg; controls: a (exact), b (exact .png), m_0/m_1 (two),
    gap_0/gap_2 (non-contiguous) and nothing for orphan."""
    targets, controls = tmp_path / "targets", tmp_path / "controls"
    for name in ("a", "b", "m", "orphan", "gap"):
        _img(targets / f"{name}.jpg", color=(90, 90, 90))
    _img(controls / "a.jpg")
    _img(controls / "b.png")
    _img(controls / "m_0.jpg")
    _img(controls / "m_1.jpg")
    _img(controls / "gap_0.jpg")
    _img(controls / "gap_2.jpg")
    (controls / "notes.txt").write_text("not an image")
    return targets, controls


# ---------------------------------------------------------------------------
# GGUF backend: registry, ToriiGate regression, context budget
# ---------------------------------------------------------------------------


def test_toriigate_constants_are_unchanged():
    """The caption stage's gguf engine keeps its validated files and tuning."""
    assert gg.GGUF_REPO == "DraconicDragon/ToriiGate-0.5-GGUF"
    assert gg.MMPROJ_FILE == "ToriiGate-0.5-fp16.mmproj.gguf"
    assert gg.GGUF_QUANTS["Q8_0"] == "ToriiGate-0.5-Q8_0.gguf"
    assert (gg.DEFAULT_QUANT, gg.CTX_SIZE, gg.N_PARALLEL, gg.MAX_PIXELS) == (
        "Q8_0", 32768, 16, 1_000_000,
    )


@pytest.mark.parametrize(
    "model, quant, repo, weights, mmproj",
    [
        (None, "Q5_K_M", "DraconicDragon/ToriiGate-0.5-GGUF", "ToriiGate-0.5-Q5_K_M.gguf",
         "ToriiGate-0.5-fp16.mmproj.gguf"),
        (None, "bogus", "DraconicDragon/ToriiGate-0.5-GGUF", "ToriiGate-0.5-Q8_0.gguf",
         "ToriiGate-0.5-fp16.mmproj.gguf"),
        ("qwen3-vl-4b-instruct", "", "unsloth/Qwen3-VL-4B-Instruct-GGUF",
         "Qwen3-VL-4B-Instruct-Q8_0.gguf", "mmproj-F16.gguf"),
        ("qwen3-vl-8b-instruct", "", "unsloth/Qwen3-VL-8B-Instruct-GGUF",
         "Qwen3-VL-8B-Instruct-Q4_K_M.gguf", "mmproj-F16.gguf"),
        ("qwen3-vl-8b-instruct", "Q6_K", "unsloth/Qwen3-VL-8B-Instruct-GGUF",
         "Qwen3-VL-8B-Instruct-Q6_K.gguf", "mmproj-F16.gguf"),
    ],
)
def test_ensure_gguf_resolves_model_files(monkeypatch, model, quant, repo, weights, mmproj):
    calls = []

    def fake_download(repo_id, filename):
        calls.append((repo_id, filename))
        return f"/hf/{repo_id}/{filename}"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    kwargs = {} if model is None else {"model": model}  # None: the ToriiGate default
    g, m = gg.ensure_gguf(quant, **kwargs)
    assert calls == [(repo, weights), (repo, mmproj)]
    assert g.name == weights and m.name == mmproj


def test_ensure_gguf_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown GGUF model"):
        gg.ensure_gguf("Q8_0", model="nope")


@pytest.mark.parametrize(
    "kwargs, ctx, parallel",
    [({}, "32768", "16"), ({"ctx_size": 10240, "n_parallel": 4}, "10240", "4")],
)
def test_start_server_command_line(monkeypatch, tmp_path, kwargs, ctx, parallel):
    seen = {}

    def fake_popen(cmd, **kw):
        seen["cmd"] = cmd
        return object()

    monkeypatch.setattr(gg.subprocess, "Popen", fake_popen)
    gg._start_server(tmp_path, tmp_path / "m.gguf", tmp_path / "mm.gguf", 1234, **kwargs)
    cmd = seen["cmd"]
    assert cmd[cmd.index("-c") + 1] == ctx
    assert cmd[cmd.index("--parallel") + 1] == parallel
    assert cmd[cmd.index("--port") + 1] == "1234"


def test_server_budget_fits_the_worst_row():
    spec = gg.GGUF_MODELS["qwen3-vl-4b-instruct"]
    per_image = gg.image_tokens(512 * 1024, spec.px_per_token)
    assert per_image == int(512 * 1024 * 1.1 / 1024) + 8
    ctx, parallel = gg.server_budget(spec, n_images=3, max_pixels=512 * 1024, max_new_tokens=96)
    assert parallel == spec.n_parallel
    slot = ctx // parallel
    assert ctx % parallel == 0 and slot % 256 == 0
    assert slot >= 3 * per_image + 512 + 96
    # More images per row or an explicit slot count scale the context accordingly.
    ctx5, _ = gg.server_budget(spec, n_images=5, max_pixels=512 * 1024, max_new_tokens=96)
    assert ctx5 > ctx
    ctx2, p2 = gg.server_budget(
        spec, n_images=3, max_pixels=512 * 1024, max_new_tokens=96, n_parallel=2
    )
    assert (p2, ctx2) == (2, slot * 2)


def test_toriigate_slot_cannot_hold_a_pair():
    """Why the budget exists: -c 32768 / 16 slots is one ~1 Mpx image, not a pair."""
    slot = gg.CTX_SIZE // gg.N_PARALLEL
    assert slot < 2 * gg.image_tokens(1_000_000, 32 * 32)


def test_gguf_models_by_task():
    assert set(gg.gguf_models("edit_caption")) == {"qwen3-vl-4b-instruct", "qwen3-vl-8b-instruct"}
    assert "toriigate-0.5" in gg.gguf_models("caption")


def test_toriigate_caption_path_starts_server_with_fixed_tuning(tmp_path, monkeypatch):
    """Regression: the caption stage's ToriiGate run still asks for -c 32768 --parallel 16."""
    from rengu_flow.prep.captioner import CaptionerConfig, caption_folder

    d = tmp_path / "imgs"
    _img(d / "a.jpg")
    (d / "a.txt").write_text("1girl\n")
    started = []

    class _Proc:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

    def fake_start(bin_dir, gguf, mmproj, port, **kw):
        started.append(kw)
        return _Proc()

    monkeypatch.setattr(gg, "ensure_binary", lambda: tmp_path)
    monkeypatch.setattr(gg, "ensure_gguf", lambda q: (tmp_path / "m.gguf", tmp_path / "mm.gguf"))
    monkeypatch.setattr(gg, "_start_server", fake_start)
    monkeypatch.setattr(gg, "_wait_health", lambda *a, **k: None)
    monkeypatch.setattr(gg, "_encode_image", lambda p: "b64")
    monkeypatch.setattr(gg, "_request_caption", lambda port, b64, prompt, cfg: "a caption")

    report = caption_folder(d, CaptionerConfig(model="toriigate-0.5", engine="gguf"))
    assert report["captioned"] == 1
    assert [{k: kw[k] for k in ("ctx_size", "n_parallel")} for kw in started] == [
        {"ctx_size": 32768, "n_parallel": 16}
    ]


def test_request_caption_body_is_single_image(monkeypatch):
    sent = {}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

    def fake_urlopen(req, timeout=None):
        sent["body"] = json.loads(req.data)
        return _Resp()

    monkeypatch.setattr(gg.urllib.request, "urlopen", fake_urlopen)

    class Cfg:
        temperature = None
        top_p = None
        max_new_tokens = 64

    assert gg._request_caption(1, "QUJD", "describe", Cfg()) == "ok"
    body = sent["body"]
    content = body["messages"][0]["content"]
    assert [p["type"] for p in content] == ["image_url", "text"]
    assert content[0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert (body["temperature"], body["top_p"], body["max_tokens"]) == (0.5, 1.0, 64)


def test_encode_image_caps_pixels(tmp_path):
    import base64
    import io

    p = _img(tmp_path / "big.png", size=(2000, 1000))
    out = Image.open(io.BytesIO(base64.b64decode(gg._encode_image(p, 100_000))))
    assert out.width * out.height <= 100_000
    assert abs(out.width / out.height - 2.0) < 0.05


# ---------------------------------------------------------------------------
# CaptionStore with controls
# ---------------------------------------------------------------------------


def test_caption_store_pairs_controls(edit_set):
    targets, controls = edit_set
    cs = CaptionStore.open(targets, control_path=controls)
    assert cs.control_path == controls
    assert cs.controls["a.jpg"] == [controls / "a.jpg"]
    assert cs.controls["b.jpg"] == [controls / "b.png"]
    assert cs.controls["m.jpg"] == [controls / "m_0.jpg", controls / "m_1.jpg"]
    assert set(cs.unpaired) == {"orphan.jpg", "gap.jpg"}
    assert "No control image" in cs.unpaired["orphan.jpg"]
    assert "contiguous" in cs.unpaired["gap.jpg"]
    # Unpaired targets are still images of the set (the editor can see and fix them).
    assert "orphan.jpg" in cs.images


def test_caption_store_without_controls_is_unchanged(edit_set):
    targets, _ = edit_set
    cs = CaptionStore.open(targets)
    assert cs.control_path is None and cs.controls == {} and cs.unpaired == {}


def test_caption_store_missing_control_dir(edit_set, tmp_path):
    targets, _ = edit_set
    with pytest.raises(FileNotFoundError, match="Control folder"):
        CaptionStore.open(targets, control_path=tmp_path / "nope")


def test_quarantine_moves_controls_and_restores_them(edit_set):
    targets, controls = edit_set
    (targets / "m.txt").write_text("merge them\n")
    cs = CaptionStore.open(targets, control_path=controls)
    qdir = cs.quarantine(["m.jpg"])
    assert not (controls / "m_0.jpg").exists() and not (controls / "m_1.jpg").exists()
    assert (qdir / "controls" / "m_0.jpg").is_file()
    manifest = json.loads((qdir / "manifest.json").read_text())
    assert manifest["control_path"] == str(controls)
    assert manifest["entries"]["m.jpg"]["controls"] == ["m_0.jpg", "m_1.jpg"]
    assert "m.jpg" not in cs.controls

    restored = CaptionStore.restore_quarantine(targets, qdir.name)
    assert restored == ["m.jpg"]
    assert (controls / "m_0.jpg").is_file() and (controls / "m_1.jpg").is_file()
    assert (targets / "m.txt").read_text() == "merge them\n"
    assert CaptionStore.open(targets, control_path=controls).controls["m.jpg"]


def test_quarantine_keeps_a_control_another_target_uses(tmp_path):
    """a_1.png is the exact control of target a_1 AND control #1 of target a."""
    targets, controls = tmp_path / "t", tmp_path / "c"
    _img(targets / "a.jpg")
    _img(targets / "a_1.jpg")
    _img(controls / "a_0.png")
    _img(controls / "a_1.png")
    cs = CaptionStore.open(targets, control_path=controls)
    assert cs.controls["a.jpg"] == [controls / "a_0.png", controls / "a_1.png"]
    assert cs.controls["a_1.jpg"] == [controls / "a_1.png"]
    qdir = cs.quarantine(["a.jpg"])
    assert not (controls / "a_0.png").exists()
    assert (controls / "a_1.png").is_file()  # still a_1's control
    manifest = json.loads((qdir / "manifest.json").read_text())
    assert manifest["entries"]["a.jpg"]["controls"] == ["a_0.png"]


def test_quarantine_without_controls_writes_no_control_keys(tmp_path):
    d = tmp_path / "d"
    _img(d / "x.jpg")
    cs = CaptionStore.open(d)
    manifest = json.loads((cs.quarantine(["x.jpg"]) / "manifest.json").read_text())
    assert "control_path" not in manifest
    assert manifest["entries"]["x.jpg"] == {"captions": []}


# ---------------------------------------------------------------------------
# Prompt layout + reply cleanup
# ---------------------------------------------------------------------------


def test_build_edit_content_single_control():
    parts = build_edit_content(1)
    assert [p["type"] for p in parts] == ["text", "image", "text", "image", "text"]
    assert [p["index"] for p in parts if p["type"] == "image"] == [0, 1]
    assert parts[0]["text"] == "Source image:" and parts[2]["text"] == "Result image:"
    assert parts[-1]["text"] == DEFAULT_EDIT_PROMPT
    assert "image 1" not in parts[-1]["text"]


def test_build_edit_content_multi_control_names_images():
    parts = build_edit_content(2, "  Custom rule.  ")
    labels = [p["text"] for p in parts if p["type"] == "text"]
    assert labels[:3] == ["Image 1 (source):", "Image 2 (source):", "Result image:"]
    # Controls first, in order, then the target.
    assert [p["index"] for p in parts if p["type"] == "image"] == [0, 1, 2]
    assert '"image 1", "image 2"' in labels[-1]
    assert labels[-1].endswith("Custom rule.")


def test_build_edit_content_needs_a_control():
    with pytest.raises(ValueError):
        build_edit_content(0)


def test_render_edit_prompt_marks_images():
    text = render_edit_prompt(2, "Do it.")
    assert text.count("<image>") == 3
    assert text.endswith("Do it.")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Convert the image to black and white.", "Convert the image to black and white."),
        ('"Make the sky orange."', "Make the sky orange."),
        ("Instruction: Make the sky orange.", "Make the sky orange."),
        ("**Edit instruction:** Remove the car.", "Remove the car."),
        ("Here is the instruction: 'Add a hat.'", "Add a hat."),
        ("“Turn it into a sketch.”", "Turn it into a sketch."),
        ("Editing instruction - Desaturate\nthe photo.", "Desaturate the photo."),
        ("<think>hmm</think>Make it night.", "Make it night."),
        ('"Unbalanced start', "Unbalanced start"),
        ("", ""),
    ],
)
def test_clean_instruction(raw, expected):
    assert clean_instruction(raw) == expected


# ---------------------------------------------------------------------------
# Stage runner (server + requests mocked)
# ---------------------------------------------------------------------------


class _Proc:
    returncode = None

    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    """Mock llama-server: records the server args and every request's content."""
    state = {"start": [], "requests": [], "reply": lambda content: '"Make it black and white."'}

    def fake_start(bin_dir, gguf, mmproj, port, **kw):
        state["start"].append({"gguf": gguf, **kw})
        return _Proc()

    def fake_chat(port, content, config, *, default_temperature, default_top_p):
        state["requests"].append(content)
        state["defaults"] = (default_temperature, default_top_p)
        return state["reply"](content)

    monkeypatch.setattr(gg, "ensure_binary", lambda: tmp_path)
    monkeypatch.setattr(
        gg, "ensure_gguf", lambda q, model="toriigate-0.5": (tmp_path / f"{model}-{q}.gguf", tmp_path / "mm.gguf")
    )
    monkeypatch.setattr(gg, "_start_server", fake_start)
    monkeypatch.setattr(gg, "_wait_health", lambda *a, **k: None)
    monkeypatch.setattr(gg, "_encode_image", lambda p, max_pixels=0: f"B64:{Path(p).parent.name}/{Path(p).name}")
    monkeypatch.setattr(gg, "request_chat", fake_chat)
    return state


def _stage(controls, **kw) -> EditCaptionStageConfig:
    return EditCaptionStageConfig(control_path=str(controls), **kw)


def _image_urls(content):
    prefix = "data:image/jpeg;base64,"
    return [p["image_url"]["url"].removeprefix(prefix) for p in content if p["type"] == "image_url"]


def test_edit_caption_writes_line1_and_reports(edit_set, fake_server):
    targets, controls = edit_set
    (targets / "a.txt").write_text("old instruction\nsecond line kept\n")
    report = edit_caption_folder(targets, _stage(controls, overwrite=True))

    assert report["captioned"] == 3
    assert report["failed"] == [] and report["skipped"] == 0
    assert len(report["unpaired"]) == 2
    assert report["unpaired"][0].startswith("gap.jpg: ")
    assert report["model"] == "qwen3-vl-4b-instruct" and report["quantization"] == "Q8_0"
    assert report["max_images_per_row"] == 3  # m: two controls + target
    assert (targets / "a.txt").read_text() == "Make it black and white.\nsecond line kept\n"
    assert (targets / "b.txt").read_text() == "Make it black and white.\n"
    assert not (targets / "orphan.txt").exists()

    start = fake_server["start"][0]
    assert start["gguf"].name == "qwen3-vl-4b-instruct-Q8_0.gguf"
    assert (start["ctx_size"], start["n_parallel"]) == (report["ctx_size"], report["n_parallel"])
    expected = gg.server_budget(
        gg.GGUF_MODELS["qwen3-vl-4b-instruct"], n_images=3, max_pixels=512 * 1024,
        max_new_tokens=96,
    )
    assert (start["ctx_size"], start["n_parallel"]) == expected
    assert fake_server["defaults"] == (0.7, 0.8)


def test_edit_caption_sends_controls_then_target(edit_set, fake_server):
    targets, controls = edit_set
    edit_caption_folder(targets, _stage(controls))
    by_target = {_image_urls(c)[-1]: _image_urls(c) for c in fake_server["requests"]}
    assert by_target["B64:targets/m.jpg"] == [
        "B64:controls/m_0.jpg", "B64:controls/m_1.jpg", "B64:targets/m.jpg",
    ]
    assert by_target["B64:targets/b.jpg"] == ["B64:controls/b.png", "B64:targets/b.jpg"]


def test_edit_caption_skips_existing_unless_overwrite(edit_set, fake_server):
    targets, controls = edit_set
    (targets / "a.txt").write_text("keep me\n")
    report = edit_caption_folder(targets, _stage(controls))
    assert report["skipped"] == 1 and report["captioned"] == 2
    assert (targets / "a.txt").read_text() == "keep me\n"


def test_edit_caption_empty_reply_is_failed(edit_set, fake_server):
    targets, controls = edit_set
    fake_server["reply"] = lambda content: '  "" '
    report = edit_caption_folder(targets, _stage(controls))
    assert report["captioned"] == 0
    assert sorted(report["failed"]) == ["a.jpg", "b.jpg", "m.jpg"]


def test_edit_caption_nothing_to_do_starts_no_server(tmp_path, fake_server):
    targets, controls = tmp_path / "t", tmp_path / "c"
    _img(targets / "x.jpg")
    controls.mkdir()
    report = edit_caption_folder(targets, _stage(controls))
    assert report["captioned"] == 0 and len(report["unpaired"]) == 1
    assert fake_server["start"] == []


def test_edit_caption_json_format(edit_set, fake_server):
    targets, controls = edit_set
    edit_caption_folder(targets, _stage(controls), fmt="json")
    data = json.loads((targets / "captions.json").read_text())
    assert data["a.jpg"] == ["Make it black and white."]
    assert not (targets / "a.txt").exists()


def test_edit_caption_stop_signal(edit_set, fake_server):
    targets, controls = edit_set
    report = edit_caption_folder(targets, _stage(controls), should_stop=lambda: True)
    assert report["stopped"] is True
    assert report["captioned"] == 0


def test_run_stage_edit_caption_writes_report(edit_set, fake_server, tmp_path, capsys):
    targets, controls = edit_set
    config = parse_prep_config(
        {"path": str(targets), "edit_caption": {"control_path": str(controls), "n_parallel": 2}}
    )
    code = run_stage(config, "edit_caption", tmp_path / "job")
    assert code == 0
    report = json.loads((tmp_path / "job" / "report.json").read_text())
    assert report["stage"] == "edit_caption"
    assert report["captioned"] == 3 and report["n_parallel"] == 2
    assert report["control_path"] == str(controls)
    out = capsys.readouterr().out
    assert "prep edit_caption exits with return code = 0" in out
    assert '"phase":"prep:edit_caption"' in out.replace(" ", "")


# ---------------------------------------------------------------------------
# Config / CLI / models / routes
# ---------------------------------------------------------------------------


def test_validate_edit_caption(edit_set, tmp_path):
    targets, controls = edit_set

    def cfg(**ed):
        return parse_prep_config({"path": str(targets), "edit_caption": ed})

    cfg(control_path=str(controls)).validate_for_stage("edit_caption")
    with pytest.raises(ValueError, match="needs a model"):
        cfg(control_path=str(controls), model="").validate_for_stage("edit_caption")
    with pytest.raises(ValueError, match="Unknown edit_caption model"):
        cfg(control_path=str(controls), model="toriigate-0.5").validate_for_stage("edit_caption")
    with pytest.raises(ValueError, match="control_path"):
        cfg().validate_for_stage("edit_caption")
    with pytest.raises(FileNotFoundError, match="Control folder"):
        cfg(control_path=str(tmp_path / "missing")).validate_for_stage("edit_caption")
    with pytest.raises(ValueError, match="different folder"):
        cfg(control_path=str(targets)).validate_for_stage("edit_caption")


def test_validate_caption_requires_model(edit_set):
    targets, _ = edit_set
    config = parse_prep_config({"path": str(targets), "caption": {"model": ""}})
    with pytest.raises(ValueError, match=r"\[caption\].model"):
        config.validate_for_stage("caption")
    parse_prep_config({"path": str(targets)}).validate_for_stage("caption")  # default model ok


def test_cli_edit_caption_args():
    parser = argparse.ArgumentParser()
    prep_cmd.add_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "prep", "edit_caption", "--path", "/t", "--control-path", "/c",
        "--model", "qwen3-vl-8b-instruct", "--gguf-quant", "Q6_K", "--max-pixels", "262144",
        "--parallel", "2", "--prompt", "Say it.", "--overwrite",
    ])
    config = prep_cmd._build_config(args)
    ed = config.edit_caption
    assert config.path == "/t"
    assert (ed.control_path, ed.model, ed.gguf_quantization) == ("/c", "qwen3-vl-8b-instruct", "Q6_K")
    assert (ed.max_pixels, ed.n_parallel, ed.prompt, ed.overwrite) == (262144, 2, "Say it.", True)


def test_list_models_edit_caption(monkeypatch):
    from rengu_flow.prep import models

    monkeypatch.setattr(models, "_is_downloaded", lambda repo, filename, repo_type="model": True)
    entries = {m["id"]: m for m in models.list_models("edit_caption")}
    assert set(entries) == {"qwen3-vl-4b-instruct", "qwen3-vl-8b-instruct"}
    assert entries["qwen3-vl-4b-instruct"]["default_quant"] == "Q8_0"
    assert entries["qwen3-vl-8b-instruct"]["filename"] == "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
    assert entries["qwen3-vl-4b-instruct"]["downloaded"] is True


def test_ensure_model_edit_caption(monkeypatch, tmp_path):
    from rengu_flow.prep import models

    seen = []
    monkeypatch.setattr(
        gg, "ensure_gguf",
        lambda q, model="toriigate-0.5": seen.append((q, model)) or (tmp_path / "w.gguf", tmp_path / "mm.gguf"),
    )
    assert models.ensure_model("qwen3-vl-8b-instruct", "edit_caption") == tmp_path / "w.gguf"
    assert seen == [("Q4_K_M", "qwen3-vl-8b-instruct")]
    with pytest.raises(ValueError, match="Unknown edit_caption model"):
        models.ensure_model("toriigate-0.5", "edit_caption")


# ------------------------------------------------------------------ llama-server log


class _ExitedProc:
    returncode = 1

    def poll(self):
        return 1


def test_a_server_that_dies_on_start_says_why(tmp_path):
    """stderr used to go to DEVNULL, so a failed start read only "exited early (code 1)"."""
    log = tmp_path / gg.SERVER_LOG_NAME
    log.write_text(
        "".join(f"line {i}\n" for i in range(50))
        + "ggml_vulkan: Device memory allocation of size 123 failed.\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError) as excinfo:
        gg._wait_health(1, _ExitedProc(), timeout=5, log_path=log)
    message = str(excinfo.value)
    assert "exited early (code 1)" in message
    assert "Device memory allocation of size 123 failed." in message
    assert str(log) in message
    assert "line 49" in message and "line 0\n" not in message  # the tail, not the whole log


def test_the_server_log_goes_to_the_job_dir(monkeypatch, tmp_path):
    seen = {}

    def fake_popen(cmd, **kw):
        seen.update(kw)
        return object()

    monkeypatch.setattr(gg.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(gg, "_server_log_dir", None)
    gg.set_server_log_dir(tmp_path)
    try:
        gg._start_server(tmp_path, tmp_path / "m.gguf", tmp_path / "mm.gguf", 1234)
    finally:
        gg.set_server_log_dir(None)
    assert seen["stderr"] is gg.subprocess.STDOUT
    assert seen["stdout"] is not gg.subprocess.DEVNULL
    assert Path(seen["stdout"].name) == tmp_path / gg.SERVER_LOG_NAME
    assert seen["stdout"].closed  # the parent's copy; the child holds its own


def test_run_stage_points_the_server_log_at_its_job_dir(monkeypatch, tmp_path):
    from rengu_flow.prep import runner
    from rengu_flow.prep.config import PrepConfig

    seen = {}

    def fake_tag(*_args):
        seen["dir"] = gg._server_log_dir
        return {}

    monkeypatch.setitem(runner._STAGE_RUNNERS, "tag", fake_tag)
    monkeypatch.setattr(PrepConfig, "validate_for_stage", lambda self, stage: None)
    monkeypatch.setattr(gg, "_server_log_dir", None)
    runner.run_stage(PrepConfig(path=str(tmp_path)), "tag", tmp_path / "job")
    assert seen["dir"] == tmp_path / "job"
