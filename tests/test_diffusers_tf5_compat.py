from rengu_flow.utils.diffusers_tf5_compat import strip_text_model_state_dict_prefix


def test_strip_text_model_prefix():
    sd = {
        "text_model.embeddings.position_embedding.weight": 1,
        "encoder.layers.0.weight": 2,
    }
    out = strip_text_model_state_dict_prefix(sd)
    assert "embeddings.position_embedding.weight" in out
    assert "text_model.embeddings.position_embedding.weight" not in out
    assert out["encoder.layers.0.weight"] == 2


def test_strip_text_model_prefix_noop_when_absent():
    sd = {"embeddings.weight": 1}
    assert strip_text_model_state_dict_prefix(sd) is sd


def test_patch_reaches_the_single_file_loader():
    """``from_single_file`` calls the CLIP builder through ``diffusers.loaders.single_file``'s own
    by-name import: patching only ``single_file_utils`` leaves the loader on the transformers-4
    code path (``AttributeError: 'CLIPTextModel' object has no attribute 'text_model'``)."""
    import pytest

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("diffusers")
    from packaging.version import Version

    if Version(transformers.__version__) < Version("5.0.0"):
        pytest.skip("patch only applies to transformers 5")
    import diffusers.loaders.single_file as single_file
    import diffusers.loaders.single_file_utils as sfu

    from rengu_flow.utils.diffusers_tf5_compat import apply_diffusers_transformers_v5_single_file_patch

    apply_diffusers_transformers_v5_single_file_patch()
    assert single_file.create_diffusers_clip_model_from_ldm is sfu.create_diffusers_clip_model_from_ldm
    assert single_file.create_diffusers_clip_model_from_ldm.__module__ == "rengu_flow.utils.diffusers_tf5_compat"
