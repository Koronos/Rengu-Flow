from renga_flow.utils.diffusers_tf5_compat import strip_text_model_state_dict_prefix


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
