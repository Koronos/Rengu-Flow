"""Infer MiniTrainDIT config from checkpoint state dict."""

from __future__ import annotations


def get_dit_config(state_dict, key_prefix=""):
    dit_config = {}
    dit_config["max_img_h"] = 1024
    dit_config["max_img_w"] = 1024
    dit_config["max_frames"] = 128
    concat_padding_mask = True
    dit_config["in_channels"] = (
        state_dict["{}x_embedder.proj.1.weight".format(key_prefix)].shape[1] // 4
    ) - int(concat_padding_mask)
    dit_config["out_channels"] = 16
    dit_config["patch_spatial"] = 2
    dit_config["patch_temporal"] = 1
    dit_config["model_channels"] = state_dict["{}x_embedder.proj.1.weight".format(key_prefix)].shape[0]
    dit_config["concat_padding_mask"] = concat_padding_mask
    dit_config["crossattn_emb_channels"] = 1024
    dit_config["pos_emb_cls"] = "rope3d"
    dit_config["pos_emb_learnable"] = True
    dit_config["pos_emb_interpolation"] = "crop"
    dit_config["min_fps"] = 1
    dit_config["max_fps"] = 30
    dit_config["use_adaln_lora"] = True
    dit_config["adaln_lora_dim"] = 256

    if dit_config["model_channels"] == 2048:
        dit_config["num_blocks"] = 28
        dit_config["num_heads"] = 16
    elif dit_config["model_channels"] == 5120:
        dit_config["num_blocks"] = 36
        dit_config["num_heads"] = 40
    elif dit_config["model_channels"] == 1280:
        dit_config["num_blocks"] = 20
        dit_config["num_heads"] = 20

    if dit_config["in_channels"] == 16:
        dit_config["extra_per_block_abs_pos_emb"] = False
        dit_config["rope_h_extrapolation_ratio"] = 4.0
        dit_config["rope_w_extrapolation_ratio"] = 4.0
        dit_config["rope_t_extrapolation_ratio"] = 1.0
    elif dit_config["in_channels"] == 17:
        dit_config["extra_per_block_abs_pos_emb"] = False
        dit_config["rope_h_extrapolation_ratio"] = 3.0
        dit_config["rope_w_extrapolation_ratio"] = 3.0
        dit_config["rope_t_extrapolation_ratio"] = 1.0

    dit_config["extra_h_extrapolation_ratio"] = 1.0
    dit_config["extra_w_extrapolation_ratio"] = 1.0
    dit_config["extra_t_extrapolation_ratio"] = 1.0
    dit_config["rope_enable_fps_modulation"] = False
    return dit_config
