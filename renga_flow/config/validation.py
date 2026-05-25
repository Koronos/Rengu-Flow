"""Minimal validation of config: required sections for the standard flow."""


class ConfigValidationError(ValueError):
    """Raised when required config keys are missing."""


_REQUIRED_TOP_LEVEL = ("model", "optimizer", "dataset")


def validate_config(config: dict) -> None:
    """Check that config has the minimum required sections for the orchestrator.

    Required: 'model', 'optimizer', 'dataset'. 'model' must have 'type' and 'dtype';
    'optimizer' must have 'type'.

    The 'adapter' section is optional. When absent, full-model finetuning is used
    (all trainable parameters; save via save_full_model). When present, adapter
    training (e.g. LoRA/LoKr) is used and only adapter weights are saved.

    Raises:
        ConfigValidationError: If any required key is missing.
    """
    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in config]
    if missing:
        raise ConfigValidationError(
            f"Config missing required sections: {missing}. Required: {list(_REQUIRED_TOP_LEVEL)}."
        )
    if "type" not in config["model"]:
        raise ConfigValidationError("config['model'] must contain 'type'.")
    if "dtype" not in config["model"]:
        raise ConfigValidationError("config['model'] must contain 'dtype'.")
    model_type = str(config["model"]["type"]).lower()
    if model_type in ("cosmos_predict2", "anima"):
        model = config["model"]
        for key in ("transformer_path", "vae_path"):
            if key not in model:
                raise ConfigValidationError(f"config['model'] must contain '{key}' for {model_type}.")
        if model_type == "anima":
            if "llm_path" not in model:
                raise ConfigValidationError("config['model'] must contain 'llm_path' when type is 'anima'.")
        elif "llm_path" not in model and "t5_path" not in model:
            raise ConfigValidationError(
                "config['model'] must contain 'llm_path' (Qwen3 / Anima checkpoints) or 't5_path'."
            )
    if "type" not in config["optimizer"]:
        raise ConfigValidationError("config['optimizer'] must contain 'type'.")
    optimizer = config["optimizer"]
    if optimizer.get("gradient_release") and config.get("pipeline_stages", 1) != 1:
        raise ConfigValidationError(
            "gradient_release requires pipeline_stages = 1 (one GPU pipeline stage)."
        )
    if "adapter" in config:
        adapter = config["adapter"]
        if "type" not in adapter:
            raise ConfigValidationError("config['adapter'] must contain 'type' when adapter is present.")
        if adapter["type"] not in ("lora", "lokr"):
            raise ConfigValidationError(
                f"config['adapter']['type'] must be 'lora' or 'lokr', got {adapter['type']!r}."
            )
        if "rank" not in adapter and "dim" not in adapter:
            raise ConfigValidationError(
                "config['adapter'] must contain 'rank' or 'dim' when adapter is present."
            )
