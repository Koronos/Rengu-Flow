"""Dataset preparation: tagging, captioning, watermark cleanup, and tag editing.

Runs outside the training path (own CLI stage processes / UI section). Submodules are
attached lazily (SPEC 1) so importing one helper never pulls heavy inference deps.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "caption_store": ["CaptionSet", "CaptionStore"],
        "config": ["PrepConfig", "load_prep_config", "parse_prep_config"],
        "runner": ["run_stage"],
        "captioner": [
            "CaptionBackend",
            "CaptionerConfig",
            "JoyCaptionBackend",
            "ToriiGateBackend",
            "build_prompt",
            "caption_folder",
            "list_caption_models",
        ],
        "cleanup": [
            "CleanupConfig",
            "WatermarkDetector",
            "LamaInpainter",
            "boxes_to_mask",
            "clean_folder",
        ],
        "tag_ops": ["TagEditOp", "TagFilter", "apply_ops", "tag_frequencies"],
        "tagger": [
            "TaggerModelSpec",
            "KNOWN_TAGGERS",
            "OnnxTagger",
            "merge_model_results",
            "run_ensemble",
        ],
        "models": ["list_models", "ensure_model"],
    },
)
