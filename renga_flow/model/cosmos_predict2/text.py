"""Text encoders and tokenization for Cosmos Predict2 (Qwen3 + T5 aux)."""

from __future__ import annotations

import os

import transformers
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoModelForCausalLM, AutoTokenizer, T5EncoderModel, T5TokenizerFast

from renga_flow.model.cosmos_predict2.paths import qwen3_config_dir, t5_config_dir
from renga_flow.utils.common import iterate_safetensors, load_state_dict


def tokenize(tokenizer, prompts):
    return tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )


def compute_text_embeddings(text_encoder, input_ids, attn_mask):
    input_ids = input_ids.to(text_encoder.device)
    attn_mask = attn_mask.to(text_encoder.device)
    outputs = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state
    encoded_text[~attn_mask.bool()] = 0
    return encoded_text


def load_text_stack(model_config):
    """Load primary tokenizer/encoder and bundled T5 tokenizer (auxiliary captions).

    Returns:
        tokenizer, t5_tokenizer, text_encoder, is_generic_llm, pipeline_name
    """
    dtype = model_config["dtype"]
    pipeline_name = "cosmos_predict2"
    t5_dir = t5_config_dir()
    t5_tokenizer = T5TokenizerFast(
        vocab_file=os.path.join(t5_dir, "spiece.model"),
        tokenizer_file=os.path.join(t5_dir, "tokenizer.json"),
    )

    if "t5_path" in model_config:
        tokenizer = t5_tokenizer
        t5_state_dict = load_state_dict(model_config["t5_path"])
        text_encoder = T5EncoderModel.from_pretrained(
            None,
            config=os.path.join(t5_dir, "config.json"),
            state_dict=t5_state_dict,
            torch_dtype="auto",
            local_files_only=True,
        )
        return tokenizer, t5_tokenizer, text_encoder, False, pipeline_name

    if "llm_path" not in model_config:
        raise RuntimeError("model config must contain llm_path or t5_path")

    llm_path = model_config["llm_path"]
    if os.path.isdir(llm_path):
        tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
        text_encoder = AutoModelForCausalLM.from_pretrained(llm_path, dtype=dtype, local_files_only=True)
    else:
        qwen_dir = qwen3_config_dir()
        tokenizer = AutoTokenizer.from_pretrained(qwen_dir, local_files_only=True)
        llm_config = transformers.Qwen3Config.from_pretrained(qwen_dir, local_files_only=True)
        with init_empty_weights():
            text_encoder = transformers.Qwen3ForCausalLM(llm_config)
        for key, tensor in iterate_safetensors(llm_path):
            set_module_tensor_to_device(text_encoder, key, device="cpu", dtype=dtype, value=tensor)

    text_encoder = text_encoder.model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder.config.use_cache = False
    return tokenizer, t5_tokenizer, text_encoder, True, pipeline_name
