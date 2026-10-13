# Nanosaur2 modeling code

`rengu_flow/model/nanosaur2/dit.py`, `vae.py` and `text.py` are ports to plain PyTorch of the
official ComfyUI custom node `nanosaur2_support/` (`model.py`, `vae.py`, `text_encoder.py`)
published with the `well9472/Nanosaur2-670M` model on the Hugging Face Hub (snapshot
`6120c6a4e613d26f35b2d4f7c5943d6ed8ceba95`). The training recipe in `pipeline.py` follows the
same repository's `train_lora.py`.

- Author: well9472 (Hugging Face).
- License: MIT (as declared by the model repository).

Adaptations (no numeric change; see `scripts/parity_nanosaur2.py` for the parity check against
the node): ComfyUI operations / attention / fused kernels replaced by `torch.nn` layers and
`F.scaled_dot_product_attention`; the DiT forward is split into pieces shared with the pipeline
layers; Gemma3-270M runs through `transformers.Gemma3TextModel`. Module and parameter names are
unchanged, so the release checkpoints load verbatim and exports keep the official layout.

`vae.py` also writes out the DINOv2 transformer (Meta, Apache License 2.0) and the LDM
convolutional decoder (CompVis latent-diffusion, MIT License) structures the checkpoint uses;
no code was copied from ComfyUI (GPL-3.0) beyond the parameter-name correspondence.

The model weights are distributed by their author on the Hugging Face Hub; rengu-flow does not
redistribute them.
