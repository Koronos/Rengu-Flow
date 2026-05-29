"""Synthetic dataset for Phase 1 minimal loop (SDXL-compatible batch format)."""

import torch
from torch.utils.data import Dataset


class SyntheticSDXLDataset(Dataset):
    """Yields batches with latents, caption, mask for SDXL prepare_inputs (no real data/cache)."""

    def __init__(
        self,
        num_batches: int = 100,
        micro_batch_size: int = 2,
        latent_channels: int = 4,
        latent_height: int = 64,
        latent_width: int = 64,
        device: str = "cpu",
    ):
        self.num_batches = num_batches
        self.micro_batch_size = micro_batch_size
        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.device = device

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, idx: int) -> dict:
        torch.manual_seed(idx)
        latents = torch.randn(
            self.micro_batch_size,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=torch.float32,
        )
        caption = ["synthetic caption"] * self.micro_batch_size
        mask = torch.ones(self.micro_batch_size, 1, self.latent_height, self.latent_width, device=self.device, dtype=torch.float32)
        return {"latents": latents, "caption": caption, "mask": mask}
