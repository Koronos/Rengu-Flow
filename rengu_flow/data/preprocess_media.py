"""Media preprocessing for dataset cache (ported from diffusion-pipe PreprocessMediaFile)."""

from __future__ import annotations

import tarfile
from pathlib import Path

import imageio
import torch
from PIL import Image, ImageOps

from rengu_flow.data.augmentation import (
    apply_augmentation,
    augmentation_seed_for_image,
    image_spec_base,
    image_spec_variant_key,
)
from rengu_flow.data.dataset import VIDEO_EXTENSIONS, _webp_frame_count
from rengu_flow.utils.common import round_down_to_multiple, round_to_nearest_multiple
from rengu_flow.utils.logging import get_logger

logger = get_logger(__name__)


def _ensure_mask_matches_image(
    mask_pil: Image.Image,
    height: int,
    width: int,
    *,
    image_path: str,
    mask_path: str,
) -> None:
    mask_hw = (mask_pil.height, mask_pil.width)
    if mask_hw != (height, width):
        raise ValueError(
            f"Mask shape {mask_hw} was not the same as image shape {(height, width)}.\n"
            f"Image path: {image_path}\nMask path: {mask_path}"
        )


def convert_crop_and_resize(pil_img, width_and_height):
    if pil_img.mode not in ["RGB", "RGBA"] and "transparency" in pil_img.info:
        pil_img = pil_img.convert("RGBA")
    if pil_img.mode == "RGBA":
        canvas = Image.new("RGBA", pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert("RGB")
    else:
        pil_img = pil_img.convert("RGB")
    return ImageOps.fit(pil_img, width_and_height)


def extract_clips(video, target_frames, video_clip_mode):
    frames = video.shape[1]
    if frames < target_frames:
        print(
            f"video with shape {video.shape} is being skipped because it has less "
            f"({frames}) than the target_frames {target_frames}"
        )
        return []
    if video_clip_mode == "single_beginning":
        return [video[:, :target_frames, ...]]
    if video_clip_mode == "single_middle":
        start = int((frames - target_frames) / 2)
        return [video[:, start : start + target_frames, ...]]
    raise NotImplementedError(f"video_clip_mode={video_clip_mode} is not recognized")


class PreprocessMediaFile:
    def __init__(
        self,
        config,
        support_video=False,
        framerate=None,
        round_height=16,
        round_width=16,
        round_frames=4,
        augmentation_resolver=None,
    ):
        self.config = config
        self.augmentation_resolver = augmentation_resolver
        self.video_clip_mode = config.get("video_clip_mode", "single_beginning")
        from torchvision import transforms

        self.pil_to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.support_video = support_video
        self.framerate = framerate
        self.round_height = round_height
        self.round_width = round_width
        self.round_frames = round_frames
        if self.support_video:
            assert self.framerate
        self.tarfile_map = {}

    def __del__(self):
        for tar_f in self.tarfile_map.values():
            tar_f.close()

    def _apply_augmentation_if_needed(self, pil_img, mask, spec):
        resolved_pack = self.augmentation_resolver(spec)
        if not resolved_pack:
            return pil_img, mask
        resolved, aug_fingerprint = resolved_pack
        variant_key = image_spec_variant_key(spec)
        seed = augmentation_seed_for_image(
            image_spec_base(spec), aug_fingerprint, variant_key
        )
        return apply_augmentation(
            pil_img, mask, seed, resolved, variant_key=variant_key
        )

    def __call__(self, spec, mask_filepath, size_bucket=None):
        from torchvision.transforms import functional as tv_functional

        suffix = Path(spec[1]).suffix

        if spec[0] is None:
            tar_f = None
            filepath_or_file = str(spec[1])
        else:
            tar_filename = spec[0]
            if tar_filename not in self.tarfile_map:
                self.tarfile_map[tar_filename] = tarfile.TarFile(tar_filename)
            tar_f = self.tarfile_map[tar_filename]
            filepath_or_file = tar_f.extractfile(str(spec[1]))

        # Animated WebP is decoded as a video by its native frames (the pillow webp
        # reader can't resample to the model framerate). Path-based only.
        is_webp_video = (
            self.support_video
            and spec[0] is None
            and suffix == ".webp"
            and _webp_frame_count(filepath_or_file) > 1
        )
        is_video = suffix in VIDEO_EXTENSIONS or is_webp_video

        valid = True  # False = corrupt/truncated -> tombstone (zero placeholder, skipped at train)
        if is_video:
            assert self.support_video
            iter_kwargs = {} if is_webp_video else {"fps": self.framerate}
            num_frames = 0
            for _frame in imageio.v3.imiter(filepath_or_file, **iter_kwargs):
                num_frames += 1
            video = imageio.v3.imiter(filepath_or_file, **iter_kwargs)
        else:
            num_frames = 1
            mask_pil = None
            try:
                pil_img = Image.open(filepath_or_file)
                height, width = pil_img.height, pil_img.width
                if mask_filepath:
                    mask_pil = Image.open(mask_filepath).convert("RGB")
                    _ensure_mask_matches_image(
                        mask_pil, height, width, image_path=spec[1], mask_path=mask_filepath,
                    )
                if self.augmentation_resolver is not None:
                    pil_img, mask_pil = self._apply_augmentation_if_needed(
                        pil_img, mask_pil, spec
                    )
                    height, width = pil_img.height, pil_img.width
                pil_img.load()  # force full decode HERE so truncation surfaces in this try
                video = [pil_img]
            except (OSError, SyntaxError) as e:  # UnidentifiedImageError is an OSError subclass
                if size_bucket is None:
                    raise  # no bucket geometry to size a placeholder from
                logger.warning("Corrupt/truncated image tombstoned at latent encode: %s (%s)",
                               spec[1], e)
                valid = False
                video = None
                width = height = 0
                mask_pil = None

        if size_bucket is not None:
            size_bucket_width, size_bucket_height, size_bucket_frames = size_bucket
        else:
            size_bucket_width, size_bucket_height, size_bucket_frames = width, height, num_frames

        height_rounded = round_to_nearest_multiple(size_bucket_height, self.round_height)
        width_rounded = round_to_nearest_multiple(size_bucket_width, self.round_width)
        frames_rounded = round_down_to_multiple(size_bucket_frames - 1, self.round_frames) + 1
        resize_wh = (width_rounded, height_rounded)

        mask = None
        if valid and not is_video and mask_filepath:
            mask_img = mask_pil if mask_pil is not None else Image.open(mask_filepath).convert("RGB")
            mask_img = ImageOps.fit(mask_img, resize_wh)
            mask = tv_functional.to_tensor(mask_img)[0].to(torch.float16)
        elif is_video and mask_filepath:
            mask_img = Image.open(mask_filepath).convert("RGB")
            _ensure_mask_matches_image(
                mask_img,
                height,
                width,
                image_path=spec[1],
                mask_path=mask_filepath,
            )
            mask_img = ImageOps.fit(mask_img, resize_wh)
            mask = tv_functional.to_tensor(mask_img)[0].to(torch.float16)

        if valid:
            resized_video = torch.empty((num_frames, 3, height_rounded, width_rounded))
            for i, frame in enumerate(video):
                if not isinstance(frame, Image.Image):
                    frame = tv_functional.to_pil_image(frame)
                cropped_image = convert_crop_and_resize(frame, resize_wh)
                resized_video[i, ...] = self.pil_to_tensor(cropped_image)
        else:  # tombstone: correctly-shaped zeros; marked invalid, never sampled at train
            resized_video = torch.zeros((num_frames, 3, height_rounded, width_rounded))

        if hasattr(filepath_or_file, "close"):
            filepath_or_file.close()

        if not self.support_video:
            return [(resized_video.squeeze(0), mask, valid)]

        resized_video = torch.permute(resized_video, (1, 0, 2, 3))
        if not is_video:
            return [(resized_video, mask, valid)]
        videos = extract_clips(resized_video, frames_rounded, self.video_clip_mode)
        return [(video, mask, valid) for video in videos]
