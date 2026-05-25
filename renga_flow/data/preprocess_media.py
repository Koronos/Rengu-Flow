"""Media preprocessing for dataset cache (ported from diffusion-pipe PreprocessMediaFile)."""

from __future__ import annotations

import tarfile
from pathlib import Path

import imageio
import torch
from PIL import Image, ImageOps

from renga_flow.data.dataset import VIDEO_EXTENSIONS
from renga_flow.utils.common import round_down_to_multiple, round_to_nearest_multiple


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
    ):
        self.config = config
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

    def __call__(self, spec, mask_filepath, size_bucket=None):
        from torchvision.transforms import functional as tv_functional

        is_video = Path(spec[1]).suffix in VIDEO_EXTENSIONS

        if spec[0] is None:
            tar_f = None
            filepath_or_file = str(spec[1])
        else:
            tar_filename = spec[0]
            if tar_filename not in self.tarfile_map:
                self.tarfile_map[tar_filename] = tarfile.TarFile(tar_filename)
            tar_f = self.tarfile_map[tar_filename]
            filepath_or_file = tar_f.extractfile(str(spec[1]))

        if is_video:
            assert self.support_video
            num_frames = 0
            for _frame in imageio.v3.imiter(filepath_or_file, fps=self.framerate):
                num_frames += 1
            video = imageio.v3.imiter(filepath_or_file, fps=self.framerate)
        else:
            num_frames = 1
            pil_img = Image.open(filepath_or_file)
            height, width = pil_img.height, pil_img.width
            video = [pil_img]

        if size_bucket is not None:
            size_bucket_width, size_bucket_height, size_bucket_frames = size_bucket
        else:
            size_bucket_width, size_bucket_height, size_bucket_frames = width, height, num_frames

        height_rounded = round_to_nearest_multiple(size_bucket_height, self.round_height)
        width_rounded = round_to_nearest_multiple(size_bucket_width, self.round_width)
        frames_rounded = round_down_to_multiple(size_bucket_frames - 1, self.round_frames) + 1
        resize_wh = (width_rounded, height_rounded)

        if mask_filepath:
            mask_img = Image.open(mask_filepath).convert("RGB")
            img_hw = (height, width)
            mask_hw = (mask_img.height, mask_img.width)
            if mask_hw != img_hw:
                raise ValueError(
                    f"Mask shape {mask_hw} was not the same as image shape {img_hw}.\n"
                    f"Image path: {spec[1]}\nMask path: {mask_filepath}"
                )
            mask_img = ImageOps.fit(mask_img, resize_wh)
            mask = tv_functional.to_tensor(mask_img)[0].to(torch.float16)
        else:
            mask = None

        resized_video = torch.empty((num_frames, 3, height_rounded, width_rounded))
        for i, frame in enumerate(video):
            if not isinstance(frame, Image.Image):
                frame = tv_functional.to_pil_image(frame)
            cropped_image = convert_crop_and_resize(frame, resize_wh)
            resized_video[i, ...] = self.pil_to_tensor(cropped_image)

        if hasattr(filepath_or_file, "close"):
            filepath_or_file.close()

        if not self.support_video:
            return [(resized_video.squeeze(0), mask)]

        resized_video = torch.permute(resized_video, (1, 0, 2, 3))
        if not is_video:
            return [(resized_video, mask)]
        videos = extract_clips(resized_video, frames_rounded, self.video_clip_mode)
        return [(video, mask) for video in videos]
