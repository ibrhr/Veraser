from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import importlib
import logging
import os
import sys
from time import perf_counter
from typing import Any

from PIL import Image

from app.core.config import Settings
from app.models.base import VideoInpaintingResult
from app.schemas.performance import OperationSpeedMetric, build_speed_metric
from app.services.errors import ModelRuntimeError


logger = logging.getLogger(__name__)


class SttnVideoInpaintingModel:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._torch: Any | None = None
        self._transforms: Any | None = None
        self._stack_class: type[Any] | None = None
        self._to_torch_format_tensor_class: type[Any] | None = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            logger.debug("STTN model already loaded")
            return
        logger.info(
            "Loading STTN model repo_path=%s checkpoint_path=%s device=%s",
            self.settings.sttn_repo_path,
            self.settings.sttn_checkpoint_path,
            self.settings.sttn_device,
        )
        if not self.settings.sttn_repo_path.exists():
            raise ModelRuntimeError(
                f"STTN repo path does not exist: {self.settings.sttn_repo_path}. "
                "Run `python3 scripts/setup_sttn.py` first."
            )
        if not self.settings.sttn_checkpoint_path.exists():
            raise ModelRuntimeError(
                f"STTN checkpoint does not exist: {self.settings.sttn_checkpoint_path}. "
                "Run `python3 scripts/setup_sttn.py` first."
            )

        repo_path = str(self.settings.sttn_repo_path)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            import torch
            from torchvision import transforms
            from core.utils import Stack, ToTorchFormatTensor
        except Exception as exc:
            raise ModelRuntimeError(
                "Could not import STTN runtime dependencies. Install `uv sync --group gpu` first."
            ) from exc

        with self._sttn_working_directory():
            try:
                logger.info("Importing STTN model module")
                module = importlib.import_module("model.sttn")
                model = module.InpaintGenerator().to(self.settings.sttn_device)
                logger.info("Loading STTN checkpoint checkpoint_path=%s", self.settings.sttn_checkpoint_path)
                data = torch.load(self.settings.sttn_checkpoint_path, map_location=self.settings.sttn_device)
                model.load_state_dict(data["netG"])
                model.eval()
            except Exception as exc:
                raise ModelRuntimeError("Could not load STTN model or checkpoint.") from exc

        self._torch = torch
        self._transforms = transforms
        self._stack_class = Stack
        self._to_torch_format_tensor_class = ToTorchFormatTensor
        self._model = model
        self._loaded = True
        logger.info("STTN model loaded")

    def inpaint_video(
        self,
        *,
        session_id: str,
        frames_dir: Path,
        masks_dir: Path,
        output_video_path: Path,
    ) -> VideoInpaintingResult:
        self.load()
        frame_paths = sorted(frames_dir.glob("*.png"))
        mask_paths = sorted((masks_dir / "combined").glob("*.png"))
        if not frame_paths:
            raise ModelRuntimeError("No extracted frames were found for STTN inpainting.")
        if len(frame_paths) != len(mask_paths):
            raise ModelRuntimeError(
                f"STTN requires one combined mask per frame; found {len(mask_paths)} masks for {len(frame_paths)} frames."
            )
        if self._model is None or self._torch is None:
            raise ModelRuntimeError("STTN model is not loaded.")
        logger.info(
            "Starting STTN inpainting session_id=%s frames_dir=%s masks_dir=%s output_video_path=%s frame_count=%s mask_count=%s",
            session_id,
            frames_dir,
            masks_dir,
            output_video_path,
            len(frame_paths),
            len(mask_paths),
        )

        performance: list[OperationSpeedMetric] = []
        read_frames_started_at = perf_counter()
        logger.info("Reading STTN frames session_id=%s frame_count=%s", session_id, len(frame_paths))
        frames, original_size = self._read_frames(frame_paths)
        performance.append(
            build_speed_metric(
                name="sttn_read_frames",
                label="STTN read frames",
                elapsed_seconds=perf_counter() - read_frames_started_at,
                frames_processed=len(frame_paths),
            )
        )
        read_masks_started_at = perf_counter()
        logger.info("Reading STTN masks session_id=%s mask_count=%s", session_id, len(mask_paths))
        masks, binary_masks = self._read_masks(mask_paths)
        performance.append(
            build_speed_metric(
                name="sttn_read_masks",
                label="STTN read masks",
                elapsed_seconds=perf_counter() - read_masks_started_at,
                frames_processed=len(mask_paths),
            )
        )
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        inference_started_at = perf_counter()
        logger.info("Running STTN inference session_id=%s frame_count=%s", session_id, len(frame_paths))
        comp_frames = self._run_sttn(frames=frames, masks=masks, binary_masks=binary_masks)
        performance.append(
            build_speed_metric(
                name="sttn_inference",
                label="STTN inference",
                elapsed_seconds=perf_counter() - inference_started_at,
                frames_processed=len(frame_paths),
            )
        )
        write_started_at = perf_counter()
        logger.info("Writing STTN output video session_id=%s output_video_path=%s", session_id, output_video_path)
        self._write_video(comp_frames, output_video_path, original_size)
        performance.append(
            build_speed_metric(
                name="sttn_video_write",
                label="STTN video write",
                elapsed_seconds=perf_counter() - write_started_at,
                frames_processed=len(comp_frames),
            )
        )
        return VideoInpaintingResult(
            frames_total=len(frame_paths),
            frames_done=len(frame_paths),
            video_path=output_video_path,
            performance=performance,
        )

    def _read_frames(self, frame_paths: list[Path]) -> tuple[list[Any], tuple[int, int]]:
        width = self.settings.sttn_width
        height = self.settings.sttn_height
        first = Image.open(frame_paths[0]).convert("RGB")
        original_size = first.size
        frames = [first.resize((width, height))]
        logger.debug("Read STTN frame index=0 path=%s original_size=%s resized_size=%sx%s", frame_paths[0], original_size, width, height)
        for frame_index, path in enumerate(frame_paths[1:], start=1):
            frames.append(Image.open(path).convert("RGB").resize((width, height)))
            logger.debug("Read STTN frame index=%s path=%s resized_size=%sx%s", frame_index, path, width, height)
        return frames, original_size

    def _read_masks(self, mask_paths: list[Path]) -> tuple[list[Any], list[Any]]:
        import cv2
        import numpy as np

        width = self.settings.sttn_width
        height = self.settings.sttn_height
        masks = []
        binary_masks = []
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for mask_index, mask_path in enumerate(mask_paths):
            mask = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
            mask_array = (np.array(mask) > 0).astype("uint8")
            if self.settings.sttn_mask_dilation_iterations > 0:
                mask_array = cv2.dilate(mask_array, kernel, iterations=self.settings.sttn_mask_dilation_iterations)
            binary_masks.append(np.expand_dims(mask_array, 2))
            masks.append(Image.fromarray(mask_array * 255))
            logger.debug("Read STTN mask index=%s path=%s resized_size=%sx%s", mask_index, mask_path, width, height)
        return masks, binary_masks

    def _run_sttn(self, *, frames: list[Any], masks: list[Any], binary_masks: list[Any]) -> list[Any]:
        import numpy as np

        torch = self._torch
        if (
            torch is None
            or self._model is None
            or self._transforms is None
            or self._stack_class is None
            or self._to_torch_format_tensor_class is None
        ):
            raise ModelRuntimeError("STTN model is not loaded.")

        to_tensors = self._transforms.Compose([self._stack_class(), self._to_torch_format_tensor_class()])
        video_length = len(frames)
        frame_arrays = [np.array(frame).astype("uint8") for frame in frames]
        feats = to_tensors(frames).unsqueeze(0) * 2 - 1
        mask_tensors = to_tensors(masks).unsqueeze(0)
        feats = feats.to(self.settings.sttn_device)
        mask_tensors = mask_tensors.to(self.settings.sttn_device)
        comp_frames: list[Any] = [None] * video_length

        with torch.no_grad():
            encoded_feats = self._model.encoder((feats * (1 - mask_tensors).float()).view(video_length, 3, self.settings.sttn_height, self.settings.sttn_width))
            _, channels, feat_height, feat_width = encoded_feats.size()
            encoded_feats = encoded_feats.view(1, video_length, channels, feat_height, feat_width)

            for frame_index in range(0, video_length, self.settings.sttn_neighbor_stride):
                neighbor_ids = list(
                    range(
                        max(0, frame_index - self.settings.sttn_neighbor_stride),
                        min(video_length, frame_index + self.settings.sttn_neighbor_stride + 1),
                    )
                )
                ref_ids = self._get_ref_index(neighbor_ids, video_length)
                logger.debug(
                    "Running STTN window frame_index=%s neighbor_ids=%s ref_ids=%s",
                    frame_index,
                    neighbor_ids,
                    ref_ids,
                )
                pred_feat = self._model.infer(
                    encoded_feats[0, neighbor_ids + ref_ids, :, :, :],
                    mask_tensors[0, neighbor_ids + ref_ids, :, :, :],
                )
                pred_img = torch.tanh(self._model.decoder(pred_feat[: len(neighbor_ids), :, :, :])).detach()
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                for index, source_frame_index in enumerate(neighbor_ids):
                    img = np.array(pred_img[index]).astype("uint8") * binary_masks[source_frame_index] + frame_arrays[source_frame_index] * (
                        1 - binary_masks[source_frame_index]
                    )
                    if comp_frames[source_frame_index] is None:
                        comp_frames[source_frame_index] = img
                    else:
                        comp_frames[source_frame_index] = comp_frames[source_frame_index].astype("float32") * 0.5 + img.astype("float32") * 0.5
                    logger.debug("Composited STTN frame source_frame_index=%s", source_frame_index)

        logger.info("STTN inference windows complete frame_count=%s", video_length)
        return [
            np.array(comp_frames[index]).astype("uint8") * binary_masks[index] + frame_arrays[index] * (1 - binary_masks[index])
            for index in range(video_length)
        ]

    def _get_ref_index(self, neighbor_ids: list[int], length: int) -> list[int]:
        return [
            index
            for index in range(0, length, self.settings.sttn_ref_length)
            if index not in neighbor_ids
        ]

    def _write_video(self, frames: list[Any], output_video_path: Path, original_size: tuple[int, int]) -> None:
        import cv2
        import numpy as np

        output_size = (self.settings.sttn_width, self.settings.sttn_height)
        if self.settings.sttn_preserve_source_resolution:
            output_size = original_size
        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.settings.sttn_output_fps,
            output_size,
        )
        try:
            for frame_index, frame in enumerate(frames):
                frame_array = np.array(frame).astype("uint8")
                if output_size != (self.settings.sttn_width, self.settings.sttn_height):
                    frame_array = cv2.resize(frame_array, output_size, interpolation=cv2.INTER_LINEAR)
                writer.write(cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR))
                logger.debug("Wrote STTN video frame index=%s output_video_path=%s", frame_index, output_video_path)
        finally:
            writer.release()
            logger.info("Released STTN video writer output_video_path=%s output_size=%s", output_video_path, output_size)

    @contextmanager
    def _sttn_working_directory(self):
        previous_cwd = Path.cwd()
        os.chdir(self.settings.sttn_repo_path)
        try:
            yield
        finally:
            os.chdir(previous_cwd)
