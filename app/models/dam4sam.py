from pathlib import Path
from contextlib import contextmanager
import logging
import os
import sys
from time import perf_counter
from typing import Any

from PIL import Image

from app.core.config import Settings
from app.models.base import VideoMaskingResult
from app.schemas.performance import OperationSpeedMetric, build_speed_metric
from app.schemas.prompts import BoxPrompt, StoredObjectPrompt
from app.services.errors import ModelRuntimeError
from app.services.json_store import write_json


logger = logging.getLogger(__name__)


class D4smVideoMaskingModel:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tracker_class: type[Any] | None = None
        self._tracker: Any | None = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            logger.debug("D4SM model already loaded")
            return
        logger.info(
            "Loading D4SM model repo_path=%s checkpoint_dir=%s model_size=%s",
            self.settings.d4sm_repo_path,
            self.settings.d4sm_checkpoint_dir,
            self.settings.d4sm_model_size,
        )
        if not self.settings.d4sm_repo_path.exists():
            raise ModelRuntimeError(
                f"D4SM repo path does not exist: {self.settings.d4sm_repo_path}. "
                "Run `python3 scripts/setup_d4sm.py --model-size large` first."
            )
        if not self.settings.d4sm_checkpoint_dir.exists():
            raise ModelRuntimeError(
                f"D4SM checkpoint directory does not exist: {self.settings.d4sm_checkpoint_dir}. "
                "Run `python3 scripts/setup_d4sm.py --model-size large` first."
            )

        repo_path = str(self.settings.d4sm_repo_path.resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            from tracking_wrapper_mot import DAM4SAMMOT
        except Exception as exc:
            raise ModelRuntimeError("Could not import D4SM tracking_wrapper_mot.DAM4SAMMOT.") from exc

        self._tracker_class = DAM4SAMMOT
        self._loaded = True
        logger.info("D4SM model loaded")

    def generate_masks(
        self,
        *,
        session_id: str,
        frames_dir: Path,
        output_dir: Path,
        objects: list[StoredObjectPrompt],
    ) -> VideoMaskingResult:
        self.load()
        frame_paths = sorted(frames_dir.glob("*.png"))
        if not frame_paths:
            raise ModelRuntimeError("No extracted frames were found for D4SM masking.")
        if self._tracker_class is None:
            raise ModelRuntimeError("D4SM tracker class is not loaded.")
        logger.info(
            "Starting D4SM mask generation session_id=%s frames_dir=%s output_dir=%s frame_count=%s object_count=%s",
            session_id,
            frames_dir,
            output_dir,
            len(frame_paths),
            len(objects),
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        combined_dir = output_dir / "combined"
        per_object_dir = output_dir / "objects"
        combined_dir.mkdir(parents=True, exist_ok=True)
        per_object_dir.mkdir(parents=True, exist_ok=True)

        performance: list[OperationSpeedMetric] = []
        tracker_started_at = perf_counter()
        logger.info("Initializing D4SM tracker session_id=%s", session_id)
        tracker = self._create_tracker()
        init_image = Image.open(frame_paths[0]).convert("RGB")
        init_regions = self._objects_to_init_regions(objects)
        logger.info("Initializing D4SM objects session_id=%s init_region_count=%s", session_id, len(init_regions))
        self._initialize_tracker(tracker=tracker, image=init_image, init_regions=init_regions)
        performance.append(
            build_speed_metric(
                name="d4sm_initialize",
                label="DAM4SAM initialize",
                elapsed_seconds=perf_counter() - tracker_started_at,
                frames_processed=1,
            )
        )

        initial_mask_started_at = perf_counter()
        logger.info("Writing initial D4SM masks session_id=%s frame_index=0", session_id)
        initial_masks = self._initial_masks_from_tracker(tracker=tracker, image_size=init_image.size)
        manifest_frames = [
            self._write_frame_masks(
                masks=initial_masks,
                objects=objects,
                frame_index=0,
                combined_dir=combined_dir,
                per_object_dir=per_object_dir,
            )
        ]
        performance.append(
            build_speed_metric(
                name="initial_mask_write",
                label="Initial mask write",
                elapsed_seconds=perf_counter() - initial_mask_started_at,
                frames_processed=1,
            )
        )
        tracking_started_at = perf_counter()
        for frame_index, frame_path in enumerate(frame_paths[1:], start=1):
            logger.debug(
                "Tracking D4SM frame session_id=%s frame_index=%s frame_path=%s",
                session_id,
                frame_index,
                frame_path,
            )
            image = Image.open(frame_path).convert("RGB")
            outputs = self._track_frame(tracker=tracker, image=image, frame_index=frame_index)
            masks = outputs["masks"]
            frame_artifacts = self._write_frame_masks(
                masks=masks,
                objects=objects,
                frame_index=frame_index,
                combined_dir=combined_dir,
                per_object_dir=per_object_dir,
            )
            manifest_frames.append(frame_artifacts)
            logger.debug("Wrote D4SM frame masks session_id=%s frame_index=%s", session_id, frame_index)
        logger.info(
            "D4SM tracking complete session_id=%s tracked_frames=%s",
            session_id,
            max(len(frame_paths) - 1, 0),
        )
        performance.append(
            build_speed_metric(
                name="d4sm_tracking",
                label="DAM4SAM tracking",
                elapsed_seconds=perf_counter() - tracking_started_at,
                frames_processed=max(len(frame_paths) - 1, 0),
            )
        )

        manifest_path = output_dir / "manifest.json"
        manifest_started_at = perf_counter()
        logger.info("Writing D4SM mask manifest session_id=%s manifest_path=%s", session_id, manifest_path)
        write_json(
            manifest_path,
            {
                "session_id": session_id,
                "model": "d4sm",
                "model_size": self.settings.d4sm_model_size,
                "frames_total": len(frame_paths),
                "objects": [item.object_id for item in objects],
                "processed_video": None,
                "frames": manifest_frames,
            },
        )
        performance.append(
            build_speed_metric(
                name="mask_manifest_write",
                label="Mask manifest write",
                elapsed_seconds=perf_counter() - manifest_started_at,
            )
        )
        return VideoMaskingResult(
            frames_total=len(frame_paths),
            frames_done=len(frame_paths),
            manifest_path=manifest_path,
            performance=performance,
        )

    def write_object_preview_mask(
        self,
        *,
        session_id: str,
        first_frame_path: Path,
        output_path: Path,
        object_prompt: StoredObjectPrompt,
    ) -> Path:
        self.load()
        if self._tracker_class is None:
            raise ModelRuntimeError("D4SM tracker class is not loaded.")

        logger.info(
            "Generating D4SM first-frame preview mask session_id=%s object_id=%s first_frame_path=%s output_path=%s",
            session_id,
            object_prompt.object_id,
            first_frame_path,
            output_path,
        )
        image = Image.open(first_frame_path).convert("RGB")
        init_regions = self._objects_to_init_regions([object_prompt])
        tracker = self._create_tracker()
        self._initialize_tracker(tracker=tracker, image=image, init_regions=init_regions)
        masks = self._initial_masks_from_tracker(tracker=tracker, image_size=image.size)
        if not masks:
            raise ModelRuntimeError("D4SM did not return a first-frame preview mask.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_preview_overlay(mask=masks[0], output_path=output_path)
        logger.info(
            "Generated D4SM first-frame preview mask session_id=%s object_id=%s output_path=%s",
            session_id,
            object_prompt.object_id,
            output_path,
        )
        return output_path

    def _objects_to_init_regions(self, objects: list[StoredObjectPrompt]) -> list[dict[str, Any]]:
        init_regions = []
        for item in objects:
            logger.debug("Preparing D4SM init region object_id=%s prompt_count=%s", item.object_id, len(item.prompts))
            box_prompt = next((prompt for prompt in item.prompts if isinstance(prompt, BoxPrompt)), None)
            if box_prompt is None:
                raise ModelRuntimeError(
                    "D4SM v1 integration requires at least one bounding box per object. "
                    "Point-only initialization needs a SAM2 image-prompt mask prepass."
                )
            init_regions.append(
                {
                    "obj_id": item.object_id,
                    "bbox": [
                        box_prompt.x1,
                        box_prompt.y1,
                        box_prompt.x2 - box_prompt.x1,
                        box_prompt.y2 - box_prompt.y1,
                    ],
                }
            )
        return init_regions

    def _write_initial_frame_masks(
        self,
        *,
        init_regions: list[dict[str, Any]],
        objects: list[StoredObjectPrompt],
        image_size: tuple[int, int],
        combined_dir: Path,
        per_object_dir: Path,
    ) -> dict[str, Any]:
        import numpy as np

        width, height = image_size
        masks = []
        for region in init_regions:
            mask = np.zeros((height, width), dtype="uint8")
            x, y, box_width, box_height = region["bbox"]
            mask[y : y + box_height, x : x + box_width] = 1
            masks.append(mask)
        return self._write_frame_masks(
            masks=masks,
            objects=objects,
            frame_index=0,
            combined_dir=combined_dir,
            per_object_dir=per_object_dir,
        )

    def _write_frame_masks(
        self,
        *,
        masks: list[Any],
        objects: list[StoredObjectPrompt],
        frame_index: int,
        combined_dir: Path,
        per_object_dir: Path,
    ) -> dict[str, Any]:
        import numpy as np

        frame_name = f"{frame_index:06d}.png"
        combined_mask = None
        object_paths = {}
        for object_number, (object_prompt, mask) in enumerate(zip(objects, masks, strict=False), start=1):
            logger.debug(
                "Writing object mask frame_index=%s object_id=%s object_number=%s",
                frame_index,
                object_prompt.object_id,
                object_number,
            )
            mask_array = (np.asarray(mask) > 0).astype("uint8")
            if combined_mask is None:
                combined_mask = np.zeros(mask_array.shape, dtype="uint8")
            combined_mask[mask_array > 0] = object_number
            object_dir = per_object_dir / object_prompt.object_id
            object_dir.mkdir(parents=True, exist_ok=True)
            object_path = object_dir / frame_name
            Image.fromarray(mask_array * 255).save(object_path)
            object_paths[object_prompt.object_id] = str(object_path)

        if combined_mask is None:
            raise ModelRuntimeError("D4SM returned no masks for the current frame.")

        combined_path = combined_dir / frame_name
        Image.fromarray(combined_mask).save(combined_path)
        return {
            "frame_index": frame_index,
            "combined_mask": str(combined_path),
            "object_masks": object_paths,
        }

    def _create_tracker(self) -> Any:
        if self._tracker_class is None:
            raise ModelRuntimeError("D4SM tracker class is not loaded.")
        checkpoint_dir = self.settings.d4sm_checkpoint_dir.resolve()
        checkpoint_path = checkpoint_dir / f"sam2.1_hiera_{self.settings.d4sm_model_size}.pt"
        if not checkpoint_path.exists():
            raise ModelRuntimeError(
                f"D4SM checkpoint was not found at {checkpoint_path}. "
                f"Run `uv run python scripts/setup_d4sm.py --model-size {self.settings.d4sm_model_size}`."
            )
        try:
            with self._d4sm_working_directory():
                return self._tracker_class(
                    model_size=self.settings.d4sm_model_size,
                    checkpoint_dir=str(checkpoint_dir),
                    offload_state_to_cpu=self.settings.d4sm_offload_state_to_cpu,
                )
        except ModuleNotFoundError as exc:
            raise ModelRuntimeError(
                f"D4SM dependency is missing: {exc.name}. "
                "Run `uv sync --group gpu` after pulling the latest code."
            ) from exc
        except Exception as exc:
            missing_module = self._missing_module_name(exc)
            if missing_module:
                raise ModelRuntimeError(
                    f"D4SM dependency is missing: {missing_module}. "
                    "Run `uv sync --group gpu` after pulling the latest code."
                ) from exc
            raise ModelRuntimeError(f"Could not initialize D4SM tracker: {exc}") from exc

    def _initialize_tracker(self, *, tracker: Any, image: Image.Image, init_regions: list[dict[str, Any]]) -> None:
        try:
            tracker.initialize(image, init_regions)
        except Exception as exc:
            raise ModelRuntimeError(f"Could not initialize D4SM objects on the first frame: {exc}") from exc

    def _track_frame(self, *, tracker: Any, image: Image.Image, frame_index: int) -> dict[str, Any]:
        try:
            return tracker.track(image)
        except Exception as exc:
            raise ModelRuntimeError(f"Could not run D4SM tracking on frame {frame_index}: {exc}") from exc

    def _missing_module_name(self, exc: BaseException) -> str | None:
        seen: set[int] = set()
        stack: list[BaseException | None] = [exc]
        while stack:
            current = stack.pop()
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            if isinstance(current, ModuleNotFoundError):
                return current.name
            stack.extend([current.__cause__, current.__context__])
        return None

    def _initial_masks_from_tracker(self, *, tracker: Any, image_size: tuple[int, int]) -> list[Any]:
        import numpy as np
        import torch

        width, height = image_size
        masks = []
        for object_id in getattr(tracker, "all_obj_ids", []):
            object_outputs = tracker.per_object_outputs_all.get(object_id, [])
            if not object_outputs:
                continue
            pred_masks = object_outputs[0].get("pred_masks")
            if pred_masks is None:
                continue
            if isinstance(pred_masks, torch.Tensor):
                mask_tensor = pred_masks.detach()
                if mask_tensor.ndim == 3:
                    mask_tensor = mask_tensor.unsqueeze(0)
                if mask_tensor.ndim != 4:
                    raise ModelRuntimeError(f"Unexpected D4SM preview mask tensor shape: {tuple(mask_tensor.shape)}")
                resized = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                masks.append((resized[0, 0] > 0).cpu().numpy().astype("uint8"))
                continue

            mask_array = np.asarray(pred_masks)
            mask_array = np.squeeze(mask_array)
            if mask_array.ndim != 2:
                raise ModelRuntimeError(f"Unexpected D4SM preview mask array shape: {mask_array.shape}")
            mask_image = Image.fromarray((mask_array > 0).astype("uint8") * 255)
            masks.append((np.asarray(mask_image.resize((width, height), Image.NEAREST)) > 0).astype("uint8"))
        return masks

    def _write_preview_overlay(self, *, mask: Any, output_path: Path) -> None:
        import numpy as np

        mask_array = (np.asarray(mask) > 0).astype("uint8")
        alpha = mask_array * 180
        overlay = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype="uint8")
        overlay[..., 0] = 20
        overlay[..., 1] = 184
        overlay[..., 2] = 166
        overlay[..., 3] = alpha
        Image.fromarray(overlay, mode="RGBA").save(output_path)

    @contextmanager
    def _d4sm_working_directory(self):
        previous_cwd = Path.cwd()
        os.chdir(self.settings.d4sm_repo_path.resolve())
        try:
            yield
        finally:
            os.chdir(previous_cwd)


Dam4SamVideoMaskingModel = D4smVideoMaskingModel
