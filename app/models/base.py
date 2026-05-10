from pathlib import Path
from typing import Protocol

from app.schemas.performance import OperationSpeedMetric
from app.schemas.prompts import StoredObjectPrompt


class VideoMaskingResult:
    def __init__(
        self,
        *,
        frames_total: int,
        frames_done: int,
        manifest_path: Path,
        performance: list[OperationSpeedMetric] | None = None,
    ) -> None:
        self.frames_total = frames_total
        self.frames_done = frames_done
        self.manifest_path = manifest_path
        self.performance = performance or []


class VideoInpaintingResult:
    def __init__(
        self,
        *,
        frames_total: int,
        frames_done: int,
        video_path: Path,
        performance: list[OperationSpeedMetric] | None = None,
    ) -> None:
        self.frames_total = frames_total
        self.frames_done = frames_done
        self.video_path = video_path
        self.performance = performance or []


class VideoMaskingModel(Protocol):
    def load(self) -> None:
        """Load model weights and runtime resources."""

    def generate_masks(
        self,
        *,
        session_id: str,
        frames_dir: Path,
        output_dir: Path,
        objects: list[StoredObjectPrompt],
    ) -> VideoMaskingResult:
        """Generate object masks for every video frame."""


class VideoInpaintingModel(Protocol):
    def load(self) -> None:
        """Load model weights and runtime resources."""

    def inpaint_video(
        self,
        *,
        session_id: str,
        frames_dir: Path,
        masks_dir: Path,
        output_video_path: Path,
    ) -> VideoInpaintingResult:
        """Inpaint masked video frames and write a processed video."""
