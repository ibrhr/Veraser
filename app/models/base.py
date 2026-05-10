from typing import Protocol

from app.schemas.prompts import StoredObjectPrompt


class VideoMaskingModel(Protocol):
    def load(self) -> None:
        """Load model weights and runtime resources."""

    def segment_first_frame(
        self,
        *,
        session_id: str,
        first_frame_path: str,
        objects: list[StoredObjectPrompt],
    ) -> None:
        """Segment prompted objects on the first frame."""

    def propagate_video_masks(self, *, session_id: str, video_path: str) -> None:
        """Propagate first-frame masks across the full video."""
