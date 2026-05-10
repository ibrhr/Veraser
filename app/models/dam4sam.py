from app.core.config import Settings
from app.schemas.prompts import StoredObjectPrompt


class Dam4SamVideoMaskingModel:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        raise NotImplementedError("DAM4SAM runtime loading is not implemented yet.")

    def segment_first_frame(
        self,
        *,
        session_id: str,
        first_frame_path: str,
        objects: list[StoredObjectPrompt],
    ) -> None:
        raise NotImplementedError("DAM4SAM first-frame segmentation is not implemented yet.")

    def propagate_video_masks(self, *, session_id: str, video_path: str) -> None:
        raise NotImplementedError("DAM4SAM video mask propagation is not implemented yet.")
