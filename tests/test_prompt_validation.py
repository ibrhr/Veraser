from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.models.base import VideoMaskingModel
from app.schemas.prompts import SubmitObjectPromptsRequest
from app.services.errors import InvalidPromptError
from app.services.mask_prompt_service import MaskPromptService
from app.services.session_service import VideoSessionService


class NoopModel(VideoMaskingModel):
    def load(self) -> None:
        return None

    def segment_first_frame(self, *, session_id: str, first_frame_path: str, objects: list[Any]) -> None:
        return None

    def propagate_video_masks(self, *, session_id: str, video_path: str) -> None:
        return None


def make_service(tmp_path: Path) -> MaskPromptService:
    settings = Settings(session_storage_dir=tmp_path)
    session_service = VideoSessionService(settings=settings, frame_service=None)  # type: ignore[arg-type]
    return MaskPromptService(session_service=session_service, model=NoopModel())


def test_point_prompt_must_be_inside_first_frame(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    request = SubmitObjectPromptsRequest.model_validate(
        {"objects": [{"prompts": [{"type": "point", "x": 1920, "y": 100}]}]}
    )

    with pytest.raises(InvalidPromptError):
        service._validate_prompt_bounds(request, width=1920, height=1080)


def test_box_prompt_can_touch_right_and_bottom_edges(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    request = SubmitObjectPromptsRequest.model_validate(
        {"objects": [{"prompts": [{"type": "box", "x1": 0, "y1": 0, "x2": 1920, "y2": 1080}]}]}
    )

    service._validate_prompt_bounds(request, width=1920, height=1080)
