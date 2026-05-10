from pathlib import Path

import pytest

from app.core.config import Settings
from app.schemas.prompts import SubmitObjectPromptsRequest
from app.services.errors import InvalidPromptError
from app.services.mask_artifact_service import MaskArtifactService
from app.services.mask_prompt_service import MaskPromptService
from app.services.session_service import VideoSessionService


def make_service(tmp_path: Path) -> MaskPromptService:
    settings = Settings(session_storage_dir=tmp_path)
    session_service = VideoSessionService(settings=settings, frame_service=None)  # type: ignore[arg-type]
    return MaskPromptService(
        session_service=session_service,
        artifact_service=MaskArtifactService(session_service=session_service),
    )


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
