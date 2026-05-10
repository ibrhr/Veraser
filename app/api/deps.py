from functools import lru_cache

from app.core.config import Settings
from app.models.dam4sam import Dam4SamVideoMaskingModel
from app.services.frame_service import VideoFrameService
from app.services.mask_prompt_service import MaskPromptService
from app.services.session_service import VideoSessionService


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_video_frame_service() -> VideoFrameService:
    return VideoFrameService(settings=get_settings())


@lru_cache
def get_video_masking_model() -> Dam4SamVideoMaskingModel:
    return Dam4SamVideoMaskingModel(settings=get_settings())


@lru_cache
def get_video_session_service() -> VideoSessionService:
    return VideoSessionService(
        settings=get_settings(),
        frame_service=get_video_frame_service(),
    )


@lru_cache
def get_mask_prompt_service() -> MaskPromptService:
    return MaskPromptService(
        session_service=get_video_session_service(),
        model=get_video_masking_model(),
    )
