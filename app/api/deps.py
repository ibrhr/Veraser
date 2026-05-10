from functools import lru_cache

from app.core.config import Settings
from app.models.dam4sam import D4smVideoMaskingModel
from app.models.sttn import SttnVideoInpaintingModel
from app.services.frame_service import VideoFrameService
from app.services.inpainting_job_service import InpaintingJobService
from app.services.mask_artifact_service import MaskArtifactService
from app.services.masking_job_service import MaskingJobService
from app.services.mask_prompt_service import MaskPromptService
from app.services.session_service import VideoSessionService
from app.services.video_frame_extraction_service import VideoFrameExtractionService


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_video_frame_service() -> VideoFrameService:
    return VideoFrameService(settings=get_settings())


@lru_cache
def get_video_masking_model() -> D4smVideoMaskingModel:
    return D4smVideoMaskingModel(settings=get_settings())


@lru_cache
def get_video_inpainting_model() -> SttnVideoInpaintingModel:
    return SttnVideoInpaintingModel(settings=get_settings())


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
        artifact_service=get_mask_artifact_service(),
        preview_model=get_video_masking_model(),
    )


@lru_cache
def get_video_frame_extraction_service() -> VideoFrameExtractionService:
    return VideoFrameExtractionService()


@lru_cache
def get_mask_artifact_service() -> MaskArtifactService:
    return MaskArtifactService(session_service=get_video_session_service())


@lru_cache
def get_masking_job_service() -> MaskingJobService:
    return MaskingJobService(
        session_service=get_video_session_service(),
        frame_extraction_service=get_video_frame_extraction_service(),
        artifact_service=get_mask_artifact_service(),
        model=get_video_masking_model(),
    )


@lru_cache
def get_inpainting_job_service() -> InpaintingJobService:
    return InpaintingJobService(
        session_service=get_video_session_service(),
        masking_job_service=get_masking_job_service(),
        artifact_service=get_mask_artifact_service(),
        model=get_video_inpainting_model(),
    )
