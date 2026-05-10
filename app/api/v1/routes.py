from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.api.deps import (
    get_inpainting_job_service,
    get_mask_artifact_service,
    get_mask_prompt_service,
    get_masking_job_service,
    get_video_session_service,
)
from app.schemas.masking import InpaintingJobResponse, MaskArtifactManifest, MaskingJobResponse
from app.schemas.prompts import (
    ObjectPrompt,
    ObjectPromptListResponse,
    ObjectPromptResponse,
    SubmitObjectPromptsRequest,
    SubmitObjectPromptsResponse,
)
from app.schemas.sessions import VideoSessionDetail, VideoSessionResponse
from app.services.errors import (
    InpaintingJobNotFoundError,
    InvalidPromptError,
    MaskingJobNotFoundError,
    ModelRuntimeError,
    ObjectPromptNotFoundError,
    SessionNotFoundError,
    UploadRejectedError,
)
from app.services.inpainting_job_service import InpaintingJobService
from app.services.mask_artifact_service import MaskArtifactService
from app.services.masking_job_service import MaskingJobService
from app.services.mask_prompt_service import MaskPromptService
from app.services.session_service import VideoSessionService

router = APIRouter(prefix="/api/v1", tags=["video masking"])


@router.post(
    "/video-sessions",
    response_model=VideoSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_video_session(
    video: UploadFile = File(...),
    service: VideoSessionService = Depends(get_video_session_service),
) -> VideoSessionResponse:
    try:
        return await service.create_session(video)
    except UploadRejectedError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/video-sessions/{session_id}",
    response_model=VideoSessionDetail,
)
async def get_video_session(
    session_id: str,
    service: VideoSessionService = Depends(get_video_session_service),
) -> VideoSessionDetail:
    try:
        return service.get_session_detail(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/video-sessions/{session_id}/first-frame")
async def get_first_frame(
    session_id: str,
    service: VideoSessionService = Depends(get_video_session_service),
) -> FileResponse:
    try:
        first_frame_path = service.get_first_frame_path(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return FileResponse(
        Path(first_frame_path),
        media_type="image/png",
        filename=f"{session_id}-first-frame.png",
        content_disposition_type="inline",
    )


@router.post(
    "/video-sessions/{session_id}/objects",
    response_model=SubmitObjectPromptsResponse,
)
async def submit_object_prompts(
    session_id: str,
    request: SubmitObjectPromptsRequest,
    service: MaskPromptService = Depends(get_mask_prompt_service),
) -> SubmitObjectPromptsResponse:
    try:
        return service.submit_prompts(session_id=session_id, request=request)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except InvalidPromptError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.get(
    "/video-sessions/{session_id}/objects",
    response_model=ObjectPromptListResponse,
)
async def list_object_prompts(
    session_id: str,
    service: MaskPromptService = Depends(get_mask_prompt_service),
) -> ObjectPromptListResponse:
    try:
        return service.list_objects(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.put(
    "/video-sessions/{session_id}/objects/{object_id}",
    response_model=ObjectPromptResponse,
)
async def update_object_prompt(
    session_id: str,
    object_id: str,
    request: ObjectPrompt,
    service: MaskPromptService = Depends(get_mask_prompt_service),
) -> ObjectPromptResponse:
    try:
        return service.update_object(session_id=session_id, object_id=object_id, request=request)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ObjectPromptNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except InvalidPromptError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.delete(
    "/video-sessions/{session_id}/objects/{object_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_object_prompt(
    session_id: str,
    object_id: str,
    service: MaskPromptService = Depends(get_mask_prompt_service),
) -> None:
    try:
        service.delete_object(session_id=session_id, object_id=object_id)
    except (SessionNotFoundError, ObjectPromptNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/video-sessions/{session_id}/objects/{object_id}/preview-mask")
async def get_object_preview_mask(
    session_id: str,
    object_id: str,
    service: MaskPromptService = Depends(get_mask_prompt_service),
) -> FileResponse:
    try:
        preview_path = service.get_preview_mask_path(session_id=session_id, object_id=object_id)
    except (SessionNotFoundError, ObjectPromptNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return FileResponse(
        preview_path,
        media_type="image/png",
        filename=f"{object_id}-preview-mask.png",
        content_disposition_type="inline",
        headers={"Cache-Control": "no-store"},
    )


@router.post(
    "/video-sessions/{session_id}/masking-jobs",
    response_model=MaskingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_masking_job(
    session_id: str,
    background_tasks: BackgroundTasks,
    service: MaskingJobService = Depends(get_masking_job_service),
) -> MaskingJobResponse:
    try:
        job = service.create_job(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ModelRuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    background_tasks.add_task(service.run_job, session_id, job.job_id)
    return job


@router.get(
    "/video-sessions/{session_id}/masking-jobs/{job_id}",
    response_model=MaskingJobResponse,
)
async def get_masking_job(
    session_id: str,
    job_id: str,
    service: MaskingJobService = Depends(get_masking_job_service),
) -> MaskingJobResponse:
    try:
        return service.get_job(session_id, job_id)
    except MaskingJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/video-sessions/{session_id}/masking-jobs/{job_id}/masks/manifest",
    response_model=MaskArtifactManifest,
)
async def get_mask_manifest(
    session_id: str,
    job_id: str,
    artifact_service: MaskArtifactService = Depends(get_mask_artifact_service),
    job_service: MaskingJobService = Depends(get_masking_job_service),
) -> MaskArtifactManifest:
    try:
        job = job_service.get_job(session_id, job_id)
    except MaskingJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if job.status != "succeeded":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Masking job is {job.status}; manifest is not ready.",
        )
    manifest = artifact_service.read_manifest(session_id, job_id)
    return MaskArtifactManifest(
        session_id=session_id,
        job_id=job_id,
        frames_total=manifest.get("frames_total", 0),
        objects=manifest.get("objects", []),
        combined_masks_url=f"/api/v1/video-sessions/{session_id}/masking-jobs/{job_id}/masks/combined/{{frame_index}}",
        processed_video_url=f"/api/v1/video-sessions/{session_id}/masking-jobs/{job_id}/processed-video",
    )


@router.get("/video-sessions/{session_id}/masking-jobs/{job_id}/masks/combined/{frame_index}")
async def get_combined_mask(
    session_id: str,
    job_id: str,
    frame_index: int,
    artifact_service: MaskArtifactService = Depends(get_mask_artifact_service),
    job_service: MaskingJobService = Depends(get_masking_job_service),
) -> FileResponse:
    try:
        job = job_service.get_job(session_id, job_id)
    except MaskingJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if job.status != "succeeded":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Masking job is {job.status}; masks are not ready.",
        )

    mask_path = artifact_service.combined_mask_path(session_id, job_id, frame_index)
    if not mask_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mask frame was not found.")
    return FileResponse(
        mask_path,
        media_type="image/png",
        filename=mask_path.name,
        content_disposition_type="inline",
    )


@router.get("/video-sessions/{session_id}/masking-jobs/{job_id}/processed-video")
async def get_processed_video(
    session_id: str,
    job_id: str,
    artifact_service: MaskArtifactService = Depends(get_mask_artifact_service),
    job_service: MaskingJobService = Depends(get_masking_job_service),
) -> FileResponse:
    try:
        job = job_service.get_job(session_id, job_id)
    except MaskingJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if job.status != "succeeded":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Masking job is {job.status}; processed video is not ready.",
        )

    video_path = artifact_service.processed_video_path(session_id, job_id)
    if not video_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processed video has not been generated by an inpainting backend yet.",
        )
    return FileResponse(video_path, media_type="video/mp4", filename=video_path.name)


@router.post(
    "/video-sessions/{session_id}/masking-jobs/{job_id}/inpainting-job",
    response_model=InpaintingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_inpainting_job(
    session_id: str,
    job_id: str,
    background_tasks: BackgroundTasks,
    service: InpaintingJobService = Depends(get_inpainting_job_service),
) -> InpaintingJobResponse:
    try:
        job = service.create_job(session_id=session_id, masking_job_id=job_id)
    except (SessionNotFoundError, MaskingJobNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ModelRuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    background_tasks.add_task(service.run_job, session_id, job_id)
    return job


@router.get(
    "/video-sessions/{session_id}/masking-jobs/{job_id}/inpainting-job",
    response_model=InpaintingJobResponse,
)
async def get_inpainting_job(
    session_id: str,
    job_id: str,
    service: InpaintingJobService = Depends(get_inpainting_job_service),
) -> InpaintingJobResponse:
    try:
        return service.get_job(session_id=session_id, masking_job_id=job_id)
    except InpaintingJobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.delete("/video-sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video_session(
    session_id: str,
    service: VideoSessionService = Depends(get_video_session_service),
) -> None:
    try:
        service.delete_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
