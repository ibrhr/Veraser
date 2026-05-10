from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.api.deps import get_mask_prompt_service, get_video_session_service
from app.schemas.prompts import SubmitObjectPromptsRequest, SubmitObjectPromptsResponse
from app.schemas.sessions import VideoSessionDetail, VideoSessionResponse
from app.services.errors import InvalidPromptError, SessionNotFoundError, UploadRejectedError
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


@router.delete("/video-sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video_session(
    session_id: str,
    service: VideoSessionService = Depends(get_video_session_service),
) -> None:
    try:
        service.delete_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
