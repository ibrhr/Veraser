from datetime import UTC, datetime
from pathlib import Path
import shutil
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import Settings
from app.schemas.prompts import StoredObjectPrompt
from app.schemas.sessions import FirstFrameInfo, VideoMetadata, VideoSessionDetail, VideoSessionResponse
from app.services.errors import SessionNotFoundError, UploadRejectedError
from app.services.frame_service import VideoFrameService
from app.services.json_store import read_json, write_json


class VideoSessionService:
    def __init__(self, *, settings: Settings, frame_service: VideoFrameService) -> None:
        self.settings = settings
        self.frame_service = frame_service

    async def create_session(self, upload: UploadFile) -> VideoSessionResponse:
        self._validate_upload(upload)

        session_id = str(uuid4())
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=False)

        video_path = session_dir / self._safe_video_filename(upload.filename)
        size_bytes = await self._write_upload(upload=upload, path=video_path)
        if size_bytes > self.settings.max_upload_bytes:
            shutil.rmtree(session_dir, ignore_errors=True)
            raise UploadRejectedError("Uploaded video exceeds the configured size limit.")

        first_frame_path = session_dir / f"first_frame.{self.settings.first_frame_format}"
        width, height = self.frame_service.extract_first_frame(
            video_path=video_path,
            output_path=first_frame_path,
        )

        created_at = datetime.now(UTC)
        response = VideoSessionResponse(
            session_id=session_id,
            created_at=created_at,
            video=VideoMetadata(
                filename=upload.filename or video_path.name,
                content_type=upload.content_type,
                size_bytes=size_bytes,
                width=width,
                height=height,
            ),
            first_frame=FirstFrameInfo(
                width=width,
                height=height,
                url=f"/api/v1/video-sessions/{session_id}/first-frame",
            ),
        )

        write_json(
            self._metadata_path(session_id),
            {
                **response.model_dump(mode="json"),
                "paths": {
                    "video": str(video_path),
                    "first_frame": str(first_frame_path),
                },
            },
        )
        write_json(self.get_prompts_path(session_id), [])
        return response

    def get_session_detail(self, session_id: str) -> VideoSessionDetail:
        metadata = self.get_metadata(session_id)
        objects = [
            StoredObjectPrompt.model_validate(item)
            for item in read_json(self.get_prompts_path(session_id), default=[])
        ]
        return VideoSessionDetail(
            **{
                key: value
                for key, value in metadata.items()
                if key in VideoSessionResponse.model_fields
            },
            objects=objects,
        )

    def get_metadata(self, session_id: str) -> dict:
        metadata_path = self._metadata_path(session_id)
        if not metadata_path.exists():
            raise SessionNotFoundError(f"Video session {session_id} was not found.")
        return read_json(metadata_path, default={})

    def get_first_frame_path(self, session_id: str) -> str:
        metadata = self.get_metadata(session_id)
        return metadata["paths"]["first_frame"]

    def get_video_path(self, session_id: str) -> str:
        metadata = self.get_metadata(session_id)
        return metadata["paths"]["video"]

    def get_prompts_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "prompts.json"

    def delete_session(self, session_id: str) -> None:
        if not self._metadata_path(session_id).exists():
            raise SessionNotFoundError(f"Video session {session_id} was not found.")
        shutil.rmtree(self._session_dir(session_id), ignore_errors=True)

    def _validate_upload(self, upload: UploadFile) -> None:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in self.settings.accepted_video_extensions:
            raise UploadRejectedError(f"Unsupported video extension: {suffix or '<missing>'}.")
        if upload.content_type and upload.content_type not in self.settings.accepted_video_mime_types:
            raise UploadRejectedError(f"Unsupported video content type: {upload.content_type}.")

    async def _write_upload(self, *, upload: UploadFile, path: Path) -> int:
        size_bytes = 0
        with path.open("wb") as file:
            while chunk := await upload.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > self.settings.max_upload_bytes:
                    break
                file.write(chunk)
        await upload.close()
        return size_bytes

    def _session_dir(self, session_id: str) -> Path:
        return self.settings.session_storage_dir / session_id

    def _metadata_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "metadata.json"

    def _safe_video_filename(self, filename: str | None) -> str:
        suffix = Path(filename or "video").suffix.lower() or ".mp4"
        return f"source{suffix}"
