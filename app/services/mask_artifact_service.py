from pathlib import Path

from app.services.json_store import read_json
from app.services.session_service import VideoSessionService


class MaskArtifactService:
    def __init__(self, *, session_service: VideoSessionService) -> None:
        self.session_service = session_service

    def masking_root(self, session_id: str) -> Path:
        self.session_service.get_metadata(session_id)
        return self.session_service.get_session_dir(session_id) / "masking"

    def job_dir(self, session_id: str, job_id: str) -> Path:
        return self.masking_root(session_id) / "jobs" / job_id

    def frames_dir(self, session_id: str) -> Path:
        return self.masking_root(session_id) / "frames"

    def masks_dir(self, session_id: str, job_id: str) -> Path:
        return self.job_dir(session_id, job_id) / "masks"

    def manifest_path(self, session_id: str, job_id: str) -> Path:
        return self.masks_dir(session_id, job_id) / "manifest.json"

    def combined_mask_path(self, session_id: str, job_id: str, frame_index: int) -> Path:
        return self.masks_dir(session_id, job_id) / "combined" / f"{frame_index:06d}.png"

    def object_preview_mask_path(self, session_id: str, object_id: str) -> Path:
        return self.masking_root(session_id) / "previews" / f"{object_id}.png"

    def processed_video_path(self, session_id: str, job_id: str) -> Path:
        return self.job_dir(session_id, job_id) / "processed_video.mp4"

    def read_manifest(self, session_id: str, job_id: str) -> dict:
        return read_json(self.manifest_path(session_id, job_id), default={})
