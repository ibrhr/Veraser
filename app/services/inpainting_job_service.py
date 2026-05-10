from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from app.models.base import VideoInpaintingModel
from app.schemas.masking import InpaintingJobResponse
from app.schemas.performance import build_speed_metric
from app.services.errors import InpaintingJobNotFoundError, MaskingJobNotFoundError, ModelRuntimeError, SessionNotFoundError
from app.services.json_store import read_json, write_json
from app.services.mask_artifact_service import MaskArtifactService
from app.services.masking_job_service import MaskingJobService
from app.services.session_service import VideoSessionService


class InpaintingJobService:
    def __init__(
        self,
        *,
        session_service: VideoSessionService,
        masking_job_service: MaskingJobService,
        artifact_service: MaskArtifactService,
        model: VideoInpaintingModel,
    ) -> None:
        self.session_service = session_service
        self.masking_job_service = masking_job_service
        self.artifact_service = artifact_service
        self.model = model

    def create_job(self, session_id: str, masking_job_id: str) -> InpaintingJobResponse:
        self.session_service.get_metadata(session_id)
        masking_job = self.masking_job_service.get_job(session_id, masking_job_id)
        if masking_job.status != "succeeded":
            raise ModelRuntimeError(f"Masking job is {masking_job.status}; inpainting requires tracked masks.")
        if not self.artifact_service.manifest_path(session_id, masking_job_id).exists():
            raise ModelRuntimeError("Mask manifest is missing; inpainting cannot start.")

        job_id = str(uuid4())
        now = datetime.now(UTC)
        job = InpaintingJobResponse(
            job_id=job_id,
            session_id=session_id,
            masking_job_id=masking_job_id,
            status="pending",
            created_at=now,
            updated_at=now,
            processed_video_url=f"/api/v1/video-sessions/{session_id}/masking-jobs/{masking_job_id}/processed-video",
        )
        self._write_job(job)
        return job

    def get_job(self, session_id: str, masking_job_id: str) -> InpaintingJobResponse:
        job_path = self._job_path(session_id, masking_job_id)
        if not job_path.exists():
            raise InpaintingJobNotFoundError(f"Inpainting job for masking job {masking_job_id} was not found.")
        return InpaintingJobResponse.model_validate(read_json(job_path, default={}))

    def run_job(self, session_id: str, masking_job_id: str) -> None:
        try:
            self._update_job(session_id, masking_job_id, status="running", current_stage="inpainting")
            frames_dir = self.artifact_service.frames_dir(session_id)
            masks_dir = self.artifact_service.masks_dir(session_id, masking_job_id)
            output_video_path = self.artifact_service.processed_video_path(session_id, masking_job_id)
            started_at = perf_counter()
            result = self.model.inpaint_video(
                session_id=session_id,
                frames_dir=frames_dir,
                masks_dir=masks_dir,
                output_video_path=output_video_path,
            )
            fallback_metric = build_speed_metric(
                name="video_inpainting",
                label="Video inpainting",
                elapsed_seconds=perf_counter() - started_at,
                frames_processed=result.frames_done,
            )
            performance = result.performance or [fallback_metric]
            self._update_job(
                session_id,
                masking_job_id,
                status="succeeded",
                frames_total=result.frames_total,
                frames_done=result.frames_done,
                current_stage="complete",
                performance=performance,
            )
        except Exception as exc:
            try:
                self._update_job(
                    session_id,
                    masking_job_id,
                    status="failed",
                    current_stage="failed",
                    error=str(exc),
                )
            except (InpaintingJobNotFoundError, MaskingJobNotFoundError, SessionNotFoundError):
                pass

    def _job_path(self, session_id: str, masking_job_id: str):
        return self.artifact_service.job_dir(session_id, masking_job_id) / "inpainting_job.json"

    def _write_job(self, job: InpaintingJobResponse) -> None:
        write_json(self._job_path(job.session_id, job.masking_job_id), job.model_dump(mode="json"))

    def _update_job(self, session_id: str, masking_job_id: str, **changes: object) -> InpaintingJobResponse:
        job = self.get_job(session_id, masking_job_id)
        updated = job.model_copy(update={**changes, "updated_at": datetime.now(UTC)})
        self._write_job(updated)
        return updated
