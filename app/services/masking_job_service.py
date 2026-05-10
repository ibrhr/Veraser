from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from app.models.base import VideoMaskingModel
from app.schemas.masking import MaskingJobResponse
from app.schemas.performance import OperationSpeedMetric, build_speed_metric
from app.schemas.prompts import StoredObjectPrompt
from app.services.errors import MaskingJobNotFoundError, ModelRuntimeError, SessionNotFoundError
from app.services.json_store import read_json, write_json
from app.services.mask_artifact_service import MaskArtifactService
from app.services.session_service import VideoSessionService
from app.services.video_frame_extraction_service import VideoFrameExtractionService


class MaskingJobService:
    def __init__(
        self,
        *,
        session_service: VideoSessionService,
        frame_extraction_service: VideoFrameExtractionService,
        artifact_service: MaskArtifactService,
        model: VideoMaskingModel,
    ) -> None:
        self.session_service = session_service
        self.frame_extraction_service = frame_extraction_service
        self.artifact_service = artifact_service
        self.model = model

    def create_job(self, session_id: str) -> MaskingJobResponse:
        self.session_service.get_metadata(session_id)
        objects = self._get_objects(session_id)
        if not objects:
            raise ModelRuntimeError("At least one prompted object is required before masking.")

        job_id = str(uuid4())
        now = datetime.now(UTC)
        job = MaskingJobResponse(
            job_id=job_id,
            session_id=session_id,
            status="pending",
            created_at=now,
            updated_at=now,
            manifest_url=f"/api/v1/video-sessions/{session_id}/masking-jobs/{job_id}/masks/manifest",
            processed_video_url=f"/api/v1/video-sessions/{session_id}/masking-jobs/{job_id}/processed-video",
        )
        self._write_job(job)
        return job

    def get_job(self, session_id: str, job_id: str) -> MaskingJobResponse:
        job_path = self._job_path(session_id, job_id)
        if not job_path.exists():
            raise MaskingJobNotFoundError(f"Masking job {job_id} was not found.")
        return MaskingJobResponse.model_validate(read_json(job_path, default={}))

    def run_job(self, session_id: str, job_id: str) -> None:
        try:
            job = self._update_job(
                session_id,
                job_id,
                status="running",
                current_stage="extracting_frames",
            )
            performance: list[OperationSpeedMetric] = []
            video_path = Path(self.session_service.get_video_path(session_id))
            extraction_started_at = perf_counter()
            frames_total = self.frame_extraction_service.extract_frames(
                video_path=video_path,
                output_dir=self.artifact_service.frames_dir(session_id),
            )
            performance.append(
                build_speed_metric(
                    name="frame_extraction",
                    label="Frame extraction",
                    elapsed_seconds=perf_counter() - extraction_started_at,
                    frames_processed=frames_total,
                )
            )
            job = self._update_job(
                session_id,
                job_id,
                frames_total=frames_total,
                current_stage="generating_masks",
                performance=performance,
            )
            masking_started_at = perf_counter()
            result = self.model.generate_masks(
                session_id=session_id,
                frames_dir=self.artifact_service.frames_dir(session_id),
                output_dir=self.artifact_service.masks_dir(session_id, job_id),
                objects=self._get_objects(session_id),
            )
            fallback_masking_metric = build_speed_metric(
                name="mask_generation",
                label="Mask generation",
                elapsed_seconds=perf_counter() - masking_started_at,
                frames_processed=result.frames_done,
            )
            performance.extend(result.performance or [fallback_masking_metric])
            self._update_job(
                session_id,
                job_id,
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
                    job_id,
                    status="failed",
                    current_stage="failed",
                    error=str(exc),
                )
            except (MaskingJobNotFoundError, SessionNotFoundError):
                pass

    def _get_objects(self, session_id: str) -> list[StoredObjectPrompt]:
        prompt_path = self.session_service.get_prompts_path(session_id)
        return [
            StoredObjectPrompt.model_validate(item)
            for item in read_json(prompt_path, default=[])
        ]

    def _job_path(self, session_id: str, job_id: str) -> Path:
        return self.artifact_service.job_dir(session_id, job_id) / "job.json"

    def _write_job(self, job: MaskingJobResponse) -> None:
        write_json(
            self._job_path(job.session_id, job.job_id),
            job.model_dump(mode="json"),
        )

    def _update_job(self, session_id: str, job_id: str, **changes: object) -> MaskingJobResponse:
        job = self.get_job(session_id, job_id)
        updated = job.model_copy(
            update={
                **changes,
                "updated_at": datetime.now(UTC),
            }
        )
        self._write_job(updated)
        return updated
