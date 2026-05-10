from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from app.models.base import VideoInpaintingResult, VideoMaskingResult
from app.schemas.masking import InpaintingJobResponse, MaskingJobResponse
from app.schemas.prompts import SubmitObjectPromptsRequest
from app.services.json_store import write_json
from app.services.inpainting_job_service import InpaintingJobService
from app.services.mask_artifact_service import MaskArtifactService
from app.services.mask_prompt_service import MaskPromptService
from app.services.masking_job_service import MaskingJobService
from app.services.session_service import VideoSessionService
from app.services.video_frame_extraction_service import VideoFrameExtractionService
from app.core.config import Settings


class NoopFrameExtractionService(VideoFrameExtractionService):
    def extract_frames(self, *, video_path: Path, output_dir: Path) -> int:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "000000.png").write_bytes(b"fake")
        return 1


class NoopMaskingModel:
    def load(self) -> None:
        return None

    def generate_masks(self, *, session_id: str, frames_dir: Path, output_dir: Path, objects: list) -> VideoMaskingResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        write_json(
            manifest_path,
            {
                "session_id": session_id,
                "model": "noop",
                "frames_total": 1,
                "objects": [item.object_id for item in objects],
                "processed_video": None,
                "frames": [],
            },
        )
        return VideoMaskingResult(frames_total=1, frames_done=1, manifest_path=manifest_path)


class NoopInpaintingModel:
    def load(self) -> None:
        return None

    def inpaint_video(
        self,
        *,
        session_id: str,
        frames_dir: Path,
        masks_dir: Path,
        output_video_path: Path,
    ) -> VideoInpaintingResult:
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        output_video_path.write_bytes(b"fake-mp4")
        return VideoInpaintingResult(frames_total=1, frames_done=1, video_path=output_video_path)


def make_session(tmp_path: Path) -> tuple[str, VideoSessionService]:
    settings = Settings(session_storage_dir=tmp_path)
    session_service = VideoSessionService(settings=settings, frame_service=None)  # type: ignore[arg-type]
    session_id = "session-1"
    session_dir = tmp_path / session_id
    session_dir.mkdir(parents=True)
    video_path = session_dir / "source.mp4"
    video_path.write_bytes(b"fake")
    write_json(
        session_dir / "metadata.json",
        {
            "session_id": session_id,
            "status": "ready_for_prompts",
            "created_at": datetime.now(UTC).isoformat(),
            "video": {"filename": "source.mp4", "size_bytes": 4, "width": 10, "height": 10},
            "first_frame": {"width": 10, "height": 10, "content_type": "image/png", "url": "/frame"},
            "paths": {"video": str(video_path), "first_frame": str(session_dir / "first_frame.png")},
        },
    )
    write_json(session_dir / "prompts.json", [])
    return session_id, session_service


def test_masking_job_lifecycle_succeeds(tmp_path: Path) -> None:
    session_id, session_service = make_session(tmp_path)
    artifact_service = MaskArtifactService(session_service=session_service)
    prompt_service = MaskPromptService(session_service=session_service, artifact_service=artifact_service)
    prompt_service.submit_prompts(
        session_id=session_id,
        request=SubmitObjectPromptsRequest.model_validate(
            {"objects": [{"prompts": [{"type": "box", "x1": 1, "y1": 1, "x2": 5, "y2": 5}]}]}
        ),
    )
    job_service = MaskingJobService(
        session_service=session_service,
        frame_extraction_service=NoopFrameExtractionService(),
        artifact_service=artifact_service,
        model=NoopMaskingModel(),
    )

    job = job_service.create_job(session_id)
    job_service.run_job(session_id, job.job_id)

    stored_job = job_service.get_job(session_id, job.job_id)
    assert isinstance(stored_job, MaskingJobResponse)
    assert stored_job.status == "succeeded"
    assert stored_job.frames_done == 1


def test_inpainting_job_lifecycle_succeeds_after_masking(tmp_path: Path) -> None:
    session_id, session_service = make_session(tmp_path)
    artifact_service = MaskArtifactService(session_service=session_service)
    prompt_service = MaskPromptService(session_service=session_service, artifact_service=artifact_service)
    prompt_service.submit_prompts(
        session_id=session_id,
        request=SubmitObjectPromptsRequest.model_validate(
            {"objects": [{"prompts": [{"type": "box", "x1": 1, "y1": 1, "x2": 5, "y2": 5}]}]}
        ),
    )
    masking_service = MaskingJobService(
        session_service=session_service,
        frame_extraction_service=NoopFrameExtractionService(),
        artifact_service=artifact_service,
        model=NoopMaskingModel(),
    )
    masking_job = masking_service.create_job(session_id)
    masking_service.run_job(session_id, masking_job.job_id)
    inpainting_service = InpaintingJobService(
        session_service=session_service,
        masking_job_service=masking_service,
        artifact_service=artifact_service,
        model=NoopInpaintingModel(),
    )

    inpainting_job = inpainting_service.create_job(session_id, masking_job.job_id)
    inpainting_service.run_job(session_id, masking_job.job_id)

    stored_job = inpainting_service.get_job(session_id, masking_job.job_id)
    assert isinstance(stored_job, InpaintingJobResponse)
    assert stored_job.job_id == inpainting_job.job_id
    assert stored_job.status == "succeeded"
    assert artifact_service.processed_video_path(session_id, masking_job.job_id).exists()


def test_inpainting_requires_succeeded_masking_job(tmp_path: Path) -> None:
    session_id, session_service = make_session(tmp_path)
    artifact_service = MaskArtifactService(session_service=session_service)
    prompt_service = MaskPromptService(session_service=session_service, artifact_service=artifact_service)
    prompt_service.submit_prompts(
        session_id=session_id,
        request=SubmitObjectPromptsRequest.model_validate(
            {"objects": [{"prompts": [{"type": "box", "x1": 1, "y1": 1, "x2": 5, "y2": 5}]}]}
        ),
    )
    masking_service = MaskingJobService(
        session_service=session_service,
        frame_extraction_service=NoopFrameExtractionService(),
        artifact_service=artifact_service,
        model=NoopMaskingModel(),
    )
    masking_job = masking_service.create_job(session_id)
    inpainting_service = InpaintingJobService(
        session_service=session_service,
        masking_job_service=masking_service,
        artifact_service=artifact_service,
        model=NoopInpaintingModel(),
    )

    try:
        inpainting_service.create_job(session_id, masking_job.job_id)
    except Exception as exc:
        assert "inpainting requires tracked masks" in str(exc)
    else:
        raise AssertionError("Expected inpainting to require a succeeded masking job.")


def test_object_prompt_preview_update_and_delete(tmp_path: Path) -> None:
    session_id, session_service = make_session(tmp_path)
    artifact_service = MaskArtifactService(session_service=session_service)
    prompt_service = MaskPromptService(session_service=session_service, artifact_service=artifact_service)

    response = prompt_service.submit_prompts(
        session_id=session_id,
        request=SubmitObjectPromptsRequest.model_validate(
            {"objects": [{"prompts": [{"type": "box", "x1": 1, "y1": 1, "x2": 5, "y2": 5}]}]}
        ),
    )
    object_id = response.objects[0].object_id

    preview_path = prompt_service.get_preview_mask_path(session_id=session_id, object_id=object_id)
    assert preview_path.exists()
    with Image.open(preview_path) as preview:
        assert preview.mode == "RGBA"
        assert preview.getpixel((2, 2))[3] > 0

    updated = prompt_service.update_object(
        session_id=session_id,
        object_id=object_id,
        request=response.objects[0].model_copy(
            update={"prompts": [{"type": "box", "x1": 2, "y1": 2, "x2": 6, "y2": 6}]}
        ),
    )
    assert updated.object.object_id == object_id

    prompt_service.delete_object(session_id=session_id, object_id=object_id)
    assert prompt_service.list_objects(session_id).objects == []
    assert not preview_path.exists()
