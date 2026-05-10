from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

from PIL import Image

from app.models.base import VideoInpaintingResult, VideoMaskingResult
from app.models.dam4sam import D4smVideoMaskingModel
from app.models.sttn import SttnVideoInpaintingModel
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


class ExplodingFrameExtractionService(VideoFrameExtractionService):
    def extract_frames(self, *, video_path: Path, output_dir: Path) -> int:
        raise AssertionError("Frame extraction should not run for video-native masking models.")


class NativeVideoMaskingModel(NoopMaskingModel):
    video_path: Path | None = None

    def generate_masks_from_video(
        self,
        *,
        session_id: str,
        video_path: Path,
        output_dir: Path,
        objects: list,
    ) -> VideoMaskingResult:
        self.video_path = video_path
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        write_json(
            manifest_path,
            {
                "session_id": session_id,
                "model": "native-video",
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


class RecordingD4smTracker:
    calls: list[dict[str, object]] = []

    def __init__(self, *, model_size: str, checkpoint_dir: str, offload_state_to_cpu: bool) -> None:
        self.__class__.calls.append(
            {
                "model_size": model_size,
                "checkpoint_dir": checkpoint_dir,
                "offload_state_to_cpu": offload_state_to_cpu,
                "cwd": Path.cwd(),
            }
        )


class RecordingInferenceContext:
    enter_count = 0
    exit_count = 0

    def __enter__(self) -> None:
        self.__class__.enter_count += 1

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.__class__.exit_count += 1


class RecordingInferenceTracker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def initialize(self, image: Image.Image, init_regions: list[dict[str, object]]) -> None:
        self.calls.append(f"initialize:{image.size}:{len(init_regions)}")

    def track(self, image: Image.Image) -> dict[str, list[object]]:
        self.calls.append(f"track:{image.size}")
        return {"masks": []}


class RecordingSttnGenerator:
    def to(self, device: str) -> "RecordingSttnGenerator":
        return self

    def load_state_dict(self, state: dict[str, object]) -> None:
        return None

    def eval(self) -> None:
        return None


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


def test_masking_job_uses_video_native_model_without_extracting_frames(tmp_path: Path) -> None:
    session_id, session_service = make_session(tmp_path)
    artifact_service = MaskArtifactService(session_service=session_service)
    prompt_service = MaskPromptService(session_service=session_service, artifact_service=artifact_service)
    prompt_service.submit_prompts(
        session_id=session_id,
        request=SubmitObjectPromptsRequest.model_validate(
            {"objects": [{"prompts": [{"type": "box", "x1": 1, "y1": 1, "x2": 5, "y2": 5}]}]}
        ),
    )
    model = NativeVideoMaskingModel()
    job_service = MaskingJobService(
        session_service=session_service,
        frame_extraction_service=ExplodingFrameExtractionService(),
        artifact_service=artifact_service,
        model=model,
    )

    job = job_service.create_job(session_id)
    job_service.run_job(session_id, job.job_id)

    stored_job = job_service.get_job(session_id, job.job_id)
    assert stored_job.status == "succeeded"
    assert model.video_path == Path(session_service.get_video_path(session_id))
    assert not artifact_service.frames_dir(session_id).exists()


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


def test_d4sm_tracker_uses_absolute_checkpoint_dir_inside_repo_cwd(tmp_path: Path) -> None:
    repo_dir = tmp_path / "var" / "models" / "d4sm"
    checkpoint_dir = repo_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "sam2.1_hiera_large.pt").write_bytes(b"fake")

    model = D4smVideoMaskingModel(
        Settings(
            d4sm_repo_path=repo_dir,
            d4sm_checkpoint_dir=checkpoint_dir,
            d4sm_model_size="large",
        )
    )
    model._tracker_class = RecordingD4smTracker
    RecordingD4smTracker.calls = []

    model._create_tracker()

    assert RecordingD4smTracker.calls == [
        {
            "model_size": "large",
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "offload_state_to_cpu": True,
            "cwd": repo_dir.resolve(),
        }
    ]


def test_sttn_load_uses_absolute_checkpoint_path_inside_repo_cwd(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "var" / "models" / "sttn"
    checkpoint_path = repo_dir / "checkpoints" / "sttn.pth"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"fake")
    monkeypatch.chdir(tmp_path)

    load_calls: list[dict[str, object]] = []

    def fake_torch_load(path: Path, *, map_location: str) -> dict[str, dict[str, object]]:
        load_calls.append(
            {
                "path": path,
                "map_location": map_location,
                "cwd": Path.cwd(),
            }
        )
        return {"netG": {}}

    fake_torch = SimpleNamespace(load=fake_torch_load)
    fake_torchvision = ModuleType("torchvision")
    fake_torchvision.transforms = SimpleNamespace()
    fake_core = ModuleType("core")
    fake_core_utils = ModuleType("core.utils")
    fake_core_utils.Stack = object
    fake_core_utils.ToTorchFormatTensor = object
    fake_model = ModuleType("model")
    fake_sttn_module = ModuleType("model.sttn")
    fake_sttn_module.InpaintGenerator = RecordingSttnGenerator
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torchvision", fake_torchvision)
    monkeypatch.setitem(sys.modules, "core", fake_core)
    monkeypatch.setitem(sys.modules, "core.utils", fake_core_utils)
    monkeypatch.setitem(sys.modules, "model", fake_model)
    monkeypatch.setitem(sys.modules, "model.sttn", fake_sttn_module)

    model = SttnVideoInpaintingModel(
        Settings(
            sttn_repo_path=Path("var/models/sttn"),
            sttn_checkpoint_path=Path("var/models/sttn/checkpoints/sttn.pth"),
        )
    )

    model.load()

    assert load_calls == [
        {
            "path": checkpoint_path.resolve(),
            "map_location": "cuda:0",
            "cwd": repo_dir.resolve(),
        }
    ]
    assert Path.cwd() == tmp_path


def test_d4sm_tracking_runs_under_torch_inference_mode(tmp_path: Path, monkeypatch) -> None:
    RecordingInferenceContext.enter_count = 0
    RecordingInferenceContext.exit_count = 0
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=RecordingInferenceContext))

    model = D4smVideoMaskingModel(Settings(session_storage_dir=tmp_path))
    tracker = RecordingInferenceTracker()
    image = Image.new("RGB", (10, 10))

    model._initialize_tracker(tracker=tracker, image=image, init_regions=[])
    result = model._track_frame(tracker=tracker, image=image, frame_index=1)

    assert result == {"masks": []}
    assert tracker.calls == ["initialize:(10, 10):0", "track:(10, 10)"]
    assert RecordingInferenceContext.enter_count == 2
    assert RecordingInferenceContext.exit_count == 2


def test_d4sm_cuda_cache_cleanup_uses_torch_empty_cache(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.append("empty_cache"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    model = D4smVideoMaskingModel(Settings(session_storage_dir=tmp_path))
    model._release_cuda_memory()

    assert calls == ["empty_cache"]
