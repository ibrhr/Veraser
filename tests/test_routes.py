import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_inpainting_job_service, get_masking_job_service
from app.main import create_app
from app.services.errors import SessionNotFoundError


class MissingSessionMaskingJobService:
    def get_job(self, session_id: str, job_id: str):
        raise SessionNotFoundError(f"Video session {session_id} was not found.")


class MissingSessionInpaintingJobService:
    def get_job(self, session_id: str, masking_job_id: str):
        raise SessionNotFoundError(f"Video session {session_id} was not found.")


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_masking_job_service] = MissingSessionMaskingJobService
    app.dependency_overrides[get_inpainting_job_service] = MissingSessionInpaintingJobService
    return TestClient(app)


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/video-sessions/missing/masking-jobs/job-1",
        "/api/v1/video-sessions/missing/masking-jobs/job-1/masks/manifest",
        "/api/v1/video-sessions/missing/masking-jobs/job-1/masks/combined/0",
        "/api/v1/video-sessions/missing/masking-jobs/job-1/processed-video",
        "/api/v1/video-sessions/missing/masking-jobs/job-1/inpainting-job",
    ],
)
def test_stale_job_polling_returns_404(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 404
    assert response.json() == {"detail": "Video session missing was not found."}
