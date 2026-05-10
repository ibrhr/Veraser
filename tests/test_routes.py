import asyncio

import pytest
from fastapi import HTTPException

from app.api.v1 import routes
from app.services.errors import SessionNotFoundError


class MissingSessionMaskingJobService:
    def get_job(self, session_id: str, job_id: str):
        raise SessionNotFoundError(f"Video session {session_id} was not found.")


class MissingSessionInpaintingJobService:
    def get_job(self, session_id: str, masking_job_id: str):
        raise SessionNotFoundError(f"Video session {session_id} was not found.")


def assert_raises_404(coro) -> None:
    with pytest.raises(HTTPException) as error:
        asyncio.run(coro)
    assert error.value.status_code == 404
    assert error.value.detail == "Video session missing was not found."


def test_stale_masking_job_polling_returns_404() -> None:
    assert_raises_404(
        routes.get_masking_job(
            session_id="missing",
            job_id="job-1",
            service=MissingSessionMaskingJobService(),
        )
    )


def test_stale_mask_manifest_polling_returns_404() -> None:
    assert_raises_404(
        routes.get_mask_manifest(
            session_id="missing",
            job_id="job-1",
            artifact_service=None,  # type: ignore[arg-type]
            job_service=MissingSessionMaskingJobService(),
        )
    )


def test_stale_combined_mask_polling_returns_404() -> None:
    assert_raises_404(
        routes.get_combined_mask(
            session_id="missing",
            job_id="job-1",
            frame_index=0,
            artifact_service=None,  # type: ignore[arg-type]
            job_service=MissingSessionMaskingJobService(),
        )
    )


def test_stale_processed_video_polling_returns_404() -> None:
    assert_raises_404(
        routes.get_processed_video(
            session_id="missing",
            job_id="job-1",
            artifact_service=None,  # type: ignore[arg-type]
            job_service=MissingSessionMaskingJobService(),
        )
    )


def test_stale_inpainting_job_polling_returns_404() -> None:
    assert_raises_404(
        routes.get_inpainting_job(
            session_id="missing",
            job_id="job-1",
            service=MissingSessionInpaintingJobService(),
        )
    )
