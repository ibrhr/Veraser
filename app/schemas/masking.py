from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.performance import OperationSpeedMetric


MaskingJobStatus = Literal["pending", "running", "succeeded", "failed"]
InpaintingJobStatus = Literal["pending", "running", "succeeded", "failed"]


class MaskingJobResponse(BaseModel):
    job_id: str
    session_id: str
    status: MaskingJobStatus
    created_at: datetime
    updated_at: datetime
    frames_total: int | None = None
    frames_done: int = 0
    current_stage: str = "queued"
    error: str | None = None
    performance: list[OperationSpeedMetric] = Field(default_factory=list)
    manifest_url: str
    processed_video_url: str


class MaskArtifactManifest(BaseModel):
    session_id: str
    job_id: str
    frames_total: int
    objects: list[str] = Field(default_factory=list)
    combined_masks_url: str
    processed_video_url: str


class InpaintingJobResponse(BaseModel):
    job_id: str
    session_id: str
    masking_job_id: str
    status: InpaintingJobStatus
    created_at: datetime
    updated_at: datetime
    frames_total: int | None = None
    frames_done: int = 0
    current_stage: str = "queued"
    error: str | None = None
    performance: list[OperationSpeedMetric] = Field(default_factory=list)
    processed_video_url: str
