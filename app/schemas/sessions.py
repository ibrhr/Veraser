from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.performance import OperationSpeedMetric
from app.schemas.prompts import StoredObjectPrompt


class VideoMetadata(BaseModel):
    filename: str
    content_type: str | None = None
    size_bytes: int
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None


class FirstFrameInfo(BaseModel):
    width: int
    height: int
    content_type: Literal["image/png"] = "image/png"
    url: str


class VideoSessionResponse(BaseModel):
    session_id: str
    status: Literal["ready_for_prompts"] = "ready_for_prompts"
    created_at: datetime
    video: VideoMetadata
    first_frame: FirstFrameInfo
    performance: list[OperationSpeedMetric] = Field(default_factory=list)


class VideoSessionDetail(VideoSessionResponse):
    objects: list[StoredObjectPrompt] = Field(default_factory=list)
