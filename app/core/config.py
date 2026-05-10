from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Veraser"
    api_version: str = "v1"
    session_storage_dir: Path = Path("var/video-sessions")
    max_upload_bytes: int = 1_073_741_824
    accepted_video_extensions: set[str] = Field(
        default_factory=lambda: {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    )
    accepted_video_mime_types: set[str] = Field(
        default_factory=lambda: {
            "video/mp4",
            "video/quicktime",
            "video/x-matroska",
            "video/x-msvideo",
            "video/webm",
        }
    )
    first_frame_format: str = "png"

    dam4sam_checkpoint_path: Path | None = None
    dam4sam_device: str = "cuda"
    dam4sam_lazy_load: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VERASER_",
        extra="ignore",
    )
