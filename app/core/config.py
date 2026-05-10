from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Veraser"
    api_version: str = "v1"
    log_level: str = "INFO"
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
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )

    d4sm_repo_path: Path = Path("var/models/d4sm")
    d4sm_checkpoint_dir: Path = Path("var/models/d4sm/checkpoints")
    d4sm_model_size: str = "large"
    d4sm_device: str = "cuda:0"
    d4sm_lazy_load: bool = True
    d4sm_offload_state_to_cpu: bool = False

    sttn_repo_path: Path = Path("var/models/sttn")
    sttn_checkpoint_path: Path = Path("var/models/sttn/checkpoints/sttn.pth")
    sttn_device: str = "cuda:0"
    sttn_width: int = 432
    sttn_height: int = 240
    sttn_ref_length: int = 10
    sttn_neighbor_stride: int = 5
    sttn_mask_dilation_iterations: int = 4
    sttn_output_fps: int = 24
    sttn_preserve_source_resolution: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VERASER_",
        extra="ignore",
    )
