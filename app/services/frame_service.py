import logging
from pathlib import Path

from PIL import Image

from app.core.config import Settings


logger = logging.getLogger(__name__)


class VideoFrameService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract_first_frame(self, *, video_path: Path, output_path: Path) -> tuple[int, int]:
        import imageio.v3 as iio

        logger.info("Reading first video frame video_path=%s", video_path)
        frame = iio.imread(video_path, index=0)
        image = Image.fromarray(frame)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format=self.settings.first_frame_format.upper())
        logger.info(
            "Saved first video frame output_path=%s width=%s height=%s",
            output_path,
            image.width,
            image.height,
        )
        return image.width, image.height
