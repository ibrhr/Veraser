from pathlib import Path

from PIL import Image

from app.core.config import Settings


class VideoFrameService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract_first_frame(self, *, video_path: Path, output_path: Path) -> tuple[int, int]:
        import imageio.v3 as iio

        frame = iio.imread(video_path, index=0)
        image = Image.fromarray(frame)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format=self.settings.first_frame_format.upper())
        return image.width, image.height
