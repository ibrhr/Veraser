import logging
from pathlib import Path

from PIL import Image


logger = logging.getLogger(__name__)


class VideoFrameExtractionService:
    def extract_frames(self, *, video_path: Path, output_dir: Path) -> int:
        import imageio.v3 as iio

        output_dir.mkdir(parents=True, exist_ok=True)
        existing_frames = sorted(output_dir.glob("*.png"))
        if existing_frames:
            logger.info(
                "Using existing extracted frames video_path=%s output_dir=%s frame_count=%s",
                video_path,
                output_dir,
                len(existing_frames),
            )
            return len(existing_frames)

        logger.info("Extracting video frames video_path=%s output_dir=%s", video_path, output_dir)
        frame_count = 0
        for frame_count, frame in enumerate(iio.imiter(video_path), start=1):
            image = Image.fromarray(frame)
            frame_path = output_dir / f"{frame_count - 1:06d}.png"
            image.save(frame_path)
            logger.debug("Extracted frame index=%s path=%s", frame_count - 1, frame_path)
        logger.info("Finished extracting video frames video_path=%s output_dir=%s frame_count=%s", video_path, output_dir, frame_count)
        return frame_count
