from pathlib import Path

from PIL import Image


class VideoFrameExtractionService:
    def extract_frames(self, *, video_path: Path, output_dir: Path) -> int:
        import imageio.v3 as iio

        output_dir.mkdir(parents=True, exist_ok=True)
        existing_frames = sorted(output_dir.glob("*.png"))
        if existing_frames:
            return len(existing_frames)

        frame_count = 0
        for frame_count, frame in enumerate(iio.imiter(video_path), start=1):
            image = Image.fromarray(frame)
            image.save(output_dir / f"{frame_count - 1:06d}.png")
        return frame_count
