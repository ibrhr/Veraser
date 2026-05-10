from pydantic import BaseModel


class OperationSpeedMetric(BaseModel):
    name: str
    label: str
    elapsed_seconds: float
    frames_processed: int | None = None
    fps: float | None = None
    seconds_per_frame: float | None = None


def build_speed_metric(
    *,
    name: str,
    label: str,
    elapsed_seconds: float,
    frames_processed: int | None = None,
) -> OperationSpeedMetric:
    rounded_seconds = round(max(elapsed_seconds, 0.0), 4)
    fps = None
    seconds_per_frame = None
    if frames_processed and rounded_seconds > 0:
        fps = round(frames_processed / rounded_seconds, 4)
        seconds_per_frame = round(rounded_seconds / frames_processed, 4)

    return OperationSpeedMetric(
        name=name,
        label=label,
        elapsed_seconds=rounded_seconds,
        frames_processed=frames_processed,
        fps=fps,
        seconds_per_frame=seconds_per_frame,
    )
