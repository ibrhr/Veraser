Veraser
=======

Veraser is a FastAPI service for full object removal from video.

The project scope is the complete removal pipeline:

1. Accept a source video.
2. Extract the first frame for user-guided object selection.
3. Use a SAM2-compatible video masking model to segment and track selected objects
   across the video.
4. Use a frame-by-frame video inpainting model to remove the masked objects while
   preserving temporal consistency as much as the selected model allows.
5. Return the processed video and intermediate artifacts needed for review,
   debugging, or iteration.

The first masking backend target is a MSA2 variant called DAM4SAM. The API is
structured so DAM4SAM can be plugged in behind a model adapter without changing
the public session and prompt workflow.

Current status
--------------

This repository currently contains the API boilerplate and service boundaries,
not the actual masking or inpainting runtimes.

Implemented:

- Upload/session handling.
- First-frame extraction.
- Pixel-coordinate object prompts using points and bounding boxes.
- Multiple prompted objects per session.
- Multiple points per object.
- A DAM4SAM adapter boundary for future SAM2-compatible video masking.

Planned:

- DAM4SAM model loading and first-frame segmentation.
- Video mask propagation.
- Frame-by-frame video inpainting integration.
- Processed video artifact storage and retrieval.
- Background job execution for long-running masking and inpainting work.

Run the API locally:

```bash
uv run uvicorn app.main:app --reload
```
