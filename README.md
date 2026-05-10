Veraser
=======

Veraser is a demo-ready FastAPI service and React GUI for full object removal
from video.

The project scope is the complete removal pipeline:

1. Accept a source video.
2. Extract the first frame for user-guided object selection.
3. Use a SAM2-compatible video masking model to segment and track selected objects
   across the video.
4. Use a frame-by-frame video inpainting model to remove the masked objects while
   preserving temporal consistency as much as the selected model allows.
5. Return the processed video and intermediate artifacts needed for review,
   debugging, or iteration.

The first masking backend target is the multi-object D4SM implementation of
DAM4SAM, a SAM2-compatible tracker. The API is structured so masking and
inpainting runtimes can be replaced behind adapters without changing the public
session and prompt workflow.

Quick start
-----------

Run the API-only demo:

```bash
uv sync
uv run uvicorn app.main:app --reload
```

Run the GUI in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, upload a video, draw boxes or point prompts for
one or more objects, preview the first-frame mask, then start the masking job.

Prepare the DAM4SAM/D4SM runtime on a CUDA machine:

```bash
uv sync --group gpu
python3 scripts/setup_d4sm.py --model-size large
python3 scripts/setup_sttn.py
```

This clones D4SM and STTN, then downloads their checkpoints under `var/models/`.
API-only tests and GUI development do not require GPU dependencies or model
weights.

Current status
--------------

This repository currently contains the API, demo GUI, prompt workflow, masking
job boundary, D4SM adapter setup, and STTN inpainting adapter setup. Full CUDA
runtime validation is still planned.

Implemented:

- Upload/session handling.
- First-frame extraction.
- Pixel-coordinate object prompts using points and bounding boxes.
- Multiple prompted objects per session.
- Multiple points per object.
- First-frame mask preview for UI verification.
- Full-video mask job endpoints and artifact paths.
- A D4SM/DAM4SAM adapter boundary for SAM2-compatible video masking.
- A separate STTN inpainting job boundary that consumes tracked masks.
- Processed video artifact storage and retrieval.
- React + Vite + TypeScript + Konva demo GUI.

Planned:

- Full D4SM/DAM4SAM runtime validation on a CUDA machine.
- Full STTN runtime validation on a CUDA machine.
- Production queue execution for long-running masking and inpainting work.

D4SM / DAM4SAM integration
--------------------------

The masking integration targets the multi-object DAM4SAM implementation from
`alanlukezic/d4sm`. Veraser keeps that checkout and its checkpoints in the local
runtime cache by default:

```text
var/models/d4sm/
var/models/d4sm/checkpoints/
```

Prepare that cache on a CUDA machine with:

```bash
python3 scripts/setup_d4sm.py --model-size large
```

The script clones `https://github.com/alanlukezic/d4sm.git` into
`var/models/d4sm` and downloads the selected SAM 2.1 checkpoint from Meta's
public checkpoint host into `var/models/d4sm/checkpoints`. Use
`--model-size tiny`, `small`, `base_plus`, `large`, or `all`.

The optional `gpu` dependency group installs the CUDA runtime packages needed for
a dedicated ML environment with a compatible NVIDIA driver:

```bash
uv sync --group gpu
```

The `gpu` group installs PyTorch/TorchVision from the CUDA 12.4 PyTorch wheel
index, plus the Python packages used by the D4SM wrapper, including Hydra,
OpenCV, NumPy, OmegaConf, and the VOT toolkit. This project does not require
installing those dependencies to run API-only tests.

The default configuration points at the local cache above. Override it only when
you keep D4SM elsewhere:

```bash
export VERASER_D4SM_REPO_PATH=/path/to/d4sm
export VERASER_D4SM_CHECKPOINT_DIR=/path/to/checkpoints
export VERASER_D4SM_MODEL_SIZE=large
export VERASER_D4SM_DEVICE=cuda:0
```

The API records prompts separately from inference. After adding prompted objects,
frontend clients can list, update, delete, and preview object selections before
starting full-video tracking.

STTN integration
----------------

The inpainting integration targets `researchmm/STTN`, the ECCV 2020
Spatial-Temporal Transformer Network for video inpainting. Veraser keeps the
checkout and checkpoint in the local runtime cache by default:

```text
var/models/sttn/
var/models/sttn/checkpoints/sttn.pth
```

Prepare that cache on a CUDA machine with:

```bash
uv sync --group gpu
python3 scripts/setup_sttn.py
```

The script clones `https://github.com/researchmm/STTN.git` into
`var/models/sttn` and downloads the official pretrained YouTube-VOS checkpoint
from the Google Drive file linked by the STTN README. The `gpu` group includes
`gdown` for this checkpoint download.

STTN inference is isolated behind `SttnVideoInpaintingModel`. Its defaults match
the official test script: 432 x 240 inference, `ref_length=10`,
`neighbor_stride=5`, four mask dilation iterations, and MP4 output. Override
paths or runtime parameters only when needed:

```bash
export VERASER_STTN_REPO_PATH=/path/to/STTN
export VERASER_STTN_CHECKPOINT_PATH=/path/to/sttn.pth
export VERASER_STTN_DEVICE=cuda:0
export VERASER_STTN_WIDTH=432
export VERASER_STTN_HEIGHT=240
```

Demo API workflow
-----------------

1. Upload a video:

```http
POST /api/v1/video-sessions
```

2. Display the first frame returned by the upload response:

```http
GET /api/v1/video-sessions/{session_id}/first-frame
```

3. Add one or more selected objects using point or box prompts:

```http
POST /api/v1/video-sessions/{session_id}/objects
```

4. Visually verify and correct selections:

```http
GET /api/v1/video-sessions/{session_id}/objects
GET /api/v1/video-sessions/{session_id}/objects/{object_id}/preview-mask
PUT /api/v1/video-sessions/{session_id}/objects/{object_id}
DELETE /api/v1/video-sessions/{session_id}/objects/{object_id}
```

5. Start full-video mask tracking:

```http
POST /api/v1/video-sessions/{session_id}/masking-jobs
```

6. Poll the returned `job_id`:

```http
GET /api/v1/video-sessions/{session_id}/masking-jobs/{job_id}
```

7. Retrieve and visually inspect tracked masks:

```http
GET /api/v1/video-sessions/{session_id}/masking-jobs/{job_id}/masks/manifest
GET /api/v1/video-sessions/{session_id}/masking-jobs/{job_id}/masks/combined/{frame_index}
```

8. Start inpainting after the tracked mask is correct, poll it, then fetch the
   processed video:

```http
POST /api/v1/video-sessions/{session_id}/masking-jobs/{job_id}/inpainting-job
GET /api/v1/video-sessions/{session_id}/masking-jobs/{job_id}/inpainting-job
GET /api/v1/video-sessions/{session_id}/masking-jobs/{job_id}/processed-video
```

Development commands
--------------------

```bash
# API server
uv run uvicorn app.main:app --reload

# Backend tests
uv run pytest

# Runtime model setup on CUDA hosts
uv sync --group gpu
python3 scripts/setup_d4sm.py --model-size large
python3 scripts/setup_sttn.py

# Frontend dev server
cd frontend
npm run dev

# Frontend production build
cd frontend
npm run build
```

The current development worker uses FastAPI background tasks; the service
boundary is structured so this can move to a dedicated GPU queue later.
