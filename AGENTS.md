# Repository Guidelines

## Project Structure & Module Organization

This repository is a FastAPI service plus React demo GUI for video object
removal. Backend source code lives in `app/`:

- `app/main.py` creates the FastAPI application.
- `app/api/` contains versioned HTTP routes and dependency wiring.
- `app/core/` contains environment-backed configuration.
- `app/schemas/` contains Pydantic request and response models.
- `app/services/` contains application services for sessions, frames, prompts, jobs, and storage.
- `app/models/` contains model interfaces plus DAM4SAM and STTN adapter boundaries.
- `tests/` contains pytest tests.
- `scripts/` contains setup utilities, including `setup_d4sm.py` and `setup_sttn.py`.
- `frontend/` contains the React + Vite + TypeScript + Konva demo client.

Runtime uploads, generated masks, model checkouts, and checkpoints should stay
under `var/`, which is ignored by git.

## Build, Test, and Development Commands

- `uv run uvicorn app.main:app --reload` starts the local API server.
- `uv run pytest` runs the test suite.
- `python3 -m compileall app tests scripts` performs a quick syntax/import pass.
- `uv run python scripts/setup_d4sm.py --model-size large` clones D4SM and downloads the
  default SAM 2.1 checkpoint into `var/models/d4sm`.
- `uv run python scripts/setup_sttn.py` clones STTN and downloads `sttn.pth` into
  `var/models/sttn/checkpoints`.
- `cd frontend && npm install` installs GUI dependencies.
- `cd frontend && npm run dev` starts the Vite dev server at `localhost:5173`.
- `cd frontend && npm run build` type-checks and builds the GUI.

Use `uv` for dependency-managed commands so the Python version and lockfile stay
consistent with the project.

## Coding Style & Naming Conventions

Use Python 3.13 syntax and standard 4-space indentation. Keep modules focused by
layer: routes should handle HTTP concerns, services should contain workflow logic,
schemas should define API shapes, and model adapters should isolate ML runtime
integration.

Prefer clear names such as `VideoSessionService`, `SubmitObjectPromptsRequest`,
`D4smVideoMaskingModel`, and `SttnVideoInpaintingModel`. Use snake_case for
functions, variables, and file names; use PascalCase for classes and Pydantic
models.

Frontend code should use TypeScript, React function components, and local API
helpers in `frontend/src/api.ts`. Keep Konva coordinate handling in image pixel
space so prompts match backend validation.

## Testing Guidelines

Tests use `pytest`. Place tests in `tests/` and name files `test_*.py`. Test
functions should also start with `test_`.

Focus coverage on API contracts, coordinate validation, session lifecycle
behavior, masking/inpainting job lifecycles, artifact paths, and model-adapter
boundaries. Do not require real DAM4SAM/D4SM, STTN, or inpainting weights in unit
tests; mock or stub model integrations.

## Commit & Pull Request Guidelines

This repository currently has no commit history, so there is no established commit
message convention. Use short, imperative commit messages, for example:

- `Add video session API`
- `Validate object prompt coordinates`
- `Document object removal pipeline`

Pull requests should include a concise description, the commands run for testing,
and any API contract changes. Include example request/response payloads when
changing endpoints. For future UI changes, include screenshots or short screen
recordings.

## Security & Configuration Tips

Configuration is read from environment variables with the `VERASER_` prefix. Do
not commit secrets, model weights, uploaded videos, generated frames, or processed
video artifacts. Keep large local assets under ignored runtime directories such as
`var/`.

The default D4SM paths are `var/models/d4sm` and
`var/models/d4sm/checkpoints`. The default STTN paths are `var/models/sttn` and
`var/models/sttn/checkpoints/sttn.pth`. Override them with `VERASER_D4SM_*` and
`VERASER_STTN_*` variables only when using external checkouts or checkpoints.
