# Repository Guidelines

## Project Structure & Module Organization

This repository is a FastAPI service for video object removal. Source code lives in
`app/`:

- `app/main.py` creates the FastAPI application.
- `app/api/` contains versioned HTTP routes and dependency wiring.
- `app/core/` contains environment-backed configuration.
- `app/schemas/` contains Pydantic request and response models.
- `app/services/` contains application services for sessions, frames, prompts, and storage.
- `app/models/` contains model interfaces and the DAM4SAM adapter boundary.
- `tests/` contains pytest tests.

Runtime uploads and generated artifacts should stay under `var/`, which is ignored
by git.

## Build, Test, and Development Commands

- `uv run uvicorn app.main:app --reload` starts the local API server.
- `uv run pytest` runs the test suite.
- `python3 -m compileall app tests` performs a quick syntax/import compilation pass.

Use `uv` for dependency-managed commands so the Python version and lockfile stay
consistent with the project.

## Coding Style & Naming Conventions

Use Python 3.13 syntax and standard 4-space indentation. Keep modules focused by
layer: routes should handle HTTP concerns, services should contain workflow logic,
schemas should define API shapes, and model adapters should isolate ML runtime
integration.

Prefer clear names such as `VideoSessionService`, `SubmitObjectPromptsRequest`,
and `Dam4SamVideoMaskingModel`. Use snake_case for functions, variables, and file
names; use PascalCase for classes and Pydantic models.

## Testing Guidelines

Tests use `pytest`. Place tests in `tests/` and name files `test_*.py`. Test
functions should also start with `test_`.

Focus coverage on API contracts, coordinate validation, session lifecycle behavior,
and model-adapter boundaries. Do not require real DAM4SAM or inpainting weights in
unit tests; mock or stub model integrations until runtime support exists.

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
