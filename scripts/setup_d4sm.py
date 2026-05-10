#!/usr/bin/env python3
"""Prepare the local D4SM/DAM4SAM runtime used by Veraser."""

from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path


D4SM_REPO_URL = "https://github.com/alanlukezic/d4sm.git"
SAM21_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
CHECKPOINTS = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "large": "sam2.1_hiera_large.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone D4SM and download SAM 2.1 checkpoints into Veraser's local model cache."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("var/models/d4sm"),
        help="Local D4SM checkout directory. Default: var/models/d4sm",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Checkpoint directory. Default: <repo-dir>/checkpoints",
    )
    parser.add_argument(
        "--model-size",
        choices=[*CHECKPOINTS.keys(), "all"],
        default="large",
        help="SAM 2.1 checkpoint to download. Default: large",
    )
    parser.add_argument(
        "--skip-repo",
        action="store_true",
        help="Only download checkpoints; do not clone or update the D4SM repository.",
    )
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Only clone/update the D4SM repository; do not download checkpoints.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(f"$ {' '.join(command)}")
    subprocess.run(command, check=True)


def ensure_repo(repo_dir: Path) -> None:
    if repo_dir.exists():
        if not (repo_dir / ".git").exists():
            raise SystemExit(f"{repo_dir} exists but is not a git checkout.")
        run(["git", "-C", str(repo_dir), "pull", "--ff-only"])
        return

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", D4SM_REPO_URL, str(repo_dir)])


def download_checkpoint(checkpoint_dir: Path, filename: str) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / filename
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already present: {destination}")
        return

    url = f"{SAM21_BASE_URL}/{filename}"
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, temporary_path)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Wrote {destination}")


def write_env_hint(repo_dir: Path, checkpoint_dir: Path, model_size: str) -> None:
    print("\nInstall the Python runtime dependencies before starting the API:")
    print("uv sync --group gpu")
    print("\nSet these values if you use non-default paths:")
    print(f"export VERASER_D4SM_REPO_PATH={repo_dir.resolve()}")
    print(f"export VERASER_D4SM_CHECKPOINT_DIR={checkpoint_dir.resolve()}")
    if model_size != "all":
        print(f"export VERASER_D4SM_MODEL_SIZE={model_size}")
    print("\nDefault Veraser config already points to var/models/d4sm.")


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir
    checkpoint_dir = args.checkpoint_dir or repo_dir / "checkpoints"

    if not args.skip_repo:
        ensure_repo(repo_dir)

    if not args.skip_checkpoints:
        model_sizes = CHECKPOINTS.keys() if args.model_size == "all" else [args.model_size]
        for model_size in model_sizes:
            download_checkpoint(checkpoint_dir, CHECKPOINTS[model_size])

    write_env_hint(repo_dir, checkpoint_dir, args.model_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
