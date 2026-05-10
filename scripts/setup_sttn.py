#!/usr/bin/env python3
"""Prepare the local STTN runtime used by Veraser."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STTN_REPO_URL = "https://github.com/researchmm/STTN.git"
STTN_CHECKPOINT_FILE_ID = "1ZAMV8547wmZylKRt5qR_tC5VlosXD4Wv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone STTN and download the pretrained checkpoint into Veraser's local model cache."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("var/models/sttn"),
        help="Local STTN checkout directory. Default: var/models/sttn",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint path. Default: <repo-dir>/checkpoints/sttn.pth",
    )
    parser.add_argument(
        "--skip-repo",
        action="store_true",
        help="Only download the checkpoint; do not clone or update the STTN repository.",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Only clone/update STTN; do not download the checkpoint.",
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
    run(["git", "clone", STTN_REPO_URL, str(repo_dir)])


def download_checkpoint(checkpoint_path: Path) -> None:
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        print(f"Already present: {checkpoint_path}")
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    run(
        [
            sys.executable,
            "-m",
            "gdown",
            "--id",
            STTN_CHECKPOINT_FILE_ID,
            "--output",
            str(temporary_path),
        ]
    )
    temporary_path.replace(checkpoint_path)
    print(f"Wrote {checkpoint_path}")


def write_env_hint(repo_dir: Path, checkpoint_path: Path) -> None:
    print("\nSet these values if you use non-default paths:")
    print(f"export VERASER_STTN_REPO_PATH={repo_dir.resolve()}")
    print(f"export VERASER_STTN_CHECKPOINT_PATH={checkpoint_path.resolve()}")
    print("export VERASER_STTN_DEVICE=cuda:0")
    print("\nDefault Veraser config already points to var/models/sttn.")


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir
    checkpoint_path = args.checkpoint_path or repo_dir / "checkpoints" / "sttn.pth"

    if not args.skip_repo:
        ensure_repo(repo_dir)
    if not args.skip_checkpoint:
        download_checkpoint(checkpoint_path)

    write_env_hint(repo_dir, checkpoint_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
