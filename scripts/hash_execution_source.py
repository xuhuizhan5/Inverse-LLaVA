#!/usr/bin/env python3
"""Hash the files that can change scientific execution behavior."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Admission checks call this script before the package is installed.  Resolve the
# adjacent source tree explicitly so the command has the same behavior in a clean
# shell, an unpacked archive, and an editable development environment.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from invllava.artifacts.source import execution_source_files, execution_source_sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--details", action="store_true")
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the relative staged-file inventory, one path per line",
    )
    args = parser.parse_args()
    if args.details and args.list:
        parser.error("--details and --list are mutually exclusive")
    if args.list:
        root = args.root.resolve()
        for path in execution_source_files(root):
            print(path.relative_to(root).as_posix())
        return
    digest, file_count, total_bytes = execution_source_sha256(args.root)
    if args.details:
        print(f"sha256={digest} files={file_count} bytes={total_bytes}")
    else:
        print(digest)


if __name__ == "__main__":
    main()
