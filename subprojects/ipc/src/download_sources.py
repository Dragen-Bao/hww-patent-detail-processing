"""Download the pinned official IPC 2026.01 source files."""

from __future__ import annotations

import argparse
from pathlib import Path

from ipc_sources import download_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    result = download_sources(args.output_dir, timeout=args.timeout)
    print(f"manifest={result['manifest']}")
    for source in result["sources"]:
        print(f"{source['name']}: {source['bytes']} bytes sha256={source['sha256']}")


if __name__ == "__main__":
    main()
