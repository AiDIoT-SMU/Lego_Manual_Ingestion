"""
Clear all ingestion output for a specific manual or all manuals.

Usage:
    uv run python scripts/clear_ingestion.py              # clears ALL manuals
    uv run python scripts/clear_ingestion.py 6262059      # clears one manual
"""

import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INGESTION_DIRS = [
    ROOT / "data" / "manuals",
    ROOT / "data" / "processed",
    ROOT / "data" / "cropped",
]


def clear_manual(manual_id: str) -> None:
    removed_any = False
    for base in INGESTION_DIRS:
        target = base / manual_id
        if target.exists():
            shutil.rmtree(target)
            print(f"  deleted  {target.relative_to(ROOT)}")
            removed_any = True
    if not removed_any:
        print(f"  nothing found for manual_id='{manual_id}'")


def clear_all() -> None:
    for base in INGESTION_DIRS:
        for child in base.iterdir():
            if child.name == ".gitkeep" or not child.is_dir():
                continue
            shutil.rmtree(child)
            print(f"  deleted  {child.relative_to(ROOT)}")


def main() -> None:
    if len(sys.argv) > 1:
        manual_id = sys.argv[1]
        print(f"Clearing ingestion for manual: {manual_id}")
        clear_manual(manual_id)
    else:
        confirm = input("Clear ALL ingested manuals? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return
        print("Clearing all ingestion data…")
        clear_all()
    print("Done.")


if __name__ == "__main__":
    main()
