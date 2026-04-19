"""
EXPERIMENTAL FEATURE

Item registry for consensus-analysis dashboard workflow.

Discovers build items from the repository's root `data/` directory.
Each item is resolved from:
1) `data/<item>/config.json` (preferred), or
2) fallback folder conventions.

NOTE: This is an experimental feature and is not part of the main VLM pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from loguru import logger


@dataclass(frozen=True)
class ItemDefinition:
    """Resolved item metadata and asset locations."""

    id: str
    label: str
    item_dir: Path
    dependencies_path: Path
    anchors_dir: Path
    ground_truth_path: Optional[Path]
    manual_pages_dir: Optional[Path]
    precomputed_result_path: Optional[Path]
    config_path: Optional[Path]
    warnings: list[str]


class ItemRegistry:
    """Discovers and resolves items from the root data directory."""

    _IGNORED_DIR_NAMES = {
        "manuals",
        "processed",
        "cropped",
        "brick_library",
        "ldraw_library",
        "lego studio",
        "analysis_runs",
    }

    def __init__(self, project_root: Optional[Path] = None, data_root: Optional[Path] = None):
        root = project_root or Path(__file__).resolve().parents[2]
        self.project_root = root.resolve()
        self.data_root = (data_root or (self.project_root / "data")).resolve()

    def list_items(self) -> list[ItemDefinition]:
        """Return all discoverable items under `data/`."""
        if not self.data_root.exists():
            logger.warning(f"Data root does not exist: {self.data_root}")
            return []

        items: list[ItemDefinition] = []
        for child in sorted(self.data_root.iterdir()):
            if not child.is_dir():
                continue
            if child.name.lower() in self._IGNORED_DIR_NAMES:
                continue

            item = self._resolve_item(child)
            if item is not None:
                items.append(item)

        return items

    def get_item(self, item_id: str) -> ItemDefinition:
        """Resolve and return one item by id."""
        for item in self.list_items():
            if item.id == item_id:
                return item
        raise ValueError(f"Unknown item '{item_id}'.")

    def _resolve_item(self, item_dir: Path) -> Optional[ItemDefinition]:
        config_path = item_dir / "config.json"
        warnings: list[str] = []

        if config_path.exists():
            cfg = self._load_config(config_path)
            if cfg is None:
                return None

            item_id = str(cfg.get("id") or item_dir.name)
            label = str(cfg.get("label") or item_id.replace("_", " ").title())

            dependencies_path = self._resolve_path(
                raw_path=cfg.get("dependencies_path"),
                item_dir=item_dir,
                required=True,
                expect_dir=False,
                warnings=warnings,
                field_name="dependencies_path",
            )
            anchors_dir = self._resolve_path(
                raw_path=cfg.get("anchors_dir"),
                item_dir=item_dir,
                required=True,
                expect_dir=True,
                warnings=warnings,
                field_name="anchors_dir",
            )
            ground_truth_path = self._resolve_path(
                raw_path=cfg.get("ground_truth_path"),
                item_dir=item_dir,
                required=False,
                expect_dir=False,
                warnings=warnings,
                field_name="ground_truth_path",
            )
            manual_pages_dir = self._resolve_path(
                raw_path=cfg.get("manual_pages_dir"),
                item_dir=item_dir,
                required=False,
                expect_dir=True,
                warnings=warnings,
                field_name="manual_pages_dir",
            )
            precomputed_result_path = self._resolve_path(
                raw_path=cfg.get("precomputed_result_path"),
                item_dir=item_dir,
                required=False,
                expect_dir=False,
                warnings=warnings,
                field_name="precomputed_result_path",
            )

            if dependencies_path is None or anchors_dir is None:
                warnings.append("Missing required dependencies/anchors for configured item.")
                logger.warning(f"Skipping item '{item_id}' because required paths are unresolved.")
                return None

            return ItemDefinition(
                id=item_id,
                label=label,
                item_dir=item_dir.resolve(),
                dependencies_path=dependencies_path,
                anchors_dir=anchors_dir,
                ground_truth_path=ground_truth_path,
                manual_pages_dir=manual_pages_dir,
                precomputed_result_path=precomputed_result_path,
                config_path=config_path.resolve(),
                warnings=warnings,
            )

        # Fallback folder conventions
        input_dependencies = sorted((item_dir / "input").glob("*dependencies*.json")) if (item_dir / "input").exists() else []
        dependencies_path = self._find_first_file(
            candidates=[
                item_dir / "input" / "dependencies.json",
                item_dir / "dependencies.json",
                *input_dependencies,
                *sorted(item_dir.glob("*dependencies*.json")),
            ]
        )
        anchors_dir = self._find_first_dir(
            candidates=[
                item_dir / "anchors_video",
                item_dir / "anchors",
            ]
        )
        ground_truth_path = self._find_first_file(
            candidates=[
                item_dir / "annotations" / "ground_truth.json",
                item_dir / "annotations" / "ground_truth_verified.json",
                item_dir / "ground_truth.json",
            ]
        )
        manual_pages_dir = self._find_first_dir(
            candidates=[
                item_dir / "manual_pages",
            ]
        )

        # Fallback requires dependencies + anchors to be considered a usable item.
        if dependencies_path is None or anchors_dir is None:
            return None

        return ItemDefinition(
            id=item_dir.name,
            label=item_dir.name.replace("_", " ").title(),
            item_dir=item_dir.resolve(),
            dependencies_path=dependencies_path.resolve(),
            anchors_dir=anchors_dir.resolve(),
            ground_truth_path=ground_truth_path.resolve() if ground_truth_path else None,
            manual_pages_dir=manual_pages_dir.resolve() if manual_pages_dir else None,
            precomputed_result_path=None,
            config_path=None,
            warnings=warnings,
        )

    def _load_config(self, config_path: Path) -> Optional[dict[str, Any]]:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to parse item config '{config_path}': {exc}")
            return None

    def _resolve_path(
        self,
        raw_path: Any,
        item_dir: Path,
        required: bool,
        expect_dir: bool,
        warnings: list[str],
        field_name: str,
    ) -> Optional[Path]:
        if not raw_path:
            if required:
                warnings.append(f"Missing required field '{field_name}'.")
            return None

        raw = Path(str(raw_path))
        candidate_paths: list[Path] = []

        if raw.is_absolute():
            candidate_paths.append(raw)
        else:
            # Prefer project-root relative paths to match contract examples.
            candidate_paths.append((self.project_root / raw).resolve())
            candidate_paths.append((item_dir / raw).resolve())
            candidate_paths.append((self.data_root / raw).resolve())

        for candidate in candidate_paths:
            if expect_dir and candidate.is_dir():
                return candidate.resolve()
            if not expect_dir and candidate.is_file():
                return candidate.resolve()

        msg = f"Could not resolve '{field_name}' from '{raw_path}'."
        if required:
            warnings.append(msg)
        else:
            logger.info(msg)
        return None

    @staticmethod
    def _find_first_file(candidates: list[Path]) -> Optional[Path]:
        for path in candidates:
            if path.is_file():
                return path
        return None

    @staticmethod
    def _find_first_dir(candidates: list[Path]) -> Optional[Path]:
        for path in candidates:
            if path.is_dir():
                return path
        return None
