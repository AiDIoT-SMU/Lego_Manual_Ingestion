import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pytest.importorskip("loguru")

from backend.services.item_registry import ItemRegistry


os.environ.setdefault("GEMINI_API_KEY", "test-key")


def test_item_registry_prefers_config_and_resolves_paths(tmp_path: Path):
    project_root = tmp_path
    data_root = project_root / "data"
    item_dir = data_root / "sample_item"
    (item_dir / "input").mkdir(parents=True)
    (item_dir / "anchors_video").mkdir(parents=True)
    (item_dir / "annotations").mkdir(parents=True)

    deps = item_dir / "input" / "deps.json"
    deps.write_text("{}", encoding="utf-8")
    gt = item_dir / "annotations" / "ground_truth.json"
    gt.write_text("{}", encoding="utf-8")
    anchor = item_dir / "anchors_video" / "step_01_anchor.jpg"
    anchor.write_bytes(b"jpg")

    config = {
        "id": "sample-item",
        "label": "Sample Item",
        "dependencies_path": "data/sample_item/input/deps.json",
        "anchors_dir": "data/sample_item/anchors_video",
        "ground_truth_path": "data/sample_item/annotations/ground_truth.json",
    }
    (item_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    registry = ItemRegistry(project_root=project_root, data_root=data_root)
    items = registry.list_items()
    assert len(items) == 1
    assert items[0].id == "sample-item"
    assert items[0].dependencies_path == deps.resolve()


def test_item_registry_fallback_conventions(tmp_path: Path):
    project_root = tmp_path
    data_root = project_root / "data"
    item_dir = data_root / "fallback_item"
    (item_dir / "input").mkdir(parents=True)
    (item_dir / "anchors").mkdir(parents=True)

    deps = item_dir / "input" / "my_dependencies.json"
    deps.write_text("{}", encoding="utf-8")
    (item_dir / "anchors" / "step_01_anchor.jpg").write_bytes(b"jpg")

    registry = ItemRegistry(project_root=project_root, data_root=data_root)
    items = registry.list_items()

    assert len(items) == 1
    assert items[0].id == "fallback_item"
    assert items[0].dependencies_path == deps.resolve()
    assert items[0].anchors_dir == (item_dir / "anchors").resolve()
