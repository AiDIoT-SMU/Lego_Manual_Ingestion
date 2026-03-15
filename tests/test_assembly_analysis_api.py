import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("fastapi")
pytest.importorskip("loguru")
from fastapi.testclient import TestClient

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from backend.main import app  # noqa: E402


def test_item_selection_video_upload_and_synced_details():
    client = TestClient(app)

    items_resp = client.get("/api/assembly/items")
    assert items_resp.status_code == 200
    items = items_resp.json()["items"]
    changi = next((x for x in items if x["id"] == "changi"), None)
    assert changi is not None

    details_payload = {
        "experiment_id": "uploaded_fixture",
        "experiment_name": "Uploaded Fixture",
        "architecture": "uploaded",
        "metadata": {"source": "test"},
        "per_second_results": [
            {"timestamp_sec": 0, "predicted_step": "0", "gate_triggered": False, "vlm_called": False},
            {"timestamp_sec": 2, "predicted_step": "1", "gate_triggered": True, "vlm_called": False},
        ],
    }

    analyze_resp = client.post(
        "/api/assembly/analyze",
        data={"item_id": "changi"},
        files=[
            ("video_file", ("demo.mp4", b"fake-video-bytes", "video/mp4")),
            (
                "details_json_file",
                (
                    "consensus_gate_absolute_error_only_with_error_detection.json",
                    json.dumps(details_payload).encode("utf-8"),
                    "application/json",
                ),
            ),
        ],
    )
    assert analyze_resp.status_code == 200

    payload = analyze_resp.json()
    assert payload["analysis_id"]
    assert payload["item"]["id"] == "changi"
    assert payload["mode"] == "uploaded-details"
    assert isinstance(payload["timeline"], list)
    assert len(payload["timeline"]) > 0

    analysis_id = payload["analysis_id"]
    late_second_resp = client.get(f"/api/assembly/analysis/{analysis_id}/second/1000")
    assert late_second_resp.status_code == 200
    late_details = late_second_resp.json()["details"]
    assert late_details["timestamp_sec"] == 1000
    assert "detected_step" in late_details
    assert "trace" in late_details


def test_analysis_requires_consensus_json_upload():
    client = TestClient(app)
    resp = client.post(
        "/api/assembly/analyze",
        data={"item_id": "changi"},
        files=[("video_file", ("demo.mp4", b"fake-video-bytes", "video/mp4"))],
    )
    assert resp.status_code == 400
    assert "must be provided" in resp.json()["detail"]
