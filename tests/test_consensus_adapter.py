import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pytest.importorskip("loguru")

from backend.services.consensus_adapter import (
    build_detection_lookup,
    get_confidence,
    get_detected_step,
    get_method,
    get_non_progress_reason,
    resolve_record_for_second,
)


os.environ.setdefault("GEMINI_API_KEY", "test-key")


def test_parser_fallbacks_detected_step_confidence_method():
    record = {
        "predicted_step": "3",
        "vlm_confidence": 0.41,
        "gate_similarity": 0.66,
        "error_detection_ran": True,
        "vlm_called": False,
    }
    assert get_detected_step(record) == 3
    assert get_confidence(record) == pytest.approx(0.41)
    assert get_method(record) == "vlm"

    fallback_record = {
        "predicted_step": "not-a-number",
        "gate_similarity": 0.52,
        "error_detection_ran": True,
        "vlm_called": False,
    }
    assert get_detected_step(fallback_record) == 0
    assert get_confidence(fallback_record) == pytest.approx(0.52)
    assert get_method(fallback_record) == "err-only"


def test_non_progress_reason_legacy_step_mismatch_protection():
    record = {
        "gate_triggered": True,
        "step_complete": False,
        "vlm_reasoning": "Step 5 is missing the left support.",
        "non_progress_trigger": "blocked",
    }
    reason = get_non_progress_reason(record, expected_next_step=4)
    assert "Step 4 not verified yet" in reason


def test_timeline_carry_forward_for_missing_seconds():
    per_second_results = [
        {"timestamp_sec": 0, "predicted_step": "0"},
        {"timestamp_sec": 2, "predicted_step": "1"},
        {"timestamp_sec": 5, "predicted_step": "2"},
    ]
    lookup = build_detection_lookup(per_second_results)
    sorted_seconds = sorted(lookup)

    assert resolve_record_for_second(0, lookup, sorted_seconds)["predicted_step"] == "0"
    assert resolve_record_for_second(1, lookup, sorted_seconds)["predicted_step"] == "0"
    assert resolve_record_for_second(4, lookup, sorted_seconds)["predicted_step"] == "1"
    assert resolve_record_for_second(9, lookup, sorted_seconds)["predicted_step"] == "2"
