"""
Consensus parser + timeline adapter extracted from BuildGuide renderer logic.

These helpers keep behavior parity with:
`eval/render_video_consensus.py`
without rendering overlays onto frames.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from pathlib import Path
from typing import Any, Optional

from backend.services.item_registry import ItemDefinition


def load_step_labels(deps_path: Path) -> tuple[dict[int, str], dict[int, str], list[int]]:
    """Load step labels from dependencies JSON."""
    with open(deps_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", {})
    build_order = data.get("build_order", sorted(int(k) for k in nodes))
    total_steps = max(build_order) if build_order else 0

    completed_labels: dict[int, str] = {}
    guidance_labels: dict[int, str] = {}

    for step_str, node in nodes.items():
        step = int(step_str)
        parts = node.get("new_parts_to_add", [])
        actions = node.get("actions", [])
        sub_desc = (node.get("subassembly_hint") or {}).get("description", "")

        if step == 0:
            completed_labels[step] = node.get("notes", "Empty workspace")
        elif actions:
            a = actions[0]
            completed_labels[step] = f"{a.get('action_verb', 'Add').capitalize()} {a.get('target', '')}"
        elif parts:
            completed_labels[step] = f"Add {', '.join(parts)}"
        else:
            completed_labels[step] = sub_desc or f"Step {step}"

    for step in build_order:
        next_step = step + 1
        if next_step > total_steps:
            guidance_labels[step] = "Assembly complete!"
        elif str(next_step) in nodes:
            next_node = nodes[str(next_step)]
            next_actions = next_node.get("actions", [])
            next_parts = next_node.get("new_parts_to_add", [])
            if next_actions:
                a = next_actions[0]
                guidance_labels[step] = f"{a.get('action_verb', 'Add').capitalize()} {a.get('target', '')}"
            elif next_parts:
                guidance_labels[step] = f"Add {', '.join(next_parts)}"
            else:
                guidance_labels[step] = f"Do step {next_step}"
        else:
            guidance_labels[step] = "Assembly complete!"

    return completed_labels, guidance_labels, [int(x) for x in build_order]


def build_detection_lookup(per_second_results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Build dict keyed by integer second -> raw per-second record."""
    lookup: dict[int, dict[str, Any]] = {}
    for rec in per_second_results:
        t = int(rec["timestamp_sec"])
        lookup[t] = rec
    return lookup


def get_detected_step(record: dict[str, Any]) -> int:
    """Extract detected step as integer from an eval record."""
    step_str = record.get("smoothed_step") or record.get("predicted_step", "0")
    try:
        return int(step_str)
    except (ValueError, TypeError):
        return 0


def get_confidence(record: dict[str, Any]) -> float:
    """Extract best confidence from record."""
    val = record.get("smoothed_confidence")
    if val is None:
        val = record.get("vlm_confidence")
    if val is None:
        val = record.get("gate_similarity")
    if val is None:
        err = record.get("error_detection_result") or {}
        if isinstance(err, dict):
            val = err.get("confidence")
    return float(val) if val is not None else 0.0


def get_method(record: dict[str, Any]) -> str:
    """Extract which method produced the final result."""
    method = record.get("method_used")
    if method:
        return str(method)
    if (
        bool(record.get("error_detection_ran"))
        and not bool(record.get("vlm_called"))
        and (record.get("gate_similarity") is None)
    ):
        return "err-only"
    if record.get("vlm_called"):
        return "vlm"
    if "gate_triggered" in record:
        return "gate"
    return "vlm"


def get_non_progress_reason(record: dict[str, Any], expected_next_step: Optional[int] = None) -> str:
    """Extract one-line non-progress reason with backward-compat behavior."""
    if not record:
        return ""

    trigger = str(record.get("non_progress_trigger", "none")).lower()
    reason = str(record.get("non_progress_reason") or "").strip()
    visible = record.get("non_progress_visible")
    used_legacy_reasoning = False

    if visible is None:
        visible = trigger in {"blocked", "stale"} and bool(reason)

    if not reason and record.get("gate_triggered") and (record.get("step_complete") is False):
        reason = str(record.get("vlm_reasoning") or "").strip()
        if reason:
            visible = True
            used_legacy_reasoning = True

    if not visible:
        return ""

    if used_legacy_reasoning and expected_next_step is not None:
        referenced_steps = [int(m) for m in re.findall(r"\bstep\s+(\d+)\b", reason.lower())]
        if referenced_steps and any(step != expected_next_step for step in referenced_steps):
            return (
                f"Step {expected_next_step} not verified yet; "
                "required parts are missing or not fully attached."
            )

    reason = " ".join(reason.split())
    sentences = [s.strip() for s in reason.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    if sentences:
        keywords = (
            "missing",
            "not yet",
            "not complete",
            "only",
            "still being held",
            "not attached",
            "unable",
            "cannot",
            "blocked",
            "second eye",
        )
        chosen = None
        for sentence in sentences:
            low = sentence.lower()
            if any(k in low for k in keywords):
                chosen = sentence
                break
        reason = chosen or sentences[0]

    if len(reason) > 120:
        reason = reason[:117].rstrip() + "..."
    return reason


def get_error_overlay_lines(record: dict[str, Any]) -> list[str]:
    """Build compact error-detection summary lines."""
    if not record or not record.get("error_detection_ran"):
        return []
    er = record.get("error_detection_result") or {}
    if not isinstance(er, dict):
        return []

    err_type = str(er.get("error_type", "none")).strip() or "none"
    conf = er.get("confidence")
    step_id = str(er.get("step_id", record.get("predicted_step", ""))).strip()

    try:
        conf_text = f"{float(conf):.2f}" if conf is not None else "--"
    except (TypeError, ValueError):
        conf_text = "--"

    lines = [f"ErrorDetect: step={step_id} type={err_type} conf={conf_text}"]

    evidence = er.get("evidence") or {}
    if isinstance(evidence, dict):
        missing_parts = evidence.get("missing_parts", [])
        if isinstance(missing_parts, list):
            missing_clean = [str(x).strip() for x in missing_parts if str(x).strip()]
        else:
            missing_clean = []
        if missing_clean:
            lines.append("Missing: " + "; ".join(missing_clean))

        prev_step = evidence.get("image3_previous_correct_step")
        prev_frame = evidence.get("image3_previous_correct_frame_number")
        if prev_step is not None:
            if prev_frame is not None:
                lines.append(f"Image3: prev-correct step {prev_step} @ frame {prev_frame}")
            else:
                lines.append(f"Image3: prev-correct step {prev_step}")
    return lines


def load_ground_truth(path: Optional[Path]) -> tuple[dict[str, str], set[int]]:
    """Load ground-truth mapping if available."""
    if path is None or not path.exists():
        return {}, set()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    simplified = data.get("simplified_ground_truth", {})
    non_matchable = set(data.get("non_matchable_frames", []))
    return simplified, {int(x) for x in non_matchable}


def resolve_record_for_second(
    second: int,
    detection_lookup: dict[int, dict[str, Any]],
    sorted_seconds: Optional[list[int]] = None,
) -> Optional[dict[str, Any]]:
    """Return record for second, carrying forward the latest known <= second."""
    if not detection_lookup:
        return None
    keys = sorted_seconds or sorted(detection_lookup)
    idx = bisect_right(keys, int(second)) - 1
    if idx < 0:
        return None
    return detection_lookup[keys[idx]]


def _coerce_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compute_thumbnail_path(item: ItemDefinition, next_step: int) -> Optional[Path]:
    # Prefer manual pages if a step-scoped image exists.
    if item.manual_pages_dir and item.manual_pages_dir.exists():
        patterns = [
            f"*step*{next_step:02d}*",
            f"*step*{next_step}*",
            f"*{next_step:02d}*",
            f"*{next_step}*",
        ]
        for pattern in patterns:
            for candidate in sorted(item.manual_pages_dir.glob(pattern)):
                if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} and candidate.is_file():
                    return candidate.resolve()

    if item.anchors_dir.exists():
        for candidate in sorted(item.anchors_dir.glob(f"step_{next_step:02d}_anchor.*")):
            if candidate.is_file():
                return candidate.resolve()
        for candidate in sorted(item.anchors_dir.glob(f"step_{next_step}_anchor.*")):
            if candidate.is_file():
                return candidate.resolve()
    return None


def build_synced_details(
    *,
    second: int,
    record: Optional[dict[str, Any]],
    item: ItemDefinition,
    completed_labels: dict[int, str],
    guidance_labels: dict[int, str],
    build_order: list[int],
    gt_simplified: dict[str, str],
    non_matchable_frames: set[int],
) -> dict[str, Any]:
    """Build one UI-facing details record for a given second."""
    total_steps = max(build_order) if build_order else 0

    if record is None:
        detected_step = 0
        confidence = 0.0
        method = "ensemble"
        non_progress_reason = ""
        error_lines: list[str] = []
        source_record: dict[str, Any] = {}
    else:
        detected_step = get_detected_step(record)
        confidence = get_confidence(record)
        method = get_method(record)
        non_progress_reason = get_non_progress_reason(record, expected_next_step=detected_step + 1)
        error_lines = get_error_overlay_lines(record)
        source_record = record

    next_step = detected_step + 1 if detected_step < total_steps else total_steps
    next_step = max(0, next_step)

    gt_step_raw = gt_simplified.get(str(second))
    gt_step_int = _coerce_int(gt_step_raw)
    correct_value = source_record.get("correct")
    within_one_value = source_record.get("within_one")

    if correct_value is None and gt_step_int is not None:
        correct_value = detected_step == gt_step_int
    if within_one_value is None and gt_step_int is not None:
        within_one_value = abs(detected_step - gt_step_int) <= 1

    is_matchable = bool(gt_step_raw is not None and second not in non_matchable_frames)

    progress_ratio = 0.0
    if total_steps > 0:
        progress_ratio = max(0.0, min(1.0, detected_step / float(total_steps)))

    thumbnail_path = _compute_thumbnail_path(item, next_step if next_step > 0 else 1)

    return {
        "timestamp_sec": int(second),
        "detected_step": detected_step,
        "next_step": next_step,
        "confidence": confidence,
        "method": method,
        "completed_label": completed_labels.get(detected_step),
        "guidance_label": guidance_labels.get(detected_step),
        "progress": {
            "current_step": detected_step,
            "total_steps": total_steps,
            "ratio": progress_ratio,
            "build_order": build_order,
        },
        "ground_truth": {
            "step": gt_step_raw,
            "correct": correct_value,
            "within_one": within_one_value,
            "is_matchable": is_matchable,
        },
        "non_progress_reason": non_progress_reason,
        "error_summary_lines": error_lines,
        "thumbnail_path": str(thumbnail_path) if thumbnail_path else None,
        "trace": {
            "gate_triggered": source_record.get("gate_triggered"),
            "gate_similarity": source_record.get("gate_similarity"),
            "vlm_called": source_record.get("vlm_called"),
            "vlm_confidence": source_record.get("vlm_confidence"),
            "vlm_reasoning": source_record.get("vlm_reasoning"),
            "non_progress_reason": source_record.get("non_progress_reason"),
            "non_progress_reason_raw": source_record.get("non_progress_reason_raw"),
            "non_progress_reason_source": source_record.get("non_progress_reason_source"),
            "non_progress_trigger": source_record.get("non_progress_trigger"),
            "non_progress_visible": source_record.get("non_progress_visible"),
            "error_detection_ran": source_record.get("error_detection_ran"),
            "error_detection_source": source_record.get("error_detection_source"),
            "error_detection_result": source_record.get("error_detection_result"),
            "error_detected": source_record.get("error_detected"),
            "completed_action_detected": source_record.get("completed_action_detected"),
            "processing_time_ms": source_record.get("processing_time_ms"),
        },
    }

