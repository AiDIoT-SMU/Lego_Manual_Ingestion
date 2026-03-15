"""
Orchestrates item-driven consensus analysis for dashboard workflows.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

from config.settings import get_settings
from backend.services.consensus_adapter import (
    build_detection_lookup,
    build_synced_details,
    load_ground_truth,
    load_step_labels,
    resolve_record_for_second,
)
from backend.services.item_registry import ItemDefinition, ItemRegistry


class AssemblyAnalysisService:
    """Service to list items, run analysis, and expose synced timeline records."""

    def __init__(self, project_root: Path | None = None):
        self.settings = get_settings()
        self.project_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
        self.registry = ItemRegistry(project_root=self.project_root, data_root=self.settings.data_dir)
        self.runs_root = (self.settings.data_dir / "analysis_runs").resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def list_items(self) -> list[dict]:
        """Return UI-friendly item metadata for dropdown population."""
        items = self.registry.list_items()
        return [self._serialize_item(item) for item in items]

    def run_analysis(
        self,
        *,
        item_id: str,
        uploaded_video_path: Path,
        original_filename: str,
        details_json_file: tuple[str, bytes] | None = None,
    ) -> dict:
        """Run analysis for an item + uploaded video and persist run artifacts."""
        try:
            item = self.registry.get_item(item_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        run_id = self._make_run_id(item.id)
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        video_ext = Path(original_filename).suffix or ".mp4"
        stored_video_path = run_dir / f"uploaded_video{video_ext}"
        shutil.copy2(uploaded_video_path, stored_video_path)

        result_path = run_dir / "result.json"
        details_payload, details_warnings, details_uploaded = self._extract_uploaded_details_payload(
            details_json_file=details_json_file,
            run_dir=run_dir,
        )

        if details_payload is not None:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(details_payload, f, indent=2)
            run_result = {
                "mode": "uploaded-details",
                "warnings": details_warnings,
                "result_path": str(result_path),
                "result": details_payload,
            }
        else:
            detail = (
                "Consensus details JSON must be provided as `details_json_file` "
                "and contain valid per_second_results."
            )
            if details_warnings:
                detail = f"{detail} {' '.join(details_warnings)}"
            raise HTTPException(status_code=400, detail=detail)

        result_payload = run_result["result"]
        per_second_results = result_payload.get("per_second_results", [])
        detection_lookup = build_detection_lookup(per_second_results)
        sorted_seconds = sorted(detection_lookup)

        completed_labels, guidance_labels, build_order = load_step_labels(item.dependencies_path)
        gt_simplified, non_matchable_frames = load_ground_truth(item.ground_truth_path)

        timeline = self._build_timeline(
            sorted_seconds=sorted_seconds,
            detection_lookup=detection_lookup,
            item=item,
            completed_labels=completed_labels,
            guidance_labels=guidance_labels,
            build_order=build_order,
            gt_simplified=gt_simplified,
            non_matchable_frames=non_matchable_frames,
        )

        manifest = {
            "analysis_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "item": self._serialize_item(item),
            "video_path": str(stored_video_path),
            "result_path": str(result_path),
            "mode": run_result["mode"],
            "warnings": run_result["warnings"],
            "uploaded_details_files": details_uploaded,
        }
        with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return {
            "analysis_id": run_id,
            "item": self._serialize_item(item),
            "mode": run_result["mode"],
            "warnings": run_result["warnings"],
            "per_second_results": per_second_results,
            "timeline": timeline,
            "video_path": str(stored_video_path),
            "metadata": result_payload.get("metadata", {}),
            "uploaded_details_files": details_uploaded,
        }

    def get_analysis(self, analysis_id: str) -> dict:
        """Load stored analysis run by id."""
        run_dir = self.runs_root / analysis_id
        manifest_path = run_dir / "manifest.json"
        result_path = run_dir / "result.json"

        if not manifest_path.exists() or not result_path.exists():
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(result_path, "r", encoding="utf-8") as f:
            result_payload = json.load(f)

        item = self._deserialize_item(manifest["item"])
        per_second_results = result_payload.get("per_second_results", [])
        detection_lookup = build_detection_lookup(per_second_results)
        sorted_seconds = sorted(detection_lookup)
        completed_labels, guidance_labels, build_order = load_step_labels(item.dependencies_path)
        gt_simplified, non_matchable_frames = load_ground_truth(item.ground_truth_path)

        timeline = self._build_timeline(
            sorted_seconds=sorted_seconds,
            detection_lookup=detection_lookup,
            item=item,
            completed_labels=completed_labels,
            guidance_labels=guidance_labels,
            build_order=build_order,
            gt_simplified=gt_simplified,
            non_matchable_frames=non_matchable_frames,
        )

        return {
            "analysis_id": analysis_id,
            "item": manifest["item"],
            "mode": manifest.get("mode"),
            "warnings": manifest.get("warnings", []),
            "per_second_results": per_second_results,
            "timeline": timeline,
            "video_path": manifest.get("video_path"),
            "metadata": result_payload.get("metadata", {}),
            "uploaded_details_files": manifest.get("uploaded_details_files", []),
        }

    def get_synced_second(self, analysis_id: str, second: int) -> dict:
        """Return carry-forward synced details for a specific second."""
        run_dir = self.runs_root / analysis_id
        manifest_path = run_dir / "manifest.json"
        result_path = run_dir / "result.json"

        if not manifest_path.exists() or not result_path.exists():
            raise HTTPException(status_code=404, detail=f"Analysis '{analysis_id}' not found.")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(result_path, "r", encoding="utf-8") as f:
            result_payload = json.load(f)

        item = self._deserialize_item(manifest["item"])
        per_second_results = result_payload.get("per_second_results", [])
        detection_lookup = build_detection_lookup(per_second_results)
        sorted_seconds = sorted(detection_lookup)

        completed_labels, guidance_labels, build_order = load_step_labels(item.dependencies_path)
        gt_simplified, non_matchable_frames = load_ground_truth(item.ground_truth_path)

        record = resolve_record_for_second(second=second, detection_lookup=detection_lookup, sorted_seconds=sorted_seconds)
        details = build_synced_details(
            second=second,
            record=record,
            item=item,
            completed_labels=completed_labels,
            guidance_labels=guidance_labels,
            build_order=build_order,
            gt_simplified=gt_simplified,
            non_matchable_frames=non_matchable_frames,
        )
        return {
            "analysis_id": analysis_id,
            "second": second,
            "details": details,
        }

    def resolve_asset_path(self, raw_path: str) -> Path:
        """Resolve and validate asset path to prevent path traversal."""
        path = Path(raw_path)
        if not path.is_absolute():
            path = (self.project_root / path).resolve()
        else:
            path = path.resolve()

        allowed_roots = [
            self.settings.data_dir.resolve(),
            self.project_root.resolve(),
        ]
        for root in allowed_roots:
            if root.exists() and path.is_relative_to(root):
                return path
        raise HTTPException(status_code=400, detail="Asset path is outside allowed roots.")

    def _build_timeline(
        self,
        *,
        sorted_seconds: list[int],
        detection_lookup: dict[int, dict],
        item: ItemDefinition,
        completed_labels: dict[int, str],
        guidance_labels: dict[int, str],
        build_order: list[int],
        gt_simplified: dict[str, str],
        non_matchable_frames: set[int],
    ) -> list[dict]:
        if sorted_seconds:
            max_second = max(sorted_seconds)
        elif gt_simplified:
            max_second = max(int(k) for k in gt_simplified.keys())
        else:
            max_second = 0

        timeline: list[dict] = []
        for second in range(max_second + 1):
            record = resolve_record_for_second(
                second=second,
                detection_lookup=detection_lookup,
                sorted_seconds=sorted_seconds,
            )
            timeline.append(
                build_synced_details(
                    second=second,
                    record=record,
                    item=item,
                    completed_labels=completed_labels,
                    guidance_labels=guidance_labels,
                    build_order=build_order,
                    gt_simplified=gt_simplified,
                    non_matchable_frames=non_matchable_frames,
                )
            )
        return timeline

    @staticmethod
    def _make_run_id(item_id: str) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{item_id}_{stamp}_{uuid4().hex[:8]}"

    @staticmethod
    def _serialize_item(item: ItemDefinition) -> dict:
        return {
            "id": item.id,
            "label": item.label,
            "dependencies_path": str(item.dependencies_path),
            "anchors_dir": str(item.anchors_dir),
            "ground_truth_path": str(item.ground_truth_path) if item.ground_truth_path else None,
            "manual_pages_dir": str(item.manual_pages_dir) if item.manual_pages_dir else None,
            "precomputed_result_path": str(item.precomputed_result_path) if item.precomputed_result_path else None,
            "config_path": str(item.config_path) if item.config_path else None,
            "warnings": item.warnings,
        }

    @staticmethod
    def _deserialize_item(data: dict) -> ItemDefinition:
        return ItemDefinition(
            id=str(data["id"]),
            label=str(data["label"]),
            item_dir=Path(data.get("item_dir") or "."),
            dependencies_path=Path(data["dependencies_path"]),
            anchors_dir=Path(data["anchors_dir"]),
            ground_truth_path=Path(data["ground_truth_path"]) if data.get("ground_truth_path") else None,
            manual_pages_dir=Path(data["manual_pages_dir"]) if data.get("manual_pages_dir") else None,
            precomputed_result_path=Path(data["precomputed_result_path"]) if data.get("precomputed_result_path") else None,
            config_path=Path(data["config_path"]) if data.get("config_path") else None,
            warnings=list(data.get("warnings", [])),
        )

    @staticmethod
    def _extract_uploaded_details_payload(
        *,
        details_json_file: tuple[str, bytes] | None,
        run_dir: Path,
    ) -> tuple[dict[str, Any] | None, list[str], list[dict[str, str]]]:
        if details_json_file is None:
            return None, [], []

        details_dir = run_dir / "details_inputs"
        details_dir.mkdir(parents=True, exist_ok=True)

        warnings: list[str] = []
        filename, content = details_json_file
        safe_name = Path(filename).name or "uploaded_details.json"
        dest_path = details_dir / safe_name
        dest_path.write_bytes(content)
        uploaded_files_meta = [{"filename": safe_name, "path": str(dest_path)}]

        try:
            payload = json.loads(content.decode("utf-8"))
        except Exception as exc:
            warnings.append(f"Ignoring details JSON '{safe_name}': invalid JSON ({exc}).")
            return None, warnings, uploaded_files_meta

        normalized = AssemblyAnalysisService._normalize_uploaded_details_payload(payload, source_name=safe_name)
        if normalized is None:
            warnings.append(
                f"Ignoring details JSON '{safe_name}': expected JSON with `per_second_results` or a list of records."
            )
            return None, warnings, uploaded_files_meta

        return normalized, warnings, uploaded_files_meta

    @staticmethod
    def _normalize_uploaded_details_payload(payload: Any, source_name: str) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            per_second = payload.get("per_second_results")
            if AssemblyAnalysisService._is_valid_per_second_results(per_second):
                return payload
            # Accept alternate key if user uploads a slimmed details object.
            if AssemblyAnalysisService._is_valid_per_second_results(payload.get("results")):
                return {
                    "experiment_id": "uploaded_details",
                    "experiment_name": f"Uploaded Details ({source_name})",
                    "architecture": "uploaded",
                    "metadata": {"source_file": source_name},
                    "per_second_results": payload["results"],
                }
            return None

        if AssemblyAnalysisService._is_valid_per_second_results(payload):
            return {
                "experiment_id": "uploaded_details",
                "experiment_name": f"Uploaded Details ({source_name})",
                "architecture": "uploaded",
                "metadata": {"source_file": source_name},
                "per_second_results": payload,
            }
        return None

    @staticmethod
    def _is_valid_per_second_results(value: Any) -> bool:
        if not isinstance(value, list):
            return False
        for rec in value:
            if not isinstance(rec, dict):
                return False
            if "timestamp_sec" not in rec:
                return False
            try:
                int(rec["timestamp_sec"])
            except (TypeError, ValueError):
                return False
        return True
