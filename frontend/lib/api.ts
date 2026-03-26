const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface ManualMeta {
  id: string;
  step_count: number;
  created_at: number; // unix timestamp
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface PartInfo {
  description: string;
  bounding_box: BoundingBox | null;
  cropped_image_path: string | null;
}

export interface SubassemblyInfo {
  description: string;
  bounding_box: BoundingBox | null;
  cropped_image_path: string | null;
}

export interface Step {
  step_number: number;
  parts_required: PartInfo[];
  subassemblies: SubassemblyInfo[];
  actions: string[];
  source_page_path: string;
  notes: string | null;
}

export interface ManualSteps {
  manual_id: string;
  steps: Step[];
}

export interface PartCatalogEntry {
  description: string;          // Clean description without "(1x)" suffix
  images: string[];             // May be empty if detection failed
  used_in_steps: number[];
  total_quantity: number;       // Total count across all steps
}

export interface PartsCatalog {
  manual_id: string;
  total_unique_parts: number;   // Count of unique parts
  parts: PartCatalogEntry[];
}

export interface BrickGeometryReference {
  mesh_file: string;
  point_cloud_file: string;
  library_key: string;
}

export interface RotationAngles {
  roll_deg: number;
  pitch_deg: number;
  yaw_deg: number;
}

export interface Brick {
  brick_id: number;
  part_number: string;
  color_id: number;
  position: [number, number, number];
  rotation_matrix: [[number, number, number], [number, number, number], [number, number, number]];
  pose_4x4: [[number, number, number, number], [number, number, number, number], [number, number, number, number], [number, number, number, number]];
  rotation_angles_deg: RotationAngles;
  geometry_reference: BrickGeometryReference;
}

export interface DigitalTwinStep {
  step_number: number;
  step_name: string;
  num_bricks: number;
  bricks: Brick[];
}

export interface DigitalTwin {
  manual_id: string;
  steps: DigitalTwinStep[];
}

export interface AnalysisItem {
  id: string;
  label: string;
  dependencies_path: string;
  anchors_dir: string;
  ground_truth_path: string | null;
  manual_pages_dir: string | null;
  precomputed_result_path: string | null;
  config_path: string | null;
  warnings: string[];
}

export interface AnalysisGroundTruthDetails {
  step: string | null;
  correct: boolean | null;
  within_one: boolean | null;
  is_matchable: boolean;
}

export interface AnalysisProgressDetails {
  current_step: number;
  total_steps: number;
  ratio: number;
  build_order: number[];
}

export interface AnalysisTraceDetails {
  gate_triggered: boolean | null;
  gate_similarity: number | null;
  vlm_called: boolean | null;
  vlm_confidence: number | null;
  vlm_reasoning: string | null;
  non_progress_reason: string | null;
  non_progress_reason_raw: string | null;
  non_progress_reason_source: string | null;
  non_progress_trigger: string | null;
  non_progress_visible: boolean | null;
  error_detection_ran: boolean | null;
  error_detection_source: string | null;
  error_detection_result: Record<string, unknown> | null;
  error_detected: boolean | null;
  completed_action_detected: boolean | null;
  processing_time_ms: number | null;
}

export interface AnalysisTimelineRecord {
  timestamp_sec: number;
  detected_step: number;
  next_step: number;
  confidence: number;
  method: string;
  completed_label: string | null;
  guidance_label: string | null;
  progress: AnalysisProgressDetails;
  ground_truth: AnalysisGroundTruthDetails;
  non_progress_reason: string;
  error_summary_lines: string[];
  thumbnail_path: string | null;
  trace: AnalysisTraceDetails;
}

export interface AnalysisResult {
  analysis_id: string;
  item: AnalysisItem;
  mode: string;
  warnings: string[];
  per_second_results: Record<string, unknown>[];
  timeline: AnalysisTimelineRecord[];
  video_path: string;
  metadata: Record<string, unknown>;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Convert a stored cropped_image_path to the backend-served URL.
 *
 * The JSON stores paths like:
 *   "cropped/6262059/parts/step_1_part_0.png"
 * The backend mounts data/cropped/ at /images, so the correct URL is:
 *   "/images/6262059/parts/step_1_part_0.png"
 */
export function croppedPathToUrl(filePath: string): string {
  // Strip any leading path segments up to and including "cropped/"
  const after = filePath.replace(/^(.*\/)?cropped\//, "");
  return "/images/" + after;
}

// ─── Fetch functions ──────────────────────────────────────────────────────────

export async function fetchManuals(): Promise<ManualMeta[]> {
  const res = await fetch(`${API_BASE}/api/manuals`);
  if (!res.ok) throw new Error(`Failed to fetch manuals: ${res.status}`);
  return res.json();
}

export async function fetchSteps(manualId: string): Promise<ManualSteps> {
  const res = await fetch(`${API_BASE}/api/manuals/${manualId}/steps`);
  if (!res.ok) throw new Error(`Failed to fetch steps: ${res.status}`);
  return res.json();
}

export async function fetchParts(manualId: string): Promise<PartsCatalog> {
  const res = await fetch(`${API_BASE}/api/manuals/${manualId}/parts`);
  if (!res.ok) throw new Error(`Failed to fetch parts: ${res.status}`);
  return res.json();
}

export async function fetchDigitalTwin(manualId: string): Promise<DigitalTwin> {
  const res = await fetch(`${API_BASE}/api/manuals/${manualId}/digital-twin`);
  if (!res.ok) throw new Error(`Failed to fetch digital twin: ${res.status}`);
  return res.json();
}

export async function ingestUrl(
  manualId: string,
  url: string,
  instructionPages?: number[]
): Promise<{ status: string; manual_id: string; message: string }> {
  const form = new FormData();
  form.append("manual_id", manualId);
  form.append("url", url);
  if (instructionPages?.length) {
    form.append("instruction_pages", JSON.stringify(instructionPages));
  }

  const res = await fetch(`${API_BASE}/api/ingest/url`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error((data as { detail?: string }).detail ?? `Server error ${res.status}`);
  }

  return res.json();
}

export async function ingestImages(
  manualId: string,
  instructionImages: File[],
  options?: {
    assembledImage?: File;
    partsImages?: File[];
  }
): Promise<{ status: string; manual_id: string; message: string }> {
  const form = new FormData();
  form.append("manual_id", manualId);

  instructionImages.forEach((f) => form.append("images", f));
  form.append(
    "image_numbers",
    JSON.stringify(instructionImages.map((_, i) => i + 1))
  );

  if (options?.assembledImage) {
    form.append("assembled_image", options.assembledImage);
  }
  options?.partsImages?.forEach((f) => form.append("parts_images", f));

  const res = await fetch(`${API_BASE}/api/ingest/images`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error((data as { detail?: string }).detail ?? `Server error ${res.status}`);
  }

  return res.json();
}

// ─── Assembly Analysis ────────────────────────────────────────────────────────

export async function fetchAnalysisItems(): Promise<{ items: AnalysisItem[] }> {
  const res = await fetch(`${API_BASE}/api/assembly/items`);
  if (!res.ok) throw new Error(`Failed to fetch analysis items: ${res.status}`);
  return res.json();
}

export async function analyzeAssemblyVideo(
  itemId: string,
  videoFile: File,
  detailsJsonFile: File
): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("item_id", itemId);
  form.append("video_file", videoFile);
  form.append("details_json_file", detailsJsonFile);

  const res = await fetch(`${API_BASE}/api/assembly/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error((data as { detail?: string }).detail ?? `Server error ${res.status}`);
  }

  return res.json();
}

export function analysisAssetUrl(path: string): string {
  return `${API_BASE}/api/assembly/asset?path=${encodeURIComponent(path)}`;
}

// ─── Video Analysis ────────────────────────────────────────────────────────────

export interface StepTimelineEntry {
  step_number: number;
  start_time: number;
  end_time: number;
  duration_seconds: number;
  confidence_avg: number;
  frame_numbers: number[];
}

export interface PartUsage {
  first_seen_timestamp: number;
  last_seen_timestamp: number;
  marked_as_used: boolean;
  usage_confidence: number;
  frames_visible: number[];
}

export interface VideoAnalysis {
  video_id: string;
  manual_id: string;
  video_filename: string;
  total_duration_seconds: number;
  total_frames_extracted: number;
  processed_at: string;
  step_timeline: StepTimelineEntry[];
  parts_used: Record<string, PartUsage>;
  frame_analyses: Array<{
    frame_number: number;
    timestamp_seconds: number;
    detected_step: number | null;
    step_confidence: number;
    detected_parts: string[];
    parts_confidence: number;
  }>;
}

export async function uploadVideo(
  manualId: string,
  videoFile: File
): Promise<{ video_id: string; status: string; message: string }> {
  const form = new FormData();
  form.append("manual_id", manualId);
  form.append("video_file", videoFile);

  const res = await fetch(`${API_BASE}/api/video/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error((data as { detail?: string }).detail ?? `Upload failed: ${res.status}`);
  }

  return res.json();
}

export async function fetchVideoAnalysis(
  manualId: string,
  videoId: string
): Promise<VideoAnalysis> {
  const res = await fetch(`${API_BASE}/api/video/analysis/${manualId}/${videoId}`);
  if (!res.ok) throw new Error(`Failed to fetch video analysis: ${res.status}`);
  return res.json();
}

export async function listVideos(
  manualId: string
): Promise<Array<{ video_id: string; filename: string; duration_seconds: number; processed_at: string }>> {
  const res = await fetch(`${API_BASE}/api/video/list/${manualId}`);
  if (!res.ok) throw new Error(`Failed to list videos: ${res.status}`);
  return res.json();
}

// ─── Video-Enhanced Assembly Instructions ─────────────────────────────────────

export interface SpatialDescription {
  target_part: string;
  placement_part: string;
  location: string;
  position_detail: string;
  orientation: string | null;
  relative_to: string | null;
}

export interface SubStep {
  sub_step_number: string;
  action_type: 'pick' | 'place' | 'attach' | 'rotate' | 'verify';
  description: string;
  parts_involved: string[];
  spatial_description: SpatialDescription | null;
  frame_range: {
    start_frame: number;
    end_frame: number;
    start_time: number;
    end_time: number;
  };
  confidence: number;
}

export interface Correction {
  field: string;
  original_value: any;
  corrected_value: any;
  reason: string;
  confidence: number;
}

export interface VideoEnhancedStep extends Step {
  original_manual_step: number;
  sub_steps: SubStep[];
  corrections: Correction[];
}

export interface VideoEnhancedManual {
  manual_id: string;
  source_video_id: string;
  created_at: string;
  video_metadata: {
    duration_seconds: number;
    frame_count: number;
    filename: string;
  };
  steps: VideoEnhancedStep[];
  manual_step_mapping: Record<string, number[]>;
}

export async function uploadAndEnhanceVideo(
  manualId: string,
  videoFile: File
): Promise<{ video_id: string; status: string; message: string }> {
  const form = new FormData();
  form.append("manual_id", manualId);
  form.append("video_file", videoFile);

  const res = await fetch(`${API_BASE}/api/video/upload-and-enhance`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error((data as { detail?: string }).detail ?? `Upload and enhance failed: ${res.status}`);
  }

  return res.json();
}

export async function enhanceManualWithVideo(
  manualId: string,
  videoId: string
): Promise<{ status: string; message: string }> {
  const res = await fetch(`${API_BASE}/api/video/enhance/${manualId}/${videoId}`, {
    method: 'POST',
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error((data as { detail?: string }).detail ?? `Enhancement failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchVideoEnhancedSteps(
  manualId: string
): Promise<VideoEnhancedManual> {
  const res = await fetch(`${API_BASE}/api/video/manuals/${manualId}/video-enhanced`);
  if (!res.ok) throw new Error(`Failed to fetch video-enhanced steps: ${res.status}`);
  return res.json();
}

export async function listVideoEnhancements(
  manualId: string
): Promise<Array<{
  video_id: string;
  created_at: string;
  sub_steps_count: number;
  corrections_count: number;
}>> {
  const res = await fetch(`${API_BASE}/api/video/manuals/${manualId}/video-enhanced/list`);
  if (!res.ok) throw new Error(`Failed to list video enhancements: ${res.status}`);
  return res.json();
}
