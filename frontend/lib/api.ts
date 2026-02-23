const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

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
  description: string;
  images: string[];
  used_in_steps: number[];
}

export interface PartsCatalog {
  manual_id: string;
  parts: PartCatalogEntry[];
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
