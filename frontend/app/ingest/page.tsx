"use client";

import { useState, useRef, useCallback } from "react";

// ─── Types ───────────────────────────────────────────────────────────────────

type InputMode = "url" | "pdf" | "upload" | "video";
type ImageRole = "instruction" | "assembled" | "parts";
type Status = "idle" | "submitting" | "success" | "error";

interface StagedFile {
  file: File;
  preview: string;
  role: ImageRole;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function parsePageInput(raw: string): number[] {
  const pages: number[] = [];
  for (const part of raw.split(",")) {
    const range = part.trim();
    const match = range.match(/^(\d+)-(\d+)$/);
    if (match) {
      const from = parseInt(match[1], 10);
      const to = parseInt(match[2], 10);
      for (let i = from; i <= to; i++) pages.push(i);
    } else if (/^\d+$/.test(range)) {
      pages.push(parseInt(range, 10));
    }
  }
  return [...new Set(pages)].sort((a, b) => a - b);
}

const ROLE_LABELS: Record<ImageRole, string> = {
  instruction: "Instruction",
  assembled: "Final Assembly",
  parts: "Parts Catalog",
};

const ROLE_COLORS: Record<ImageRole, string> = {
  instruction: "bg-blue-500",
  assembled: "bg-green-500",
  parts: "bg-orange-500",
};

// ─── Sub-components ───────────────────────────────────────────────────────────

function RoleBadge({
  role,
  onChange,
  assembledTaken,
}: {
  role: ImageRole;
  onChange: (r: ImageRole) => void;
  assembledTaken: boolean;
}) {
  return (
    <select
      value={role}
      onChange={(e) => onChange(e.target.value as ImageRole)}
      className="w-full text-xs font-medium px-2 py-1 rounded bg-gray-800 border border-gray-700 text-gray-100 focus:outline-none focus:ring-1 focus:ring-yellow-400 cursor-pointer"
    >
      <option value="instruction">Instruction</option>
      <option value="assembled" disabled={assembledTaken && role !== "assembled"}>
        Final Assembly{assembledTaken && role !== "assembled" ? " (taken)" : ""}
      </option>
      <option value="parts">Parts Catalog</option>
    </select>
  );
}

function FileDropzone({ onFiles }: { onFiles: (files: File[]) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const files = Array.from(e.dataTransfer.files).filter((f) =>
        f.type.startsWith("image/")
      );
      if (files.length) onFiles(files);
    },
    [onFiles]
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
        dragging
          ? "border-yellow-400 bg-yellow-400/5"
          : "border-gray-700 hover:border-gray-500"
      }`}
    >
      <p className="text-gray-400 text-sm">
        Drag & drop images here, or{" "}
        <span className="text-yellow-400 underline">click to select</span>
      </p>
      <p className="text-gray-600 text-xs mt-1">PNG, JPG, JPEG, WEBP</p>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          if (files.length) onFiles(files);
          e.target.value = "";
        }}
      />
    </div>
  );
}

function PdfDropzone({
  file,
  onFile,
}: {
  file: File | null;
  onFile: (f: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = Array.from(e.dataTransfer.files).find(
        (x) => x.type === "application/pdf"
      );
      if (f) onFile(f);
    },
    [onFile]
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
        dragging
          ? "border-yellow-400 bg-yellow-400/5"
          : file
          ? "border-green-500 bg-green-500/5"
          : "border-gray-700 hover:border-gray-500"
      }`}
    >
      {file ? (
        <div>
          <p className="text-green-400 text-sm font-medium">{file.name}</p>
          <p className="text-gray-500 text-xs mt-1">
            {(file.size / 1024 / 1024).toFixed(2)} MB — click to replace
          </p>
        </div>
      ) : (
        <div>
          <p className="text-gray-400 text-sm">
            Drag & drop a PDF here, or{" "}
            <span className="text-yellow-400 underline">click to select</span>
          </p>
          <p className="text-gray-600 text-xs mt-1">PDF only</p>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}

function VideoDropzone({
  onFile,
  currentFile
}: {
  onFile: (file: File) => void;
  currentFile: File | null;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const files = Array.from(e.dataTransfer.files);
      const videoFile = files.find(f =>
        f.type.startsWith("video/") ||
        /\.(mp4|mov|avi)$/i.test(f.name)
      );
      if (videoFile) onFile(videoFile);
    },
    [onFile]
  );

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
        dragging ? "border-yellow-400 bg-yellow-400/5" :
        currentFile ? "border-green-500 bg-green-500/5" :
        "border-gray-700 hover:border-gray-500"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo,.mp4,.mov,.avi"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          e.target.value = "";
        }}
        className="hidden"
      />
      {currentFile ? (
        <div>
          <p className="text-green-400 text-sm font-medium">✓ {currentFile.name}</p>
          <p className="text-gray-500 text-xs mt-1">
            {(currentFile.size / 1024 / 1024).toFixed(2)} MB — click to replace
          </p>
        </div>
      ) : (
        <div>
          <p className="text-gray-400 text-sm">
            Drag & drop a video here, or{" "}
            <span className="text-yellow-400 underline">click to select</span>
          </p>
          <p className="text-gray-600 text-xs mt-1">Supports MP4, MOV, AVI</p>
        </div>
      )}
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function IngestPage() {
  const [mode, setMode] = useState<InputMode>("url");

  // Shared
  const [manualId, setManualId] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("");

  // URL mode
  const [url, setUrl] = useState("");
  const [urlPages, setUrlPages] = useState("");

  // PDF upload mode
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [pdfPages, setPdfPages] = useState("");

  // Upload mode
  const [staged, setStaged] = useState<StagedFile[]>([]);

  // Video mode
  const [videoFile, setVideoFile] = useState<File | null>(null);

  // ── Helpers ────────────────────────────────────────────────────────────────

  const assembledTaken = staged.some((f) => f.role === "assembled");

  function addFiles(files: File[]) {
    const next: StagedFile[] = files.map((file) => ({
      file,
      preview: URL.createObjectURL(file),
      role: "instruction",
    }));
    setStaged((prev) => [...prev, ...next]);
  }

  function removeFile(idx: number) {
    setStaged((prev) => {
      URL.revokeObjectURL(prev[idx].preview);
      return prev.filter((_, i) => i !== idx);
    });
  }

  function setRole(idx: number, role: ImageRole) {
    setStaged((prev) =>
      prev.map((f, i) => (i === idx ? { ...f, role } : f))
    );
  }

  // ── Submit ─────────────────────────────────────────────────────────────────

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!manualId.trim()) {
      setStatus("error");
      setMessage("Manual ID is required.");
      return;
    }

    setStatus("submitting");
    setMessage("");

    try {
      if (mode === "url") {
        await submitUrl();
      } else if (mode === "pdf") {
        await submitPdf();
      } else if (mode === "video") {
        await submitVideo();
        return; // submitVideo handles its own status/redirect
      } else {
        await submitUpload();
      }
      setStatus("success");
      setMessage(`Ingestion started for manual "${manualId}". Processing runs in the background.`);
    } catch (err) {
      setStatus("error");
      setMessage(err instanceof Error ? err.message : "Submission failed.");
    }
  }

  async function submitUrl() {
    if (!url.trim()) throw new Error("URL is required.");

    const pages = parsePageInput(urlPages);
    const form = new FormData();
    form.append("manual_id", manualId.trim());
    form.append("url", url.trim());
    if (pages.length) form.append("instruction_pages", JSON.stringify(pages));

    const res = await fetch("/api/ingest/url", { method: "POST", body: form });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data?.detail ?? `Server error ${res.status}`);
    }
  }

  async function submitPdf() {
    if (!pdfFile) throw new Error("A PDF file is required.");

    const pages = parsePageInput(pdfPages);
    const form = new FormData();
    form.append("manual_id", manualId.trim());
    form.append("pdf_file", pdfFile);
    if (pages.length) form.append("instruction_pages", JSON.stringify(pages));

    const res = await fetch("/api/ingest/pdf", { method: "POST", body: form });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data?.detail ?? `Server error ${res.status}`);
    }
  }

  async function submitUpload() {
    const instructions = staged.filter((f) => f.role === "instruction");
    if (!instructions.length) throw new Error("At least one Instruction image is required.");

    const assembled = staged.find((f) => f.role === "assembled");
    const parts = staged.filter((f) => f.role === "parts");

    const form = new FormData();
    form.append("manual_id", manualId.trim());

    instructions.forEach((f) => form.append("images", f.file));
    // All instruction images will be processed (no sub-selection needed)
    const nums = instructions.map((_, i) => i + 1);
    form.append("image_numbers", JSON.stringify(nums));

    if (assembled) form.append("assembled_image", assembled.file);
    parts.forEach((f) => form.append("parts_images", f.file));

    const res = await fetch("/api/ingest/images", { method: "POST", body: form });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data?.detail ?? `Server error ${res.status}`);
    }
  }

  async function submitVideo() {
    if (!videoFile) {
      setMessage("Please select a video file");
      setStatus("error");
      return;
    }

    setStatus("submitting");
    setMessage("Uploading video and starting enhancement...");

    try {
      const { uploadAndEnhanceVideo } = await import("@/lib/api");

      // Upload video and directly start enhancement (bypasses video_analyzer)
      // This processes the ENTIRE video (no 1000 frame limit)
      const result = await uploadAndEnhanceVideo(manualId, videoFile);

      setStatus("success");
      setMessage(
        `Video uploaded successfully! Enhancement is processing in the background.\n\n` +
        `Video ID: ${result.video_id}\n\n` +
        `The system is now:\n` +
        `1. Extracting frames from the entire video (every 30 frames)\n` +
        `2. Running VLM action detection on all frames\n` +
        `3. Extracting spatial placement information\n` +
        `4. Reconciling with manual and generating corrections\n\n` +
        `This may take several minutes depending on video length. ` +
        `The video_enhanced.json will be created when complete.`
      );
    } catch (err) {
      setStatus("error");
      setMessage(err instanceof Error ? err.message : "Enhancement failed");
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-1">Ingest Manual</h1>
      <p className="text-gray-400 text-sm mb-8">
        Process a LEGO instruction manual via URL, PDF upload, or image upload.
      </p>

      <form onSubmit={handleSubmit} className="space-y-6">

        {/* Manual ID */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Manual ID <span className="text-red-400">*</span>
          </label>
          <input
            type="text"
            value={manualId}
            onChange={(e) => setManualId(e.target.value)}
            placeholder="e.g. 6262059"
            className="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-yellow-400"
          />
        </div>

        {/* Mode toggle */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Input source
          </label>
          <div className="flex rounded-lg overflow-hidden border border-gray-700 w-fit">
            {(["url", "pdf", "upload", "video"] as InputMode[]).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={`px-5 py-2 text-sm font-medium transition-colors ${
                  mode === m
                    ? "bg-yellow-400 text-gray-900"
                    : "bg-gray-800 text-gray-400 hover:text-white"
                }`}
              >
                {m === "url" ? "URL" : m === "pdf" ? "Upload PDF" : m === "video" ? "Video" : "Upload Images"}
              </button>
            ))}
          </div>
        </div>

        {/* ── URL mode ── */}
        {mode === "url" && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                PDF URL <span className="text-red-400">*</span>
              </label>
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://www.lego.com/cdn/.../6262059.pdf"
                className="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-yellow-400"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Instruction pages
                <span className="text-gray-500 font-normal ml-2">(optional — blank = all pages)</span>
              </label>
              <input
                type="text"
                value={urlPages}
                onChange={(e) => setUrlPages(e.target.value)}
                placeholder="e.g. 13-20, 25, 30"
                className="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-yellow-400"
              />
              <p className="text-xs text-gray-500 mt-1">
                Ranges (13-20) and individual numbers (25, 30) are both supported.
              </p>
            </div>
          </div>
        )}

        {/* ── PDF upload mode ── */}
        {mode === "pdf" && (
          <div className="space-y-4">
            <PdfDropzone file={pdfFile} onFile={setPdfFile} />
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                Instruction pages
                <span className="text-gray-500 font-normal ml-2">(optional — blank = all pages)</span>
              </label>
              <input
                type="text"
                value={pdfPages}
                onChange={(e) => setPdfPages(e.target.value)}
                placeholder="e.g. 13-20, 25, 30"
                className="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-yellow-400"
              />
              <p className="text-xs text-gray-500 mt-1">
                Ranges (13-20) and individual numbers (25, 30) are both supported. Leave blank to process all pages.
              </p>
            </div>
          </div>
        )}

        {/* ── Upload mode ── */}
        {mode === "upload" && (
          <div className="space-y-4">
            <FileDropzone onFiles={addFiles} />

            {staged.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-3">
                  <p className="text-sm font-medium text-gray-300">
                    {staged.length} image{staged.length !== 1 ? "s" : ""} staged
                  </p>
                  {/* Legend */}
                  <div className="flex gap-3 text-xs text-gray-400">
                    {(Object.entries(ROLE_COLORS) as [ImageRole, string][]).map(
                      ([role, color]) => (
                        <span key={role} className="flex items-center gap-1">
                          <span className={`inline-block w-2 h-2 rounded-full ${color}`} />
                          {ROLE_LABELS[role]}
                        </span>
                      )
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3 sm:grid-cols-4">
                  {staged.map((f, i) => (
                    <div key={i} className="relative group">
                      {/* Thumbnail */}
                      <div
                        className={`aspect-square rounded-lg overflow-hidden border-2 transition-colors ${
                          ROLE_COLORS[f.role].replace("bg-", "border-")
                        }`}
                      >
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={f.preview}
                          alt={f.file.name}
                          className="w-full h-full object-cover"
                        />
                      </div>

                      {/* Remove button */}
                      <button
                        type="button"
                        onClick={() => removeFile(i)}
                        className="absolute top-1 right-1 w-5 h-5 rounded-full bg-gray-900/80 text-gray-300 text-xs flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600 hover:text-white"
                      >
                        ×
                      </button>

                      {/* Role badge */}
                      <div
                        className={`absolute top-1 left-1 text-xs font-bold px-1.5 py-0.5 rounded text-white ${ROLE_COLORS[f.role]}`}
                      >
                        {f.role === "instruction"
                          ? `#${staged.filter((s, si) => s.role === "instruction" && si <= i).length}`
                          : f.role === "assembled"
                          ? "FA"
                          : "PC"}
                      </div>

                      {/* Role selector */}
                      <div className="mt-1.5">
                        <RoleBadge
                          role={f.role}
                          onChange={(r) => setRole(i, r)}
                          assembledTaken={assembledTaken}
                        />
                      </div>

                      {/* Filename */}
                      <p className="text-xs text-gray-500 truncate mt-0.5 px-0.5">
                        {f.file.name}
                      </p>
                    </div>
                  ))}
                </div>

                {/* Summary */}
                <div className="mt-4 p-3 rounded-lg bg-gray-800/50 border border-gray-700 text-xs text-gray-400 flex flex-wrap gap-4">
                  <span>
                    <span className="text-blue-400 font-semibold">
                      {staged.filter((f) => f.role === "instruction").length}
                    </span>{" "}
                    instruction
                  </span>
                  <span>
                    <span className="text-green-400 font-semibold">
                      {staged.filter((f) => f.role === "assembled").length}
                    </span>{" "}
                    final assembly
                  </span>
                  <span>
                    <span className="text-orange-400 font-semibold">
                      {staged.filter((f) => f.role === "parts").length}
                    </span>{" "}
                    parts catalog
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Video mode ── */}
        {mode === "video" && (
          <div className="space-y-4">
            <p className="text-sm text-gray-400">
              Upload an assembly video to enrich this manual with spatial sub-steps, detailed
              placement instructions, and corrections derived from the video.
              The manual must already be ingested.
            </p>

            <VideoDropzone
              onFile={(file) => setVideoFile(file)}
              currentFile={videoFile}
            />

            {videoFile && (
              <div className="text-sm text-gray-300 p-3 rounded-lg bg-gray-800/50 border border-gray-700">
                <span className="text-green-400 font-medium">✓ Ready to upload:</span> {videoFile.name} ({(videoFile.size / 1024 / 1024).toFixed(2)} MB)
              </div>
            )}
          </div>
        )}

        {/* Status messages */}
        {status === "error" && (
          <div className="px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm">
            {message}
          </div>
        )}
        {status === "success" && (
          <div className="px-4 py-3 rounded-lg bg-green-500/10 border border-green-500/30 text-green-300 text-sm">
            ✓ {message}
          </div>
        )}

        {/* Submit */}
        <button
          type="submit"
          disabled={status === "submitting"}
          className="w-full py-3 rounded-lg bg-yellow-400 text-gray-900 font-semibold hover:bg-yellow-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {status === "submitting"
            ? "Submitting…"
            : mode === "video"
            ? "Upload & Enhance Manual"
            : "Start Ingestion"}
        </button>
      </form>
    </div>
  );
}
