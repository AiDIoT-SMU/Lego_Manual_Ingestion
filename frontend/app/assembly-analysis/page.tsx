"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  analyzeAssemblyVideo,
  fetchAnalysisItems,
  type AnalysisItem,
  type AnalysisResult,
  type AnalysisTimelineRecord,
} from "@/lib/api";

function fmtPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function fmtNumber(value: number | null | undefined, decimals = 2): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A";
  return value.toFixed(decimals);
}

function fmtBool(value: boolean | null | undefined): string {
  if (value === null || value === undefined) return "N/A";
  return value ? "Yes" : "No";
}

function fmtMethod(method: string): string {
  if (!method) return "N/A";
  return method
    .replace(/[_-]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (m) => m.toUpperCase());
}

function pickSyncedRecord(timeline: AnalysisTimelineRecord[], sec: number): AnalysisTimelineRecord | null {
  if (!timeline.length) return null;
  const clamped = Math.max(0, Math.floor(sec));
  if (clamped < timeline.length) return timeline[clamped];
  return timeline[timeline.length - 1];
}

function statusBadgeClass(value: boolean | null | undefined): string {
  if (value === null || value === undefined) {
    return "border-gray-700 bg-gray-800 text-gray-300";
  }
  return value
    ? "border-green-500/30 bg-green-500/10 text-green-300"
    : "border-red-500/30 bg-red-500/10 text-red-300";
}

type Moment = {
  second: number;
  label: string;
  type: "step" | "error";
};

type ParsedErrorCheck = {
  checked: boolean;
  hasError: boolean;
  errorType: string | null;
  step: string | null;
  confidence: number | null;
  missingParts: string[];
};

function humanizeErrorType(errorType: string | null): string {
  if (!errorType) return "N/A";
  return errorType
    .replace(/[_-]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (m) => m.toUpperCase());
}

function parseErrorCheck(record: AnalysisTimelineRecord): ParsedErrorCheck {
  const summaryLines = record.error_summary_lines ?? [];
  const rawResult = record.trace.error_detection_result;
  const resultObj =
    rawResult && typeof rawResult === "object" && !Array.isArray(rawResult)
      ? (rawResult as Record<string, unknown>)
      : null;

  let errorType: string | null =
    typeof resultObj?.error_type === "string" ? (resultObj.error_type as string) : null;
  let step: string | null =
    resultObj?.step_id !== undefined && resultObj?.step_id !== null ? String(resultObj.step_id) : null;
  let confidence: number | null =
    typeof resultObj?.confidence === "number" ? (resultObj.confidence as number) : null;
  let missingParts: string[] = [];

  const evidence =
    resultObj?.evidence && typeof resultObj.evidence === "object" && !Array.isArray(resultObj.evidence)
      ? (resultObj.evidence as Record<string, unknown>)
      : null;
  if (evidence?.missing_parts && Array.isArray(evidence.missing_parts)) {
    missingParts = evidence.missing_parts
      .map((item) => String(item).trim())
      .filter((item) => item.length > 0);
  }

  for (const line of summaryLines) {
    if (!errorType || !step || confidence === null) {
      const match = line.match(/ErrorDetect:\s*step=([^\s]+)\s+type=([^\s]+)\s+conf=([^\s]+)/i);
      if (match) {
        step = step ?? match[1];
        errorType = errorType ?? match[2];
        if (confidence === null) {
          const parsedConf = Number(match[3]);
          confidence = Number.isFinite(parsedConf) ? parsedConf : null;
        }
      }
    }

    if (missingParts.length === 0 && line.toLowerCase().startsWith("missing:")) {
      const payload = line.slice(line.indexOf(":") + 1).trim();
      if (payload) {
        missingParts = payload
          .split(";")
          .map((part) => part.trim())
          .filter((part) => part.length > 0);
      }
    }
  }

  const checked = Boolean(record.trace.error_detection_ran) || Boolean(errorType) || summaryLines.length > 0;
  const normalizedType = (errorType ?? "").toLowerCase().trim();
  const hasError = checked && normalizedType !== "" && normalizedType !== "none";

  return {
    checked,
    hasError,
    errorType,
    step,
    confidence,
    missingParts,
  };
}

function InfoTile({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800 p-3">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-xl font-semibold text-gray-100">{value}</p>
      {hint && <p className="mt-1 text-xs text-gray-400">{hint}</p>}
    </div>
  );
}

export default function AssemblyAnalysisPage() {
  const [items, setItems] = useState<AnalysisItem[]>([]);
  const [itemsLoading, setItemsLoading] = useState(true);
  const [itemsError, setItemsError] = useState("");
  const [selectedItemId, setSelectedItemId] = useState("");

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [detailsJsonFile, setDetailsJsonFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [runState, setRunState] = useState<"idle" | "running" | "error">("idle");
  const [runError, setRunError] = useState("");

  const [currentSecond, setCurrentSecond] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    fetchAnalysisItems()
      .then((data) => {
        setItems(data.items);
        if (data.items.length > 0) {
          setSelectedItemId(data.items[0].id);
        }
      })
      .catch((err: Error) => setItemsError(err.message))
      .finally(() => setItemsLoading(false));
  }, []);

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  const timeline = analysis?.timeline ?? [];
  const maxSecond = Math.max(0, timeline.length - 1);

  const synced = useMemo(() => {
    if (!analysis) return null;
    return pickSyncedRecord(analysis.timeline, currentSecond);
  }, [analysis, currentSecond]);

  const keyMoments = useMemo(() => {
    if (!analysis) return [] as Moment[];

    const moments: Moment[] = [];
    let prevStep: number | null = null;
    for (const record of analysis.timeline) {
      const stepChanged = prevStep === null || record.detected_step !== prevStep;
      const errorCheck = parseErrorCheck(record);
      const hasErrorSignal = errorCheck.hasError;

      if (stepChanged) {
        moments.push({
          second: record.timestamp_sec,
          label: `Step ${record.detected_step}`,
          type: "step",
        });
      } else if (hasErrorSignal) {
        moments.push({
          second: record.timestamp_sec,
          label: "Error Signal",
          type: "error",
        });
      }
      prevStep = record.detected_step;
    }
    return moments.slice(0, 12);
  }, [analysis]);

  function seekToSecond(second: number, opts?: { pause?: boolean }) {
    const clamped = Math.max(0, Math.min(maxSecond, Math.floor(second)));
    setCurrentSecond(clamped);
    if (videoRef.current) {
      videoRef.current.currentTime = clamped;
      if (opts?.pause) {
        videoRef.current.pause();
      }
    }
  }

  async function handleRunAnalysis(e: React.FormEvent) {
    e.preventDefault();
    if (!selectedItemId) {
      setRunError("Select an item first.");
      setRunState("error");
      return;
    }
    if (!videoFile) {
      setRunError("Upload a video file first.");
      setRunState("error");
      return;
    }
    if (!detailsJsonFile) {
      setRunError("Consensus details JSON must be provided.");
      setRunState("error");
      return;
    }

    setRunState("running");
    setRunError("");
    setAnalysis(null);
    setCurrentSecond(0);

    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(URL.createObjectURL(videoFile));

    try {
      const result = await analyzeAssemblyVideo(selectedItemId, videoFile, detailsJsonFile);
      setAnalysis(result);
      setRunState("idle");
      seekToSecond(0);
    } catch (err) {
      setRunState("error");
      setRunError(err instanceof Error ? err.message : "Analysis failed.");
    }
  }

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Assembly Analysis</h1>
        <p className="text-gray-400 text-sm mt-1">
          Upload an assembly video and analyse the assembly process
        </p>
      </div>

      <form
        onSubmit={handleRunAnalysis}
        className="mb-8 grid gap-4 rounded-xl border border-gray-700 bg-gray-900 p-5 md:grid-cols-[1.05fr_1fr_1fr_auto]"
      >
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Item</label>
          <select
            value={selectedItemId}
            onChange={(e) => setSelectedItemId(e.target.value)}
            className="w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-100 focus:outline-none focus:ring-2 focus:ring-yellow-400"
            disabled={itemsLoading || items.length === 0}
          >
            {items.map((item) => (
              <option key={item.id} value={item.id}>
                {item.label}
              </option>
            ))}
          </select>
          {itemsLoading && <p className="mt-1 text-xs text-gray-500">Loading items…</p>}
          {itemsError && <p className="mt-1 text-xs text-red-300">{itemsError}</p>}
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Upload Video</label>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)}
            className="block w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-100 file:mr-3 file:rounded file:border-0 file:bg-gray-700 file:px-3 file:py-1 file:text-xs file:text-gray-100"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Upload Output JSON</label>
          <input
            type="file"
            accept=".json,application/json"
            onChange={(e) => setDetailsJsonFile(e.target.files?.[0] ?? null)}
            className="block w-full px-3 py-2 rounded-lg bg-gray-800 border border-gray-700 text-gray-100 file:mr-3 file:rounded file:border-0 file:bg-gray-700 file:px-3 file:py-1 file:text-xs file:text-gray-100"
          />
        </div>

        <button
          type="submit"
          disabled={runState === "running"}
          className="h-fit self-end px-5 py-2 bg-yellow-400 text-gray-900 font-semibold rounded-lg hover:bg-yellow-300 transition-colors disabled:cursor-not-allowed disabled:opacity-60"
        >
          {runState === "running" ? "Analyzing..." : "Run Analysis"}
        </button>
      </form>

      {runState === "error" && (
        <div className="mb-6 px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm">
          {runError}
        </div>
      )}

      {analysis && (
        <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
          <section className="rounded-xl border border-gray-700 bg-gray-900 p-4">
            <div className="mb-3 flex items-center justify-between">
              <p className="text-xs uppercase tracking-wider text-gray-400">
                Assembly Video{videoFile?.name ? ` · ${videoFile.name}` : ""}
              </p>
            </div>

            <video
              ref={videoRef}
              src={videoUrl}
              controls
              className="w-full rounded-xl border border-gray-700 bg-black"
              onTimeUpdate={(e) => setCurrentSecond(Math.floor((e.currentTarget as HTMLVideoElement).currentTime))}
            />

            <div className="mt-4">
              <div className="mb-2 flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => seekToSecond(currentSecond - 5)}
                  className="px-2.5 py-1 rounded bg-gray-800 border border-gray-700 text-xs text-gray-200 hover:bg-gray-700"
                >
                  -5s
                </button>
                <button
                  type="button"
                  onClick={() => seekToSecond(currentSecond + 5)}
                  className="px-2.5 py-1 rounded bg-gray-800 border border-gray-700 text-xs text-gray-200 hover:bg-gray-700"
                >
                  +5s
                </button>
              </div>
              <input
                type="range"
                min={0}
                max={Math.max(1, maxSecond)}
                step={1}
                value={Math.min(currentSecond, maxSecond)}
                onChange={(e) => seekToSecond(Number(e.target.value))}
                className="w-full accent-yellow-400"
              />
            </div>

            <div className="mt-4">
              <p className="mb-2 text-xs uppercase tracking-wider text-gray-400">Key Moments</p>
              {keyMoments.length === 0 && <p className="text-xs text-gray-500">No key moments detected.</p>}
              {keyMoments.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {keyMoments.map((moment, idx) => (
                    <button
                      key={`${moment.second}-${idx}`}
                      type="button"
                      onClick={() => seekToSecond(moment.second, { pause: true })}
                      className={`px-2.5 py-1 rounded border text-xs transition-colors ${
                        moment.type === "error"
                          ? "border-red-500/40 bg-red-500/10 text-red-300 hover:bg-red-500/20"
                          : "border-gray-700 bg-gray-800 text-gray-200 hover:bg-gray-700"
                      }`}
                    >
                      {moment.second}s · {moment.label}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {analysis.warnings.length > 0 && (
              <div className="mt-4 px-3 py-2 rounded-lg border border-yellow-500/30 bg-yellow-500/10 text-xs text-yellow-200">
                {analysis.warnings.join(" ")}
              </div>
            )}
          </section>

          <section className="rounded-xl border border-gray-700 bg-gray-900 p-5">
            <p className="mb-3 text-xs uppercase tracking-wider text-gray-400">Live Assembly Snapshot</p>
            {!synced && <p className="text-sm text-gray-500">No synced record available.</p>}
            {synced && (
              <div className="space-y-4">
                {(() => {
                  const errorCheck = parseErrorCheck(synced);
                  const methodValue =
                    synced.method === "err-only"
                      ? "Error Detection"
                      : fmtMethod(synced.method);

                  return (
                    <>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <InfoTile label="Detected Step" value={String(synced.detected_step)} />
                  <InfoTile label="Next Step" value={String(synced.next_step)} />
                  <InfoTile label="Confidence" value={fmtNumber(synced.confidence)} hint={fmtPercent(synced.confidence)} />
                  <InfoTile label="Method" value={methodValue} />
                </div>

                <div className="rounded-lg border border-gray-700 bg-gray-800 p-3">
                  <div className="mb-1 flex items-center justify-between text-xs text-gray-500">
                    <span>Build Progress</span>
                    <span>
                      Step {synced.progress.current_step}/{synced.progress.total_steps}
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-gray-900">
                    <div
                      className="h-2 rounded-full bg-yellow-400 transition-all"
                      style={{ width: `${Math.max(3, synced.progress.ratio * 100)}%` }}
                    />
                  </div>
                  <p className="mt-2 text-sm text-gray-300">
                    {synced.guidance_label || "No guidance available."}
                  </p>
                </div>

                <div className="rounded-lg border border-gray-700 bg-gray-800 p-3">
                  <p className="mb-2 text-xs uppercase tracking-wider text-gray-500">Ground Truths</p>
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-gray-300">
                      Current Step: {synced.ground_truth.step ?? "N/A"}
                    </span>
                    <span className={`rounded border px-2 py-1 ${statusBadgeClass(synced.ground_truth.correct)}`}>
                      Correct: {fmtBool(synced.ground_truth.correct)}
                    </span>
                  </div>
                </div>

                <div className="rounded-lg border border-gray-700 bg-gray-800 p-3 text-sm">
                  <p className="mb-2 text-xs uppercase tracking-wider text-gray-500">Error Detected</p>
                  {errorCheck.hasError && (
                    <div className="space-y-2">
                      <div className="flex flex-wrap gap-2 text-xs">
                        <span className="rounded border border-red-500/30 bg-red-500/10 px-2 py-1 text-red-200">
                          Error: {humanizeErrorType(errorCheck.errorType)}
                        </span>
                        <span className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-gray-300">
                          Step: {errorCheck.step ?? synced.detected_step}
                        </span>
                        <span className="rounded border border-gray-700 bg-gray-900 px-2 py-1 text-gray-300">
                          Confidence: {fmtNumber(errorCheck.confidence)}
                        </span>
                      </div>

                      {errorCheck.missingParts.length > 0 && (
                        <div className="rounded border border-red-500/20 bg-red-500/5 p-2 text-xs text-red-100">
                          <p className="mb-1 font-semibold text-red-200">Missing Parts</p>
                          <ul className="space-y-1">
                            {errorCheck.missingParts.map((part, idx) => (
                              <li key={`${part}-${idx}`}>• {part}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {synced.non_progress_reason && (
                        <p className="text-xs text-gray-300">{synced.non_progress_reason}</p>
                      )}
                    </div>
                  )}
                </div>
                    </>
                  );
                })()}
              </div>
            )}
          </section>

          {synced && (
            <section className="lg:col-span-2 rounded-xl border border-gray-700 bg-gray-900 p-5">
              <p className="mb-3 text-xs uppercase tracking-wider text-gray-400">Inference Audit Trail</p>
              <div className="grid gap-3 text-xs md:grid-cols-2 font-mono">
                <div className="rounded-lg border border-gray-700 bg-gray-800 p-3 space-y-1">
                  <p>gate_triggered: {fmtBool(synced.trace.gate_triggered)}</p>
                  <p>gate_similarity: {fmtNumber(synced.trace.gate_similarity)}</p>
                  <p>vlm_called: {fmtBool(synced.trace.vlm_called)}</p>
                  <p>vlm_confidence: {fmtNumber(synced.trace.vlm_confidence)}</p>
                  <p>completed_action_detected: {fmtBool(synced.trace.completed_action_detected)}</p>
                  <p>processing_time_ms: {fmtNumber(synced.trace.processing_time_ms, 1)}</p>
                </div>
                <div className="rounded-lg border border-gray-700 bg-gray-800 p-3 space-y-1">
                  <p>non_progress_trigger: {synced.trace.non_progress_trigger ?? "N/A"}</p>
                  <p>non_progress_visible: {fmtBool(synced.trace.non_progress_visible)}</p>
                  <p>error_detection_ran: {fmtBool(synced.trace.error_detection_ran)}</p>
                  <p>error_detected: {fmtBool(synced.trace.error_detected)}</p>
                  <p>error_detection_source: {synced.trace.error_detection_source ?? "N/A"}</p>
                </div>
                <div className="rounded-lg border border-gray-700 bg-gray-800 p-3 md:col-span-2">
                  <p className="mb-1 text-gray-400">vlm_reasoning</p>
                  <p className="whitespace-pre-wrap text-gray-200">
                    {synced.trace.vlm_reasoning?.trim() || "None"}
                  </p>
                  <details className="mt-3">
                    <summary className="cursor-pointer text-gray-400">error_detection_result (raw)</summary>
                    <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-gray-200">
                      {JSON.stringify(synced.trace.error_detection_result, null, 2)}
                    </pre>
                  </details>
                </div>
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
