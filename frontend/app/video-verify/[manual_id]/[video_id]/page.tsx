"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  fetchVideoAnalysis,
  fetchParts,
  fetchVideoEnhancedSteps,
  enhanceManualWithVideo,
  croppedPathToUrl,
  type VideoAnalysis,
  type PartsCatalog,
  type VideoEnhancedManual,
} from "@/lib/api";

export default function VideoAnalysisPage() {
  const params = useParams();
  const manualId = params.manual_id as string;
  const videoId = params.video_id as string;

  const [analysis, setAnalysis] = useState<VideoAnalysis | null>(null);
  const [partsCatalog, setPartsCatalog] = useState<PartsCatalog | null>(null);
  const [enhancedManual, setEnhancedManual] = useState<VideoEnhancedManual | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [processingStatus, setProcessingStatus] = useState<{
    status: string;
    message: string;
    progress: number;
  } | null>(null);

  const [enhancing, setEnhancing] = useState(false);
  const [enhanceMessage, setEnhanceMessage] = useState("");

  const [currentTime, setCurrentTime] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const enhanceIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // ── Poll for analysis completion ───────────────────────────────────────────

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const fetchData = async () => {
      try {
        const data = await fetchVideoAnalysis(manualId, videoId);

        if ((data as any).status === "processing") {
          setProcessingStatus({
            status: (data as any).status,
            message: (data as any).message || "Processing...",
            progress: (data as any).progress || 0,
          });
          return;
        }

        if ((data as any).status === "failed") {
          setError((data as any).error || "Video processing failed");
          setLoading(false);
          clearInterval(intervalId);
          return;
        }

        // Analysis complete — load parts catalog and check for enhancement in parallel
        setAnalysis(data);
        clearInterval(intervalId);

        const [catalog] = await Promise.all([
          fetchParts(manualId).catch(() => null),
          fetchVideoEnhancedSteps(manualId)
            .then(setEnhancedManual)
            .catch(() => {}),
        ]);
        setPartsCatalog(catalog);
        setLoading(false);
        setProcessingStatus(null);
      } catch (err) {
        if (err instanceof Error && err.message.includes("404")) {
          setProcessingStatus({
            status: "waiting",
            message: "Waiting for processing to start...",
            progress: 0,
          });
        } else {
          setError(err instanceof Error ? err.message : "Failed to load analysis");
          setLoading(false);
          clearInterval(intervalId);
        }
      }
    };

    fetchData();
    intervalId = setInterval(fetchData, 3000);
    return () => clearInterval(intervalId);
  }, [manualId, videoId]);

  // ── Cleanup enhance polling on unmount ─────────────────────────────────────

  useEffect(() => {
    return () => {
      if (enhanceIntervalRef.current) clearInterval(enhanceIntervalRef.current);
    };
  }, []);

  // ── Helpers ────────────────────────────────────────────────────────────────

  const getCurrentStep = () => {
    if (!analysis) return null;
    return analysis.step_timeline.find(
      (step) => currentTime >= step.start_time && currentTime <= step.end_time
    );
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const seekTo = (time: number) => {
    if (videoRef.current) videoRef.current.currentTime = time;
  };

  const partImageFor = useCallback(
    (description: string): string | null => {
      if (!partsCatalog) return null;
      const entry = partsCatalog.parts.find((p) => p.description === description);
      if (!entry || !entry.images.length) return null;
      return croppedPathToUrl(entry.images[0]);
    },
    [partsCatalog]
  );

  // Parts discovered so far (timestamp has been reached in video)
  const discoveredParts = analysis
    ? Object.entries(analysis.parts_used).filter(
        ([, u]) => u.marked_as_used && currentTime >= u.first_seen_timestamp
      )
    : [];

  // ── Enhance handler ────────────────────────────────────────────────────────

  const handleEnhance = async () => {
    setEnhancing(true);
    setEnhanceMessage("Enhancement started. Polling for results every 10 seconds…");

    try {
      await enhanceManualWithVideo(manualId, videoId);
    } catch (err) {
      setEnhancing(false);
      setEnhanceMessage(err instanceof Error ? err.message : "Enhancement request failed.");
      return;
    }

    // Poll for enhancement completion
    enhanceIntervalRef.current = setInterval(async () => {
      try {
        const data = await fetchVideoEnhancedSteps(manualId);
        setEnhancedManual(data);
        setEnhancing(false);
        setEnhanceMessage("");
        if (enhanceIntervalRef.current) clearInterval(enhanceIntervalRef.current);
      } catch {
        // Not ready yet — keep polling
      }
    }, 10000);
  };

  // ── Loading / error states ─────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col items-center justify-center py-20 px-6">
          <svg className="animate-spin h-12 w-12 text-yellow-400 mb-4" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <p className="text-lg font-semibold mb-2">Processing video…</p>
          {processingStatus && (
            <div className="w-full max-w-md mt-4">
              <p className="text-sm text-gray-300 mb-1">{processingStatus.message}</p>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-yellow-400 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${processingStatus.progress}%` }}
                />
              </div>
              <p className="text-xs text-gray-400 mt-1 text-right">
                {Math.round(processingStatus.progress)}%
              </p>
            </div>
          )}
          {!processingStatus && (
            <p className="text-sm text-gray-400">
              Analyzing frames and detecting steps. This may take a few minutes.
            </p>
          )}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-6">
        <div className="px-6 py-4 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300">
          <p className="font-semibold mb-2 text-lg">Processing Failed</p>
          <p className="text-sm">{error}</p>
        </div>
        <Link href="/video-verify" className="mt-4 inline-block text-yellow-400 hover:text-yellow-300 text-sm">
          ← Back to upload
        </Link>
      </div>
    );
  }

  if (!analysis) return <div>No analysis data</div>;

  const currentStep = getCurrentStep();

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Link href="/video-verify" className="text-gray-500 hover:text-gray-300 transition-colors text-sm">
          ← Back to upload
        </Link>
        <span className="text-gray-700">/</span>
        <h1 className="text-2xl font-bold truncate">{analysis.video_filename}</h1>
        <span className="ml-auto text-sm text-gray-500 shrink-0">Manual: {manualId}</span>
      </div>

      {/* Main layout */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6">

        {/* ── Left column ─────────────────────────────────────────────────── */}
        <div className="space-y-6">

          {/* Video player with live overlay */}
          <div className="bg-gray-900 rounded-lg overflow-hidden relative">
            <video
              ref={videoRef}
              src={`http://localhost:8000/videos/${manualId}/${videoId}.mp4`}
              controls
              className="w-full"
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
            />

            {/* Live overlay: current step + ticking parts */}
            <div className="absolute bottom-14 right-3 w-52 bg-black/75 backdrop-blur-sm rounded-lg p-3 pointer-events-none">
              <p className="text-[10px] text-gray-400 uppercase tracking-widest mb-1.5">Live</p>
              {currentStep ? (
                <p className="text-yellow-400 text-xs font-bold mb-2 truncate">
                  Step {currentStep.step_number} &nbsp;
                  <span className="text-yellow-200/60 font-normal">
                    {(currentStep.confidence_avg * 100).toFixed(0)}%
                  </span>
                </p>
              ) : (
                <p className="text-gray-500 text-xs mb-2">No step detected</p>
              )}

              {discoveredParts.length > 0 ? (
                <div className="space-y-1 max-h-28 overflow-hidden">
                  {discoveredParts.slice(0, 6).map(([desc]) => (
                    <div key={desc} className="flex items-center gap-1.5">
                      <span className="text-green-400 text-[10px] shrink-0">✓</span>
                      <span className="text-white/80 text-[10px] truncate">{desc}</span>
                    </div>
                  ))}
                  {discoveredParts.length > 6 && (
                    <p className="text-gray-500 text-[10px]">+{discoveredParts.length - 6} more</p>
                  )}
                </div>
              ) : (
                <p className="text-gray-600 text-[10px]">No parts detected yet</p>
              )}
            </div>
          </div>

          {/* Current step indicator */}
          {currentStep && (
            <div className="px-4 py-3 rounded-lg bg-yellow-400/10 border border-yellow-400/30 text-yellow-300">
              <p className="text-sm font-semibold">Currently on Step {currentStep.step_number}</p>
              <p className="text-xs text-yellow-200 mt-1">
                {formatTime(currentStep.start_time)} – {formatTime(currentStep.end_time)} &nbsp;·&nbsp;
                Confidence: {(currentStep.confidence_avg * 100).toFixed(0)}%
              </p>
            </div>
          )}

          {/* Step timeline */}
          <div className="bg-gray-800 rounded-lg p-5">
            <h2 className="text-lg font-semibold mb-4">Step Timeline</h2>
            {analysis.step_timeline.length === 0 ? (
              <p className="text-sm text-gray-400">
                No steps detected. Try recording with better lighting or a closer camera angle.
              </p>
            ) : (
              <div className="space-y-2">
                {analysis.step_timeline.map((step) => {
                  const isActive = currentStep?.step_number === step.step_number;
                  return (
                    <div
                      key={step.step_number}
                      onClick={() => seekTo(step.start_time)}
                      className={`p-3 rounded-lg border cursor-pointer transition-all ${
                        isActive
                          ? "border-yellow-400 bg-yellow-400/10 shadow-lg"
                          : "border-gray-700 hover:border-gray-600"
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <span className={`font-semibold ${isActive ? "text-yellow-400" : ""}`}>
                          Step {step.step_number}
                        </span>
                        <span className="text-sm text-gray-400">
                          {formatTime(step.start_time)} – {formatTime(step.end_time)}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 mt-1">
                        <span className="text-xs text-gray-500">
                          Duration: {step.duration_seconds.toFixed(1)}s
                        </span>
                        <span className="text-xs text-gray-500">
                          Confidence: {(step.confidence_avg * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Spatial Enhancement section */}
          <div className="bg-gray-800 rounded-lg p-5">
            <h2 className="text-lg font-semibold mb-1">Spatial Enhancement</h2>
            <p className="text-sm text-gray-400 mb-4">
              Generate detailed sub-steps and spatial placement instructions by analyzing this video against the manual.
            </p>

            {enhancedManual ? (
              <div className="space-y-4">
                <p className="text-xs text-green-400 mb-1">
                  ✓ Enhanced &mdash;{" "}
                  {enhancedManual.steps.reduce((n, s) => n + s.sub_steps.length, 0)} sub-steps
                  across {enhancedManual.steps.length} steps
                </p>
                {enhancedManual.steps.map((step) => (
                  <div key={step.step_number} className="border border-gray-700 rounded-lg p-3">
                    <h3 className="text-sm font-semibold text-yellow-400 mb-2">
                      Step {step.step_number}
                    </h3>
                    <div className="space-y-2">
                      {step.sub_steps.map((sub) => (
                        <div key={sub.sub_step_number} className="pl-2 border-l border-gray-700">
                          <p className="text-xs text-gray-200">
                            <span className="text-gray-500 mr-1">{sub.sub_step_number}.</span>
                            {sub.description}
                          </p>
                          {sub.spatial_description && (
                            <p className="text-[11px] text-gray-500 mt-0.5 italic">
                              Place {sub.spatial_description.target_part} on{" "}
                              {sub.spatial_description.placement_part}
                              {sub.spatial_description.location
                                ? ` — ${sub.spatial_description.location}`
                                : ""}
                              {sub.spatial_description.position_detail
                                ? ` (${sub.spatial_description.position_detail})`
                                : ""}
                            </p>
                          )}
                          <div className="flex items-center gap-2 mt-0.5">
                            <span className={`text-[10px] px-1.5 py-0.5 rounded capitalize ${
                              sub.action_type === "place" ? "bg-blue-500/20 text-blue-300" :
                              sub.action_type === "attach" ? "bg-green-500/20 text-green-300" :
                              sub.action_type === "pick" ? "bg-yellow-500/20 text-yellow-300" :
                              sub.action_type === "verify" ? "bg-purple-500/20 text-purple-300" :
                              "bg-gray-700 text-gray-400"
                            }`}>
                              {sub.action_type}
                            </span>
                            <span className="text-[10px] text-gray-600">
                              {(sub.confidence * 100).toFixed(0)}% confidence
                            </span>
                          </div>
                        </div>
                      ))}
                      {step.corrections.length > 0 && (
                        <div className="mt-2 p-2 rounded bg-orange-500/10 border border-orange-500/20">
                          <p className="text-[11px] text-orange-300 font-medium mb-1">
                            {step.corrections.length} correction{step.corrections.length > 1 ? "s" : ""}
                          </p>
                          {step.corrections.map((c, i) => (
                            <p key={i} className="text-[10px] text-gray-400">
                              {c.field}: {String(c.original_value)} → {String(c.corrected_value)}
                            </p>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div>
                <button
                  onClick={handleEnhance}
                  disabled={enhancing}
                  className="px-4 py-2 rounded-lg bg-yellow-400 text-gray-900 font-semibold text-sm hover:bg-yellow-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {enhancing ? "Enhancing… (this takes several minutes)" : "Enhance Manual with This Video"}
                </button>
                {enhanceMessage && (
                  <p className="text-xs text-gray-400 mt-2">{enhanceMessage}</p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ── Right column: parts checklist ───────────────────────────────── */}
        <div className="bg-gray-800 rounded-lg p-5 h-fit sticky top-6">
          <h2 className="text-lg font-semibold mb-4">Parts Checklist</h2>

          {Object.keys(analysis.parts_used).length === 0 ? (
            <p className="text-sm text-gray-400">
              No parts detected. Ensure parts are clearly visible in the video.
            </p>
          ) : (
            <div className="space-y-3">
              {Object.entries(analysis.parts_used).map(([partDesc, usage]) => {
                const seen = usage.marked_as_used && currentTime >= usage.first_seen_timestamp;
                const imgUrl = partImageFor(partDesc);
                return (
                  <div
                    key={partDesc}
                    className={`p-3 rounded-lg border transition-all ${
                      seen
                        ? "border-green-500/50 bg-green-500/10"
                        : usage.marked_as_used
                        ? "border-gray-600 bg-gray-700/40"
                        : "border-gray-700"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {/* Part image */}
                      {imgUrl ? (
                        <img
                          src={imgUrl}
                          alt={partDesc}
                          className="w-12 h-12 object-contain rounded bg-gray-700/50 border border-gray-600 shrink-0"
                        />
                      ) : (
                        <div className="w-12 h-12 rounded bg-gray-700/50 border border-gray-600 shrink-0 flex items-center justify-center text-gray-600 text-xl">
                          ☐
                        </div>
                      )}

                      {/* Tick + info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start gap-2">
                          <span className={`mt-0.5 shrink-0 text-lg leading-none ${seen ? "text-green-400" : "text-gray-600"}`}>
                            {seen ? "✓" : "○"}
                          </span>
                          <p className="text-sm font-medium leading-snug">{partDesc}</p>
                        </div>

                        {usage.marked_as_used && (
                          <div className="text-xs text-gray-400 mt-1 ml-6">
                            {seen ? (
                              <>First seen at {formatTime(usage.first_seen_timestamp)}</>
                            ) : (
                              <>Appears at {formatTime(usage.first_seen_timestamp)}</>
                            )}
                            <br />
                            Confidence: {(usage.usage_confidence * 100).toFixed(0)}%
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Stats */}
          <div className="mt-6 pt-4 border-t border-gray-700 text-xs text-gray-400 space-y-1">
            <div className="flex justify-between">
              <span>Total Parts:</span>
              <span className="text-white">{Object.keys(analysis.parts_used).length}</span>
            </div>
            <div className="flex justify-between">
              <span>Seen in video:</span>
              <span className="text-green-400">
                {Object.values(analysis.parts_used).filter((u) => u.marked_as_used).length}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Discovered (so far):</span>
              <span className="text-yellow-400">{discoveredParts.length}</span>
            </div>
            <div className="flex justify-between">
              <span>Frames Analyzed:</span>
              <span className="text-white">{analysis.total_frames_extracted}</span>
            </div>
            <div className="flex justify-between">
              <span>Duration:</span>
              <span className="text-white">{formatTime(analysis.total_duration_seconds)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
