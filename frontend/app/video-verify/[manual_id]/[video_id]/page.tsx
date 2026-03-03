"use client";

import { useEffect, useState, useRef } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { fetchVideoAnalysis, type VideoAnalysis } from "@/lib/api";

export default function VideoAnalysisPage() {
  const params = useParams();
  const manualId = params.manual_id as string;
  const videoId = params.video_id as string;

  const [analysis, setAnalysis] = useState<VideoAnalysis | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [processingStatus, setProcessingStatus] = useState<{
    status: string;
    message: string;
    progress: number;
  } | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Fetch analysis results
  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const fetchData = async () => {
      try {
        const data = await fetchVideoAnalysis(manualId, videoId);

        // Check if still processing
        if ((data as any).status === "processing") {
          setProcessingStatus({
            status: (data as any).status,
            message: (data as any).message || "Processing...",
            progress: (data as any).progress || 0
          });
          return; // Keep polling
        }

        // Check if processing failed
        if ((data as any).status === "failed") {
          setError((data as any).error || "Video processing failed");
          setLoading(false);
          clearInterval(intervalId);
          return;
        }

        // Processing completed successfully
        setAnalysis(data);
        setLoading(false);
        setProcessingStatus(null);
        clearInterval(intervalId);
      } catch (err) {
        // If 404, video might still be processing
        if (err instanceof Error && err.message.includes("404")) {
          // Keep polling
          console.log("Analysis not ready, waiting...");
          setProcessingStatus({
            status: "waiting",
            message: "Waiting for processing to start...",
            progress: 0
          });
        } else {
          setError(err instanceof Error ? err.message : "Failed to load analysis");
          setLoading(false);
          clearInterval(intervalId);
        }
      }
    };

    // Poll every 3 seconds until analysis is ready
    fetchData();
    intervalId = setInterval(fetchData, 3000);

    return () => clearInterval(intervalId);
  }, [manualId, videoId]);

  // Get current step based on video playhead
  const getCurrentStep = () => {
    if (!analysis) return null;
    return analysis.step_timeline.find(
      (step) => currentTime >= step.start_time && currentTime <= step.end_time
    );
  };

  // Format time as MM:SS
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // Seek to specific time
  const seekTo = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
    }
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col items-center justify-center py-20 px-6">
          <svg className="animate-spin h-12 w-12 text-yellow-400 mb-4" viewBox="0 0 24 24">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
          <p className="text-lg font-semibold mb-2">Processing video...</p>

          {processingStatus && (
            <div className="w-full max-w-md mt-4">
              <div className="mb-2">
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
          <p className="text-sm mb-3">{error}</p>

          <details className="mt-4">
            <summary className="cursor-pointer text-xs text-gray-400 hover:text-gray-300">
              Show technical details
            </summary>
            <pre className="mt-2 p-3 bg-black/30 rounded text-xs overflow-x-auto">
              {error}
            </pre>
          </details>
        </div>
        <Link
          href="/video-verify"
          className="mt-4 inline-block text-yellow-400 hover:text-yellow-300 text-sm"
        >
          ← Back to upload
        </Link>
      </div>
    );
  }

  if (!analysis) {
    return <div>No analysis data</div>;
  }

  const currentStep = getCurrentStep();

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <Link
          href="/video-verify"
          className="text-gray-500 hover:text-gray-300 transition-colors text-sm"
        >
          ← Back to upload
        </Link>
        <span className="text-gray-700">/</span>
        <h1 className="text-2xl font-bold">{analysis.video_filename}</h1>
        <span className="ml-auto text-sm text-gray-500">
          Manual: {manualId}
        </span>
      </div>

      {/* Main Layout: Video Player + Timeline | Parts Checklist */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-6">
        {/* Left Column: Video + Timeline */}
        <div className="space-y-6">
          {/* Video Player */}
          <div className="bg-gray-900 rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              src={`http://localhost:8000/videos/${manualId}/${videoId}.mp4`}
              controls
              className="w-full"
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
            />
          </div>

          {/* Current Step Indicator */}
          {currentStep && (
            <div className="px-4 py-3 rounded-lg bg-yellow-400/10 border border-yellow-400/30 text-yellow-300">
              <p className="text-sm font-semibold">
                Currently on Step {currentStep.step_number}
              </p>
              <p className="text-xs text-yellow-200 mt-1">
                {formatTime(currentStep.start_time)} - {formatTime(currentStep.end_time)} •{" "}
                Confidence: {(currentStep.confidence_avg * 100).toFixed(0)}%
              </p>
            </div>
          )}

          {/* Step Timeline */}
          <div className="bg-gray-800 rounded-lg p-5">
            <h2 className="text-lg font-semibold mb-4">Step Timeline</h2>
            {analysis.step_timeline.length === 0 ? (
              <p className="text-sm text-gray-400">
                No steps detected. Try recording with better lighting or closer camera angle.
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
                          {formatTime(step.start_time)} - {formatTime(step.end_time)}
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
        </div>

        {/* Right Column: Parts Checklist */}
        <div className="bg-gray-800 rounded-lg p-5">
          <h2 className="text-lg font-semibold mb-4">Parts Checklist</h2>
          {Object.keys(analysis.parts_used).length === 0 ? (
            <p className="text-sm text-gray-400">
              No parts detected. Ensure parts are clearly visible in the video.
            </p>
          ) : (
            <div className="space-y-3">
              {Object.entries(analysis.parts_used).map(([partDesc, usage]) => (
                <div
                  key={partDesc}
                  className={`p-3 rounded-lg border transition-all ${
                    usage.marked_as_used
                      ? "border-green-500/50 bg-green-500/10"
                      : "border-gray-700"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5 shrink-0">
                      {usage.marked_as_used ? (
                        <span className="text-green-400 text-2xl leading-none">✓</span>
                      ) : (
                        <span className="text-gray-600 text-2xl leading-none">○</span>
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium leading-tight">{partDesc}</div>
                      {usage.marked_as_used && (
                        <div className="text-xs text-gray-400 mt-1.5">
                          First seen: {formatTime(usage.first_seen_timestamp)}
                          <br />
                          Confidence: {(usage.usage_confidence * 100).toFixed(0)}%
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Statistics */}
          <div className="mt-6 pt-4 border-t border-gray-700">
            <div className="text-xs text-gray-400 space-y-1">
              <div className="flex justify-between">
                <span>Total Parts:</span>
                <span className="text-white">{Object.keys(analysis.parts_used).length}</span>
              </div>
              <div className="flex justify-between">
                <span>Parts Used:</span>
                <span className="text-green-400">
                  {Object.values(analysis.parts_used).filter((u) => u.marked_as_used).length}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Frames Analyzed:</span>
                <span className="text-white">{analysis.total_frames_extracted}</span>
              </div>
              <div className="flex justify-between">
                <span>Video Duration:</span>
                <span className="text-white">
                  {formatTime(analysis.total_duration_seconds)}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
