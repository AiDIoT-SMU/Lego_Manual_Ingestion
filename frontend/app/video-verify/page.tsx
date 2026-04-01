"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { uploadVideo } from "@/lib/api";

export default function VideoVerifyPage() {
  const [manualId, setManualId] = useState("");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();

  const handleUpload = async () => {
    if (!manualId) {
      setError("Please enter a manual ID");
      return;
    }

    if (!videoFile) {
      setError("Please select a video file");
      return;
    }

    setUploading(true);
    setError("");

    try {
      const response = await uploadVideo(manualId, videoFile);

      // Redirect to results page
      router.push(`/video-verify/${manualId}/${response.video_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setUploading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">Video Assembly Verification</h1>
      <p className="text-gray-400 text-sm mb-8">
        Upload a video of your LEGO assembly to verify each step and track part usage
      </p>

      <div className="space-y-6">
        {/* Manual ID Input */}
        <div>
          <label className="block text-sm font-medium mb-2">
            Manual ID
          </label>
          <input
            type="text"
            value={manualId}
            onChange={(e) => setManualId(e.target.value)}
            placeholder="e.g., 6262059"
            className="w-full px-4 py-3 rounded-lg border border-gray-700 bg-gray-800 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-yellow-400"
            disabled={uploading}
          />
          <p className="text-xs text-gray-500 mt-1">
            The manual ID must already be ingested in the system
          </p>
        </div>

        {/* Video File Input */}
        <div>
          <label className="block text-sm font-medium mb-2">
            Assembly Video
          </label>
          <div className="relative">
            <input
              type="file"
              accept="video/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  setVideoFile(file);
                  setError("");
                }
              }}
              className="block w-full text-sm text-gray-400
                file:mr-4 file:py-3 file:px-4
                file:rounded-lg file:border-0
                file:text-sm file:font-semibold
                file:bg-yellow-400 file:text-gray-900
                hover:file:bg-yellow-300
                file:cursor-pointer
                cursor-pointer"
              disabled={uploading}
            />
          </div>
          {videoFile && (
            <p className="text-xs text-gray-400 mt-2">
              Selected: {videoFile.name} ({(videoFile.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          )}
          <p className="text-xs text-gray-500 mt-1">
            Supported formats: MP4, MOV, AVI
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={uploading || !manualId || !videoFile}
          className="w-full px-6 py-4 bg-yellow-400 text-gray-900 font-semibold rounded-lg hover:bg-yellow-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-lg"
        >
          {uploading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
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
              Uploading & Processing...
            </span>
          ) : (
            "Upload & Analyze Video"
          )}
        </button>

        {uploading && (
          <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/30 text-blue-300 text-sm">
            <p className="font-semibold mb-1">Processing your video...</p>
            <p className="text-xs text-blue-200">
              This may take a few minutes depending on video length. The page will redirect
              automatically when complete.
            </p>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="mt-12 p-6 rounded-lg bg-gray-800 border border-gray-700">
        <h2 className="text-lg font-semibold mb-3">How it works</h2>
        <ol className="space-y-2 text-sm text-gray-300">
          <li className="flex gap-3">
            <span className="text-yellow-400 font-bold">1.</span>
            <span>Upload a video of you assembling the LEGO model</span>
          </li>
          <li className="flex gap-3">
            <span className="text-yellow-400 font-bold">2.</span>
            <span>Our AI analyzes every 50 frames to detect which step you're on</span>
          </li>
          <li className="flex gap-3">
            <span className="text-yellow-400 font-bold">3.</span>
            <span>Parts are tracked to verify you're using the correct pieces</span>
          </li>
          <li className="flex gap-3">
            <span className="text-yellow-400 font-bold">4.</span>
            <span>View results with a timeline showing each step and a parts checklist</span>
          </li>
        </ol>
      </div>
    </div>
  );
}
