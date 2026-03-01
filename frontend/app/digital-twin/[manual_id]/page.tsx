"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";
import { fetchDigitalTwin, type DigitalTwin, type Brick } from "@/lib/api";

// Dynamically import the 3D viewer to avoid SSR issues
const DigitalTwinViewer = dynamic(
  () => import("@/components/DigitalTwinViewer"),
  { ssr: false, loading: () => <div className="flex items-center justify-center h-[500px] bg-gray-900 rounded-lg border border-gray-700"><p className="text-gray-500 text-sm">Loading 3D viewer...</p></div> }
);

// Color mapping for LEGO color IDs (common colors)
const LEGO_COLORS: Record<number, string> = {
  0: "Black",
  1: "Blue",
  2: "Green",
  3: "Dark Turquoise",
  4: "Red",
  5: "Dark Pink",
  6: "Brown",
  7: "Light Gray",
  8: "Dark Gray",
  9: "Light Blue",
  10: "Bright Green",
  11: "Light Turquoise",
  12: "Salmon",
  13: "Pink",
  14: "Yellow",
  15: "White",
  16: "Light Green",
  17: "Light Yellow",
  18: "Tan",
  19: "Light Violet",
  20: "Purple",
  21: "Dark Blue Violet",
  22: "Orange",
  23: "Magenta",
  24: "Lime",
  25: "Dark Tan",
};

function formatPosition(pos: [number, number, number]): string {
  return `(${pos[0].toFixed(1)}, ${pos[1].toFixed(1)}, ${pos[2].toFixed(1)})`;
}

function formatRotation(angles: { roll_deg: number; pitch_deg: number; yaw_deg: number }): string {
  return `R:${angles.roll_deg.toFixed(1)}° P:${angles.pitch_deg.toFixed(1)}° Y:${angles.yaw_deg.toFixed(1)}°`;
}

function BrickCard({ brick }: { brick: Brick }) {
  const colorName = LEGO_COLORS[brick.color_id] || `Color ${brick.color_id}`;
  const partName = brick.part_number.replace(".dat", "");

  return (
    <div className="p-4 rounded-lg border border-gray-700 bg-gray-800/50 hover:border-yellow-500/30 transition-colors">
      <div className="flex items-start justify-between mb-2">
        <div>
          <span className="text-sm font-semibold text-yellow-400">
            Brick #{brick.brick_id}
          </span>
          <p className="text-xs text-gray-500 mt-0.5">
            {partName}
          </p>
        </div>
        <span className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300">
          {colorName}
        </span>
      </div>

      <div className="space-y-1.5 text-xs">
        <div className="flex justify-between">
          <span className="text-gray-500">Position:</span>
          <span className="text-gray-300 font-mono text-[10px]">
            {formatPosition(brick.position)}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Rotation:</span>
          <span className="text-gray-300 font-mono text-[10px]">
            {formatRotation(brick.rotation_angles_deg)}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Mesh:</span>
          <span className="text-gray-300 font-mono text-[10px]">
            {brick.geometry_reference.mesh_file}
          </span>
        </div>
      </div>
    </div>
  );
}

export default function DigitalTwinDetailPage({
  params,
}: {
  params: { manual_id: string };
}) {
  const { manual_id } = params;
  const [data, setData] = useState<DigitalTwin | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());

  useEffect(() => {
    fetchDigitalTwin(manual_id)
      .then((dt) => {
        setData(dt);
        // Expand first step by default
        if (dt.steps.length > 0) {
          setExpandedSteps(new Set([dt.steps[0].step_number]));
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [manual_id]);

  const toggleStep = (stepNumber: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(stepNumber)) {
        next.delete(stepNumber);
      } else {
        next.add(stepNumber);
      }
      return next;
    });
  };

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-3 mb-8">
        <Link
          href="/digital-twin"
          className="text-gray-500 hover:text-gray-300 transition-colors text-sm"
        >
          ← All manuals
        </Link>
        <span className="text-gray-700">/</span>
        <h1 className="text-2xl font-bold">Digital Twin: {manual_id}</h1>
        {data && (
          <span className="ml-auto text-sm text-gray-500">
            {data.steps.length} steps
          </span>
        )}
      </div>

      {loading && <p className="text-gray-500 text-sm">Loading digital twin data…</p>}

      {error && (
        <div className="px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm">
          {error}
        </div>
      )}

      {data && (
        <div className="space-y-4">
          {data.steps.map((step) => {
            const isExpanded = expandedSteps.has(step.step_number);
            return (
              <div
                key={step.step_number}
                className="rounded-xl border border-gray-700 bg-gray-900 overflow-hidden"
              >
                {/* Step header - clickable */}
                <button
                  onClick={() => toggleStep(step.step_number)}
                  className="w-full px-5 py-4 border-b border-gray-700 flex items-center justify-between bg-gray-800/50 hover:bg-gray-800 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <span className="font-semibold text-yellow-400">
                      Step {step.step_number}
                    </span>
                    <span className="text-sm text-gray-500">
                      {step.step_name}
                    </span>
                    <span className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300">
                      {step.num_bricks} brick{step.num_bricks !== 1 ? "s" : ""}
                    </span>
                  </div>
                  <span className={`text-gray-500 transition-transform ${isExpanded ? "rotate-180" : ""}`}>
                    ▼
                  </span>
                </button>

                {/* Step content - collapsible */}
                {isExpanded && (
                  <div className="p-5 space-y-5">
                    {/* 3D Visualization */}
                    <div>
                      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
                        3D Visualization
                      </h3>
                      <DigitalTwinViewer bricks={step.bricks} height={500} />
                    </div>

                    {/* Brick Details */}
                    <div>
                      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
                        Brick Details
                      </h3>
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                        {step.bricks.map((brick) => (
                          <BrickCard key={brick.brick_id} brick={brick} />
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Info footer */}
      {data && data.steps.length > 0 && (
        <div className="mt-8 p-4 rounded-lg border border-gray-700 bg-gray-900/50">
          <p className="text-xs text-gray-500">
            <span className="font-semibold text-gray-400">Note:</span> Digital twin data includes precise 3D positions, rotations, and geometry references for each LEGO brick.
            Click on each step to expand and view brick details.
          </p>
        </div>
      )}
    </div>
  );
}
