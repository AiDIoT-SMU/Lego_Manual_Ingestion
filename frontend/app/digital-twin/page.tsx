"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchManuals, type ManualMeta } from "@/lib/api";

export default function DigitalTwinIndexPage() {
  const [manuals, setManuals] = useState<ManualMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchManuals()
      .then(setManuals)
      .catch(() => setError("Could not reach the backend. Is it running?"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Digital Twin Viewer</h1>
          <p className="text-gray-400 text-sm mt-1">
            View 3D digital twin data for each assembly step.
          </p>
        </div>
        <Link
          href="/ingest"
          className="px-4 py-2 text-sm bg-yellow-400 text-gray-900 font-semibold rounded-lg hover:bg-yellow-300 transition-colors"
        >
          + Ingest Manual
        </Link>
      </div>

      {loading && (
        <div className="text-gray-500 text-sm">Loading manuals…</div>
      )}

      {error && (
        <div className="px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm">
          {error}
        </div>
      )}

      {!loading && !error && manuals.length === 0 && (
        <div className="flex flex-col items-center justify-center py-24 gap-4 text-center border border-dashed border-gray-700 rounded-2xl">
          <p className="text-4xl">🧊</p>
          <p className="text-gray-300 font-medium">No manuals ingested yet</p>
          <p className="text-gray-500 text-sm max-w-xs">
            Ingest a LEGO instruction manual to see its digital twin data here.
          </p>
          <Link
            href="/ingest"
            className="mt-2 px-5 py-2 bg-yellow-400 text-gray-900 font-semibold rounded-lg hover:bg-yellow-300 transition-colors text-sm"
          >
            Ingest your first manual
          </Link>
        </div>
      )}

      {!loading && manuals.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {manuals.map((m) => (
            <Link
              key={m.id}
              href={`/digital-twin/${m.id}`}
              className="group p-5 rounded-xl border border-gray-700 hover:border-yellow-400/50 bg-gray-900 hover:bg-gray-800 transition-all"
            >
              <div className="flex items-start justify-between gap-2">
                <div>
                  <p className="font-semibold text-gray-100 group-hover:text-yellow-400 transition-colors">
                    {m.id}
                  </p>
                  <p className="text-sm text-gray-500 mt-0.5">
                    {m.step_count} step{m.step_count !== 1 ? "s" : ""}
                  </p>
                </div>
                <span className="text-gray-600 group-hover:text-yellow-400 transition-colors text-lg">
                  →
                </span>
              </div>
              <p className="text-xs text-gray-600 mt-3">
                {new Date(m.created_at * 1000).toLocaleDateString()}
              </p>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
