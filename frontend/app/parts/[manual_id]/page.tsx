"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchParts, croppedPathToUrl, type PartsCatalog } from "@/lib/api";

export default function PartsDetailPage({
  params,
}: {
  params: { manual_id: string };
}) {
  const { manual_id } = params;
  const [data, setData] = useState<PartsCatalog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchParts(manual_id)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [manual_id]);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-3 mb-8">
        <Link
          href="/parts"
          className="text-gray-500 hover:text-gray-300 transition-colors text-sm"
        >
          ← All manuals
        </Link>
        <span className="text-gray-700">/</span>
        <h1 className="text-2xl font-bold">{manual_id}</h1>
        {data && (
          <span className="ml-auto text-sm text-gray-500">
            {data.parts.length} unique part{data.parts.length !== 1 ? "s" : ""}
          </span>
        )}
      </div>

      {loading && <p className="text-gray-500 text-sm">Loading parts…</p>}

      {error && (
        <div className="px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm">
          {error}
        </div>
      )}

      {data && data.parts.length === 0 && (
        <div className="text-center py-16 text-gray-500 text-sm border border-dashed border-gray-700 rounded-2xl">
          No parts data found for this manual.
        </div>
      )}

      {data && data.parts.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {data.parts.map((part, i) => (
            <div
              key={i}
              className="p-4 rounded-xl border border-gray-700 bg-gray-900 flex gap-4 items-start"
            >
              {/* Image */}
              <div className="w-16 h-16 shrink-0 rounded-lg border border-gray-700 bg-gray-800 overflow-hidden flex items-center justify-center">
                {part.images.length > 0 ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={croppedPathToUrl(part.images[0])}
                    alt={part.description}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span className="text-2xl">🧱</span>
                )}
              </div>

              {/* Info */}
              <div className="min-w-0">
                <p className="text-sm font-medium text-gray-100 leading-snug">
                  {part.description}
                </p>
                <p className="text-xs text-gray-500 mt-1.5">
                  Used in step{part.used_in_steps.length !== 1 ? "s" : ""}:{" "}
                  <span className="text-gray-400">
                    {part.used_in_steps.join(", ")}
                  </span>
                </p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
