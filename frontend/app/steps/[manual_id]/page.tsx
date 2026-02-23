"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { fetchSteps, croppedPathToUrl, type ManualSteps } from "@/lib/api";

// ── Lightbox ──────────────────────────────────────────────────────────────────

function Lightbox({
  src,
  alt,
  onClose,
}: {
  src: string;
  alt: string;
  onClose: () => void;
}) {
  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative max-w-3xl max-h-[85vh] p-2 bg-gray-900 rounded-xl border border-gray-700 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute -top-3 -right-3 w-7 h-7 rounded-full bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm flex items-center justify-center transition-colors z-10"
          aria-label="Close"
        >
          ✕
        </button>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={src}
          alt={alt}
          className="max-w-full max-h-[80vh] object-contain rounded-lg"
        />
        {alt && (
          <p className="text-center text-xs text-gray-400 mt-2 px-2">{alt}</p>
        )}
      </div>
    </div>
  );
}

// ── Clickable image thumbnail ─────────────────────────────────────────────────

function Thumbnail({
  src,
  alt,
  onClick,
}: {
  src: string;
  alt: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="shrink-0 w-20 h-20 rounded-lg border border-gray-700 bg-gray-800 overflow-hidden hover:border-yellow-500/60 hover:scale-105 transition-all cursor-zoom-in"
      title="Click to enlarge"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={alt}
        className="w-full h-full object-cover"
      />
    </button>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function StepsDetailPage({
  params,
}: {
  params: { manual_id: string };
}) {
  const { manual_id } = params;
  const [data, setData] = useState<ManualSteps | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [lightbox, setLightbox] = useState<{ src: string; alt: string } | null>(null);

  const openLightbox = useCallback((src: string, alt: string) => {
    setLightbox({ src, alt });
  }, []);

  const closeLightbox = useCallback(() => setLightbox(null), []);

  useEffect(() => {
    fetchSteps(manual_id)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [manual_id]);

  return (
    <>
      {lightbox && (
        <Lightbox src={lightbox.src} alt={lightbox.alt} onClose={closeLightbox} />
      )}

      <div>
        {/* Header */}
        <div className="flex items-center gap-3 mb-8">
          <Link
            href="/steps"
            className="text-gray-500 hover:text-gray-300 transition-colors text-sm"
          >
            ← All manuals
          </Link>
          <span className="text-gray-700">/</span>
          <h1 className="text-2xl font-bold">{manual_id}</h1>
          {data && (
            <span className="ml-auto text-sm text-gray-500">
              {data.steps.length} steps
            </span>
          )}
        </div>

        {loading && <p className="text-gray-500 text-sm">Loading steps…</p>}

        {error && (
          <div className="px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/30 text-red-300 text-sm">
            {error}
          </div>
        )}

        {data && (
          <div className="space-y-6">
            {data.steps.map((step) => (
              <div
                key={step.step_number}
                className="rounded-xl border border-gray-700 bg-gray-900 overflow-hidden"
              >
                {/* Step header */}
                <div className="px-5 py-3 border-b border-gray-700 flex items-center justify-between bg-gray-800/50">
                  <span className="font-semibold text-yellow-400">
                    Step {step.step_number}
                  </span>
                  {step.notes && (
                    <span className="text-xs text-gray-400 italic">
                      {step.notes}
                    </span>
                  )}
                </div>

                <div className="p-5 grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Actions */}
                  {step.actions.length > 0 && (
                    <div>
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                        Actions
                      </p>
                      <ol className="space-y-1">
                        {step.actions.map((a, i) => (
                          <li key={i} className="text-sm text-gray-200 flex gap-2">
                            <span className="text-gray-600 shrink-0">{i + 1}.</span>
                            {a}
                          </li>
                        ))}
                      </ol>
                    </div>
                  )}

                  {/* Parts required */}
                  {step.parts_required.length > 0 && (
                    <div>
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                        Parts Required
                      </p>
                      <ul className="space-y-3">
                        {step.parts_required.map((part, i) => (
                          <li key={i} className="flex items-center gap-3">
                            {part.cropped_image_path ? (
                              <Thumbnail
                                src={croppedPathToUrl(part.cropped_image_path)}
                                alt={part.description}
                                onClick={() =>
                                  openLightbox(
                                    croppedPathToUrl(part.cropped_image_path!),
                                    part.description
                                  )
                                }
                              />
                            ) : (
                              <div className="shrink-0 w-20 h-20 rounded-lg border border-gray-700 bg-gray-800 flex items-center justify-center text-gray-600 text-xs">
                                no image
                              </div>
                            )}
                            <span className="text-sm text-gray-200">
                              {part.description}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Subassemblies */}
                  {step.subassemblies.length > 0 && (
                    <div>
                      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
                        Subassemblies
                      </p>
                      <ul className="space-y-3">
                        {step.subassemblies.map((sub, i) => (
                          <li key={i} className="flex items-center gap-3">
                            {sub.cropped_image_path ? (
                              <Thumbnail
                                src={croppedPathToUrl(sub.cropped_image_path)}
                                alt={sub.description}
                                onClick={() =>
                                  openLightbox(
                                    croppedPathToUrl(sub.cropped_image_path!),
                                    sub.description
                                  )
                                }
                              />
                            ) : (
                              <div className="shrink-0 w-20 h-20 rounded-lg border border-gray-700 bg-gray-800 flex items-center justify-center text-gray-600 text-xs">
                                no image
                              </div>
                            )}
                            <span className="text-sm text-gray-200">
                              {sub.description}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
