import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-6 text-center">
      <h1 className="text-4xl font-bold tracking-tight">LEGO Assembler</h1>
      <p className="text-gray-400 max-w-md">
        Process LEGO instruction manuals using Gemini VLM to extract assembly
        steps, parts, and bounding boxes.
      </p>
      <Link
        href="/ingest"
        className="px-6 py-3 bg-yellow-400 text-gray-900 font-semibold rounded-lg hover:bg-yellow-300 transition-colors"
      >
        Start Ingesting
      </Link>
    </div>
  );
}
