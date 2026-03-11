import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LEGO Assembler",
  description: "LEGO instruction manual processor",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen antialiased">
        <nav className="border-b border-gray-800 bg-gray-900">
          <div className="max-w-5xl mx-auto px-6 py-4 flex items-center gap-8">
            <span className="font-bold text-lg tracking-tight">
              🧱 LEGO Assembler
            </span>
            <a
              href="/ingest"
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Ingest
            </a>
            <a
              href="/steps"
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Steps
            </a>
            <a
              href="/parts"
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Parts
            </a>
            <a
              href="/digital-twin"
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Digital Twin
            </a>
            <a
              href="/video-verify"
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Video Verify
            </a>
          </div>
        </nav>
        <main className="max-w-5xl mx-auto px-6 py-10">{children}</main>
      </body>
    </html>
  );
}
