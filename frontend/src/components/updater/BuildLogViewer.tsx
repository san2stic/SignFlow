import { useEffect, useRef, useState } from "react";
import { Terminal, Copy, Check } from "lucide-react";

interface BuildLogViewerProps {
  lines: string[];
}

export function BuildLogViewer({ lines }: BuildLogViewerProps): JSX.Element {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = useState(false);

  // Auto-scroll vers le bas à chaque nouvelle ligne
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines]);

  const handleCopy = async (): Promise<void> => {
    if (lines.length === 0) return;
    try {
      await navigator.clipboard.writeText(lines.join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // clipboard non disponible, silencieux
    }
  };

  return (
    <div className="card flex flex-col overflow-hidden">
      {/* Header terminal */}
      <div className="flex items-center justify-between border-b border-slate-700/80 bg-slate-900/80 px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <span className="h-3 w-3 rounded-full bg-red-500/80" />
            <span className="h-3 w-3 rounded-full bg-amber-500/80" />
            <span className="h-3 w-3 rounded-full bg-emerald-500/80" />
          </div>
          <div className="flex items-center gap-2 ml-2">
            <Terminal className="h-4 w-4 text-slate-400" />
            <span className="font-mono text-xs text-slate-400">build.log</span>
          </div>
        </div>
        <button
          disabled={lines.length === 0}
          onClick={() => void handleCopy()}
          className="flex items-center gap-1.5 rounded-btn bg-slate-800/60 px-2.5 py-1 text-xs text-slate-400 transition-all duration-200 hover:bg-slate-700/60 hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
          title="Copier les logs"
        >
          {copied ? (
            <>
              <Check className="h-3.5 w-3.5 text-emerald-400" />
              <span className="text-emerald-400">Copié !</span>
            </>
          ) : (
            <>
              <Copy className="h-3.5 w-3.5" />
              <span>Copier les logs</span>
            </>
          )}
        </button>
      </div>

      {/* Zone de log scrollable */}
      <div
        ref={scrollRef}
        className="h-[400px] max-h-[400px] overflow-y-auto bg-slate-950 p-4 font-mono text-xs leading-relaxed"
        style={{ scrollBehavior: "smooth" }}
      >
        {lines.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center space-y-2">
              <Terminal className="mx-auto h-8 w-8 text-slate-600" />
              <p className="text-slate-500 text-xs">
                Aucun log disponible
              </p>
              <p className="text-slate-600 text-[11px]">
                Le terminal s&apos;activera lors du prochain build
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-0.5">
            {lines.map((line, index) => (
              <div
                key={index}
                className={`whitespace-pre-wrap break-all ${
                  line.startsWith("ERROR") || line.includes("Error") || line.includes("FAILED")
                    ? "text-red-400"
                    : line.startsWith("WARNING") || line.includes("WARN")
                    ? "text-amber-400"
                    : line.startsWith("SUCCESS") || line.includes("SUCCESS") || line.includes("✓")
                    ? "text-emerald-400"
                    : line.startsWith("---") || line.startsWith("===")
                    ? "text-slate-400"
                    : "text-green-400"
                }`}
              >
                <span className="mr-3 select-none text-slate-600 text-[10px]">
                  {String(index + 1).padStart(4, " ")}
                </span>
                {line || " "}
              </div>
            ))}
            {/* Curseur clignotant */}
            <div className="flex items-center gap-1 text-green-400 mt-1">
              <span className="select-none text-slate-600 text-[10px] mr-3">
                {String(lines.length + 1).padStart(4, " ")}
              </span>
              <span className="inline-block h-3.5 w-2 animate-pulse bg-green-400/80" />
            </div>
          </div>
        )}
      </div>

      {/* Footer: compteur de lignes */}
      {lines.length > 0 && (
        <div className="border-t border-slate-700/60 bg-slate-900/70 px-4 py-2">
          <p className="font-mono text-[11px] text-slate-500">
            {lines.length} ligne{lines.length > 1 ? "s" : ""}
          </p>
        </div>
      )}
    </div>
  );
}
