/**
 * AnnotationTimeline — timeline multi-piste (signes, NMM, phrases).
 */

import { useRef } from "react";
import type { VideoAnnotation } from "../../api/studio";
import { AnnotationTrack } from "./AnnotationTrack";

interface AnnotationTimelineProps {
  annotations: VideoAnnotation[];
  durationMs: number;
  currentTimeMs: number;
  selectedAnnotationId: number | null;
  onSelectAnnotation: (ann: VideoAnnotation) => void;
  onUpdateAnnotation: (id: number, start_ms: number, end_ms: number) => void;
  onSeek: (timeMs: number) => void;
}

function msToTimecode(ms: number): string {
  const totalSeconds = ms / 1000;
  const m = Math.floor(totalSeconds / 60);
  const s = Math.floor(totalSeconds % 60);
  const cs = Math.floor((ms % 1000) / 10);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(cs).padStart(2, "0")}`;
}

export function AnnotationTimeline({
  annotations,
  durationMs,
  currentTimeMs,
  selectedAnnotationId,
  onSelectAnnotation,
  onUpdateAnnotation,
  onSeek,
}: AnnotationTimelineProps): JSX.Element {
  const scrubberRef = useRef<HTMLDivElement>(null);

  const playheadPercent = durationMs > 0 ? (currentTimeMs / durationMs) * 100 : 0;

  // Filter annotations by track type
  const signAnnotations = annotations.filter((a) => !a.nmm_tags || !Object.values(a.nmm_tags).some(Boolean));
  // NMM annotations = those with at least one NMM tag set to true
  const nmmAnnotations = annotations.filter(
    (a) => a.nmm_tags && Object.values(a.nmm_tags).some((v) => v === true)
  );
  // Grammar / phrase annotations (e.g. label starts with PHRASE or contains spaces)
  const grammarAnnotations = annotations.filter(
    (a) => a.sign_label.includes(" ") || a.sign_label.startsWith("PHRASE")
  );

  // Scrubber click
  const handleScrubberClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    onSeek(ratio * durationMs);
  };

  // Tick marks
  const tickCount = 10;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => (i / tickCount) * durationMs);

  return (
    <div className="select-none rounded-b-lg border border-slate-800/70 bg-slate-950/80">
      {/* Ruler / scrubber */}
      <div
        ref={scrubberRef}
        className="relative h-6 cursor-pointer border-b border-slate-800/60 bg-slate-900/60"
        onClick={handleScrubberClick}
      >
        {/* Ticks */}
        {ticks.map((t) => (
          <div
            key={t}
            className="absolute top-0 flex h-full flex-col items-center"
            style={{ left: `${(t / durationMs) * 100}%` }}
          >
            <div className="h-2 w-px bg-slate-600" />
            <span className="mt-0.5 text-[8px] text-slate-500">{msToTimecode(t)}</span>
          </div>
        ))}

        {/* Playhead indicator */}
        <div
          className="absolute top-0 h-full w-px bg-primary/80 shadow-[0_0_4px_rgba(99,102,241,0.8)]"
          style={{ left: `${playheadPercent}%` }}
        />
      </div>

      {/* Tracks */}
      <div className="relative">
        {/* Playhead line across tracks */}
        <div
          className="pointer-events-none absolute inset-y-0 w-px bg-primary/50"
          style={{ left: `calc(5rem + (100% - 5rem) * ${playheadPercent / 100})` }}
        />

        <AnnotationTrack
          label="SIGNES"
          annotations={signAnnotations}
          durationMs={durationMs}
          selectedAnnotationId={selectedAnnotationId}
          onSelectAnnotation={onSelectAnnotation}
          onUpdateAnnotation={onUpdateAnnotation}
          onTimelineClick={onSeek}
          color="bg-primary"
        />
        <AnnotationTrack
          label="NMM"
          annotations={nmmAnnotations}
          durationMs={durationMs}
          selectedAnnotationId={selectedAnnotationId}
          onSelectAnnotation={onSelectAnnotation}
          onUpdateAnnotation={onUpdateAnnotation}
          onTimelineClick={onSeek}
          color="bg-amber-500"
        />
        <AnnotationTrack
          label="GRAMMAIRE"
          annotations={grammarAnnotations}
          durationMs={durationMs}
          selectedAnnotationId={selectedAnnotationId}
          onSelectAnnotation={onSelectAnnotation}
          onUpdateAnnotation={onUpdateAnnotation}
          onTimelineClick={onSeek}
          color="bg-violet-500"
        />
      </div>
    </div>
  );
}
