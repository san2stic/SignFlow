/**
 * AnnotationTrack — une piste dans la timeline multi-piste.
 */

import type { VideoAnnotation } from "../../api/studio";
import { AnnotationSegment } from "./AnnotationSegment";

interface AnnotationTrackProps {
  label: string;
  annotations: VideoAnnotation[];
  durationMs: number;
  selectedAnnotationId: number | null;
  onSelectAnnotation: (ann: VideoAnnotation) => void;
  onUpdateAnnotation: (id: number, start_ms: number, end_ms: number) => void;
  onTimelineClick?: (timeMs: number) => void;
  color?: string;
}

const TRACK_COLORS: Record<string, string> = {
  SIGNES: "bg-primary",
  NMM: "bg-amber-500",
  GRAMMAIRE: "bg-violet-500",
};

export function AnnotationTrack({
  label,
  annotations,
  durationMs,
  selectedAnnotationId,
  onSelectAnnotation,
  onUpdateAnnotation,
  onTimelineClick,
  color,
}: AnnotationTrackProps): JSX.Element {
  const trackColor = color ?? TRACK_COLORS[label] ?? "bg-slate-500";

  const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!onTimelineClick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    onTimelineClick(ratio * durationMs);
  };

  return (
    <div className="flex items-center border-b border-slate-800/60 last:border-0">
      {/* Track label */}
      <div className="w-20 shrink-0 py-1 pr-2 text-right text-[11px] font-medium text-slate-400">
        {label}
      </div>

      {/* Track body */}
      <div
        className="relative flex-1 cursor-crosshair overflow-hidden"
        style={{ height: 36 }}
        onClick={handleTrackClick}
      >
        {/* Background grid */}
        <div className="absolute inset-0 bg-slate-900/40" />

        {/* Segments */}
        {annotations.map((ann) => (
          <AnnotationSegment
            key={ann.id}
            annotation={ann}
            durationMs={durationMs}
            isSelected={ann.id === selectedAnnotationId}
            onClick={() => onSelectAnnotation(ann)}
            onUpdate={onUpdateAnnotation}
            color={trackColor}
          />
        ))}
      </div>
    </div>
  );
}
