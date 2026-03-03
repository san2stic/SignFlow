/**
 * AnnotationSegment — un segment cliquable+draggable sur la timeline.
 */

import { useRef, useState } from "react";
import type { VideoAnnotation } from "../../api/studio";
import { NMMIndicator } from "./NMMIndicator";

interface AnnotationSegmentProps {
  annotation: VideoAnnotation;
  durationMs: number;
  isSelected: boolean;
  onClick: () => void;
  onUpdate: (id: number, start_time_ms: number, end_time_ms: number) => void;
  color?: string;
}

export function AnnotationSegment({
  annotation,
  durationMs,
  isSelected,
  onClick,
  onUpdate,
  color = "bg-primary",
}: AnnotationSegmentProps): JSX.Element {
  const ref = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<"move" | "left" | "right" | null>(null);
  const startX = useRef(0);
  const startStart = useRef(0);
  const startEnd = useRef(0);

  const left = durationMs > 0 ? (annotation.start_time_ms / durationMs) * 100 : 0;
  const width =
    durationMs > 0
      ? ((annotation.end_time_ms - annotation.start_time_ms) / durationMs) * 100
      : 0;

  const handleMouseDown = (
    e: React.MouseEvent,
    handle: "move" | "left" | "right"
  ) => {
    e.stopPropagation();
    e.preventDefault();
    setDragging(handle);
    startX.current = e.clientX;
    startStart.current = annotation.start_time_ms;
    startEnd.current = annotation.end_time_ms;

    const onMove = (me: MouseEvent) => {
      if (!ref.current?.parentElement) return;
      const parentWidth = ref.current.parentElement.offsetWidth;
      const delta = ((me.clientX - startX.current) / parentWidth) * durationMs;

      let newStart = startStart.current;
      let newEnd = startEnd.current;

      if (handle === "move") {
        const dur = startEnd.current - startStart.current;
        newStart = Math.max(0, Math.min(durationMs - dur, startStart.current + delta));
        newEnd = newStart + dur;
      } else if (handle === "left") {
        newStart = Math.max(0, Math.min(startEnd.current - 100, startStart.current + delta));
      } else {
        newEnd = Math.min(durationMs, Math.max(startStart.current + 100, startEnd.current + delta));
      }

      onUpdate(annotation.id, Math.round(newStart), Math.round(newEnd));
    };

    const onUp = () => {
      setDragging(null);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  };

  const ringColor = isSelected ? "ring-2 ring-white/80" : "ring-0";
  const opacity = annotation.is_verified ? "opacity-100" : "opacity-75";

  return (
    <div
      ref={ref}
      onClick={onClick}
      onMouseDown={(e) => handleMouseDown(e, "move")}
      className={`absolute top-1 flex h-7 cursor-pointer select-none items-center overflow-hidden rounded-sm ${color} ${ringColor} ${opacity} transition-opacity hover:opacity-100 ${dragging ? "cursor-grabbing" : "cursor-grab"}`}
      style={{
        left: `${left}%`,
        width: `${Math.max(width, 1)}%`,
      }}
      title={`${annotation.sign_label} [${annotation.start_time_ms.toFixed(0)}ms – ${annotation.end_time_ms.toFixed(0)}ms]`}
    >
      {/* Left handle */}
      <div
        className="absolute left-0 top-0 h-full w-1.5 cursor-ew-resize bg-white/30 hover:bg-white/60"
        onMouseDown={(e) => handleMouseDown(e, "left")}
      />

      {/* Label */}
      <span className="pointer-events-none flex-1 truncate px-2 text-[10px] font-bold text-white">
        {annotation.sign_label}
        {annotation.nmm_tags && (
          <span className="ml-1 inline-flex">
            <NMMIndicator tags={annotation.nmm_tags} compact />
          </span>
        )}
      </span>

      {/* Right handle */}
      <div
        className="absolute right-0 top-0 h-full w-1.5 cursor-ew-resize bg-white/30 hover:bg-white/60"
        onMouseDown={(e) => handleMouseDown(e, "right")}
      />
    </div>
  );
}
