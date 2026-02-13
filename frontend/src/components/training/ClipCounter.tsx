interface ClipCounterClip {
  quality: "valid" | "low";
}

interface ClipCounterProps {
  clips: ClipCounterClip[];
  minClips: number;
}

export function ClipCounter({ clips, minClips }: ClipCounterProps): JSX.Element {
  const validCount = clips.filter((clip) => clip.quality === "valid").length;
  const slots = Array.from({ length: Math.max(minClips, clips.length) });

  return (
    <div className="flex flex-wrap items-center gap-2">
      {slots.map((_, index) => {
        const clip = clips[index];
        const tone = !clip ? "bg-slate-700/80" : clip.quality === "valid" ? "bg-secondary" : "bg-red-500";
        const label = !clip ? "Pending slot" : clip.quality === "valid" ? "Valid clip" : "Low quality clip";

        return (
          <span
            key={index}
            className={`h-8 w-8 rounded-full border border-slate-600 ${tone}`}
            aria-label={label}
            title={label}
          />
        );
      })}
      <span className="text-xs text-slate-400 sm:text-sm">
        {validCount}/{minClips} minimum
      </span>
    </div>
  );
}
