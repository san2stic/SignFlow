interface QualityIndicatorProps {
  visibleHands: number;
}

export function QualityIndicator({ visibleHands }: QualityIndicatorProps): JSX.Element {
  const status =
    visibleHands >= 2
      ? {
          dotColor: "bg-secondary",
          panelColor: "bg-secondary/15 border-secondary/30",
          text: "Perfect - both hands detected"
        }
      : visibleHands === 1
        ? {
            dotColor: "bg-accent",
            panelColor: "bg-accent/15 border-accent/30",
            text: "Good - one hand detected"
          }
        : {
            dotColor: "bg-red-500",
            panelColor: "bg-red-500/15 border-red-500/30",
            text: "No hands detected"
          };

  return (
    <div className={`flex items-center gap-2 rounded-btn border px-3 py-2 ${status.panelColor}`}>
      <span className={`h-2.5 w-2.5 rounded-full ${status.dotColor}`} aria-hidden />
      <p className="text-xs text-slate-200 sm:text-sm">{status.text}</p>
    </div>
  );
}
