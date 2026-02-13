interface ConfidenceBadgeProps {
  confidence: number;
}

export function ConfidenceBadge({ confidence }: ConfidenceBadgeProps): JSX.Element {
  const pct = Math.round(confidence * 100);
  const color = pct >= 70 ? "bg-secondary" : pct >= 50 ? "bg-accent" : "bg-red-500";

  return (
    <div className="space-y-2">
      <p className="text-sm text-slate-300">Confiance: {pct}%</p>
      <div className="h-2 w-full rounded-full bg-slate-700">
        <div className={`h-2 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
