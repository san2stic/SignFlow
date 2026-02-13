interface DeploymentReadinessBadgeProps {
  ready: boolean;
  accuracy: number | null;
  threshold: number;
}

export function DeploymentReadinessBadge({
  ready,
  accuracy,
  threshold
}: DeploymentReadinessBadgeProps): JSX.Element {
  const panelClass = ready ? "bg-secondary/15 border-secondary/30" : "bg-accent/15 border-accent/30";
  const textClass = ready ? "text-secondary" : "text-accent";
  const label = ready ? "Ready to deploy" : "Below threshold";
  const icon = ready ? "OK" : "WARN";

  const detail =
    accuracy === null
      ? "Waiting for validation metrics"
      : ready
        ? `${(accuracy * 100).toFixed(1)}%`
        : `${(accuracy * 100).toFixed(1)}% < ${(threshold * 100).toFixed(0)}%`;

  return (
    <div className={`flex items-center gap-2 rounded-btn border px-3 py-2 ${panelClass}`}>
      <span className={`text-xs font-semibold uppercase tracking-wide ${textClass}`}>{icon}</span>
      <p className={`text-xs sm:text-sm ${textClass}`}>
        {label} ({detail})
      </p>
    </div>
  );
}
