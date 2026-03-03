import type { DeploymentStatus } from "../../api/updater";

interface StatusBadgeProps {
  status: DeploymentStatus;
  size?: "sm" | "md" | "lg";
}

const STATUS_CONFIG: Record<
  DeploymentStatus,
  { label: string; classes: string; dot: string; pulse: boolean }
> = {
  idle: {
    label: "Inactif",
    classes: "bg-slate-700/60 text-slate-300 ring-1 ring-slate-600/60",
    dot: "bg-slate-400",
    pulse: false
  },
  fetching: {
    label: "Récupération",
    classes: "bg-sky-900/40 text-sky-300 ring-1 ring-sky-500/40",
    dot: "bg-sky-400",
    pulse: true
  },
  pulling: {
    label: "Pull en cours",
    classes: "bg-sky-900/40 text-sky-300 ring-1 ring-sky-500/40",
    dot: "bg-sky-400",
    pulse: true
  },
  building: {
    label: "Build en cours",
    classes: "bg-amber-900/40 text-amber-300 ring-1 ring-amber-500/40",
    dot: "bg-amber-400",
    pulse: true
  },
  deploying: {
    label: "Déploiement",
    classes: "bg-blue-900/40 text-blue-300 ring-1 ring-blue-500/40",
    dot: "bg-blue-400",
    pulse: true
  },
  success: {
    label: "Succès",
    classes: "bg-emerald-900/40 text-emerald-300 ring-1 ring-emerald-500/40",
    dot: "bg-emerald-400",
    pulse: false
  },
  error: {
    label: "Erreur",
    classes: "bg-red-900/40 text-red-300 ring-1 ring-red-500/40",
    dot: "bg-red-400",
    pulse: false
  },
  rolled_back: {
    label: "Rollback",
    classes: "bg-orange-900/40 text-orange-300 ring-1 ring-orange-500/40",
    dot: "bg-orange-400",
    pulse: false
  }
};

const SIZE_CLASSES = {
  sm: "px-2 py-0.5 text-xs gap-1.5",
  md: "px-3 py-1 text-sm gap-2",
  lg: "px-4 py-1.5 text-sm gap-2"
};

const DOT_SIZE_CLASSES = {
  sm: "h-1.5 w-1.5",
  md: "h-2 w-2",
  lg: "h-2.5 w-2.5"
};

export function StatusBadge({ status, size = "md" }: StatusBadgeProps): JSX.Element {
  const config = STATUS_CONFIG[status];

  return (
    <span
      className={`inline-flex items-center rounded-full font-medium transition-all duration-200 ${config.classes} ${SIZE_CLASSES[size]}`}
    >
      <span
        className={`rounded-full ${DOT_SIZE_CLASSES[size]} ${config.dot} ${config.pulse ? "animate-pulse" : ""} shrink-0`}
      />
      {config.label}
    </span>
  );
}
