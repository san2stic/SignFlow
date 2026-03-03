import { useState } from "react";
import { GitBranch, Clock, RefreshCw, AlertCircle, CheckCircle2, Wifi, WifiOff } from "lucide-react";
import type { UpdaterStatus } from "../../api/updater";
import { triggerUpdate } from "../../api/updater";
import type { DeploymentStatus } from "../../api/updater";

interface GitStatusPanelProps {
  status: UpdaterStatus | null;
  onTrigger: () => void;
}

const PIPELINE_STATES: DeploymentStatus[] = ["fetching", "pulling", "building", "deploying"];

function isPipelineRunning(state: DeploymentStatus | undefined): boolean {
  return state !== undefined && PIPELINE_STATES.includes(state);
}

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString("fr-BE", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit"
    });
  } catch {
    return iso;
  }
}

function shortHash(hash: string | null): string {
  if (!hash) return "—";
  return hash.slice(0, 7);
}

export function GitStatusPanel({ status, onTrigger }: GitStatusPanelProps): JSX.Element {
  const [triggering, setTriggering] = useState(false);
  const [triggerError, setTriggerError] = useState<string | null>(null);

  const isOutOfDate =
    status?.local_commit !== null &&
    status?.remote_commit !== null &&
    status?.local_commit !== status?.remote_commit;

  const pipelineRunning = isPipelineRunning(status?.state);

  const handleTrigger = async (): Promise<void> => {
    setTriggering(true);
    setTriggerError(null);
    try {
      await triggerUpdate();
      onTrigger();
    } catch (err) {
      setTriggerError(err instanceof Error ? err.message : "Erreur lors du déclenchement.");
    } finally {
      setTriggering(false);
    }
  };

  return (
    <div className="card p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-primary/20 to-secondary/20">
          <GitBranch className="h-5 w-5 text-primary" />
        </div>
        <div>
          <h2 className="font-display text-lg font-semibold text-white">État Git</h2>
          <p className="text-xs text-text-secondary">Branche & commits</p>
        </div>
        {/* Enabled/Disabled badge */}
        <span
          className={`ml-auto inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ${
            status?.auto_update_enabled
              ? "bg-emerald-900/30 text-emerald-300 ring-emerald-500/30"
              : "bg-slate-700/40 text-slate-400 ring-slate-600/40"
          }`}
        >
          {status?.auto_update_enabled ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
          {status?.auto_update_enabled ? "Activé" : "Désactivé"}
        </span>
      </div>

      {/* Branch info */}
      <div className="rounded-btn bg-slate-900/60 p-3 space-y-3">
        {/* Branch */}
        <div className="flex items-center gap-2">
          <GitBranch className="h-4 w-4 shrink-0 text-primary" />
          <span className="text-xs text-text-secondary">Branche</span>
          <span className="ml-auto font-mono text-sm font-semibold text-white">
            {status?.git_branch ?? "—"}
          </span>
        </div>

        {/* Local commit */}
        <div className="flex items-start gap-2">
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
          <span className="text-xs text-text-secondary">Commit local</span>
          <span className="ml-auto font-mono text-sm text-emerald-300">
            {shortHash(status?.local_commit ?? null)}
          </span>
        </div>

        {/* Remote commit */}
        <div className="flex items-start gap-2">
          <span className="mt-0.5 h-4 w-4 shrink-0 flex items-center justify-center">
            {isOutOfDate ? (
              <AlertCircle className="h-4 w-4 text-amber-400" />
            ) : (
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
            )}
          </span>
          <span className="text-xs text-text-secondary">Commit remote</span>
          <span
            className={`ml-auto font-mono text-sm ${isOutOfDate ? "text-amber-300" : "text-emerald-300"}`}
          >
            {shortHash(status?.remote_commit ?? null)}
          </span>
        </div>

        {/* Out-of-date indicator */}
        {isOutOfDate && (
          <div className="rounded-btn bg-amber-900/20 border border-amber-500/20 px-3 py-2">
            <p className="text-xs text-amber-300 flex items-center gap-2">
              <AlertCircle className="h-3.5 w-3.5 shrink-0" />
              Une mise à jour est disponible
            </p>
          </div>
        )}
      </div>

      {/* Poll & last check info */}
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-btn bg-slate-900/60 p-3">
          <p className="mb-1 text-[11px] uppercase tracking-wide text-text-tertiary">Dernier check</p>
          <p className="flex items-center gap-1.5 text-xs text-text-secondary">
            <Clock className="h-3.5 w-3.5 shrink-0" />
            {formatDate(status?.last_check_at ?? null)}
          </p>
        </div>
        <div className="rounded-btn bg-slate-900/60 p-3">
          <p className="mb-1 text-[11px] uppercase tracking-wide text-text-tertiary">Polling</p>
          <p className="text-xs text-text-secondary">
            {status?.poll_interval_s !== undefined
              ? `Toutes les ${status.poll_interval_s}s`
              : "—"}
          </p>
        </div>
      </div>

      {/* Trigger error */}
      {triggerError && (
        <div className="rounded-btn border border-red-500/30 bg-red-900/20 px-3 py-2">
          <p className="text-xs text-red-300">{triggerError}</p>
        </div>
      )}

      {/* Deploy button */}
      <button
        disabled={triggering || pipelineRunning || !status?.auto_update_enabled}
        onClick={() => void handleTrigger()}
        className="touch-btn w-full bg-gradient-to-r from-primary/30 to-secondary/30 text-primary ring-1 ring-primary/40 hover:from-primary/40 hover:to-secondary/40 disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-200 flex items-center justify-center gap-2"
      >
        <RefreshCw className={`h-4 w-4 ${triggering || pipelineRunning ? "animate-spin" : ""}`} />
        <span className="font-semibold">
          {triggering ? "Déclenchement…" : pipelineRunning ? "Pipeline en cours…" : "🚀 Déployer maintenant"}
        </span>
      </button>
    </div>
  );
}
