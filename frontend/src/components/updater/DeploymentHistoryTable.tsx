import { useState } from "react";
import { RotateCcw, Clock, User, GitCommit } from "lucide-react";
import type { DeploymentHistory, DeploymentStatus } from "../../api/updater";
import { rollbackTo } from "../../api/updater";
import { StatusBadge } from "./StatusBadge";

interface DeploymentHistoryTableProps {
  history: DeploymentHistory[];
  onRollback: () => void;
}

const PIPELINE_STATES: DeploymentStatus[] = ["fetching", "pulling", "building", "deploying"];

function isPipelineRunning(history: DeploymentHistory[]): boolean {
  return history.some((h) => PIPELINE_STATES.includes(h.status));
}

function formatDuration(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(0).padStart(2, "0");
  return `${m}m${s}s`;
}

function formatRelativeDate(iso: string): string {
  try {
    const d = new Date(iso);
    const now = Date.now();
    const diff = now - d.getTime();
    const seconds = Math.floor(diff / 1000);
    if (seconds < 60) return "À l'instant";
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `Il y a ${minutes} min`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `Il y a ${hours}h`;
    const days = Math.floor(hours / 24);
    return `Il y a ${days}j`;
  } catch {
    return iso;
  }
}

function formatAbsoluteDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString("fr-BE", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  } catch {
    return iso;
  }
}

function shortHash(hash: string | null): string {
  if (!hash) return "—";
  return hash.slice(0, 7);
}

function truncate(str: string | null, max: number): string {
  if (!str) return "—";
  if (str.length <= max) return str;
  return str.slice(0, max) + "…";
}

export function DeploymentHistoryTable({
  history,
  onRollback
}: DeploymentHistoryTableProps): JSX.Element {
  const [rollingBackId, setRollingBackId] = useState<number | null>(null);
  const [rollbackError, setRollbackError] = useState<string | null>(null);

  const pipelineRunning = isPipelineRunning(history);

  const handleRollback = async (id: number): Promise<void> => {
    setRollingBackId(id);
    setRollbackError(null);
    try {
      await rollbackTo(id);
      onRollback();
    } catch (err) {
      setRollbackError(
        err instanceof Error ? err.message : "Erreur lors du rollback."
      );
    } finally {
      setRollingBackId(null);
    }
  };

  return (
    <div className="card overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-700/60 bg-slate-900/60 px-5 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary/20 to-secondary/20">
            <Clock className="h-4 w-4 text-primary" />
          </div>
          <h2 className="font-display text-base font-semibold text-white">
            Historique des déploiements
          </h2>
        </div>
        <span className="rounded-full bg-slate-700/60 px-2.5 py-1 text-xs text-slate-300 ring-1 ring-slate-600/60">
          {history.length} entrée{history.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Error banner */}
      {rollbackError && (
        <div className="border-b border-red-500/20 bg-red-900/20 px-5 py-3">
          <p className="text-xs text-red-300">{rollbackError}</p>
        </div>
      )}

      {/* Table */}
      {history.length === 0 ? (
        <div className="flex min-h-[200px] items-center justify-center">
          <div className="text-center space-y-2">
            <Clock className="mx-auto h-8 w-8 text-slate-600" />
            <p className="text-sm text-text-muted">Aucun déploiement enregistré</p>
          </div>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700/60 bg-slate-900/40">
                <th className="px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  #
                </th>
                <th className="px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  Statut
                </th>
                <th className="px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  Commit
                </th>
                <th className="px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  Source
                </th>
                <th className="px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  Build
                </th>
                <th className="px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  Date
                </th>
                <th className="px-4 py-3 text-right text-[11px] font-semibold uppercase tracking-wide text-slate-400">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {history.map((entry) => (
                <tr
                  key={entry.id}
                  className="transition-colors duration-150 hover:bg-slate-800/30"
                >
                  {/* ID */}
                  <td className="px-4 py-3 font-mono text-xs text-slate-500">
                    #{entry.id}
                  </td>

                  {/* Status */}
                  <td className="px-4 py-3">
                    <StatusBadge status={entry.status} size="sm" />
                  </td>

                  {/* Commit info */}
                  <td className="px-4 py-3">
                    <div className="flex flex-col gap-0.5">
                      <div className="flex items-center gap-1.5">
                        <GitCommit className="h-3.5 w-3.5 shrink-0 text-slate-400" />
                        <span className="font-mono text-xs text-primary">
                          {shortHash(entry.commit_hash)}
                        </span>
                      </div>
                      <p className="max-w-[200px] truncate text-xs text-text-secondary" title={entry.commit_message ?? undefined}>
                        {truncate(entry.commit_message, 60)}
                      </p>
                      {entry.commit_author && (
                        <div className="flex items-center gap-1 text-[11px] text-slate-500">
                          <User className="h-3 w-3 shrink-0" />
                          <span>{entry.commit_author}</span>
                        </div>
                      )}
                    </div>
                  </td>

                  {/* Triggered by */}
                  <td className="px-4 py-3">
                    <span
                      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ${
                        entry.triggered_by === "manual"
                          ? "bg-purple-900/30 text-purple-300 ring-purple-500/30"
                          : "bg-slate-700/40 text-slate-300 ring-slate-600/40"
                      }`}
                    >
                      {entry.triggered_by === "manual" ? "Manuel" : "Auto"}
                    </span>
                  </td>

                  {/* Build duration */}
                  <td className="px-4 py-3 font-mono text-xs text-text-secondary">
                    {formatDuration(entry.build_duration_s)}
                  </td>

                  {/* Date */}
                  <td className="px-4 py-3">
                    <span
                      className="text-xs text-text-secondary"
                      title={formatAbsoluteDate(entry.created_at)}
                    >
                      {formatRelativeDate(entry.created_at)}
                    </span>
                  </td>

                  {/* Actions */}
                  <td className="px-4 py-3 text-right">
                    {entry.status === "success" && (
                      <button
                        disabled={pipelineRunning || rollingBackId === entry.id}
                        onClick={() => void handleRollback(entry.id)}
                        className="inline-flex items-center gap-1.5 rounded-btn bg-orange-900/20 px-2.5 py-1.5 text-xs font-medium text-orange-300 ring-1 ring-orange-500/30 transition-all duration-200 hover:bg-orange-900/40 disabled:cursor-not-allowed disabled:opacity-40"
                        title="Rollback vers ce déploiement"
                      >
                        <RotateCcw
                          className={`h-3.5 w-3.5 ${rollingBackId === entry.id ? "animate-spin" : ""}`}
                        />
                        <span>{rollingBackId === entry.id ? "…" : "Rollback"}</span>
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
