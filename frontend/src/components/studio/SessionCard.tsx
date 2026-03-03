/**
 * SessionCard — carte dans la liste des sessions d'annotation.
 */

import { Calendar, CheckCircle, Film, Tag } from "lucide-react";
import type { AnnotationSessionWithStats } from "../../api/studio";

interface SessionCardProps {
  session: AnnotationSessionWithStats;
  onClick: () => void;
}

const STATUS_STYLES: Record<string, string> = {
  active: "bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-500/40",
  completed: "bg-sky-500/20 text-sky-300 ring-1 ring-sky-500/40",
  archived: "bg-slate-500/20 text-slate-400 ring-1 ring-slate-500/40",
};

const STATUS_LABELS: Record<string, string> = {
  active: "Active",
  completed: "Terminée",
  archived: "Archivée",
};

export function SessionCard({ session, onClick }: SessionCardProps): JSX.Element {
  const statusStyle = STATUS_STYLES[session.status] ?? STATUS_STYLES.active;
  const formattedDate = new Date(session.created_at).toLocaleDateString("fr-BE", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });

  return (
    <button
      onClick={onClick}
      className="w-full rounded-xl border border-slate-700/70 bg-slate-900/60 p-4 text-left transition-all hover:border-primary/50 hover:bg-slate-800/60 hover:shadow-lg hover:shadow-primary/5"
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <h3 className="flex-1 truncate font-semibold text-white">{session.name}</h3>
        <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${statusStyle}`}>
          {STATUS_LABELS[session.status] ?? session.status}
        </span>
      </div>

      {/* Description */}
      {session.description && (
        <p className="mt-1 line-clamp-2 text-sm text-slate-400">{session.description}</p>
      )}

      {/* Stats */}
      <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-400">
        <span className="flex items-center gap-1">
          <Film className="h-3.5 w-3.5" />
          {session.video_count} vidéo{session.video_count !== 1 ? "s" : ""}
        </span>
        <span className="flex items-center gap-1">
          <Tag className="h-3.5 w-3.5" />
          {session.annotation_count} annotation{session.annotation_count !== 1 ? "s" : ""}
        </span>
        {session.verified_count > 0 && (
          <span className="flex items-center gap-1 text-emerald-400">
            <CheckCircle className="h-3.5 w-3.5" />
            {session.verified_count} vérifiée{session.verified_count !== 1 ? "s" : ""}
          </span>
        )}
        <span className="flex items-center gap-1 ml-auto">
          <Calendar className="h-3.5 w-3.5" />
          {formattedDate}
        </span>
      </div>

      {/* Coverage progress bar */}
      {session.video_count > 0 && (
        <div className="mt-3">
          <div className="mb-1 flex justify-between text-[11px] text-slate-500">
            <span>Couverture vérifiée</span>
            <span>{session.coverage_percent}%</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-primary to-secondary transition-all"
              style={{ width: `${session.coverage_percent}%` }}
            />
          </div>
        </div>
      )}
    </button>
  );
}
