/**
 * NMMIndicator — badge visuel pour les marqueurs non-manuels (NMM).
 */

import type { NMMTags } from "../../api/studio";

interface NMMIndicatorProps {
  tags: NMMTags;
  compact?: boolean;
}

interface NMMDef {
  key: keyof NMMTags;
  label: string;
  icon: string;
}

const NMM_DEFS: NMMDef[] = [
  { key: "polar_question", label: "Q?", icon: "🤨" },
  { key: "wh_question", label: "WH?", icon: "🧐" },
  { key: "negation", label: "NEG", icon: "🚫" },
  { key: "eyebrow_raise", label: "↑brow", icon: "⬆️" },
  { key: "eyebrow_furrow", label: "↓brow", icon: "⬇️" },
  { key: "head_nod", label: "nod", icon: "👍" },
  { key: "head_shake", label: "shake", icon: "↔️" },
];

export function NMMIndicator({ tags, compact = false }: NMMIndicatorProps): JSX.Element {
  const activeNmm = NMM_DEFS.filter((d) => tags[d.key] === true);
  const hasMouthGesture = tags.mouth_gesture && tags.mouth_gesture !== "";

  if (activeNmm.length === 0 && !hasMouthGesture) {
    return <></>;
  }

  if (compact) {
    return (
      <div className="flex flex-wrap gap-0.5">
        {activeNmm.map((d) => (
          <span
            key={d.key}
            title={d.label}
            className="flex h-5 w-5 items-center justify-center rounded text-[10px]"
          >
            {d.icon}
          </span>
        ))}
        {hasMouthGesture && (
          <span
            title={`Bouche: ${tags.mouth_gesture}`}
            className="flex h-5 items-center justify-center rounded px-1 text-[9px] text-amber-300"
          >
            👄
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-wrap gap-1">
      {activeNmm.map((d) => (
        <span
          key={d.key}
          className="flex items-center gap-1 rounded-full bg-amber-500/20 px-2 py-0.5 text-[11px] font-medium text-amber-300 ring-1 ring-amber-500/30"
        >
          <span>{d.icon}</span>
          <span>{d.label}</span>
        </span>
      ))}
      {hasMouthGesture && (
        <span className="flex items-center gap-1 rounded-full bg-rose-500/20 px-2 py-0.5 text-[11px] font-medium text-rose-300 ring-1 ring-rose-500/30">
          <span>👄</span>
          <span>{tags.mouth_gesture}</span>
        </span>
      )}
    </div>
  );
}
