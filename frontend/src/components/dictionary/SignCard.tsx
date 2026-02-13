import type { Sign } from "../../api/signs";

interface SignCardProps {
  sign: Sign;
  onSelect?: (sign: Sign) => void;
  onTrain?: (sign: Sign) => void;
}

export function SignCard({ sign, onSelect, onTrain }: SignCardProps): JSX.Element {
  return (
    <article className="card w-full p-4">
      <button type="button" onClick={() => onSelect?.(sign)} className="w-full text-left">
        <h3 className="font-heading text-lg">{sign.name}</h3>
        <p className="mt-1 line-clamp-2 text-sm text-slate-300">{sign.description ?? "No description"}</p>
        <div className="mt-2 flex flex-wrap gap-1 text-xs text-slate-400">
          {sign.tags.map((tag) => (
            <span key={tag} className="rounded-full bg-slate-700/60 px-2 py-1">
              #{tag}
            </span>
          ))}
        </div>
        <div className="mt-2 text-xs text-slate-400">
          {sign.video_count} videos • {Math.round((sign.accuracy ?? 0) * 100)}% acc
        </div>
      </button>
      <div className="mt-3">
        <button type="button" className="touch-btn bg-primary text-white" onClick={() => onTrain?.(sign)}>
          Train
        </button>
      </div>
    </article>
  );
}
