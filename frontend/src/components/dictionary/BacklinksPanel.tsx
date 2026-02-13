import type { Backlink } from "../../api/signs";

interface BacklinksPanelProps {
  backlinks: Backlink[];
  onSelect?: (signId: string) => void;
}

export function BacklinksPanel({ backlinks, onSelect }: BacklinksPanelProps): JSX.Element {
  return (
    <div className="card p-4">
      <h3 className="font-heading text-lg">Backlinks</h3>
      {backlinks.length === 0 ? (
        <p className="mt-2 text-sm text-slate-400">No backlinks yet.</p>
      ) : (
        <ul className="mt-2 space-y-1 text-sm text-slate-200">
          {backlinks.map((item) => (
            <li key={item.id}>
              <button
                className="text-left text-primary hover:underline"
                onClick={() => onSelect?.(item.id)}
                type="button"
              >
                {item.name}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
