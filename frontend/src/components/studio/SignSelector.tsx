/**
 * SignSelector — dropdown/search pour choisir un label de signe.
 * Récupère la liste depuis le dictionnaire existant (/api/v1/signs).
 */

import { Check, ChevronsUpDown, Search } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { apiFetch } from "../../api/client";

interface SignOption {
  id: string;
  name: string;
  category?: string | null;
}

interface SignSelectorProps {
  value: string;
  onChange: (label: string) => void;
  placeholder?: string;
  disabled?: boolean;
}

export function SignSelector({
  value,
  onChange,
  placeholder = "Sélectionner un signe…",
  disabled = false,
}: SignSelectorProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [signs, setSigns] = useState<SignOption[]>([]);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Load signs from dictionary
  useEffect(() => {
    setLoading(true);
    apiFetch<{ items?: SignOption[]; data?: SignOption[] } | SignOption[]>(
      "/signs?limit=500"
    )
      .then((res) => {
        if (Array.isArray(res)) {
          setSigns(res);
        } else if ("items" in res && Array.isArray(res.items)) {
          setSigns(res.items);
        } else if ("data" in res && Array.isArray(res.data)) {
          setSigns(res.data);
        }
      })
      .catch(() => {
        // silently fail — user can type custom labels
      })
      .finally(() => setLoading(false));
  }, []);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const filtered =
    query.trim() === ""
      ? signs
      : signs.filter((s) =>
          s.name.toLowerCase().includes(query.toLowerCase())
        );

  const handleSelect = (name: string) => {
    onChange(name.toUpperCase());
    setQuery("");
    setOpen(false);
  };

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        disabled={disabled}
        onClick={() => {
          setOpen((o) => !o);
          setTimeout(() => inputRef.current?.focus(), 50);
        }}
        className="flex w-full items-center justify-between gap-2 rounded-lg border border-slate-700 bg-slate-800/70 px-3 py-2 text-sm text-white transition hover:border-primary/60 disabled:opacity-50"
      >
        <span className={value ? "text-white" : "text-slate-400"}>
          {value || placeholder}
        </span>
        <ChevronsUpDown className="h-4 w-4 shrink-0 text-slate-400" />
      </button>

      {open && (
        <div className="absolute z-50 mt-1 w-full rounded-xl border border-slate-700 bg-slate-900 shadow-2xl">
          {/* Search input */}
          <div className="flex items-center gap-2 border-b border-slate-800 px-3 py-2">
            <Search className="h-4 w-4 shrink-0 text-slate-400" />
            <input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && query.trim()) {
                  handleSelect(query.trim());
                }
                if (e.key === "Escape") setOpen(false);
              }}
              placeholder="Chercher ou taper un label…"
              className="flex-1 bg-transparent text-sm text-white outline-none placeholder:text-slate-500"
            />
          </div>

          {/* List */}
          <div className="max-h-56 overflow-y-auto">
            {loading ? (
              <div className="px-4 py-3 text-sm text-slate-400">Chargement…</div>
            ) : filtered.length === 0 ? (
              <button
                onClick={() => handleSelect(query.trim() || "UNKNOWN")}
                className="w-full px-4 py-2 text-left text-sm text-primary hover:bg-slate-800"
              >
                Créer : <strong>{query.toUpperCase() || "UNKNOWN"}</strong>
              </button>
            ) : (
              filtered.slice(0, 50).map((sign) => (
                <button
                  key={sign.id}
                  onClick={() => handleSelect(sign.name)}
                  className="flex w-full items-center justify-between px-4 py-2 text-left text-sm text-slate-200 hover:bg-slate-800"
                >
                  <span>{sign.name}</span>
                  {sign.name.toUpperCase() === value.toUpperCase() && (
                    <Check className="h-4 w-4 text-primary" />
                  )}
                </button>
              ))
            )}
            {/* Custom label option */}
            {query.trim() !== "" && filtered.length > 0 && (
              <button
                onClick={() => handleSelect(query.trim())}
                className="w-full border-t border-slate-800 px-4 py-2 text-left text-sm text-primary hover:bg-slate-800"
              >
                Utiliser : <strong>{query.toUpperCase()}</strong>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
