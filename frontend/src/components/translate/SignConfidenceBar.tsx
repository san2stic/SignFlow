/**
 * SignConfidenceBar — Barre de confiance colorée pour la traduction en temps réel.
 *
 * Couleurs :
 *   Rouge  < 0.5
 *   Jaune  0.5 – 0.7
 *   Vert   > 0.7
 */

interface Props {
  /** Valeur de confiance normalisée entre 0 et 1. */
  confidence: number;
  /** Afficher le pourcentage textuel à côté de la barre. */
  showLabel?: boolean;
  /** Hauteur de la barre en pixels (défaut : 6). */
  height?: number;
  /** Classes Tailwind supplémentaires sur le conteneur. */
  className?: string;
}

function confidenceColor(value: number): string {
  if (value >= 0.7) return "from-emerald-400 to-green-500";
  if (value >= 0.5) return "from-amber-400 to-yellow-500";
  return "from-red-400 to-rose-500";
}

function confidenceLabel(value: number): string {
  if (value >= 0.7) return "Haute";
  if (value >= 0.5) return "Moyenne";
  return "Faible";
}

export function SignConfidenceBar({
  confidence,
  showLabel = false,
  height = 6,
  className = "",
}: Props): JSX.Element {
  const clamped = Math.max(0, Math.min(1, confidence));
  const pct = Math.round(clamped * 100);
  const gradient = confidenceColor(clamped);
  const labelText = confidenceLabel(clamped);

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {/* Track */}
      <div
        className="relative flex-1 overflow-hidden rounded-full bg-surface-secondary"
        style={{ height }}
        role="progressbar"
        aria-valuenow={pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Confiance : ${pct}%`}
      >
        {/* Fill */}
        <div
          className={`absolute left-0 top-0 h-full rounded-full bg-gradient-to-r ${gradient} transition-[width] duration-300 ease-out`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {showLabel && (
        <span className="min-w-[3.5rem] text-right font-mono text-xs text-text-tertiary">
          {pct}%{" "}
          <span
            className={
              clamped >= 0.7
                ? "text-emerald-400"
                : clamped >= 0.5
                  ? "text-amber-400"
                  : "text-red-400"
            }
          >
            {labelText}
          </span>
        </span>
      )}
    </div>
  );
}
