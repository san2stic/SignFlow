import { useSettingsStore } from "../stores/settingsStore";

export function SettingsPage(): JSX.Element {
  const speakOnDetect = useSettingsStore((state) => state.speakOnDetect);
  const spellingMode = useSettingsStore((state) => state.spellingMode);
  const unknownThreshold = useSettingsStore((state) => state.unknownThreshold);
  const unknownFrameWindow = useSettingsStore((state) => state.unknownFrameWindow);
  const setSpeakOnDetect = useSettingsStore((state) => state.setSpeakOnDetect);
  const setSpellingMode = useSettingsStore((state) => state.setSpellingMode);
  const setUnknownThreshold = useSettingsStore((state) => state.setUnknownThreshold);
  const setUnknownFrameWindow = useSettingsStore((state) => state.setUnknownFrameWindow);

  return (
    <section className="space-y-5">
      <header className="card p-5">
        <p className="text-xs uppercase tracking-[0.16em] text-text-tertiary">Preferences</p>
        <h1 className="mt-2 font-display text-2xl font-semibold text-white">Parametres de traduction</h1>
        <p className="mt-1 text-sm text-text-secondary">
          Ajustez la sortie vocale, le mode epellation et la detection des signes inconnus.
        </p>
      </header>

      <div className="card space-y-4 p-5">
        <Toggle
          label="Lecture vocale a chaque prediction fiable"
          description="Prononce automatiquement le signe reconnu si la confiance est suffisante."
          enabled={speakOnDetect}
          onChange={(value) => setSpeakOnDetect(value)}
        />
        <Toggle
          label="Mode epellation"
          description="Affiche les lettres separees pour les signes detectes."
          enabled={spellingMode}
          onChange={(value) => setSpellingMode(value)}
        />

        <label className="block space-y-2 text-sm text-text-secondary">
          <span>Seuil de confiance pour signaler un signe inconnu</span>
          <input
            type="number"
            min={0.1}
            max={0.9}
            step={0.05}
            className="field-input max-w-[220px]"
            value={unknownThreshold}
            onChange={(event) => setUnknownThreshold(Number(event.target.value))}
          />
        </label>

        <label className="block space-y-2 text-sm text-text-secondary">
          <span>Nombre d images consecutives avant affichage du prompt inconnu</span>
          <input
            type="number"
            min={4}
            max={30}
            step={1}
            className="field-input max-w-[220px]"
            value={unknownFrameWindow}
            onChange={(event) => setUnknownFrameWindow(Number(event.target.value))}
          />
        </label>
      </div>
    </section>
  );
}

function Toggle({
  label,
  description,
  enabled,
  onChange
}: {
  label: string;
  description: string;
  enabled: boolean;
  onChange: (value: boolean) => void;
}): JSX.Element {
  return (
    <div className="rounded-btn border border-slate-700/80 bg-slate-900/55 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-white">{label}</p>
          <p className="mt-1 text-xs text-text-tertiary">{description}</p>
        </div>
        <button
          type="button"
          className={`touch-btn px-3 py-2 text-xs ${enabled ? "bg-gradient-to-r from-primary to-secondary text-slate-950" : "bg-slate-700 text-white"}`}
          onClick={() => onChange(!enabled)}
          aria-pressed={enabled}
        >
          {enabled ? "ACTIF" : "INACTIF"}
        </button>
      </div>
    </div>
  );
}
