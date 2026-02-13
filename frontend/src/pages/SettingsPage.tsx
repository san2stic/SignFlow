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
    <section className="space-y-4">
      <h1 className="font-heading text-2xl">Settings</h1>

      <div className="card space-y-4 p-4">
        <Toggle
          label="Speech output on detection"
          enabled={speakOnDetect}
          onChange={(value) => setSpeakOnDetect(value)}
        />
        <Toggle
          label="Spelling mode"
          enabled={spellingMode}
          onChange={(value) => setSpellingMode(value)}
        />
        <label className="flex flex-col gap-2 rounded-btn border border-slate-700 bg-slate-900/50 px-3 py-3 text-sm">
          Unknown confidence threshold
          <input
            type="number"
            min={0.1}
            max={0.9}
            step={0.05}
            className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-2 text-base"
            value={unknownThreshold}
            onChange={(event) => setUnknownThreshold(Number(event.target.value))}
          />
        </label>
        <label className="flex flex-col gap-2 rounded-btn border border-slate-700 bg-slate-900/50 px-3 py-3 text-sm">
          Unknown frame window
          <input
            type="number"
            min={4}
            max={30}
            step={1}
            className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-2 text-base"
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
  enabled,
  onChange
}: {
  label: string;
  enabled: boolean;
  onChange: (value: boolean) => void;
}): JSX.Element {
  return (
    <label className="flex items-center justify-between gap-3 rounded-btn border border-slate-700 bg-slate-900/50 px-3 py-3">
      <span className="text-sm">{label}</span>
      <button
        className={`touch-btn px-3 py-2 text-xs ${enabled ? "bg-secondary text-slate-950" : "bg-slate-700 text-white"}`}
        onClick={() => onChange(!enabled)}
      >
        {enabled ? "ON" : "OFF"}
      </button>
    </label>
  );
}
