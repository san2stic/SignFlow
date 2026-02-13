import { create } from "zustand";

interface SettingsStore {
  speakOnDetect: boolean;
  spellingMode: boolean;
  unknownThreshold: number;
  unknownFrameWindow: number;
  setSpeakOnDetect: (value: boolean) => void;
  setSpellingMode: (value: boolean) => void;
  setUnknownThreshold: (value: number) => void;
  setUnknownFrameWindow: (value: number) => void;
}

export const useSettingsStore = create<SettingsStore>((set) => ({
  speakOnDetect: false,
  spellingMode: false,
  unknownThreshold: 0.5,
  unknownFrameWindow: 12,
  setSpeakOnDetect: (value) => set({ speakOnDetect: value }),
  setSpellingMode: (value) => set({ spellingMode: value }),
  setUnknownThreshold: (value) => set({ unknownThreshold: value }),
  setUnknownFrameWindow: (value) => set({ unknownFrameWindow: value })
}));
