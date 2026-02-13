import { create } from "zustand";
import {
  getUnlabeledVideos,
  labelVideo,
  getSuggestions,
  bulkLabelVideos,
  UnlabeledVideo,
  SuggestedVideo
} from "../api/videos";
import { Sign } from "../api/signs";

interface LabelingStore {
  unlabeledVideos: UnlabeledVideo[];
  recentSigns: Sign[];
  selectedVideo: UnlabeledVideo | null;
  suggestions: SuggestedVideo[];
  isLoading: boolean;
  isLoadingSuggestions: boolean;
  error: string | null;
  loadUnlabeledVideos: () => Promise<void>;
  selectVideo: (video: UnlabeledVideo | null) => void;
  labelVideo: (videoId: string, signId: string) => Promise<void>;
  applySuggestions: (videoIds: string[], signId: string) => Promise<void>;
  clearSuggestions: () => void;
  refreshAfterLabel: () => Promise<void>;
  addToRecentSigns: (sign: Sign) => void;
}

export const useLabelingStore = create<LabelingStore>((set, get) => ({
  unlabeledVideos: [],
  recentSigns: [],
  selectedVideo: null,
  suggestions: [],
  isLoading: false,
  isLoadingSuggestions: false,
  error: null,

  loadUnlabeledVideos: async () => {
    set({ isLoading: true, error: null });
    try {
      const videos = await getUnlabeledVideos();
      set({ unlabeledVideos: videos, isLoading: false });
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to load unlabeled videos",
        isLoading: false
      });
    }
  },

  selectVideo: (video) => {
    set({ selectedVideo: video, suggestions: [] });
  },

  labelVideo: async (videoId: string, signId: string) => {
    try {
      // Call API to label video
      await labelVideo(videoId, signId);

      // Optimistically remove video from unlabeled list
      set((state) => ({
        unlabeledVideos: state.unlabeledVideos.filter((v) => v.id !== videoId)
      }));

      // Fetch suggestions for similar videos
      set({ isLoadingSuggestions: true });
      try {
        const result = await getSuggestions(videoId, 0.75);
        set({ suggestions: result.suggestions, isLoadingSuggestions: false });
      } catch (error) {
        // Don't fail the whole operation if suggestions fail
        console.error("Failed to load suggestions:", error);
        set({ suggestions: [], isLoadingSuggestions: false });
      }
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to label video"
      });
      throw error;
    }
  },

  applySuggestions: async (videoIds: string[], signId: string) => {
    try {
      // Call API to bulk label videos
      await bulkLabelVideos(videoIds, signId);

      // Optimistically remove videos from unlabeled list
      set((state) => ({
        unlabeledVideos: state.unlabeledVideos.filter(
          (v) => !videoIds.includes(v.id)
        ),
        suggestions: []
      }));
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to apply suggestions"
      });
      throw error;
    }
  },

  clearSuggestions: () => set({ suggestions: [] }),

  refreshAfterLabel: async () => {
    set({ selectedVideo: null, suggestions: [] });
    await get().loadUnlabeledVideos();
  },

  addToRecentSigns: (sign: Sign) => {
    set((state) => {
      // Remove if already exists
      const filtered = state.recentSigns.filter((s) => s.id !== sign.id);
      // Add to front and limit to 10
      const updated = [sign, ...filtered].slice(0, 10);
      return { recentSigns: updated };
    });
  }
}));
