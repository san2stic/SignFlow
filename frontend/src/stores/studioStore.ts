/**
 * Zustand store for the Studio annotation workflow.
 */

import { create } from "zustand";
import {
  type AnnotationSessionWithStats,
  type CreateAnnotationData,
  type CreateSessionData,
  type UpdateAnnotationData,
  type VideoAnnotation,
  type VideoInSession,
  autoSuggestAnnotations,
  createAnnotation,
  createSession,
  deleteAnnotation,
  fetchSession,
  fetchSessions,
  fetchSessionVideos,
  fetchVideoAnnotations,
  updateAnnotation,
} from "../api/studio";

interface StudioState {
  // Data
  sessions: AnnotationSessionWithStats[];
  currentSession: AnnotationSessionWithStats | null;
  currentVideo: VideoInSession | null;
  sessionVideos: VideoInSession[];
  annotations: VideoAnnotation[];
  selectedAnnotation: VideoAnnotation | null;

  // Playback state
  currentTime: number; // ms
  isPlaying: boolean;

  // UI state
  isLoadingSessions: boolean;
  isLoadingAnnotations: boolean;
  error: string | null;

  // Session actions
  fetchSessions: () => Promise<void>;
  createSession: (data: CreateSessionData) => Promise<AnnotationSessionWithStats>;
  selectSession: (id: number) => Promise<void>;
  refreshCurrentSession: () => Promise<void>;

  // Video actions
  fetchSessionVideos: (sessionId: number) => Promise<void>;
  selectVideo: (video: VideoInSession) => void;

  // Annotation actions
  fetchAnnotations: (videoId: string, sessionId?: number) => Promise<void>;
  addAnnotation: (
    videoId: string,
    sessionId: number,
    data: CreateAnnotationData
  ) => Promise<void>;
  updateAnnotation: (
    id: number,
    data: UpdateAnnotationData
  ) => Promise<void>;
  deleteAnnotation: (id: number) => Promise<void>;
  selectAnnotation: (annotation: VideoAnnotation | null) => void;
  autoSuggestAnnotations: (videoId: string, sessionId: number) => Promise<void>;

  // Playback actions
  setCurrentTime: (time: number) => void;
  setIsPlaying: (playing: boolean) => void;

  // Misc
  clearError: () => void;
}

export const useStudioStore = create<StudioState>((set, get) => ({
  // Initial state
  sessions: [],
  currentSession: null,
  currentVideo: null,
  sessionVideos: [],
  annotations: [],
  selectedAnnotation: null,
  currentTime: 0,
  isPlaying: false,
  isLoadingSessions: false,
  isLoadingAnnotations: false,
  error: null,

  // --- Session actions ---

  fetchSessions: async () => {
    set({ isLoadingSessions: true, error: null });
    try {
      const sessions = await fetchSessions();
      set({ sessions, isLoadingSessions: false });
    } catch (err) {
      set({ error: String(err), isLoadingSessions: false });
    }
  },

  createSession: async (data: CreateSessionData) => {
    set({ error: null });
    try {
      await createSession(data);
      // Re-fetch full list with stats
      const sessions = await fetchSessions();
      set({ sessions });
      // Return the newly created session (the last one with matching name)
      const created = sessions.find((s) => s.name === data.name) ?? sessions[0];
      return created;
    } catch (err) {
      set({ error: String(err) });
      throw err;
    }
  },

  selectSession: async (id: number) => {
    set({ error: null });
    try {
      const session = await fetchSession(id);
      set({ currentSession: session });
      // Also load videos
      const videos = await fetchSessionVideos(id);
      set({ sessionVideos: videos });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  refreshCurrentSession: async () => {
    const { currentSession } = get();
    if (!currentSession) return;
    try {
      const session = await fetchSession(currentSession.id);
      set({ currentSession: session });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  // --- Video actions ---

  fetchSessionVideos: async (sessionId: number) => {
    try {
      const videos = await fetchSessionVideos(sessionId);
      set({ sessionVideos: videos });
    } catch (err) {
      set({ error: String(err) });
    }
  },

  selectVideo: (video: VideoInSession) => {
    set({ currentVideo: video, annotations: [], currentTime: 0, isPlaying: false });
  },

  // --- Annotation actions ---

  fetchAnnotations: async (videoId: string, sessionId?: number) => {
    set({ isLoadingAnnotations: true, error: null });
    try {
      const annotations = await fetchVideoAnnotations(videoId, sessionId);
      set({ annotations, isLoadingAnnotations: false });
    } catch (err) {
      set({ error: String(err), isLoadingAnnotations: false });
    }
  },

  addAnnotation: async (
    videoId: string,
    sessionId: number,
    data: CreateAnnotationData
  ) => {
    set({ error: null });
    try {
      const annotation = await createAnnotation(videoId, sessionId, data);
      set((state) => ({ annotations: [...state.annotations, annotation] }));
    } catch (err) {
      set({ error: String(err) });
      throw err;
    }
  },

  updateAnnotation: async (id: number, data: UpdateAnnotationData) => {
    set({ error: null });
    try {
      const updated = await updateAnnotation(id, data);
      set((state) => ({
        annotations: state.annotations.map((a) => (a.id === id ? updated : a)),
        selectedAnnotation:
          state.selectedAnnotation?.id === id ? updated : state.selectedAnnotation,
      }));
    } catch (err) {
      set({ error: String(err) });
      throw err;
    }
  },

  deleteAnnotation: async (id: number) => {
    set({ error: null });
    try {
      await deleteAnnotation(id);
      set((state) => ({
        annotations: state.annotations.filter((a) => a.id !== id),
        selectedAnnotation:
          state.selectedAnnotation?.id === id ? null : state.selectedAnnotation,
      }));
    } catch (err) {
      set({ error: String(err) });
      throw err;
    }
  },

  selectAnnotation: (annotation: VideoAnnotation | null) => {
    set({ selectedAnnotation: annotation });
  },

  autoSuggestAnnotations: async (videoId: string, sessionId: number) => {
    set({ error: null, isLoadingAnnotations: true });
    try {
      const suggestions = await autoSuggestAnnotations(videoId, sessionId);
      set((state) => ({
        annotations: [
          ...state.annotations,
          ...suggestions.filter(
            (s) => !state.annotations.find((a) => a.id === s.id)
          ),
        ],
        isLoadingAnnotations: false,
      }));
    } catch (err) {
      set({ error: String(err), isLoadingAnnotations: false });
    }
  },

  // --- Playback ---

  setCurrentTime: (time: number) => set({ currentTime: time }),
  setIsPlaying: (playing: boolean) => set({ isPlaying: playing }),

  // --- Misc ---

  clearError: () => set({ error: null }),
}));
