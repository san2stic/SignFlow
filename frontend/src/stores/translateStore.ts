import { create } from "zustand";

const MIN_VISIBLE_PREDICTION_CONFIDENCE = 0.2;
const MIN_STICKY_CONFIDENCE = 0.55;
const STICKY_PREDICTION_HOLD_MS = 3000;
const TRANSIENT_PREDICTION_HOLD_MS = 1200;
const MAX_CONVERSATION_HISTORY = 50;

export interface PredictionState {
  prediction: string;
  confidence: number;
  sentenceBuffer: string;
  alternatives: Array<{ sign: string; confidence: number }>;
}

// ---------------------------------------------------------------------------
// Types pour la Phase 5 — Conversation
// ---------------------------------------------------------------------------

export interface ConversationTurn {
  id: number;
  text: string;
  raw_signs: string[];
  grammar_tags: string[];
  confidence: number;
  start_time: number;
  end_time: number;
  speaker: string;
  // Champs déduits au frontend
  turnIdStr?: string;
}

export interface ContextSummary {
  turn_count: number;
  last_turns: Array<{ id: number; text: string; time: number; confidence: number }>;
  spatial_referents: Record<string, string>;
  current_subjects: string[];
  session_age_seconds: number;
}

// ---------------------------------------------------------------------------
// Store principal
// ---------------------------------------------------------------------------

interface TranslateStore {
  // Existant (conserver pour rétrocompatibilité)
  live: PredictionState;
  displayedPrediction: string;
  displayedConfidence: number;
  displayUntilMs: number;
  history: string[];

  // Phase 5 — Conversation
  conversationHistory: ConversationTurn[];
  currentTurnId: number | null;
  translatedSentence: string;
  grammarTags: string[];
  translationMode: string;
  conversationContext: ContextSummary | null;
  isNewTurn: boolean;
  /** Phrase en cours de construction (signes reçus mais tour pas encore clôturé) */
  buildingBuffer: string;

  // Actions existantes
  setLive: (value: PredictionState) => void;
  reset: () => void;

  // Actions Phase 5
  addConversationTurn: (turn: ConversationTurn) => void;
  resetConversation: () => void;
  setGrammarMode: (mode: string) => void;
  setConversationContext: (ctx: ContextSummary) => void;
  handleSentenceComplete: (payload: {
    sentence: string;
    translated_sentence: string;
    grammar_tags: string[];
    confidence: number;
    raw_signs: string[];
    translation_mode: string;
    turn_id: number | null;
  }) => void;
  setNewTurn: (turnId: number) => void;
}

export const useTranslateStore = create<TranslateStore>((set) => ({
  live: { prediction: "NONE", confidence: 0, sentenceBuffer: "", alternatives: [] },
  displayedPrediction: "NONE",
  displayedConfidence: 0,
  displayUntilMs: 0,
  history: [],

  // Phase 5 defaults
  conversationHistory: [],
  currentTurnId: null,
  translatedSentence: "",
  grammarTags: [],
  translationMode: "rules",
  conversationContext: null,
  isNewTurn: false,
  buildingBuffer: "",

  // -------------------------------------------------------------------------
  // setLive — met à jour l'état en temps réel depuis les frames WS normales
  // -------------------------------------------------------------------------
  setLive: (value) =>
    set((state) => {
      const now = Date.now();
      const isNonEmptyPrediction = value.prediction !== "NONE" && value.prediction !== "RECORDING";
      const isVisiblePrediction = isNonEmptyPrediction && value.confidence >= MIN_VISIBLE_PREDICTION_CONFIDENCE;
      const isConfidentPrediction = isVisiblePrediction && value.confidence >= MIN_STICKY_CONFIDENCE;
      const isSameAsDisplayed = state.displayedPrediction === value.prediction;

      let displayedPrediction = state.displayedPrediction;
      let displayedConfidence = state.displayedConfidence;
      let displayUntilMs = state.displayUntilMs;

      if (isConfidentPrediction) {
        displayedPrediction = value.prediction;
        displayedConfidence = value.confidence;
        displayUntilMs = now + STICKY_PREDICTION_HOLD_MS;
      } else if (isVisiblePrediction) {
        const canPromoteTransientPrediction =
          now >= state.displayUntilMs || isSameAsDisplayed || state.displayedPrediction === "NONE";

        if (canPromoteTransientPrediction) {
          displayedPrediction = value.prediction;
          displayedConfidence = isSameAsDisplayed
            ? Math.max(state.displayedConfidence, value.confidence)
            : value.confidence;
          displayUntilMs = Math.max(state.displayUntilMs, now + TRANSIENT_PREDICTION_HOLD_MS);
        }
      } else if (now >= state.displayUntilMs) {
        displayedPrediction = "NONE";
        displayedConfidence = 0;
        displayUntilMs = 0;
      }

      const shouldPushHistory = isNonEmptyPrediction && value.confidence >= 0.7;
      const history =
        shouldPushHistory && state.history[0] !== value.prediction
          ? [value.prediction, ...state.history].slice(0, 30)
          : state.history;

      return {
        live: value,
        displayedPrediction,
        displayedConfidence,
        displayUntilMs,
        history,
        buildingBuffer: value.sentenceBuffer,
      };
    }),

  reset: () =>
    set({
      live: { prediction: "NONE", confidence: 0, sentenceBuffer: "", alternatives: [] },
      displayedPrediction: "NONE",
      displayedConfidence: 0,
      displayUntilMs: 0,
      history: [],
      buildingBuffer: "",
    }),

  // -------------------------------------------------------------------------
  // Phase 5 actions
  // -------------------------------------------------------------------------

  addConversationTurn: (turn) =>
    set((state) => {
      // Dé-duplication côté frontend
      const last = state.conversationHistory[state.conversationHistory.length - 1];
      if (last && last.id === turn.id) {
        return {};
      }
      const updated = [...state.conversationHistory, turn].slice(-MAX_CONVERSATION_HISTORY);
      return {
        conversationHistory: updated,
        currentTurnId: turn.id,
        translatedSentence: turn.text,
        grammarTags: turn.grammar_tags,
      };
    }),

  handleSentenceComplete: (payload) =>
    set((state) => {
      const turn: ConversationTurn = {
        id: payload.turn_id ?? Date.now(),
        text: payload.translated_sentence,
        raw_signs: payload.raw_signs,
        grammar_tags: payload.grammar_tags,
        confidence: payload.confidence,
        start_time: Date.now() / 1000,
        end_time: Date.now() / 1000,
        speaker: "signer",
      };

      // Dé-duplication légère
      const last = state.conversationHistory[state.conversationHistory.length - 1];
      if (last && last.text.trim().toLowerCase() === turn.text.trim().toLowerCase()) {
        return {
          translatedSentence: turn.text,
          grammarTags: turn.grammar_tags,
          translationMode: payload.translation_mode,
          currentTurnId: turn.id,
        };
      }

      const updated = [...state.conversationHistory, turn].slice(-MAX_CONVERSATION_HISTORY);
      return {
        conversationHistory: updated,
        currentTurnId: turn.id,
        translatedSentence: turn.text,
        grammarTags: turn.grammar_tags,
        translationMode: payload.translation_mode,
      };
    }),

  setNewTurn: (turnId) =>
    set({
      isNewTurn: true,
      currentTurnId: turnId,
    }),

  resetConversation: () =>
    set({
      conversationHistory: [],
      currentTurnId: null,
      translatedSentence: "",
      grammarTags: [],
      conversationContext: null,
      isNewTurn: false,
      buildingBuffer: "",
      live: { prediction: "NONE", confidence: 0, sentenceBuffer: "", alternatives: [] },
      displayedPrediction: "NONE",
      displayedConfidence: 0,
      displayUntilMs: 0,
      history: [],
    }),

  setGrammarMode: (mode) =>
    set({ translationMode: mode }),

  setConversationContext: (ctx) =>
    set({ conversationContext: ctx, isNewTurn: false }),
}));
