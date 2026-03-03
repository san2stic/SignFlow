/**
 * Unit tests for useFeedbackStore (Zustand).
 *
 * Cases:
 *  1. État initial correct
 *  2. openFeedback() → met à jour isOpen + predictedSign + confidence
 *  3. closeFeedback() → remet isOpen=false et efface les champs
 *  4. handleAcknowledgement() → met à jour lastResult + toast
 *  5. clearToast() → efface toast
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { useFeedbackStore } from "../feedbackStore";

// Reset Zustand store to initial state before each test to avoid cross-test pollution.
beforeEach(() => {
  useFeedbackStore.setState({
    isOpen: false,
    predictedSign: null,
    confidence: null,
    currentLandmarks: null,
    submitting: false,
    lastResult: null,
    toast: null,
  });
});

// ---------------------------------------------------------------------------
// Case 1 — État initial
// ---------------------------------------------------------------------------

describe("useFeedbackStore — initial state", () => {
  it("should have correct default values", () => {
    const state = useFeedbackStore.getState();

    expect(state.isOpen).toBe(false);
    expect(state.predictedSign).toBeNull();
    expect(state.confidence).toBeNull();
    expect(state.currentLandmarks).toBeNull();
    expect(state.submitting).toBe(false);
    expect(state.lastResult).toBeNull();
    expect(state.toast).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Case 2 — openFeedback()
// ---------------------------------------------------------------------------

describe("openFeedback()", () => {
  it("sets isOpen=true and stores predictedSign", () => {
    useFeedbackStore.getState().openFeedback("BONJOUR");

    const state = useFeedbackStore.getState();
    expect(state.isOpen).toBe(true);
    expect(state.predictedSign).toBe("BONJOUR");
    expect(state.confidence).toBeNull();
    expect(state.currentLandmarks).toBeNull();
  });

  it("stores confidence when provided", () => {
    useFeedbackStore.getState().openFeedback("MERCI", 0.87);

    const state = useFeedbackStore.getState();
    expect(state.isOpen).toBe(true);
    expect(state.predictedSign).toBe("MERCI");
    expect(state.confidence).toBeCloseTo(0.87);
  });

  it("stores landmarks when provided", () => {
    const landmarks = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
    useFeedbackStore.getState().openFeedback("AU-REVOIR", 0.9, landmarks);

    const state = useFeedbackStore.getState();
    expect(state.currentLandmarks).toEqual(landmarks);
  });

  it("resets submitting and lastResult on open", () => {
    // Pre-set dirty state.
    useFeedbackStore.setState({ submitting: true, lastResult: { correctionId: 99, trainingTriggered: true } });

    useFeedbackStore.getState().openFeedback("PARDON");

    const state = useFeedbackStore.getState();
    expect(state.submitting).toBe(false);
    expect(state.lastResult).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Case 3 — closeFeedback()
// ---------------------------------------------------------------------------

describe("closeFeedback()", () => {
  it("sets isOpen=false and clears sign-related state", () => {
    useFeedbackStore.setState({
      isOpen: true,
      predictedSign: "TEST",
      confidence: 0.5,
      currentLandmarks: [[1, 2, 3]],
      submitting: true,
    });

    useFeedbackStore.getState().closeFeedback();

    const state = useFeedbackStore.getState();
    expect(state.isOpen).toBe(false);
    expect(state.predictedSign).toBeNull();
    expect(state.confidence).toBeNull();
    expect(state.currentLandmarks).toBeNull();
    expect(state.submitting).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Case 4 — handleAcknowledgement()
// ---------------------------------------------------------------------------

describe("handleAcknowledgement()", () => {
  it("stores lastResult with correctionId and trainingTriggered=false", () => {
    useFeedbackStore.getState().handleAcknowledgement(42, false);

    const state = useFeedbackStore.getState();
    expect(state.lastResult).toEqual({ correctionId: 42, trainingTriggered: false });
    expect(state.submitting).toBe(false);
    expect(state.isOpen).toBe(false);
  });

  it("sets info-type toast when trainingTriggered=false", () => {
    useFeedbackStore.getState().handleAcknowledgement(10, false);

    const { toast } = useFeedbackStore.getState();
    expect(toast).not.toBeNull();
    expect(toast?.type).toBe("info");
    expect(toast?.message).toContain("Correction enregistrée");
  });

  it("stores lastResult with trainingTriggered=true", () => {
    useFeedbackStore.getState().handleAcknowledgement(7, true);

    const state = useFeedbackStore.getState();
    expect(state.lastResult).toEqual({ correctionId: 7, trainingTriggered: true });
  });

  it("sets success-type toast when trainingTriggered=true", () => {
    useFeedbackStore.getState().handleAcknowledgement(7, true);

    const { toast } = useFeedbackStore.getState();
    expect(toast).not.toBeNull();
    expect(toast?.type).toBe("success");
    expect(toast?.message).toContain("réentraînement");
  });
});

// ---------------------------------------------------------------------------
// Case 5 — clearToast()
// ---------------------------------------------------------------------------

describe("clearToast()", () => {
  it("sets toast to null", () => {
    useFeedbackStore.setState({
      toast: { message: "Test", type: "info" },
    });

    useFeedbackStore.getState().clearToast();

    expect(useFeedbackStore.getState().toast).toBeNull();
  });

  it("is idempotent when toast is already null", () => {
    useFeedbackStore.getState().clearToast();

    expect(useFeedbackStore.getState().toast).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// submitFeedback() — vérification que sendViaWs est appelé avec le bon payload
// ---------------------------------------------------------------------------

describe("submitFeedback()", () => {
  it("calls sendViaWs with the correct message structure", () => {
    useFeedbackStore.setState({
      predictedSign: "BONJOUR",
      confidence: 0.75,
      currentLandmarks: null,
    });

    const sendViaWs = vi.fn();
    useFeedbackStore.getState().submitFeedback("AU-REVOIR", sendViaWs);

    expect(sendViaWs).toHaveBeenCalledOnce();
    const msg = sendViaWs.mock.calls[0][0] as Record<string, unknown>;
    expect(msg.type).toBe("submit_feedback");
    expect(msg.predicted_sign).toBe("BONJOUR");
    expect(msg.corrected_sign).toBe("AU-REVOIR");
    expect(msg.confidence).toBeCloseTo(0.75);
  });

  it("sets submitting=true after call", () => {
    useFeedbackStore.setState({ predictedSign: "TEST" });

    const sendViaWs = vi.fn();
    useFeedbackStore.getState().submitFeedback("AUTRE", sendViaWs);

    expect(useFeedbackStore.getState().submitting).toBe(true);
  });

  it("does nothing when predictedSign is null", () => {
    useFeedbackStore.setState({ predictedSign: null });

    const sendViaWs = vi.fn();
    useFeedbackStore.getState().submitFeedback("QUELCONQUE", sendViaWs);

    expect(sendViaWs).not.toHaveBeenCalled();
    expect(useFeedbackStore.getState().submitting).toBe(false);
  });
});
