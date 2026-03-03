/**
 * Component tests for FeedbackButton.
 *
 * Cases:
 *  1. Renders null when predictedSign is an empty string
 *  2. Renders null when predictedSign is "NONE"
 *  3. Renders null when predictedSign is "RECORDING"
 *  4. Renders the button when predictedSign is a valid sign name
 *  5. Clicking the button calls openFeedback with the correct arguments
 */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock the Zustand store so component tests are fully isolated from store state.
const mockOpenFeedback = vi.fn();

vi.mock("../../stores/feedbackStore", () => ({
  useFeedbackStore: (selector: (s: { openFeedback: typeof mockOpenFeedback }) => unknown) =>
    selector({ openFeedback: mockOpenFeedback }),
}));

import { FeedbackButton } from "../FeedbackButton";

beforeEach(() => {
  mockOpenFeedback.mockClear();
});

// ---------------------------------------------------------------------------
// Case 1 — Rendu null si predictedSign est vide
// ---------------------------------------------------------------------------

describe("FeedbackButton — null rendering", () => {
  it("renders null when predictedSign is an empty string", () => {
    const { container } = render(
      <FeedbackButton predictedSign="" />
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders null when predictedSign is 'NONE'", () => {
    const { container } = render(
      <FeedbackButton predictedSign="NONE" />
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders null when predictedSign is 'RECORDING'", () => {
    const { container } = render(
      <FeedbackButton predictedSign="RECORDING" />
    );
    expect(container.firstChild).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Case 2 — Rendu du bouton si predictedSign est défini
// ---------------------------------------------------------------------------

describe("FeedbackButton — visible rendering", () => {
  it("renders a button when predictedSign is a valid sign label", () => {
    render(<FeedbackButton predictedSign="BONJOUR" />);

    const btn = screen.getByRole("button", { name: /corriger ce signe/i });
    expect(btn).toBeInTheDocument();
  });

  it("renders the 'Corriger' label text", () => {
    render(<FeedbackButton predictedSign="MERCI" />);

    expect(screen.getByText("Corriger")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Case 3 — Au clic, openFeedback est appelé avec les bons arguments
// ---------------------------------------------------------------------------

describe("FeedbackButton — click behaviour", () => {
  it("calls openFeedback with predictedSign when clicked (no confidence)", async () => {
    const user = userEvent.setup();
    render(<FeedbackButton predictedSign="AU-REVOIR" />);

    await user.click(screen.getByRole("button", { name: /corriger ce signe/i }));

    expect(mockOpenFeedback).toHaveBeenCalledOnce();
    expect(mockOpenFeedback).toHaveBeenCalledWith("AU-REVOIR", undefined, null);
  });

  it("calls openFeedback with predictedSign and confidence when provided", async () => {
    const user = userEvent.setup();
    render(<FeedbackButton predictedSign="PARDON" confidence={0.91} />);

    await user.click(screen.getByRole("button", { name: /corriger ce signe/i }));

    expect(mockOpenFeedback).toHaveBeenCalledOnce();
    expect(mockOpenFeedback).toHaveBeenCalledWith("PARDON", 0.91, null);
  });

  it("calls openFeedback with landmarks when provided", async () => {
    const user = userEvent.setup();
    const landmarks = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
    render(<FeedbackButton predictedSign="MERCI" confidence={0.7} landmarks={landmarks} />);

    await user.click(screen.getByRole("button", { name: /corriger ce signe/i }));

    expect(mockOpenFeedback).toHaveBeenCalledOnce();
    expect(mockOpenFeedback).toHaveBeenCalledWith("MERCI", 0.7, landmarks);
  });

  it("does not call openFeedback when predictedSign is NONE", async () => {
    render(<FeedbackButton predictedSign="NONE" />);
    // Button is not rendered, so no interaction possible — verify nothing was called.
    expect(mockOpenFeedback).not.toHaveBeenCalled();
  });
});
