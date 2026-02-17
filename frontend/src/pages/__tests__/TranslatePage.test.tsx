import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { listSigns } from "../../api/signs";
import { TranslatePage } from "../TranslatePage";
import { useSettingsStore } from "../../stores/settingsStore";
import { useTrainingStore } from "../../stores/trainingStore";
import { useTranslateStore } from "../../stores/translateStore";

let wsMessageHandler: ((payload: {
  prediction: string;
  confidence: number;
  alternatives: Array<{ sign: string; confidence: number }>;
  sentence_buffer: string;
  is_sentence_complete: boolean;
}) => void) | null = null;

vi.mock("../../hooks/useWebSocket", () => ({
  useWebSocket: ({ onMessage }: { onMessage: typeof wsMessageHandler }) => {
    wsMessageHandler = onMessage;
    return { connected: true, send: vi.fn() };
  }
}));

vi.mock("../../hooks/useMediaPipe", () => ({
  useMediaPipe: () => ({ frame: null, ready: true })
}));

vi.mock("../../hooks/useCamera", () => ({
  useCamera: () => ({
    videoRef: { current: null },
    attachVideoRef: vi.fn(),
    isReady: true,
    error: null,
    toggleFacing: vi.fn(),
    capturePreRollClip: () => new Blob(["clip"], { type: "video/webm" })
  })
}));

vi.mock("../../api/signs", () => ({
  listSigns: vi.fn(async () => ({
    items: [
      {
        id: "sign-1",
        name: "lsfb_bonjour",
        slug: "lsfb_bonjour",
        category: "salutations",
        tags: [],
        variants: [],
        related_signs: [],
        video_count: 0,
        training_sample_count: 0,
        usage_count: 0
      }
    ],
    page: 1,
    per_page: 20,
    total: 1
  }))
}));

describe("TranslatePage", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  beforeEach(() => {
    wsMessageHandler = null;
    vi.mocked(listSigns).mockClear();
    useSettingsStore.setState({
      speakOnDetect: false,
      spellingMode: false,
      unknownThreshold: 0.5,
      unknownFrameWindow: 2
    });
    useTranslateStore.setState({
      live: { prediction: "NONE", confidence: 0, sentenceBuffer: "", alternatives: [] },
      displayedPrediction: "NONE",
      displayedConfidence: 0,
      displayUntilMs: 0,
      history: []
    });
    useTrainingStore.setState({ pendingClip: null });
  });

  it("keeps the last confident word visible for an extended hold period", () => {
    vi.useFakeTimers();

    render(
      <MemoryRouter>
        <TranslatePage />
      </MemoryRouter>
    );

    act(() => {
      wsMessageHandler?.({
        prediction: "lsfb_bonjour",
        confidence: 0.91,
        alternatives: [],
        sentence_buffer: "lsfb_bonjour",
        is_sentence_complete: false
      });
    });

    const currentSignPanel = screen.getByText("Signe détecté").parentElement as HTMLElement;
    expect(within(currentSignPanel).getByText("lsfb_bonjour")).toBeInTheDocument();

    act(() => {
      wsMessageHandler?.({
        prediction: "NONE",
        confidence: 0,
        alternatives: [],
        sentence_buffer: "lsfb_bonjour",
        is_sentence_complete: false
      });
    });

    expect(within(currentSignPanel).getByText("lsfb_bonjour")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(2800);
      wsMessageHandler?.({
        prediction: "NONE",
        confidence: 0,
        alternatives: [],
        sentence_buffer: "lsfb_bonjour",
        is_sentence_complete: false
      });
    });

    expect(within(currentSignPanel).getByText("lsfb_bonjour")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(300);
      wsMessageHandler?.({
        prediction: "NONE",
        confidence: 0,
        alternatives: [],
        sentence_buffer: "lsfb_bonjour",
        is_sentence_complete: false
      });
    });

    expect(within(currentSignPanel).getByText("NONE")).toBeInTheDocument();
  });

  it("keeps a low-confidence detected word visible briefly to avoid flicker", () => {
    vi.useFakeTimers();

    render(
      <MemoryRouter>
        <TranslatePage />
      </MemoryRouter>
    );

    act(() => {
      wsMessageHandler?.({
        prediction: "lsfb_bonjour",
        confidence: 0.35,
        alternatives: [],
        sentence_buffer: "lsfb_bonjour",
        is_sentence_complete: false
      });
    });

    const currentSignPanel = screen.getByText("Signe détecté").parentElement as HTMLElement;
    expect(within(currentSignPanel).getByText("lsfb_bonjour")).toBeInTheDocument();

    act(() => {
      wsMessageHandler?.({
        prediction: "NONE",
        confidence: 0,
        alternatives: [],
        sentence_buffer: "lsfb_bonjour",
        is_sentence_complete: false
      });
    });

    expect(within(currentSignPanel).getByText("lsfb_bonjour")).toBeInTheDocument();

    act(() => {
      vi.advanceTimersByTime(1300);
      wsMessageHandler?.({
        prediction: "NONE",
        confidence: 0,
        alternatives: [],
        sentence_buffer: "lsfb_bonjour",
        is_sentence_complete: false
      });
    });

    expect(within(currentSignPanel).getByText("NONE")).toBeInTheDocument();
  });

  it("opens unknown-sign prompt and pushes pre-roll clip to training handoff", async () => {
    const user = userEvent.setup();

    render(
      <MemoryRouter>
        <TranslatePage />
      </MemoryRouter>
    );

    expect(wsMessageHandler).not.toBeNull();

    act(() => {
      wsMessageHandler?.({
        prediction: "NONE",
        confidence: 0.2,
        alternatives: [],
        sentence_buffer: "",
        is_sentence_complete: false
      });
      wsMessageHandler?.({
        prediction: "NONE",
        confidence: 0.2,
        alternatives: [],
        sentence_buffer: "",
        is_sentence_complete: false
      });
    });

    expect(await screen.findByText("Signe inconnu détecté")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Ajouter un nouveau signe" }));

    const pendingClip = useTrainingStore.getState().pendingClip;
    expect(pendingClip).not.toBeNull();
    expect(pendingClip?.file.size).toBeGreaterThan(0);
    expect(pendingClip?.assignedSign).toBeUndefined();
  });

  it("allows assigning unknown clip to an existing sign before training", async () => {
    const user = userEvent.setup();

    render(
      <MemoryRouter>
        <TranslatePage />
      </MemoryRouter>
    );

    act(() => {
      wsMessageHandler?.({
        prediction: "NONE",
        confidence: 0.2,
        alternatives: [],
        sentence_buffer: "",
        is_sentence_complete: false
      });
      wsMessageHandler?.({
        prediction: "NONE",
        confidence: 0.2,
        alternatives: [],
        sentence_buffer: "",
        is_sentence_complete: false
      });
    });

    await user.click(await screen.findByRole("button", { name: "Assigner à un signe existant" }));
    await user.click(await screen.findByRole("button", { name: /lsfb_bonjour/i }));

    const pendingClip = useTrainingStore.getState().pendingClip;
    expect(pendingClip).not.toBeNull();
    expect(pendingClip?.assignedSign?.signId).toBe("sign-1");
    expect(pendingClip?.assignedSign?.signName).toBe("lsfb_bonjour");
    expect(vi.mocked(listSigns)).toHaveBeenCalled();
  });

  it("renders spelled letters when spelling mode is enabled", () => {
    useSettingsStore.setState({
      speakOnDetect: false,
      spellingMode: true,
      unknownThreshold: 0.5,
      unknownFrameWindow: 3
    });

    render(
      <MemoryRouter>
        <TranslatePage />
      </MemoryRouter>
    );

    act(() => {
      wsMessageHandler?.({
        prediction: "lsfb_bonjour",
        confidence: 0.91,
        alternatives: [],
        sentence_buffer: "lsfb_bonjour",
        is_sentence_complete: false
      });
    });

    expect(screen.getByText("B O N J O U R")).toBeInTheDocument();
  });

  it("auto-closes unknown prompt when a confident prediction arrives", async () => {
    render(
      <MemoryRouter>
        <TranslatePage />
      </MemoryRouter>
    );

    act(() => {
      wsMessageHandler?.({
        prediction: "NONE",
        confidence: 0.2,
        alternatives: [],
        sentence_buffer: "",
        is_sentence_complete: false
      });
      wsMessageHandler?.({
        prediction: "NONE",
        confidence: 0.2,
        alternatives: [],
        sentence_buffer: "",
        is_sentence_complete: false
      });
    });

    expect(await screen.findByText("Signe inconnu détecté")).toBeInTheDocument();

    act(() => {
      wsMessageHandler?.({
        prediction: "lsfb_bonjour",
        confidence: 0.84,
        alternatives: [],
        sentence_buffer: "lsfb_bonjour",
        is_sentence_complete: false
      });
    });

    await waitFor(() => {
      expect(screen.queryByText("Signe inconnu détecté")).not.toBeInTheDocument();
    });
  });
});
