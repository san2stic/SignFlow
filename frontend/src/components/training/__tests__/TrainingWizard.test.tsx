import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRef } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { createSign, listSignVideos, uploadSignVideo } from "../../../api/signs";
import { TrainingWizard } from "../TrainingWizard";
import { useTrainingStore } from "../../../stores/trainingStore";

if (!("createObjectURL" in URL)) {
  Object.defineProperty(URL, "createObjectURL", {
    value: vi.fn(() => "blob:mock"),
    configurable: true
  });
}

let wsMessageHandler:
  | ((payload: {
      status: string;
      progress: number;
      metrics: { loss?: number; accuracy?: number; val_accuracy?: number; current_epoch?: number };
      deployment_ready?: boolean;
      deploy_threshold?: number;
      final_val_accuracy?: number | null;
      recommended_next_action?: "deploy" | "collect_more_examples" | "wait" | "review_error";
    }) => void)
  | null = null;

vi.mock("../../../api/signs", () => ({
  createSign: vi.fn(async () => ({ id: "sign-1", name: "Bonjour" })),
  listSigns: vi.fn(async () => ({ items: [] })),
  listSignVideos: vi.fn(async () => []),
  uploadSignVideo: vi.fn(async () => ({ id: "video-1" }))
}));

vi.mock("../../../hooks/useMediaPipe", () => ({
  useMediaPipe: () => ({ frame: null, ready: true })
}));

vi.mock("../../../hooks/useWebSocket", () => ({
  useWebSocket: ({ onMessage }: { onMessage: typeof wsMessageHandler }) => {
    wsMessageHandler = onMessage;
    return { connected: true, send: vi.fn() };
  }
}));

vi.mock("../../../hooks/useTraining", () => ({
  useTraining: () => ({
    startFewShot: async () => {
      useTrainingStore.getState().setSession("session-1");
      return { id: "session-1" };
    },
    stop: vi.fn(async () => ({ status: "stopping" })),
    deploy: vi.fn(async () => ({ status: "deployed" }))
  })
}));

vi.mock("../ClipRecorder", () => ({
  ClipRecorder: ({ setClips }: { setClips: (updater: (current: unknown[]) => unknown[]) => void }) => (
    <button
      onClick={() => {
        setClips(() =>
          Array.from({ length: 5 }).map((_, index) => ({
            id: `clip-${index}`,
            file: new File(["x".repeat(30_000)], `clip-${index}.webm`, { type: "video/webm" }),
            url: `blob:clip-${index}`,
            durationMs: 2500,
            quality: "valid",
            qualityReasons: [],
            source: "recorded"
          }))
        );
      }}
    >
      Inject valid clips
    </button>
  )
}));

describe("TrainingWizard", () => {
  beforeEach(() => {
    wsMessageHandler = null;
    vi.mocked(createSign).mockClear();
    vi.mocked(listSignVideos).mockReset();
    vi.mocked(listSignVideos).mockResolvedValue([]);
    vi.mocked(uploadSignVideo).mockClear();
    useTrainingStore.setState({
      activeSessionId: null,
      pendingClip: null,
      progress: {
        status: "idle",
        progress: 0,
        metrics: { loss: 0, accuracy: 0, val_accuracy: 0, current_epoch: 0 },
        deployment_ready: false,
        deploy_threshold: 0.85,
        final_val_accuracy: null,
        recommended_next_action: "wait"
      }
    });
  });

  it("keeps Validate disabled until training session is completed", async () => {
    const user = userEvent.setup();

    render(<TrainingWizard videoRef={createRef<HTMLVideoElement>()} />);

    await user.type(screen.getByLabelText("Sign Name"), "Bonjour");
    await user.click(screen.getByRole("button", { name: "Next" }));

    await user.click(screen.getByRole("button", { name: "Inject valid clips" }));
    await user.click(screen.getByRole("button", { name: "Start Training" }));

    const validateButton = await screen.findByRole("button", { name: "Validate" });
    expect(validateButton).toBeDisabled();

    act(() => {
      wsMessageHandler?.({
        status: "completed",
        progress: 100,
        metrics: { loss: 0.2, accuracy: 0.9, val_accuracy: 0.91, current_epoch: 8 },
        deployment_ready: true,
        deploy_threshold: 0.85,
        final_val_accuracy: 0.91,
        recommended_next_action: "deploy"
      });
    });

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Validate" })).toBeEnabled();
    });
  });

  it("reuses assigned existing sign target from pending handoff", async () => {
    const user = userEvent.setup();

    useTrainingStore.setState({
      pendingClip: {
        file: new File(["handoff"], "unknown.webm", { type: "video/webm" }),
        suggestedName: "lsfb_bonjour",
        assignedSign: {
          signId: "existing-sign-42",
          signName: "lsfb_bonjour"
        },
        createdAt: Date.now()
      }
    });

    render(<TrainingWizard videoRef={createRef<HTMLVideoElement>()} />);

    expect(await screen.findByText("Fine-tuning target: lsfb_bonjour")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Inject valid clips" }));
    await user.click(screen.getByRole("button", { name: "Start Training" }));

    await waitFor(() => {
      expect(vi.mocked(uploadSignVideo)).toHaveBeenCalled();
    });

    expect(vi.mocked(createSign)).not.toHaveBeenCalled();
    const signIds = vi.mocked(uploadSignVideo).mock.calls.map((call) => call[0]);
    expect(signIds.every((id) => id === "existing-sign-42")).toBe(true);
  });

  it("can start training from existing sign videos without recording new clips", async () => {
    const user = userEvent.setup();
    vi.mocked(listSignVideos).mockResolvedValue([
      {
        id: "video-existing-1",
        sign_id: "existing-sign-77",
        file_path: "/tmp/video.mp4",
        duration_ms: 2300,
        fps: 30,
        resolution: "640x480",
        type: "training",
        landmarks_extracted: true,
        created_at: new Date().toISOString()
      }
    ]);

    render(
      <TrainingWizard
        videoRef={createRef<HTMLVideoElement>()}
        initialAssignedSign={{ id: "existing-sign-77", name: "lsfb_merci", trainingSampleCount: 1 }}
      />
    );

    await user.click(screen.getByRole("button", { name: "Next" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Start Training" })).toBeEnabled();
    });

    await user.click(screen.getByRole("button", { name: "Start Training" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Validate" })).toBeDisabled();
    });
    expect(vi.mocked(createSign)).not.toHaveBeenCalled();
    expect(vi.mocked(uploadSignVideo)).not.toHaveBeenCalled();
  });
});
