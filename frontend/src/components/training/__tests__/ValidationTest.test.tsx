import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";

import { ValidationTest } from "../ValidationTest";

vi.mock("../../../hooks/useMediaPipe", () => ({
  useMediaPipe: () => ({ frame: null, ready: true })
}));

vi.mock("../../../hooks/useWebSocket", () => ({
  useWebSocket: () => ({ connected: true, send: vi.fn() })
}));

vi.mock("../../camera/CameraFeed", () => ({
  CameraFeed: () => <div>camera</div>
}));

vi.mock("../../camera/LandmarkOverlay", () => ({
  LandmarkOverlay: () => null
}));

describe("ValidationTest", () => {
  it("shows deploy action when recommendation is deploy", async () => {
    const user = userEvent.setup();
    const onDeploy = vi.fn(async () => {});

    render(
      <ValidationTest
        prediction="lsfb_bonjour"
        confidence={0.91}
        deploymentReady
        deployThreshold={0.85}
        recommendedAction="deploy"
        onDeploy={onDeploy}
        onCollectMore={vi.fn()}
        isDeploying={false}
        deployError={null}
        videoRef={createRef<HTMLVideoElement>()}
      />
    );

    await user.click(screen.getByRole("button", { name: "Deploy model" }));

    await waitFor(() => {
      expect(onDeploy).toHaveBeenCalledTimes(1);
    });
    expect(screen.getByText("Deployment complete")).toBeInTheDocument();
  });

  it("shows record-more action when recommendation requests more examples", async () => {
    const user = userEvent.setup();
    const onCollectMore = vi.fn();

    render(
      <ValidationTest
        prediction="lsfb_bonjour"
        confidence={0.72}
        deploymentReady={false}
        deployThreshold={0.85}
        recommendedAction="collect_more_examples"
        onDeploy={vi.fn(async () => {})}
        onCollectMore={onCollectMore}
        isDeploying={false}
        deployError={null}
        videoRef={createRef<HTMLVideoElement>()}
      />
    );

    await user.click(screen.getByRole("button", { name: "Record more clips" }));

    expect(onCollectMore).toHaveBeenCalledTimes(1);
  });
});
