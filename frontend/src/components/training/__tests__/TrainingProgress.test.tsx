import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { TrainingProgress } from "../TrainingProgress";
import { useTrainingStore } from "../../../stores/trainingStore";

describe("TrainingProgress", () => {
  beforeEach(() => {
    useTrainingStore.setState({
      activeSessionId: null,
      pendingClip: null,
      progress: {
        status: "training",
        progress: 42.5,
        metrics: {
          loss: 0.345,
          accuracy: 0.82,
          val_accuracy: 0.79,
          current_epoch: 21
        },
        metrics_history: [
          { epoch: 19, loss: 0.41, accuracy: 0.78, val_accuracy: 0.75 },
          { epoch: 20, loss: 0.38, accuracy: 0.8, val_accuracy: 0.77 },
          { epoch: 21, loss: 0.345, accuracy: 0.82, val_accuracy: 0.79 }
        ],
        deployment_ready: false,
        deploy_threshold: 0.85,
        final_val_accuracy: null,
        recommended_next_action: "collect_more_examples"
      }
    });
  });

  it("renders metrics, chart section and recommendation state", () => {
    render(<TrainingProgress />);

    expect(screen.getByText("Training Progress")).toBeInTheDocument();
    expect(screen.getByText("42.5%")).toBeInTheDocument();
    expect(screen.getByText("Loss: 0.345")).toBeInTheDocument();
    expect(screen.getByText("Acc: 82.0%")).toBeInTheDocument();
    expect(screen.getByText("Val: 79.0%")).toBeInTheDocument();
    expect(screen.getByText("Training metrics")).toBeInTheDocument();
    expect(screen.getByText(/Below threshold/i)).toBeInTheDocument();
    expect(screen.getByText(/Add 3-5 more clips to improve accuracy/i)).toBeInTheDocument();
  });
});
