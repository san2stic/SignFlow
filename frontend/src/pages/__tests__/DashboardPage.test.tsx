import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DashboardPage } from "../DashboardPage";

vi.mock("../../api/stats", () => ({
  getOverviewStats: vi.fn(async () => ({
    total_signs: 12,
    total_videos: 44,
    model_accuracy: 0.91,
    total_translations: 300,
    most_used_signs: [],
    recent_activity: []
  })),
  getAccuracyHistory: vi.fn(async () => [
    { id: "m1", version: "v1", accuracy: 0.8, created_at: "2026-02-01T10:00:00Z", is_active: false },
    { id: "m2", version: "v2", accuracy: 0.91, created_at: "2026-02-02T10:00:00Z", is_active: true }
  ]),
  getSignsPerCategory: vi.fn(async () => [
    { category: "salutations", count: 5 },
    { category: "questions", count: 3 }
  ])
}));

vi.mock("../../api/models", () => ({
  listModels: vi.fn(async () => [
    { id: "m2", version: "v2", is_active: true, accuracy: 0.91, created_at: "2026-02-02T10:00:00Z" },
    { id: "m1", version: "v1", is_active: false, accuracy: 0.8, created_at: "2026-02-01T10:00:00Z" }
  ]),
  activateModel: vi.fn(async () => ({ active_model_id: "m1", version: "v1" })),
  exportModel: vi.fn(async () => ({ model_id: "m2", version: "v2", format: "pt", path: "/tmp/v2.pt" }))
}));

vi.mock("../../api/training", () => ({
  listTrainingSessions: vi.fn(async () => [
    {
      id: "s1",
      status: "completed",
      progress: 100,
      mode: "few-shot",
      deployment_ready: true,
      deploy_threshold: 0.85,
      final_val_accuracy: 0.9,
      recommended_next_action: "deploy"
    }
  ])
}));

vi.mock("../../api/dictionary", () => ({
  exportDictionary: vi.fn(async () => new Blob(["zip"], { type: "application/zip" })),
  importDictionary: vi.fn(async () => ({ imported_signs: 1, imported_notes: 1, skipped: 0, errors: [] }))
}));

describe("DashboardPage", () => {
  it("renders charts and model management sections", async () => {
    render(<DashboardPage />);

    await waitFor(() => {
      expect(screen.getByText("Model Accuracy Over Time")).toBeInTheDocument();
    });

    expect(screen.getByText("Signs Per Category")).toBeInTheDocument();
    expect(screen.getByText("Model Versions")).toBeInTheDocument();
    expect(screen.getByText("Recent Trainings")).toBeInTheDocument();
    expect(screen.getByText(/v2/)).toBeInTheDocument();
  });
});
