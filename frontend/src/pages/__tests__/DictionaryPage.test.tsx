import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DictionaryPage } from "../DictionaryPage";

let currentSigns: Array<{
  id: string;
  name: string;
  slug: string;
  description?: string;
  category?: string;
  tags: string[];
  variants: string[];
  related_signs: string[];
  video_count: number;
  training_sample_count: number;
  usage_count: number;
}> = [];

vi.mock("../../api/dictionary", () => ({
  getGraph: vi.fn(async () => ({ nodes: [], edges: [] })),
  exportDictionary: vi.fn(async () => new Blob(["zip"], { type: "application/zip" })),
  importDictionary: vi.fn(async () => ({ imported_signs: 1, imported_notes: 1, skipped: 0, errors: [] }))
}));

vi.mock("../../api/signs", () => ({
  listSigns: vi.fn(async () => ({ items: currentSigns, page: 1, per_page: 20, total: currentSigns.length })),
  createSign: vi.fn(async (payload: { name: string }) => {
    const created = {
      id: `sign-${currentSigns.length + 1}`,
      name: payload.name,
      slug: payload.name.toLowerCase(),
      category: "demo",
      tags: [],
      variants: [],
      related_signs: [],
      video_count: 0,
      training_sample_count: 0,
      usage_count: 0
    };
    currentSigns = [...currentSigns, created];
    return created;
  }),
  getSign: vi.fn(async (signId: string) => currentSigns.find((item) => item.id === signId)),
  getSignBacklinks: vi.fn(async () => ({ sign_id: "", backlinks: [] }))
}));

vi.mock("../../components/dictionary/GraphView", () => ({
  GraphView: () => <div>graph</div>
}));

vi.mock("../../components/dictionary/SignDetail", () => ({
  SignDetail: () => <div>detail</div>
}));

vi.mock("../../components/dictionary/BacklinksPanel", () => ({
  BacklinksPanel: () => <div>backlinks</div>
}));

describe("DictionaryPage", () => {
  beforeEach(() => {
    currentSigns = [];
  });

  it("creates a new sign from the Dictionary page", async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <DictionaryPage />
      </MemoryRouter>
    );

    await user.click(screen.getByRole("button", { name: "New Sign" }));
    await user.type(screen.getByPlaceholderText("Bonjour"), "lsfb_test");
    await user.click(screen.getByRole("button", { name: "Create Sign" }));

    await waitFor(() => {
      expect(screen.getByText(/Created sign/)).toBeInTheDocument();
    });
    expect(currentSigns.length).toBe(1);
  });
});
