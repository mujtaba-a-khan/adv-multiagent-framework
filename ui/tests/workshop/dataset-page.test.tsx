import { render, screen, fireEvent, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import DatasetPage from "@/app/workshop/dataset/page";

// Mocks

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title?: string }) => <div data-testid="header">{title}</div>,
}));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() } }));
vi.mock("date-fns", () => ({ formatDistanceToNow: () => "2 hours ago" }));

vi.mock("@/hooks/use-dataset", () => {
  const mut = () => ({ mutate: vi.fn(), mutateAsync: vi.fn(), isPending: false });
  return {
    useDatasetPrompts: vi.fn(),
    useDatasetStats: vi.fn(),
    useAddDatasetPrompt: vi.fn().mockReturnValue(mut()),
    useUploadDataset: vi.fn().mockReturnValue(mut()),
    useUpdateDatasetPrompt: vi.fn().mockReturnValue(mut()),
    useDeleteDatasetPrompt: vi.fn().mockReturnValue(mut()),
    useDatasetSuggestions: vi.fn(),
    useConfirmSuggestion: vi.fn().mockReturnValue(mut()),
    useDismissSuggestion: vi.fn().mockReturnValue(mut()),
    useGenerateHarmless: vi.fn().mockReturnValue(mut()),
    useLoadModel: vi.fn().mockReturnValue(mut()),
    useUnloadModel: vi.fn().mockReturnValue(mut()),
    useModelStatus: vi.fn(),
  };
});

vi.mock("@/hooks/use-targets", () => ({
  useModels: vi.fn().mockReturnValue({ data: { models: ["llama3:8b", "phi4:14b"] } }),
}));

import {
  useDatasetPrompts, useDatasetStats, useDatasetSuggestions, useModelStatus,
  useConfirmSuggestion, useDismissSuggestion, useGenerateHarmless,
  useLoadModel, useUnloadModel,
} from "@/hooks/use-dataset";

// Helpers

function setupDefaultMocks(overrides?: {
  prompts?: { data?: unknown; isLoading?: boolean };
  stats?: { data?: unknown };
  suggestions?: { data?: unknown };
  modelStatus?: { data?: unknown };
}) {
  vi.mocked(useDatasetPrompts).mockReturnValue({
    data: overrides?.prompts?.data ?? { prompts: [] },
    isLoading: overrides?.prompts?.isLoading ?? false,
  } as ReturnType<typeof useDatasetPrompts>);
  vi.mocked(useDatasetStats).mockReturnValue({
    data: overrides?.stats?.data ?? null,
  } as ReturnType<typeof useDatasetStats>);
  vi.mocked(useDatasetSuggestions).mockReturnValue({
    data: overrides?.suggestions?.data ?? { suggestions: [] },
  } as ReturnType<typeof useDatasetSuggestions>);
  vi.mocked(useModelStatus).mockReturnValue({
    data: overrides?.modelStatus?.data ?? { models: [] },
  } as ReturnType<typeof useModelStatus>);
}

async function selectRadixOption(trigger: HTMLElement, text: string) {
  fireEvent.click(trigger);
  fireEvent.click(await screen.findByText(text));
}

function mp(o: Record<string, unknown> = {}) {
  return {
    id: "p1", text: "Test prompt", category: "harmful", source: "manual",
    status: "active", pair_id: null, created_at: "2025-01-01T00:00:00Z", ...o,
  };
}

const MODEL_LOADED = { data: { models: [{ name: "llama3:8b" }] } };

async function selectModel() {
  const cbs = screen.getAllByRole("combobox");
  await selectRadixOption(cbs[0], "llama3:8b");
}

function mockHook<T>(hook: T, mutate: ReturnType<typeof vi.fn>) {
  vi.mocked(hook as ReturnType<typeof vi.fn>).mockReturnValue({
    mutate, mutateAsync: vi.fn(), isPending: false,
  } as unknown);
}

// Tests

describe("DatasetPage", () => {
  beforeEach(() => vi.clearAllMocks());

  // Original 12 tests

  it("renders header and page title", () => {
    setupDefaultMocks();
    render(<DatasetPage />);
    expect(screen.getByTestId("header")).toHaveTextContent("Abliteration Dataset");
    expect(screen.getByText("Abliteration Dataset", { selector: "h2" })).toBeInTheDocument();
  });

  it("renders loading state with skeletons", () => {
    setupDefaultMocks({ prompts: { isLoading: true } });
    render(<DatasetPage />);
    expect(screen.queryByText("No prompts yet")).not.toBeInTheDocument();
  });

  it("renders empty state when no prompts", () => {
    setupDefaultMocks();
    render(<DatasetPage />);
    expect(screen.getByText("No prompts yet")).toBeInTheDocument();
    expect(screen.getByText(/Add prompts manually or upload a JSONL file/)).toBeInTheDocument();
  });

  it("renders stats card", () => {
    setupDefaultMocks({ stats: { data: { harmful_count: 10, harmless_count: 8, total: 18 } } });
    render(<DatasetPage />);
    expect(screen.getByText("10 Harmful")).toBeInTheDocument();
    expect(screen.getByText("8 Harmless")).toBeInTheDocument();
    expect(screen.getByText("18 total prompts")).toBeInTheDocument();
  });

  it("renders stats warning when present", () => {
    setupDefaultMocks({
      stats: {
        data: {
          harmful_count: 10, harmless_count: 5, total: 15,
          warning: "Mismatch: 5 harmful prompts lack pairs"
        }
      },
    });
    render(<DatasetPage />);
    expect(screen.getByText("Mismatch: 5 harmful prompts lack pairs")).toBeInTheDocument();
  });

  it("renders paired prompt cards", () => {
    setupDefaultMocks({
      prompts: {
        data: {
          prompts: [
            {
              id: "h1", text: "Harmful prompt text", category: "harmful", source: "manual",
              status: "active", pair_id: null, created_at: "2025-01-01T00:00:00Z"
            },
            {
              id: "hl1", text: "Harmless counterpart text", category: "harmless", source: "generated",
              status: "active", pair_id: "h1", created_at: "2025-01-01T00:00:00Z"
            },
          ]
        }
      },
    });
    render(<DatasetPage />);
    expect(screen.getByText("Harmful prompt text")).toBeInTheDocument();
    expect(screen.getByText("Harmless counterpart text")).toBeInTheDocument();
    expect(screen.getAllByText("harmful").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("harmless").length).toBeGreaterThanOrEqual(1);
  });

  it("renders unpaired harmful prompt with no generate button when model not loaded", () => {
    setupDefaultMocks({
      prompts: {
        data: {
          prompts: [
            {
              id: "h2", text: "Unpaired harmful", category: "harmful", source: "manual",
              status: "active", pair_id: null, created_at: "2025-01-01T00:00:00Z"
            },
          ]
        }
      },
    });
    render(<DatasetPage />);
    expect(screen.getByText("Unpaired harmful")).toBeInTheDocument();
    expect(screen.queryByText("Generate Counterpart")).not.toBeInTheDocument();
  });

  it("renders suggestions section", () => {
    setupDefaultMocks({
      suggestions: {
        data: {
          suggestions: [
            {
              id: "s1", text: "Suggested prompt text", category: "harmful", source: "session",
              status: "suggested", created_at: "2025-01-01T00:00:00Z"
            },
          ]
        }
      },
    });
    render(<DatasetPage />);
    expect(screen.getByText("Pending Suggestions (1)")).toBeInTheDocument();
    expect(screen.getByText("Suggested prompt text")).toBeInTheDocument();
    expect(screen.getAllByText("Confirm").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Dismiss").length).toBeGreaterThanOrEqual(1);
  });

  it("renders action buttons", () => {
    setupDefaultMocks();
    render(<DatasetPage />);
    expect(screen.getByText("Upload")).toBeInTheDocument();
    expect(screen.getByText("Add Prompt")).toBeInTheDocument();
  });

  it("renders model control bar with initialize button", () => {
    setupDefaultMocks();
    render(<DatasetPage />);
    expect(screen.getByText("Generation Model")).toBeInTheDocument();
    expect(screen.getByText("Initialize")).toBeInTheDocument();
  });

  it("renders filter selects", () => {
    setupDefaultMocks();
    render(<DatasetPage />);
    expect(screen.getByText("All Categories")).toBeInTheDocument();
    expect(screen.getByText("All Sources")).toBeInTheDocument();
  });

  // Model lifecycle branches

  it("renders model loaded state with Loaded indicator and Stop Model button", async () => {
    setupDefaultMocks({
      modelStatus: MODEL_LOADED,
      prompts: { data: { prompts: [mp({ id: "h1", text: "Harmful text" })] } },
    });
    render(<DatasetPage />);
    await selectModel();
    expect(screen.getByText("Loaded")).toBeInTheDocument();
    expect(screen.getByText("Stop Model")).toBeInTheDocument();
    expect(screen.queryByText("Initialize")).not.toBeInTheDocument();
  });

  it("shows Generate All button when model loaded with unpaired harmful prompts", async () => {
    setupDefaultMocks({
      modelStatus: MODEL_LOADED,
      prompts: {
        data: {
          prompts: [
            mp({ id: "h1", text: "Unpaired 1" }), mp({ id: "h2", text: "Unpaired 2" }),
          ]
        }
      },
    });
    render(<DatasetPage />);
    await selectModel();
    expect(screen.getByText("Generate All (2)")).toBeInTheDocument();
  });

  it("hides Generate All when model loaded but all prompts are paired", async () => {
    setupDefaultMocks({
      modelStatus: MODEL_LOADED,
      prompts: {
        data: {
          prompts: [
            mp({ id: "h1", text: "Harmful" }),
            mp({ id: "hl1", text: "Harmless", category: "harmless", pair_id: "h1" }),
          ]
        }
      },
    });
    render(<DatasetPage />);
    await selectModel();
    expect(screen.queryByText(/Generate All/)).not.toBeInTheDocument();
  });

  it("calls loadModel mutation when Initialize is clicked", async () => {
    const mutate = vi.fn();
    mockHook(useLoadModel, mutate);
    setupDefaultMocks();
    render(<DatasetPage />);
    await selectModel();
    await userEvent.setup().click(screen.getByText("Initialize"));
    expect(mutate).toHaveBeenCalledWith("llama3:8b", expect.objectContaining({
      onSuccess: expect.any(Function), onError: expect.any(Function),
    }));
  });

  it("calls unloadModel mutation when Stop Model is clicked", async () => {
    const mutate = vi.fn();
    mockHook(useUnloadModel, mutate);
    setupDefaultMocks({ modelStatus: MODEL_LOADED });
    render(<DatasetPage />);
    await selectModel();
    await userEvent.setup().click(screen.getByText("Stop Model"));
    expect(mutate).toHaveBeenCalledWith("llama3:8b", expect.objectContaining({
      onSuccess: expect.any(Function), onError: expect.any(Function),
    }));
  });

  // UnpairedCard showGenerate branch

  it("shows Generate Counterpart for unpaired harmful when model loaded", async () => {
    setupDefaultMocks({
      modelStatus: MODEL_LOADED,
      prompts: { data: { prompts: [mp({ id: "h1", text: "Harmful unpaired" })] } },
    });
    render(<DatasetPage />);
    await selectModel();
    expect(screen.getByText("Generate Counterpart")).toBeInTheDocument();
  });

  it("hides Generate Counterpart for harmless prompts even when model loaded", async () => {
    setupDefaultMocks({
      modelStatus: MODEL_LOADED,
      prompts: {
        data: {
          prompts: [
            mp({ id: "hl1", text: "Harmless prompt", category: "harmless" }),
          ]
        }
      },
    });
    render(<DatasetPage />);
    await selectModel();
    expect(screen.queryByText("Generate Counterpart")).not.toBeInTheDocument();
  });

  it("calls generateHarmless when Generate Counterpart is clicked", async () => {
    const mutate = vi.fn();
    mockHook(useGenerateHarmless, mutate);
    setupDefaultMocks({
      modelStatus: MODEL_LOADED,
      prompts: { data: { prompts: [mp({ id: "h1", text: "Generate for me" })] } },
    });
    render(<DatasetPage />);
    await selectModel();
    await userEvent.setup().click(screen.getByText("Generate Counterpart"));
    expect(mutate).toHaveBeenCalledWith(
      { data: { harmful_prompt: "Generate for me", pair_id: "h1" }, model: "llama3:8b" },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
  });

  // Suggestion editing branch

  it("renders suggestion edit mode with textarea when Edit is clicked", async () => {
    const user = userEvent.setup();
    setupDefaultMocks({
      suggestions: {
        data: {
          suggestions: [
            mp({ id: "s1", text: "Original suggestion", source: "session", status: "suggested" }),
          ]
        }
      },
    });
    render(<DatasetPage />);
    expect(screen.getByText("Original suggestion")).toBeInTheDocument();
    await user.click(screen.getByText("Edit"));
    expect(screen.getByRole("textbox")).toHaveValue("Original suggestion");
    expect(screen.getByText("Save & Confirm")).toBeInTheDocument();
    expect(screen.getAllByText("Cancel").length).toBeGreaterThanOrEqual(1);
  });

  it("cancels suggestion editing and restores normal view", async () => {
    const user = userEvent.setup();
    setupDefaultMocks({
      suggestions: {
        data: {
          suggestions: [
            mp({ id: "s1", text: "Editable suggestion", source: "session", status: "suggested" }),
          ]
        }
      },
    });
    render(<DatasetPage />);
    await user.click(screen.getByText("Edit"));
    expect(screen.getByText("Save & Confirm")).toBeInTheDocument();
    await user.click(screen.getAllByText("Cancel")[0]);
    expect(screen.getByText("Editable suggestion")).toBeInTheDocument();
    expect(screen.getByText("Edit")).toBeInTheDocument();
    expect(screen.queryByText("Save & Confirm")).not.toBeInTheDocument();
  });

  // isModelLoaded suggestion confirm branch

  it("calls confirmWithoutGen when confirming suggestion without model loaded", async () => {
    const mutate = vi.fn();
    mockHook(useConfirmSuggestion, mutate);
    setupDefaultMocks({
      suggestions: {
        data: {
          suggestions: [
            mp({ id: "s1", text: "Suggestion", source: "session", status: "suggested" }),
          ]
        }
      },
    });
    render(<DatasetPage />);
    await userEvent.setup().click(screen.getByText("Confirm"));
    expect(mutate).toHaveBeenCalledWith(
      { id: "s1", autoGenerateCounterpart: false },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
  });

  it("calls confirm with auto-gen when suggestion confirmed with model loaded", async () => {
    const mutate = vi.fn();
    mockHook(useConfirmSuggestion, mutate);
    setupDefaultMocks({
      modelStatus: MODEL_LOADED,
      suggestions: {
        data: {
          suggestions: [
            mp({ id: "s1", text: "Suggestion", source: "session", status: "suggested" }),
          ]
        }
      },
    });
    render(<DatasetPage />);
    await selectModel();
    await userEvent.setup().click(screen.getByText("Confirm"));
    expect(mutate).toHaveBeenCalledWith(
      { id: "s1", autoGenerateCounterpart: true, model: "llama3:8b" },
      expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
  });

  it("calls dismiss mutation when Dismiss is clicked on suggestion", async () => {
    const mutate = vi.fn();
    mockHook(useDismissSuggestion, mutate);
    setupDefaultMocks({
      suggestions: {
        data: {
          suggestions: [
            mp({ id: "s1", text: "Dismiss me", source: "session", status: "suggested" }),
          ]
        }
      },
    });
    render(<DatasetPage />);
    await userEvent.setup().click(screen.getByText("Dismiss"));
    expect(mutate).toHaveBeenCalledWith(
      "s1", expect.objectContaining({ onSuccess: expect.any(Function) }),
    );
  });

  // Stats without warning

  it("renders stats without warning message when warning is null", () => {
    setupDefaultMocks({
      stats: { data: { harmful_count: 10, harmless_count: 10, total: 20, warning: null } },
    });
    render(<DatasetPage />);
    expect(screen.getByText("10 Harmful")).toBeInTheDocument();
    expect(screen.getByText("20 total prompts")).toBeInTheDocument();
    expect(screen.queryByText(/Mismatch/)).not.toBeInTheDocument();
  });

  // Paired + unpaired cards together

  it("renders both paired and unpaired cards together", () => {
    setupDefaultMocks({
      prompts: {
        data: {
          prompts: [
            mp({ id: "h1", text: "Paired harmful" }),
            mp({ id: "hl1", text: "Paired harmless", category: "harmless", pair_id: "h1" }),
            mp({ id: "h2", text: "Solo harmful" }),
            mp({ id: "hl2", text: "Solo harmless", category: "harmless" }),
          ]
        }
      },
    });
    render(<DatasetPage />);
    expect(screen.getByText("Paired harmful")).toBeInTheDocument();
    expect(screen.getByText("Paired harmless")).toBeInTheDocument();
    expect(screen.getByText("Solo harmful")).toBeInTheDocument();
    expect(screen.getByText("Solo harmless")).toBeInTheDocument();
    expect(screen.getAllByText("harmful").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("harmless").length).toBeGreaterThanOrEqual(2);
  });

  // Add Prompt dialog auto-generate checkbox branch

  it("shows auto-generate checkbox for harmful category in add dialog", async () => {
    setupDefaultMocks();
    render(<DatasetPage />);
    await userEvent.setup().click(screen.getByText("Add Prompt"));
    expect(screen.getByText(/Auto-generate harmless counterpart via LLM/)).toBeInTheDocument();
  });

  it("hides auto-generate checkbox for harmless category in add dialog", async () => {
    setupDefaultMocks();
    render(<DatasetPage />);
    fireEvent.click(screen.getByText("Add Prompt"));
    const dialog = screen.getByText("Add a prompt to the abliteration dataset.")
      .closest("[role='dialog']");
    expect(dialog).toBeTruthy();
    const triggers = within(dialog!).getAllByRole("combobox");
    fireEvent.click(triggers[0]);
    fireEvent.click(await screen.findByRole("option", { name: "Harmless" }));
    expect(screen.queryByText(/Auto-generate harmless counterpart via LLM/)).not.toBeInTheDocument();
  });

  it("shows requires-model warning on auto-generate checkbox when no model loaded", () => {
    setupDefaultMocks();
    render(<DatasetPage />);
    fireEvent.click(screen.getByText("Add Prompt"));
    expect(screen.getByText("(requires initialized model)")).toBeInTheDocument();
  });

  // PromptHalf isHarmful text color branch

  it("applies foreground color to harmful and muted to harmless prompt text", () => {
    setupDefaultMocks({
      prompts: {
        data: {
          prompts: [
            mp({ id: "h1", text: "Harmful content" }),
            mp({ id: "hl1", text: "Harmless content", category: "harmless", pair_id: "h1" }),
          ]
        }
      },
    });
    render(<DatasetPage />);
    expect(screen.getByText("Harmful content").className).toContain("text-foreground");
    expect(screen.getByText("Harmless content").className).toContain("text-muted-foreground");
  });

  // Loading vs empty vs populated

  it("does not show empty state while loading", () => {
    setupDefaultMocks({ prompts: { isLoading: true, data: { prompts: [] } } });
    render(<DatasetPage />);
    expect(screen.queryByText("No prompts yet")).not.toBeInTheDocument();
    expect(screen.queryByText("Generate Counterpart")).not.toBeInTheDocument();
  });

  it("shows populated state when prompts exist", () => {
    setupDefaultMocks({ prompts: { data: { prompts: [mp({ id: "h1", text: "A real prompt" })] } } });
    render(<DatasetPage />);
    expect(screen.queryByText("No prompts yet")).not.toBeInTheDocument();
    expect(screen.getByText("A real prompt")).toBeInTheDocument();
  });

  // Prompt card dropdown trigger

  it("renders dropdown menu trigger on prompt cards", () => {
    setupDefaultMocks({ prompts: { data: { prompts: [mp({ id: "h1" })] } } });
    render(<DatasetPage />);
    const menuTriggers = screen.getAllByRole("button").filter(
      (btn) => btn.getAttribute("aria-haspopup") === "menu",
    );
    expect(menuTriggers.length).toBeGreaterThanOrEqual(1);
  });

  // Multiple suggestions

  it("renders multiple suggestions with correct count", () => {
    setupDefaultMocks({
      suggestions: {
        data: {
          suggestions: [
            mp({ id: "s1", text: "Suggestion A", status: "suggested" }),
            mp({ id: "s2", text: "Suggestion B", status: "suggested" }),
            mp({ id: "s3", text: "Suggestion C", status: "suggested" }),
          ]
        }
      },
    });
    render(<DatasetPage />);
    expect(screen.getByText("Pending Suggestions (3)")).toBeInTheDocument();
    expect(screen.getByText("Suggestion A")).toBeInTheDocument();
    expect(screen.getByText("Suggestion B")).toBeInTheDocument();
    expect(screen.getByText("Suggestion C")).toBeInTheDocument();
    expect(screen.getAllByText("Confirm")).toHaveLength(3);
    expect(screen.getAllByText("Dismiss")).toHaveLength(3);
    expect(screen.getAllByText("Edit")).toHaveLength(3);
  });

  // No suggestions hides section

  it("does not render suggestions section when no suggestions", () => {
    setupDefaultMocks({ suggestions: { data: { suggestions: [] } } });
    render(<DatasetPage />);
    expect(screen.queryByText(/Pending Suggestions/)).not.toBeInTheDocument();
  });

  // Stats card absent when null

  it("does not render stats card when stats data is null", () => {
    setupDefaultMocks({ stats: { data: null } });
    render(<DatasetPage />);
    expect(screen.queryByText(/total prompts/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Harmful$/)).not.toBeInTheDocument();
  });

  // Initialize button disabled

  it("renders Initialize button as disabled when no model is selected", () => {
    setupDefaultMocks();
    render(<DatasetPage />);
    expect(screen.getByText("Initialize").closest("button")).toBeDisabled();
  });

  // Source labels and timestamps

  it("renders source label and relative timestamp for prompts", () => {
    setupDefaultMocks({
      prompts: {
        data: {
          prompts: [
            mp({ id: "h1", text: "With metadata", source: "manual" }),
          ]
        }
      }
    });
    render(<DatasetPage />);
    expect(screen.getByText("With metadata")).toBeInTheDocument();
    expect(screen.getByText("Manual")).toBeInTheDocument();
    expect(screen.getByText("2 hours ago")).toBeInTheDocument();
  });
});
