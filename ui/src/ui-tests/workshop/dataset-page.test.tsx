import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import DatasetPage from "@/app/workshop/dataset/page";

// Mocks

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title?: string }) => (
    <div data-testid="header">{title}</div>
  ),
}));

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}));

vi.mock("date-fns", () => ({
  formatDistanceToNow: () => "2 hours ago",
}));

vi.mock("@/hooks/use-dataset", () => {
  const mut = () => ({
    mutate: vi.fn(),
    mutateAsync: vi.fn(),
    isPending: false,
  });
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
  useModels: vi.fn().mockReturnValue({
    data: { models: ["llama3:8b", "phi4:14b"] },
  }),
}));

import {
  useDatasetPrompts,
  useDatasetStats,
  useDatasetSuggestions,
  useModelStatus,
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

// Tests

describe("DatasetPage", () => {
  beforeEach(() => vi.clearAllMocks());

  it("renders header and page title", () => {
    setupDefaultMocks();
    render(<DatasetPage />);
    expect(screen.getByTestId("header")).toHaveTextContent(
      "Abliteration Dataset",
    );
    expect(
      screen.getByText("Abliteration Dataset", { selector: "h2" }),
    ).toBeInTheDocument();
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
    expect(
      screen.getByText(
        /Add prompts manually or upload a JSONL file/,
      ),
    ).toBeInTheDocument();
  });

  it("renders stats card", () => {
    setupDefaultMocks({
      stats: {
        data: {
          harmful_count: 10,
          harmless_count: 8,
          total: 18,
        },
      },
    });
    render(<DatasetPage />);
    expect(screen.getByText("10 Harmful")).toBeInTheDocument();
    expect(screen.getByText("8 Harmless")).toBeInTheDocument();
    expect(screen.getByText("18 total prompts")).toBeInTheDocument();
  });

  it("renders stats warning when present", () => {
    setupDefaultMocks({
      stats: {
        data: {
          harmful_count: 10,
          harmless_count: 5,
          total: 15,
          warning: "Mismatch: 5 harmful prompts lack pairs",
        },
      },
    });
    render(<DatasetPage />);
    expect(
      screen.getByText("Mismatch: 5 harmful prompts lack pairs"),
    ).toBeInTheDocument();
  });

  it("renders paired prompt cards", () => {
    setupDefaultMocks({
      prompts: {
        data: {
          prompts: [
            {
              id: "h1",
              text: "Harmful prompt text",
              category: "harmful",
              source: "manual",
              status: "active",
              pair_id: null,
              created_at: "2025-01-01T00:00:00Z",
            },
            {
              id: "hl1",
              text: "Harmless counterpart text",
              category: "harmless",
              source: "generated",
              status: "active",
              pair_id: "h1",
              created_at: "2025-01-01T00:00:00Z",
            },
          ],
        },
      },
    });
    render(<DatasetPage />);
    expect(
      screen.getByText("Harmful prompt text"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Harmless counterpart text"),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText("harmful").length,
    ).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByText("harmless").length,
    ).toBeGreaterThanOrEqual(1);
  });

  it("renders unpaired harmful prompt with no generate button when model not loaded", () => {
    setupDefaultMocks({
      prompts: {
        data: {
          prompts: [
            {
              id: "h2",
              text: "Unpaired harmful",
              category: "harmful",
              source: "manual",
              status: "active",
              pair_id: null,
              created_at: "2025-01-01T00:00:00Z",
            },
          ],
        },
      },
    });
    render(<DatasetPage />);
    expect(
      screen.getByText("Unpaired harmful"),
    ).toBeInTheDocument();
    // Generate button only shows when model is loaded
    expect(
      screen.queryByText("Generate Counterpart"),
    ).not.toBeInTheDocument();
  });

  it("renders suggestions section", () => {
    setupDefaultMocks({
      suggestions: {
        data: {
          suggestions: [
            {
              id: "s1",
              text: "Suggested prompt text",
              category: "harmful",
              source: "session",
              status: "suggested",
              created_at: "2025-01-01T00:00:00Z",
            },
          ],
        },
      },
    });
    render(<DatasetPage />);
    expect(
      screen.getByText("Pending Suggestions (1)"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Suggested prompt text"),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText("Confirm").length,
    ).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByText("Dismiss").length,
    ).toBeGreaterThanOrEqual(1);
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
    expect(
      screen.getByText("All Categories"),
    ).toBeInTheDocument();
    expect(screen.getByText("All Sources")).toBeInTheDocument();
  });
});
