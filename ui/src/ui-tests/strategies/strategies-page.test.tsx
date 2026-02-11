import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import StrategiesPage from "@/app/strategies/page";
import type { Strategy } from "@/lib/types";

// ── Mocks ────────────────────────────────────────────────────

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title?: string }) => (
    <div data-testid="header">{title}</div>
  ),
}));

vi.mock("@/hooks/use-strategies", () => ({
  useStrategies: vi.fn(),
}));

import { useStrategies } from "@/hooks/use-strategies";

// ── Helpers ──────────────────────────────────────────────────

const makeStrategy = (overrides: Partial<Strategy> = {}): Strategy => ({
  name: "roleplay",
  display_name: "Roleplay",
  category: "prompt_level",
  description: "Uses persona-based jailbreaking",
  estimated_asr: "Medium",
  min_turns: 1,
  max_turns: null,
  requires_white_box: false,
  supports_multi_turn: true,
  parameters: {},
  references: [],
  ...overrides,
});

// ── Tests ────────────────────────────────────────────────────

describe("StrategiesPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state with skeletons", () => {
    vi.mocked(useStrategies).mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useStrategies>);

    render(<StrategiesPage />);
    expect(screen.getByTestId("header")).toHaveTextContent("Strategies");
    expect(screen.getByText("Attack Strategy Catalog")).toBeInTheDocument();
  });

  it("renders empty state when no strategies match", () => {
    vi.mocked(useStrategies).mockReturnValue({
      data: { strategies: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useStrategies>);

    render(<StrategiesPage />);
    expect(
      screen.getByText("No strategies match your filters."),
    ).toBeInTheDocument();
  });

  it("renders strategy cards grouped by category", () => {
    const strategies: Strategy[] = [
      makeStrategy({
        name: "roleplay",
        display_name: "Roleplay",
        category: "prompt_level",
      }),
      makeStrategy({
        name: "pair",
        display_name: "PAIR",
        category: "optimization",
      }),
    ];

    vi.mocked(useStrategies).mockReturnValue({
      data: { strategies, total: 2 },
      isLoading: false,
    } as ReturnType<typeof useStrategies>);

    render(<StrategiesPage />);
    expect(screen.getByText("Roleplay")).toBeInTheDocument();
    expect(screen.getByText("PAIR")).toBeInTheDocument();
    expect(screen.getByText("Prompt-Level")).toBeInTheDocument();
    expect(screen.getByText("Optimization")).toBeInTheDocument();
  });

  it("renders search input", () => {
    vi.mocked(useStrategies).mockReturnValue({
      data: { strategies: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useStrategies>);

    render(<StrategiesPage />);
    expect(
      screen.getByPlaceholderText("Search strategies..."),
    ).toBeInTheDocument();
  });

  it("renders category filter button", () => {
    vi.mocked(useStrategies).mockReturnValue({
      data: { strategies: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useStrategies>);

    render(<StrategiesPage />);
    expect(screen.getByText("Category")).toBeInTheDocument();
  });

  it("shows strategy count in subtitle", () => {
    const strategies: Strategy[] = [
      makeStrategy({ name: "s1", display_name: "S1" }),
      makeStrategy({ name: "s2", display_name: "S2" }),
      makeStrategy({ name: "s3", display_name: "S3" }),
    ];

    vi.mocked(useStrategies).mockReturnValue({
      data: { strategies, total: 3 },
      isLoading: false,
    } as ReturnType<typeof useStrategies>);

    render(<StrategiesPage />);
    expect(
      screen.getByText(/3 strategies across 5 categories/),
    ).toBeInTheDocument();
  });

  it("renders strategy badges (ASR, multi-turn, turns)", () => {
    const strategies: Strategy[] = [
      makeStrategy({
        name: "tap",
        display_name: "TAP",
        estimated_asr: "High",
        supports_multi_turn: true,
        min_turns: 3,
        max_turns: 10,
      }),
    ];

    vi.mocked(useStrategies).mockReturnValue({
      data: { strategies, total: 1 },
      isLoading: false,
    } as ReturnType<typeof useStrategies>);

    render(<StrategiesPage />);
    expect(screen.getByText("ASR: High")).toBeInTheDocument();
    expect(screen.getByText("Multi-turn")).toBeInTheDocument();
  });
});
