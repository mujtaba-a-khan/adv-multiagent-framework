import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import ComparisonPage from "@/app/experiments/[id]/compare/page";

// Mocks

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title?: string }) => (
    <div data-testid="header">{title}</div>
  ),
}));

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) => <a href={href}>{children}</a>,
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "exp-1" }),
  useSearchParams: () => ({
    get: (key: string) => {
      if (key === "attack") return "a1";
      if (key === "defense") return "d1";
      return null;
    },
  }),
}));

vi.mock("@/hooks/use-comparison", () => ({
  useComparison: vi.fn(),
}));

import { useComparison } from "@/hooks/use-comparison";

// Tests

describe("ComparisonPage", () => {
  beforeEach(() => vi.clearAllMocks());

  it("renders loading state", () => {
    vi.mocked(useComparison).mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useComparison>);

    render(<ComparisonPage />);
    expect(screen.getByTestId("header")).toHaveTextContent(
      "Session Comparison",
    );
  });

  it("renders comparison data", () => {
    vi.mocked(useComparison).mockReturnValue({
      data: {
        attack_session: {
          id: "a1",
          total_turns: 10,
          total_jailbreaks: 3,
          total_refused: 5,
          total_borderline: 1,
          total_blocked: 0,
          estimated_cost_usd: 0.05,
          asr: 0.3,
        },
        defense_session: {
          id: "d1",
          total_turns: 10,
          total_jailbreaks: 1,
          total_refused: 7,
          total_borderline: 0,
          total_blocked: 4,
          estimated_cost_usd: 0.08,
          asr: 0.1,
        },
        attack_turns: [
          {
            turn_number: 1,
            attack_prompt: "Attack 1",
            target_response: "Response 1",
            judge_verdict: "JAILBREAK",
          },
        ],
        defense_turns: [
          {
            turn_number: 1,
            attack_prompt: "Attack 1",
            target_response: "Blocked",
            judge_verdict: "REFUSED",
            target_blocked: true,
          },
        ],
      },
      isLoading: false,
    } as ReturnType<typeof useComparison>);

    render(<ComparisonPage />);
    expect(
      screen.getByText("Attack vs Defense Comparison"),
    ).toBeInTheDocument();
    expect(screen.getAllByText("ASR").length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByText("Jailbreaks").length,
    ).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Attack Session")).toBeInTheDocument();
    expect(screen.getByText("Defense Session")).toBeInTheDocument();
    expect(
      screen.getByText("Turn-by-Turn Comparison"),
    ).toBeInTheDocument();
  });

  it("renders zero delta with neutral styling", () => {
    vi.mocked(useComparison).mockReturnValue({
      data: {
        attack_session: {
          id: "a1",
          total_turns: 5,
          total_jailbreaks: 2,
          total_refused: 3,
          total_borderline: 0,
          total_blocked: 0,
          estimated_cost_usd: 0.01,
          asr: 0.4,
        },
        defense_session: {
          id: "d1",
          total_turns: 5,
          total_jailbreaks: 2,
          total_refused: 3,
          total_borderline: 0,
          total_blocked: 0,
          estimated_cost_usd: 0.01,
          asr: 0.4,
        },
        attack_turns: [],
        defense_turns: [],
      },
      isLoading: false,
    } as ReturnType<typeof useComparison>);

    render(<ComparisonPage />);
    expect(screen.getByText("0.0%")).toBeInTheDocument();
  });
});
