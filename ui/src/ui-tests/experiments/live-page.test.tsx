import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import LiveBattlePage from "@/app/experiments/[id]/live/page";

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
    get: (key: string) => (key === "session" ? "s1" : null),
  }),
}));

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

vi.mock("@/hooks/use-experiments", () => ({
  useExperiment: vi.fn(),
}));

vi.mock("@/hooks/use-sessions", () => ({
  useSession: vi.fn(),
}));

vi.mock("@/hooks/use-turns", () => ({
  useTurns: vi.fn(),
}));

vi.mock("@/hooks/use-dataset", () => ({
  useAddDatasetPrompt: vi.fn().mockReturnValue({
    mutateAsync: vi.fn(),
  }),
}));

vi.mock("@/stores/ws-store", () => ({
  useWSStore: vi.fn().mockReturnValue({
    connect: vi.fn(),
    disconnect: vi.fn(),
    liveTurns: [],
    pendingTurn: null,
  }),
}));

import { useExperiment } from "@/hooks/use-experiments";
import { useSession } from "@/hooks/use-sessions";
import { useTurns } from "@/hooks/use-turns";

// Tests

describe("LiveBattlePage", () => {
  beforeEach(() => vi.clearAllMocks());

  it("renders header with battle title", () => {
    vi.mocked(useExperiment).mockReturnValue({
      data: {
        id: "exp-1",
        name: "Test Exp",
        attacker_model: "phi4:14b",
        analyzer_model: "phi4:14b",
        defender_model: "qwen3:8b",
        target_model: "llama3:8b",
        attack_objective: "Test objective",
      },
    } as ReturnType<typeof useExperiment>);
    vi.mocked(useSession).mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useSession>);
    vi.mocked(useTurns).mockReturnValue({
      data: undefined,
    } as ReturnType<typeof useTurns>);

    render(<LiveBattlePage />);
    expect(screen.getByTestId("header")).toHaveTextContent(
      "Live Battle",
    );
  });

  it("renders session info when loaded", () => {
    vi.mocked(useExperiment).mockReturnValue({
      data: {
        id: "exp-1",
        name: "My Experiment",
        attacker_model: "phi4:14b",
        analyzer_model: "phi4:14b",
        defender_model: "qwen3:8b",
        target_model: "llama3:8b",
        attack_objective: "Test objective",
      },
    } as ReturnType<typeof useExperiment>);
    vi.mocked(useSession).mockReturnValue({
      data: {
        id: "s1",
        status: "running",
        strategy_name: "pair",
        session_mode: "attack",
        max_turns: 10,
        total_turns: 3,
        total_jailbreaks: 1,
        total_refused: 1,
        total_borderline: 0,
        total_blocked: 0,
        estimated_cost_usd: 0.05,
        asr: 0.33,
        total_attacker_tokens: 500,
        total_target_tokens: 300,
        total_analyzer_tokens: 200,
      },
    } as ReturnType<typeof useSession>);
    vi.mocked(useTurns).mockReturnValue({
      data: {
        turns: [
          {
            id: "t1",
            turn_number: 1,
            attack_prompt: "Attack prompt",
            target_response: "Target response",
            judge_verdict: "REFUSED",
            is_baseline: true,
          },
        ],
        total: 1,
      },
    } as ReturnType<typeof useTurns>);

    render(<LiveBattlePage />);
    expect(screen.getByText("My Experiment")).toBeInTheDocument();
    expect(screen.getByText(/pair/)).toBeInTheDocument();
  });

  it("renders completed session state", () => {
    vi.mocked(useExperiment).mockReturnValue({
      data: {
        id: "exp-1",
        name: "Done Exp",
        attacker_model: "phi4:14b",
        analyzer_model: "phi4:14b",
        defender_model: "qwen3:8b",
        target_model: "llama3:8b",
        attack_objective: "Test objective",
      },
    } as ReturnType<typeof useExperiment>);
    vi.mocked(useSession).mockReturnValue({
      data: {
        id: "s1",
        status: "completed",
        strategy_name: "encoding",
        session_mode: "attack",
        max_turns: 5,
        total_turns: 5,
        total_jailbreaks: 2,
        total_refused: 3,
        total_borderline: 0,
        total_blocked: 0,
        estimated_cost_usd: 0.1,
        asr: 0.4,
        total_attacker_tokens: 1000,
        total_target_tokens: 800,
        total_analyzer_tokens: 600,
      },
    } as ReturnType<typeof useSession>);
    vi.mocked(useTurns).mockReturnValue({
      data: { turns: [], total: 0 },
    } as ReturnType<typeof useTurns>);

    render(<LiveBattlePage />);
    expect(screen.getByText("Done Exp")).toBeInTheDocument();
  });
});
