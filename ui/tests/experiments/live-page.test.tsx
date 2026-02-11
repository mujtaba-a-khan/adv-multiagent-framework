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
    mutate: vi.fn(),
    mutateAsync: vi.fn(),
    isPending: false,
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
import { useWSStore } from "@/stores/ws-store";

// Helpers

const baseExperiment = {
  id: "exp-1",
  name: "Test Exp",
  attacker_model: "phi4:14b",
  analyzer_model: "phi4:14b",
  defender_model: "qwen3:8b",
  target_model: "llama3:8b",
  attack_objective: "Test objective",
};

const baseSession = {
  id: "s1",
  status: "running" as const,
  strategy_name: "pair",
  session_mode: "attack" as const,
  initial_defenses: [],
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
};

const baseTurn = {
  id: "t1",
  session_id: "s1",
  turn_number: 2,
  strategy_name: "pair",
  strategy_params: {},
  attack_prompt: "Attack prompt text",
  attacker_reasoning: null,
  target_response: "Target response text",
  raw_target_response: null,
  target_blocked: false,
  judge_verdict: "refused",
  judge_confidence: 0.9,
  severity_score: null,
  specificity_score: null,
  coherence_score: null,
  vulnerability_category: null,
  attack_technique: null,
  attacker_tokens: 100,
  target_tokens: 80,
  analyzer_tokens: 50,
  created_at: "2025-01-01T00:00:00Z",
};

function setupMocks(overrides: {
  experiment?: object | null;
  session?: object | null;
  sessionLoading?: boolean;
  turns?: object[];
  pendingTurn?: object | null;
  liveTurns?: object[];
}) {
  vi.mocked(useExperiment).mockReturnValue({
    data: "experiment" in overrides ? overrides.experiment : baseExperiment,
  } as ReturnType<typeof useExperiment>);
  vi.mocked(useSession).mockReturnValue({
    data: "session" in overrides ? overrides.session : baseSession,
    isLoading: overrides.sessionLoading ?? false,
  } as ReturnType<typeof useSession>);
  vi.mocked(useTurns).mockReturnValue({
    data: overrides.turns
      ? { turns: overrides.turns, total: overrides.turns.length }
      : { turns: [], total: 0 },
  } as ReturnType<typeof useTurns>);
  if (overrides.pendingTurn !== undefined || overrides.liveTurns !== undefined) {
    vi.mocked(useWSStore).mockReturnValue({
      connect: vi.fn(),
      disconnect: vi.fn(),
      liveTurns: (overrides.liveTurns ?? []) as never[],
      pendingTurn: (overrides.pendingTurn ?? null) as never,
    });
  }
}

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

  // --- New branch-coverage tests ---

  it("renders failed session badge", () => {
    setupMocks({
      session: { ...baseSession, status: "failed" },
      turns: [],
    });
    render(<LiveBattlePage />);
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it("renders defense mode badge", () => {
    setupMocks({
      session: { ...baseSession, session_mode: "defense" },
      turns: [{ ...baseTurn }],
    });
    render(<LiveBattlePage />);
    expect(screen.getByText("Defense")).toBeInTheDocument();
  });

  it("renders skeleton metrics when session is not loaded", () => {
    setupMocks({ session: null, experiment: null, turns: [] });
    render(<LiveBattlePage />);
    // When session is falsy the sidebar renders 6 Skeleton placeholders
    expect(screen.getByText("Session Metrics")).toBeInTheDocument();
    // The metric labels like "Turns" should NOT be present
    expect(screen.queryByText("Turns")).not.toBeInTheDocument();
  });

  it("renders turn with jailbreak verdict", () => {
    const jailbreakTurn = {
      ...baseTurn,
      judge_verdict: "jailbreak",
      judge_confidence: 0.95,
    };
    setupMocks({ turns: [jailbreakTurn] });
    render(<LiveBattlePage />);
    expect(screen.getByText(/jailbreak/)).toBeInTheDocument();
    expect(screen.getByText(/95%/)).toBeInTheDocument();
  });

  it("renders turn with scores", () => {
    const scoredTurn = {
      ...baseTurn,
      severity_score: 8,
      specificity_score: 7,
      coherence_score: 6,
    };
    setupMocks({ turns: [scoredTurn] });
    render(<LiveBattlePage />);
    expect(screen.getByText("Severity: 8/10")).toBeInTheDocument();
    expect(screen.getByText("Specificity: 7/10")).toBeInTheDocument();
    expect(screen.getByText("Coherence: 6/10")).toBeInTheDocument();
  });

  it("renders turn with attacker reasoning", () => {
    const turnWithReasoning = {
      ...baseTurn,
      attacker_reasoning: "I am using a role-play strategy",
    };
    setupMocks({ turns: [turnWithReasoning] });
    render(<LiveBattlePage />);
    expect(screen.getByText("View Attacker Thinking")).toBeInTheDocument();
  });

  it("renders blocked turn with raw response", () => {
    const blockedTurn = {
      ...baseTurn,
      target_blocked: true,
      target_response: "Defense blocked this response",
      raw_target_response: "Original harmful output",
    };
    setupMocks({ turns: [blockedTurn] });
    render(<LiveBattlePage />);
    expect(screen.getByText("Target (Blocked)")).toBeInTheDocument();
    expect(screen.getByText("View Original Response")).toBeInTheDocument();
  });

  it("renders baseline turn badge", () => {
    const baselineTurn = {
      ...baseTurn,
      turn_number: 1,
    };
    setupMocks({ turns: [baselineTurn] });
    render(<LiveBattlePage />);
    expect(screen.getByText("Baseline")).toBeInTheDocument();
    expect(screen.getByText("Attacker (Baseline)")).toBeInTheDocument();
  });

  it("renders vulnerability category badge", () => {
    const vulnTurn = {
      ...baseTurn,
      vulnerability_category: "prompt_injection",
    };
    setupMocks({ turns: [vulnTurn] });
    render(<LiveBattlePage />);
    expect(screen.getByText("prompt_injection")).toBeInTheDocument();
  });

  it("renders pending turn in thinking phase", () => {
    setupMocks({
      turns: [],
      pendingTurn: {
        turn_number: 2,
        strategy_name: "pair",
        // No attack_prompt -- thinking phase
      },
    });
    render(<LiveBattlePage />);
    expect(screen.getByText(/Generating attack/)).toBeInTheDocument();
  });

  it("renders pending turn in awaiting phase", () => {
    setupMocks({
      turns: [],
      pendingTurn: {
        turn_number: 2,
        strategy_name: "pair",
        attack_prompt: "Try this attack",
        // No target_response -- awaiting phase
      },
    });
    render(<LiveBattlePage />);
    expect(screen.getByText(/Awaiting response/)).toBeInTheDocument();
    expect(screen.getByText("Try this attack")).toBeInTheDocument();
  });

  it("renders pending turn in analyzing phase", () => {
    setupMocks({
      turns: [],
      pendingTurn: {
        turn_number: 2,
        strategy_name: "pair",
        attack_prompt: "Try this attack",
        target_response: "I cannot help with that",
      },
    });
    render(<LiveBattlePage />);
    // Both the badge and the "Analyzing response..." indicator match
    const analyzingEls = screen.getAllByText(/Analyzing/);
    expect(analyzingEls.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("I cannot help with that")).toBeInTheDocument();
  });

  it("renders empty state for failed session with no turns", () => {
    setupMocks({
      session: { ...baseSession, status: "failed" },
      turns: [],
      pendingTurn: null,
    });
    render(<LiveBattlePage />);
    expect(
      screen.getByText(/This session failed before completing any turns/),
    ).toBeInTheDocument();
  });

  it("renders defense card with initial defenses", () => {
    setupMocks({
      session: {
        ...baseSession,
        session_mode: "defense",
        initial_defenses: [
          { name: "llm_judge", params: {} },
          { name: "keyword_filter", params: { threshold: 0.5 } },
        ],
      },
      turns: [{ ...baseTurn }],
    });
    render(<LiveBattlePage />);
    expect(screen.getByText("Active Defenses")).toBeInTheDocument();
    expect(screen.getByText("llm_judge:")).toBeInTheDocument();
    expect(screen.getByText("keyword_filter:")).toBeInTheDocument();
  });

  it("renders starting state with baseline preview", () => {
    setupMocks({
      session: { ...baseSession, status: "running", total_turns: 0 },
      turns: [],
      pendingTurn: null,
    });
    render(<LiveBattlePage />);
    expect(screen.getByText("Turn 1")).toBeInTheDocument();
    expect(screen.getByText("Attacker (Baseline)")).toBeInTheDocument();
    expect(screen.getByText("Test objective")).toBeInTheDocument();
  });

  it("renders pending baseline turn with Sending baseline label", () => {
    setupMocks({
      turns: [],
      pendingTurn: {
        turn_number: 1,
        is_baseline: true,
        // No attack_prompt -- thinking phase for baseline
      },
    });
    render(<LiveBattlePage />);
    expect(screen.getByText(/Sending baseline/)).toBeInTheDocument();
    expect(screen.getByText("Baseline")).toBeInTheDocument();
  });

  it("renders agent models card when experiment is loaded", () => {
    setupMocks({ turns: [] });
    render(<LiveBattlePage />);
    expect(screen.getByText("Agent Models")).toBeInTheDocument();
    // phi4:14b appears for both Attacker and Analyzer
    expect(screen.getAllByText("phi4:14b")).toHaveLength(2);
    expect(screen.getByText("llama3:8b")).toBeInTheDocument();
    expect(screen.getByText("qwen3:8b")).toBeInTheDocument();
  });

  it("renders attack mode badge with Swords icon", () => {
    setupMocks({
      session: { ...baseSession, session_mode: "attack" },
      turns: [],
    });
    render(<LiveBattlePage />);
    expect(screen.getByText("Attack")).toBeInTheDocument();
  });

  it("renders Live badge when session is running", () => {
    setupMocks({
      session: { ...baseSession, status: "running" },
      turns: [],
    });
    render(<LiveBattlePage />);
    expect(screen.getByText("Live")).toBeInTheDocument();
  });

  it("renders metrics values from session data", () => {
    setupMocks({ turns: [] });
    render(<LiveBattlePage />);
    expect(screen.getByText("3 / 10")).toBeInTheDocument();
    expect(screen.getByText("33.0%")).toBeInTheDocument();
    expect(screen.getByText("$0.0500")).toBeInTheDocument();
  });
});
