import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";

// Mocks

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) =>
    <a href={href}>{children}</a>,
}));

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title?: string }) => <div data-testid="header">{title}</div>,
}));

vi.mock("@/components/charts/verdict-pie-chart", () => ({
  VerdictPieChart: () => <div data-testid="verdict-pie-chart" />,
}));

vi.mock("@/components/charts/severity-bar-chart", () => ({
  SeverityBarChart: () => <div data-testid="severity-bar-chart" />,
}));

vi.mock("@/components/charts/turn-timeline-chart", () => ({
  TurnTimelineChart: () => <div data-testid="turn-timeline-chart" />,
}));

vi.mock("@/components/charts/cost-breakdown-chart", () => ({
  CostBreakdownChart: () => <div data-testid="cost-breakdown-chart" />,
}));

vi.mock("@/components/charts/vulnerability-chart", () => ({
  VulnerabilityChart: () => <div data-testid="vulnerability-chart" />,
}));

vi.mock("@/lib/api-client", () => ({
  exportReportJson: vi.fn(),
}));

vi.mock("@/hooks/use-reports", () => ({
  useSessionReport: vi.fn(),
}));

vi.mock("@/hooks/use-turns", () => ({
  useTurns: () => ({ data: { turns: [] } }),
}));

vi.mock("@/hooks/use-sessions", () => ({
  useSessions: () => ({ data: { sessions: [] } }),
}));

import { useSessionReport } from "@/hooks/use-reports";
import ReportDetailPage from "@/app/reports/[experimentId]/[sessionId]/page";

// Helpers

const MOCK_REPORT = {
  experiment_name: "Safety Test",
  target_model: "llama3:8b",
  strategy_name: "pair",
  attack_objective: "Test model safety boundaries",
  metrics: {
    asr: 0.4,
    total_turns: 10,
    total_jailbreaks: 4,
    avg_severity: 7.5,
    total_cost_usd: 0.0125,
    total_refused: 3,
    total_blocked: 2,
  },
  findings: [
    {
      turn_number: 3,
      strategy_name: "pair",
      vulnerability_category: "role_play",
      severity: 8,
      specificity: 7,
      attack_prompt_preview: "Pretend you are an expert...",
    },
  ],
  recommendations: [
    "Add input filtering for role-play prompts",
    "Strengthen system prompt instructions",
  ],
};

// React's use() checks thenable.status === "fulfilled" to return synchronously.
// A plain Promise.resolve() suspends because status isn't set yet at render time.
function resolvedThenable<T>(value: T): Promise<T> {
  const p = Promise.resolve(value) as Promise<T> & {
    status: string;
    value: T;
  };
  p.status = "fulfilled";
  p.value = value;
  return p;
}

function renderPage() {
  const params = resolvedThenable({
    experimentId: "exp-123",
    sessionId: "sess-456",
  });
  return render(<ReportDetailPage params={params} />);
}

// Tests

describe("ReportDetailPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders report with experiment name", async () => {
    vi.mocked(useSessionReport).mockReturnValue({
      data: MOCK_REPORT,
      isLoading: false,
    } as ReturnType<typeof useSessionReport>);

    renderPage();
    expect(
      await screen.findByText("Safety Test"),
    ).toBeInTheDocument();
  });

  it("renders metric cards", async () => {
    vi.mocked(useSessionReport).mockReturnValue({
      data: MOCK_REPORT,
      isLoading: false,
    } as ReturnType<typeof useSessionReport>);

    renderPage();
    expect(
      await screen.findByText("Attack Success Rate"),
    ).toBeInTheDocument();
    expect(screen.getByText("Total Turns")).toBeInTheDocument();
    expect(screen.getByText("Jailbreaks")).toBeInTheDocument();
    expect(screen.getByText("Avg Severity")).toBeInTheDocument();
    expect(screen.getByText("Est. Cost")).toBeInTheDocument();
  });

  it("renders export button", async () => {
    vi.mocked(useSessionReport).mockReturnValue({
      data: MOCK_REPORT,
      isLoading: false,
    } as ReturnType<typeof useSessionReport>);

    renderPage();
    expect(
      await screen.findByText("Export JSON"),
    ).toBeInTheDocument();
  });

  it("renders attack objective", async () => {
    vi.mocked(useSessionReport).mockReturnValue({
      data: MOCK_REPORT,
      isLoading: false,
    } as ReturnType<typeof useSessionReport>);

    renderPage();
    expect(
      await screen.findByText("Attack Objective"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Test model safety boundaries"),
    ).toBeInTheDocument();
  });

  it("renders chart components", async () => {
    vi.mocked(useSessionReport).mockReturnValue({
      data: MOCK_REPORT,
      isLoading: false,
    } as ReturnType<typeof useSessionReport>);

    renderPage();
    expect(
      await screen.findByText("Verdict Distribution"),
    ).toBeInTheDocument();
    expect(screen.getByText("Finding Severity")).toBeInTheDocument();
    expect(screen.getByText("Turn Timeline")).toBeInTheDocument();
    expect(screen.getByText("Vulnerability Categories")).toBeInTheDocument();
  });

  it("renders findings table", async () => {
    vi.mocked(useSessionReport).mockReturnValue({
      data: MOCK_REPORT,
      isLoading: false,
    } as ReturnType<typeof useSessionReport>);

    renderPage();
    expect(
      await screen.findByText("Jailbreak Findings"),
    ).toBeInTheDocument();
    expect(screen.getByText("T3")).toBeInTheDocument();
    expect(screen.getByText("role_play")).toBeInTheDocument();
  });

  it("renders recommendations", async () => {
    vi.mocked(useSessionReport).mockReturnValue({
      data: MOCK_REPORT,
      isLoading: false,
    } as ReturnType<typeof useSessionReport>);

    renderPage();
    expect(
      await screen.findByText("Recommendations"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Add input filtering for role-play prompts"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Strengthen system prompt instructions"),
    ).toBeInTheDocument();
  });
});
