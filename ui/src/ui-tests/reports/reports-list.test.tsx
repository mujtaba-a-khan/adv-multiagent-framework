import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import ReportsPage from "@/app/reports/page";

// Mocks 

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) => <a href={href}>{children}</a>,
}));

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title?: string }) => (
    <div data-testid="header">{title}</div>
  ),
}));

vi.mock("date-fns", () => ({
  formatDistanceToNow: () => "2 hours ago",
}));

vi.mock("@/hooks/use-reports", () => ({
  useCompletedSessions: vi.fn(),
}));

import { useCompletedSessions } from "@/hooks/use-reports";

// Helpers

function makeSession(overrides: Record<string, unknown> = {}) {
  return {
    id: "sess-1",
    experiment_id: "exp-1",
    status: "completed" as const,
    total_turns: 10,
    total_jailbreaks: 3,
    asr: 0.3,
    estimated_cost_usd: 0.0042,
    completed_at: new Date().toISOString(),
    experiment: {
      id: "exp-1",
      name: "Test Experiment",
      target_model: "llama3:8b",
    },
    ...overrides,
  };
}

// Tests

describe("ReportsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state with stat card skeletons", () => {
    vi.mocked(useCompletedSessions).mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useCompletedSessions>);

    render(<ReportsPage />);
    expect(screen.getByTestId("header")).toHaveTextContent("Reports");
    expect(screen.getByText("Completed Sessions")).toBeInTheDocument();
    expect(screen.getByText("Total Jailbreaks")).toBeInTheDocument();
    expect(screen.getByText("Avg. ASR")).toBeInTheDocument();
    expect(screen.getByText("Total Cost")).toBeInTheDocument();
  });

  it("renders empty state when no sessions", () => {
    vi.mocked(useCompletedSessions).mockReturnValue({
      data: [],
      isLoading: false,
    } as ReturnType<typeof useCompletedSessions>);

    render(<ReportsPage />);
    expect(screen.getByText("No reports yet")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Reports are generated when adversarial sessions complete.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("Go to Experiments")).toBeInTheDocument();
  });

  it("renders session reports table with data", () => {
    vi.mocked(useCompletedSessions).mockReturnValue({
      data: [makeSession()],
      isLoading: false,
    } as ReturnType<typeof useCompletedSessions>);

    render(<ReportsPage />);
    // Table headers
    expect(screen.getByText("Experiment")).toBeInTheDocument();
    expect(screen.getByText("Target Model")).toBeInTheDocument();
    expect(screen.getByText("Turns")).toBeInTheDocument();
    expect(screen.getByText("ASR")).toBeInTheDocument();
    expect(screen.getByText("Jailbreaks")).toBeInTheDocument();
    expect(screen.getByText("Cost")).toBeInTheDocument();
    expect(screen.getByText("Completed")).toBeInTheDocument();
    // Data
    expect(screen.getByText("Test Experiment")).toBeInTheDocument();
    expect(screen.getByText("llama3:8b")).toBeInTheDocument();
    expect(screen.getByText("10")).toBeInTheDocument();
    expect(screen.getByText("View")).toBeInTheDocument();
  });

  it("renders stat cards with computed values", () => {
    vi.mocked(useCompletedSessions).mockReturnValue({
      data: [
        makeSession({ total_jailbreaks: 3, asr: 0.3, estimated_cost_usd: 0.01 }),
        makeSession({
          id: "sess-2",
          total_jailbreaks: 5,
          asr: 0.5,
          estimated_cost_usd: 0.02,
        }),
      ],
      isLoading: false,
    } as ReturnType<typeof useCompletedSessions>);

    render(<ReportsPage />);
    // 2 completed sessions
    expect(screen.getByText("2")).toBeInTheDocument();
    // 3 + 5 = 8 total jailbreaks
    expect(screen.getByText("8")).toBeInTheDocument();
    // Avg ASR: (0.3 + 0.5) / 2 = 0.4 => 40.0%
    expect(screen.getByText("40.0%")).toBeInTheDocument();
    // Total cost: 0.01 + 0.02 = 0.03 => $0.0300
    expect(screen.getByText("$0.0300")).toBeInTheDocument();
  });

  it("renders session reports card title", () => {
    vi.mocked(useCompletedSessions).mockReturnValue({
      data: [],
      isLoading: false,
    } as ReturnType<typeof useCompletedSessions>);

    render(<ReportsPage />);
    expect(screen.getByText("Session Reports")).toBeInTheDocument();
    expect(
      screen.getByText(
        "View detailed analysis for completed adversarial sessions",
      ),
    ).toBeInTheDocument();
  });

  it("links view button to correct report route", () => {
    vi.mocked(useCompletedSessions).mockReturnValue({
      data: [makeSession()],
      isLoading: false,
    } as ReturnType<typeof useCompletedSessions>);

    render(<ReportsPage />);
    const viewLink = screen.getByText("View").closest("a");
    expect(viewLink).toHaveAttribute("href", "/reports/exp-1/sess-1");
  });
});
