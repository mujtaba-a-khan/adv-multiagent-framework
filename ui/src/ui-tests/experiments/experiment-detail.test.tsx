import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import ExperimentDetailPage from "@/app/experiments/[id]/page";

// Mocks

const mockPush = vi.fn();

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "exp-123" }),
  useRouter: () => ({ push: mockPush }),
}));

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) =>
    <a href={href}>{children}</a>,
}));

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title?: string }) => <div data-testid="header">{title}</div>,
}));

vi.mock("@/components/launch-session-modal", () => ({
  LaunchSessionModal: () => <div data-testid="launch-modal" />,
}));

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

vi.mock("@/hooks/use-experiments", () => ({
  useExperiment: vi.fn(),
}));

vi.mock("@/hooks/use-sessions", () => ({
  useSessions: vi.fn(),
  useCreateSession: () => ({ mutate: vi.fn(), isPending: false }),
  useStartSession: () => ({ mutate: vi.fn(), isPending: false }),
  useDeleteSession: () => ({ mutate: vi.fn() }),
}));

vi.mock("@/hooks/use-dataset", () => ({
  useAddDatasetPrompt: () => ({ mutate: vi.fn(), isPending: false }),
}));

import { useExperiment } from "@/hooks/use-experiments";
import { useSessions } from "@/hooks/use-sessions";

// Tests

const MOCK_EXPERIMENT = {
  id: "exp-123",
  name: "Test Experiment",
  description: "A test description",
  target_model: "llama3:8b",
  target_provider: "ollama",
  attacker_model: "phi4:14b",
  analyzer_model: "phi4:14b",
  defender_model: "qwen3:8b",
  attack_objective: "Test the model safety",
  created_at: new Date().toISOString(),
};

describe("ExperimentDetailPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state", () => {
    vi.mocked(useExperiment).mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useExperiment>);
    vi.mocked(useSessions).mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useSessions>);

    render(<ExperimentDetailPage />);
    expect(screen.getByTestId("header")).toBeInTheDocument();
  });

  it("renders not found state", () => {
    vi.mocked(useExperiment).mockReturnValue({
      data: undefined,
      isLoading: false,
    } as ReturnType<typeof useExperiment>);
    vi.mocked(useSessions).mockReturnValue({
      data: undefined,
      isLoading: false,
    } as ReturnType<typeof useSessions>);

    render(<ExperimentDetailPage />);
    expect(screen.getByText("Experiment not found.")).toBeInTheDocument();
    expect(screen.getByText("Back to experiments")).toBeInTheDocument();
  });

  it("renders experiment detail with config cards", () => {
    vi.mocked(useExperiment).mockReturnValue({
      data: MOCK_EXPERIMENT,
      isLoading: false,
    } as ReturnType<typeof useExperiment>);
    vi.mocked(useSessions).mockReturnValue({
      data: { sessions: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useSessions>);

    render(<ExperimentDetailPage />);
    // Name appears in both the Header mock and the page h2
    expect(screen.getAllByText("Test Experiment")).toHaveLength(2);
    expect(screen.getByText("A test description")).toBeInTheDocument();
    expect(screen.getByText("Launch Session")).toBeInTheDocument();
    expect(screen.getByText("Target")).toBeInTheDocument();
    expect(screen.getByText("Attacker")).toBeInTheDocument();
    expect(screen.getByText("Analyzer / Judge")).toBeInTheDocument();
    expect(screen.getByText("Defender")).toBeInTheDocument();
  });

  it("renders attack objective card", () => {
    vi.mocked(useExperiment).mockReturnValue({
      data: MOCK_EXPERIMENT,
      isLoading: false,
    } as ReturnType<typeof useExperiment>);
    vi.mocked(useSessions).mockReturnValue({
      data: { sessions: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useSessions>);

    render(<ExperimentDetailPage />);
    expect(screen.getByText("Attack Objective")).toBeInTheDocument();
    expect(screen.getByText("Test the model safety")).toBeInTheDocument();
    expect(screen.getByText("Add to Dataset")).toBeInTheDocument();
  });

  it("renders empty sessions state", () => {
    vi.mocked(useExperiment).mockReturnValue({
      data: MOCK_EXPERIMENT,
      isLoading: false,
    } as ReturnType<typeof useExperiment>);
    vi.mocked(useSessions).mockReturnValue({
      data: { sessions: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useSessions>);

    render(<ExperimentDetailPage />);
    expect(screen.getByText("Sessions")).toBeInTheDocument();
    expect(screen.getByText("0 sessions run")).toBeInTheDocument();
    expect(
      screen.getByText(/No sessions yet. Launch one to start/),
    ).toBeInTheDocument();
  });

  it("renders sessions table when sessions exist", () => {
    vi.mocked(useExperiment).mockReturnValue({
      data: MOCK_EXPERIMENT,
      isLoading: false,
    } as ReturnType<typeof useExperiment>);
    vi.mocked(useSessions).mockReturnValue({
      data: {
        sessions: [
          {
            id: "sess-1",
            experiment_id: "exp-123",
            status: "completed",
            session_mode: "attack",
            strategy_name: "pair",
            total_turns: 5,
            max_turns: 10,
            total_jailbreaks: 2,
            asr: 0.4,
            estimated_cost_usd: 0.05,
            max_cost_usd: 1,
            started_at: new Date().toISOString(),
            created_at: new Date().toISOString(),
          },
        ],
        total: 1,
      },
      isLoading: false,
    } as ReturnType<typeof useSessions>);

    render(<ExperimentDetailPage />);
    expect(screen.getByText("1 session run")).toBeInTheDocument();
    expect(screen.getByText("Status")).toBeInTheDocument();
    expect(screen.getByText("Mode")).toBeInTheDocument();
    expect(screen.getByText("Strategy")).toBeInTheDocument();
    expect(screen.getByText("Pair")).toBeInTheDocument();
  });
});
