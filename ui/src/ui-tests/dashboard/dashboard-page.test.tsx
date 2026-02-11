import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import DashboardPage from "@/app/page";

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

vi.mock("date-fns", () => ({
  formatDistanceToNow: () => "2 hours ago",
}));

vi.mock("@/hooks/use-experiments", () => ({
  useExperiments: vi.fn(),
}));

vi.mock("@/hooks/use-strategies", () => ({
  useStrategies: vi.fn(),
}));

vi.mock("@/hooks/use-targets", () => ({
  useHealthCheck: vi.fn(),
}));

import { useExperiments } from "@/hooks/use-experiments";
import { useStrategies } from "@/hooks/use-strategies";
import { useHealthCheck } from "@/hooks/use-targets";

// Tests

describe("DashboardPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state with skeletons", () => {
    vi.mocked(useExperiments).mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useExperiments>);
    vi.mocked(useStrategies).mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useStrategies>);
    vi.mocked(useHealthCheck).mockReturnValue({
      data: undefined,
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<DashboardPage />);
    expect(screen.getByTestId("header")).toHaveTextContent("Dashboard");
  });

  it("renders stats cards with data", () => {
    vi.mocked(useExperiments).mockReturnValue({
      data: { experiments: [], total: 12 },
      isLoading: false,
    } as ReturnType<typeof useExperiments>);
    vi.mocked(useStrategies).mockReturnValue({
      data: { strategies: [], total: 28 },
      isLoading: false,
    } as ReturnType<typeof useStrategies>);
    vi.mocked(useHealthCheck).mockReturnValue({
      data: { provider: "ollama", healthy: true },
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<DashboardPage />);
    expect(screen.getByText("12")).toBeInTheDocument();
    expect(screen.getByText("28")).toBeInTheDocument();
    expect(screen.getByText("Online")).toBeInTheDocument();
  });

  it("renders offline status", () => {
    vi.mocked(useExperiments).mockReturnValue({
      data: { experiments: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useExperiments>);
    vi.mocked(useStrategies).mockReturnValue({
      data: { strategies: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useStrategies>);
    vi.mocked(useHealthCheck).mockReturnValue({
      data: { provider: "ollama", healthy: false },
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<DashboardPage />);
    expect(screen.getByText("Offline")).toBeInTheDocument();
  });

  it("renders empty state when no experiments", () => {
    vi.mocked(useExperiments).mockReturnValue({
      data: { experiments: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useExperiments>);
    vi.mocked(useStrategies).mockReturnValue({
      data: { strategies: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useStrategies>);
    vi.mocked(useHealthCheck).mockReturnValue({
      data: undefined,
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<DashboardPage />);
    expect(
      screen.getByText("No experiments yet"),
    ).toBeInTheDocument();
  });

  it("renders experiment list when data present", () => {
    vi.mocked(useExperiments).mockReturnValue({
      data: {
        experiments: [
          {
            id: "exp-1",
            name: "Test Experiment",
            target_model: "llama3:8b",
            attack_objective: "Test objective",
            created_at: "2024-01-01T00:00:00Z",
          },
        ],
        total: 1,
      },
      isLoading: false,
    } as ReturnType<typeof useExperiments>);
    vi.mocked(useStrategies).mockReturnValue({
      data: { strategies: [], total: 28 },
      isLoading: false,
    } as ReturnType<typeof useStrategies>);
    vi.mocked(useHealthCheck).mockReturnValue({
      data: { provider: "ollama", healthy: true },
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<DashboardPage />);
    expect(screen.getByText("Test Experiment")).toBeInTheDocument();
    expect(screen.getByText("llama3:8b")).toBeInTheDocument();
    expect(screen.getByText("Test objective")).toBeInTheDocument();
  });

  it("renders quick actions section", () => {
    vi.mocked(useExperiments).mockReturnValue({
      data: { experiments: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useExperiments>);
    vi.mocked(useStrategies).mockReturnValue({
      data: { strategies: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useStrategies>);
    vi.mocked(useHealthCheck).mockReturnValue({
      data: undefined,
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<DashboardPage />);
    expect(screen.getByText("Quick Actions")).toBeInTheDocument();
    expect(
      screen.getAllByText("New Experiment").length,
    ).toBeGreaterThanOrEqual(1);
  });

  it("renders framework version", () => {
    vi.mocked(useExperiments).mockReturnValue({
      data: { experiments: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useExperiments>);
    vi.mocked(useStrategies).mockReturnValue({
      data: { strategies: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useStrategies>);
    vi.mocked(useHealthCheck).mockReturnValue({
      data: undefined,
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<DashboardPage />);
    expect(screen.getByText("v0.1")).toBeInTheDocument();
  });
});
