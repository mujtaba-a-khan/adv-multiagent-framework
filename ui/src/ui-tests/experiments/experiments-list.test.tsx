import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import ExperimentsPage from "@/app/experiments/page";

// Mocks

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) =>
    <a href={href}>{children}</a>,
}));

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title?: string }) => <div data-testid="header">{title}</div>,
}));

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

const mockMutate = vi.fn();

vi.mock("@/hooks/use-experiments", () => ({
  useExperiments: vi.fn(),
  useDeleteExperiment: () => ({ mutate: mockMutate }),
}));

import { useExperiments } from "@/hooks/use-experiments";

// Tests

describe("ExperimentsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state", () => {
    vi.mocked(useExperiments).mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useExperiments>);

    render(<ExperimentsPage />);
    // "Experiments" appears in both the Header mock and the page h2
    expect(screen.getAllByText("Experiments")).toHaveLength(2);
    expect(screen.getByText("New Experiment")).toBeInTheDocument();
  });

  it("renders empty state when no experiments", () => {
    vi.mocked(useExperiments).mockReturnValue({
      data: { experiments: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useExperiments>);

    render(<ExperimentsPage />);
    expect(screen.getByText("No experiments yet")).toBeInTheDocument();
    expect(
      screen.getByText("Create your first experiment to get started."),
    ).toBeInTheDocument();
  });

  it("renders experiment list with table headers", () => {
    vi.mocked(useExperiments).mockReturnValue({
      data: {
        experiments: [
          {
            id: "exp-1",
            name: "Test Experiment",
            attack_objective: "Test objective",
            target_model: "llama3:8b",
            attacker_model: "phi4:14b",
            analyzer_model: "phi4:14b",
            defender_model: "qwen3:8b",
            target_provider: "ollama",
            created_at: new Date().toISOString(),
          },
        ],
        total: 1,
      },
      isLoading: false,
    } as ReturnType<typeof useExperiments>);

    render(<ExperimentsPage />);
    expect(screen.getByText("Test Experiment")).toBeInTheDocument();
    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getByText("Target")).toBeInTheDocument();
    expect(screen.getByText("Attacker")).toBeInTheDocument();
    expect(screen.getByText("Judge")).toBeInTheDocument();
    expect(screen.getByText("Defender")).toBeInTheDocument();
  });

  it("renders search input", () => {
    vi.mocked(useExperiments).mockReturnValue({
      data: { experiments: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useExperiments>);

    render(<ExperimentsPage />);
    expect(
      screen.getByPlaceholderText("Search experiments..."),
    ).toBeInTheDocument();
  });

  it("renders new experiment link", () => {
    vi.mocked(useExperiments).mockReturnValue({
      data: { experiments: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useExperiments>);

    render(<ExperimentsPage />);
    const link = screen.getByText("New Experiment").closest("a");
    expect(link).toHaveAttribute("href", "/experiments/new");
  });
});
