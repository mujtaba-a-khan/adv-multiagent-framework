import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import TargetsPage from "@/app/targets/page";

// Mocks

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title?: string }) => (
    <div data-testid="header">{title}</div>
  ),
}));

const mockRefetch = vi.fn();

vi.mock("@/hooks/use-targets", () => ({
  useModels: vi.fn(),
  useHealthCheck: vi.fn(),
}));

import { useModels, useHealthCheck } from "@/hooks/use-targets";

// Tests

describe("TargetsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state", () => {
    vi.mocked(useModels).mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: mockRefetch,
      isFetching: false,
    } as unknown as ReturnType<typeof useModels>);

    vi.mocked(useHealthCheck).mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useHealthCheck>);

    render(<TargetsPage />);
    expect(screen.getByTestId("header")).toHaveTextContent(
      "Targets & Models",
    );
    expect(screen.getAllByText("Targets & Models").length).toBeGreaterThan(0);
  });

  it("renders healthy provider badge", () => {
    vi.mocked(useModels).mockReturnValue({
      data: { models: ["llama3:8b"], provider: "ollama" },
      isLoading: false,
      refetch: mockRefetch,
      isFetching: false,
    } as unknown as ReturnType<typeof useModels>);

    vi.mocked(useHealthCheck).mockReturnValue({
      data: { provider: "ollama", healthy: true },
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<TargetsPage />);
    expect(screen.getByText("Online")).toBeInTheDocument();
  });

  it("renders offline provider badge", () => {
    vi.mocked(useModels).mockReturnValue({
      data: { models: [], provider: "ollama" },
      isLoading: false,
      refetch: mockRefetch,
      isFetching: false,
    } as unknown as ReturnType<typeof useModels>);

    vi.mocked(useHealthCheck).mockReturnValue({
      data: { provider: "ollama", healthy: false },
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<TargetsPage />);
    expect(screen.getByText("Offline")).toBeInTheDocument();
  });

  it("renders empty state when no models found", () => {
    vi.mocked(useModels).mockReturnValue({
      data: { models: [], provider: "ollama" },
      isLoading: false,
      refetch: mockRefetch,
      isFetching: false,
    } as unknown as ReturnType<typeof useModels>);

    vi.mocked(useHealthCheck).mockReturnValue({
      data: { provider: "ollama", healthy: true },
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<TargetsPage />);
    expect(screen.getByText("No models found")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Ensure Ollama is running and has models pulled.",
      ),
    ).toBeInTheDocument();
  });

  it("renders model cards when models are available", () => {
    vi.mocked(useModels).mockReturnValue({
      data: {
        models: ["llama3:8b", "phi4-reasoning:14b"],
        provider: "ollama",
      },
      isLoading: false,
      refetch: mockRefetch,
      isFetching: false,
    } as unknown as ReturnType<typeof useModels>);

    vi.mocked(useHealthCheck).mockReturnValue({
      data: { provider: "ollama", healthy: true },
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<TargetsPage />);
    expect(screen.getByText("llama3")).toBeInTheDocument();
    expect(screen.getByText("phi4-reasoning")).toBeInTheDocument();
    expect(screen.getByText("8b")).toBeInTheDocument();
    expect(screen.getByText("14b")).toBeInTheDocument();
  });

  it("renders refresh button", () => {
    vi.mocked(useModels).mockReturnValue({
      data: { models: [], provider: "ollama" },
      isLoading: false,
      refetch: mockRefetch,
      isFetching: false,
    } as unknown as ReturnType<typeof useModels>);

    vi.mocked(useHealthCheck).mockReturnValue({
      data: { provider: "ollama", healthy: true },
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<TargetsPage />);
    expect(screen.getByText("Refresh")).toBeInTheDocument();
  });

  it("renders other providers section", () => {
    vi.mocked(useModels).mockReturnValue({
      data: { models: [], provider: "ollama" },
      isLoading: false,
      refetch: mockRefetch,
      isFetching: false,
    } as unknown as ReturnType<typeof useModels>);

    vi.mocked(useHealthCheck).mockReturnValue({
      data: { provider: "ollama", healthy: true },
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<TargetsPage />);
    expect(screen.getByText("Other Providers")).toBeInTheDocument();
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    expect(screen.getByText("Anthropic")).toBeInTheDocument();
    expect(screen.getByText("Google")).toBeInTheDocument();
  });

  it("shows model count text", () => {
    vi.mocked(useModels).mockReturnValue({
      data: {
        models: ["llama3:8b", "phi4-reasoning:14b"],
        provider: "ollama",
      },
      isLoading: false,
      refetch: mockRefetch,
      isFetching: false,
    } as unknown as ReturnType<typeof useModels>);

    vi.mocked(useHealthCheck).mockReturnValue({
      data: { provider: "ollama", healthy: true },
      isLoading: false,
    } as ReturnType<typeof useHealthCheck>);

    render(<TargetsPage />);
    expect(
      screen.getByText(/2 models available/),
    ).toBeInTheDocument();
  });
});
