import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import NewExperimentPage from "@/app/experiments/new/page";

// Mocks

const mockPush = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
}));

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title?: string }) => <div data-testid="header">{title}</div>,
}));

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

vi.mock("@/hooks/use-experiments", () => ({
  useCreateExperiment: () => ({
    mutate: vi.fn(),
    isPending: false,
  }),
}));

vi.mock("@/hooks/use-targets", () => ({
  useModels: () => ({
    data: { models: ["llama3:8b", "phi4:14b"] },
  }),
}));

// Tests

describe("NewExperimentPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders header with title", () => {
    render(<NewExperimentPage />);
    expect(screen.getByTestId("header")).toHaveTextContent("New Experiment");
  });

  it("renders step 0: Define Objective form", () => {
    render(<NewExperimentPage />);
    expect(screen.getByText("Define Objective")).toBeInTheDocument();
    expect(screen.getByLabelText("Experiment Name")).toBeInTheDocument();
    expect(screen.getByLabelText("Attack Objective")).toBeInTheDocument();
  });

  it("renders stepper navigation with all 4 steps", () => {
    render(<NewExperimentPage />);
    expect(screen.getByText("Objective")).toBeInTheDocument();
    expect(screen.getByText("Target")).toBeInTheDocument();
    expect(screen.getByText("Agents")).toBeInTheDocument();
    expect(screen.getByText("Review")).toBeInTheDocument();
  });

  it("renders Back button disabled on step 0", () => {
    render(<NewExperimentPage />);
    const backButton = screen.getByRole("button", { name: /back/i });
    expect(backButton).toBeDisabled();
  });

  it("renders Next button disabled when form is empty", () => {
    render(<NewExperimentPage />);
    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).toBeDisabled();
  });

  it("renders description field as optional", () => {
    render(<NewExperimentPage />);
    expect(screen.getByText("(optional)")).toBeInTheDocument();
  });
});
