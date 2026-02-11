import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import NewExperimentPage from "@/app/experiments/new/page";
import { useCreateExperiment } from "@/hooks/use-experiments";
import { useModels } from "@/hooks/use-targets";

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
  useCreateExperiment: vi.fn(),
}));

vi.mock("@/hooks/use-targets", () => ({
  useModels: vi.fn(),
}));

// Helpers

/** Fill step 0 fields and click Next to reach step 1 */
async function goToStep1(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Experiment Name"), "Test Exp");
  await user.type(screen.getByLabelText("Attack Objective"), "Test objective");
  await user.click(screen.getByRole("button", { name: /next/i }));
}

/** Select a target model via the Radix Select combobox on step 1, then click Next */
async function selectModelAndNext() {
  const triggers = screen.getAllByRole("combobox");
  fireEvent.click(triggers[0]); // Target Model select
  const option = await screen.findByText("llama3:8b");
  fireEvent.click(option);
  // After model selected, Next should be enabled
  fireEvent.click(screen.getByRole("button", { name: /next/i }));
}

// Tests

describe("NewExperimentPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useCreateExperiment).mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
    } as any);
    vi.mocked(useModels).mockReturnValue({
      data: { models: ["llama3:8b", "phi4:14b"] },
    } as any);
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

  // New branch-coverage tests

  it("enables Next when name and objective are filled", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).toBeDisabled();

    await user.type(screen.getByLabelText("Experiment Name"), "My Experiment");
    await user.type(screen.getByLabelText("Attack Objective"), "Elicit harmful content");

    expect(nextButton).toBeEnabled();
  });

  it("navigates to step 1 (Target) after filling step 0", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);

    expect(screen.getByText("Select Target")).toBeInTheDocument();
    expect(screen.queryByText("Define Objective")).not.toBeInTheDocument();
  });

  it("navigates to step 2 (Agents) after step 1", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);
    await selectModelAndNext();

    expect(screen.getByText("Assign Agent Models")).toBeInTheDocument();
    expect(screen.queryByText("Select Target")).not.toBeInTheDocument();
  });

  it("navigates to step 3 (Review) from step 2", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);
    await selectModelAndNext();

    // Step 2 canNext() is always true
    await user.click(screen.getByRole("button", { name: /next/i }));

    expect(screen.getByText("Review & Launch")).toBeInTheDocument();
    expect(screen.getByText("Name")).toBeInTheDocument();
    // "Objective" appears in both stepper nav and review row — verify at least two
    expect(screen.getAllByText("Objective").length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText("Target Model")).toBeInTheDocument();
  });

  it("navigates back from step 1 to step 0", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);
    expect(screen.getByText("Select Target")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /back/i }));
    expect(screen.getByText("Define Objective")).toBeInTheDocument();
  });

  it("shows Create Experiment button on step 3 instead of Next", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);
    await selectModelAndNext();
    await user.click(screen.getByRole("button", { name: /next/i }));

    // On step 3 the Next button should not exist
    expect(screen.queryByRole("button", { name: /^next$/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /create experiment/i })).toBeInTheDocument();
  });

  it("shows Create Experiment button with pending state", async () => {
    vi.mocked(useCreateExperiment).mockReturnValue({
      mutate: vi.fn(),
      isPending: true,
    } as any);

    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);
    await selectModelAndNext();
    await user.click(screen.getByRole("button", { name: /next/i }));

    const createButton = screen.getByRole("button", { name: /create experiment/i });
    expect(createButton).toBeDisabled();
  });

  it("renders no models message when models list is empty", async () => {
    vi.mocked(useModels).mockReturnValue({
      data: { models: [] },
    } as any);

    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);

    // Open the Target Model select dropdown
    const triggers = screen.getAllByRole("combobox");
    fireEvent.click(triggers[0]);

    const noModels = await screen.findByText(/no models found/i);
    expect(noModels).toBeInTheDocument();
  });

  it("step 1 shows system prompt field", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);

    expect(screen.getByLabelText(/target system prompt/i)).toBeInTheDocument();
  });

  it("shows presets on step 2", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);
    await selectModelAndNext();

    expect(screen.getByRole("button", { name: /optimized/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /balanced/i })).toBeInTheDocument();
  });

  it("step 1 Next is disabled when no target model selected", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);

    // On step 1 without model selected, Next should be disabled (canNext step 1 branch)
    const nextButton = screen.getByRole("button", { name: /next/i });
    expect(nextButton).toBeDisabled();
  });

  it("Back button is enabled on step 1", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);

    const backButton = screen.getByRole("button", { name: /back/i });
    expect(backButton).toBeEnabled();
  });

  it("step 2 shows all three agent model selectors", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);
    await selectModelAndNext();

    expect(screen.getByText("Attacker Model")).toBeInTheDocument();
    expect(screen.getByText("Analyzer / Judge Model")).toBeInTheDocument();
    expect(screen.getByText("Defender Model")).toBeInTheDocument();
  });

  it("review step shows provider and agent model values", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);
    await selectModelAndNext();
    await user.click(screen.getByRole("button", { name: /next/i }));

    expect(screen.getByText("Provider")).toBeInTheDocument();
    expect(screen.getByText("Attacker Model")).toBeInTheDocument();
    expect(screen.getByText("Analyzer Model")).toBeInTheDocument();
    expect(screen.getByText("Defender Model")).toBeInTheDocument();
  });

  it("calls mutate when Create Experiment is clicked", async () => {
    const mockMutate = vi.fn();
    vi.mocked(useCreateExperiment).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
    } as any);

    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);
    await selectModelAndNext();
    await user.click(screen.getByRole("button", { name: /next/i }));

    await user.click(screen.getByRole("button", { name: /create experiment/i }));
    expect(mockMutate).toHaveBeenCalledTimes(1);
  });

  it("step 1 renders provider select with default value", async () => {
    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);

    // Provider label is present
    expect(screen.getByText("Provider")).toBeInTheDocument();
    // The combobox triggers include both Target Model and Provider selects
    const triggers = screen.getAllByRole("combobox");
    expect(triggers.length).toBeGreaterThanOrEqual(2);
  });

  it("handles undefined modelsData gracefully", async () => {
    vi.mocked(useModels).mockReturnValue({
      data: undefined,
    } as any);

    const user = userEvent.setup();
    render(<NewExperimentPage />);

    await goToStep1(user);

    // Should render step 1 without crashing (models ?? [] fallback)
    expect(screen.getByText("Select Target")).toBeInTheDocument();
  });
});
