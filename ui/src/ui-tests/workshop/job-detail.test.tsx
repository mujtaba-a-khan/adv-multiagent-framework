import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import WorkshopJobDetailPage from "@/app/workshop/[jobId]/page";

// Mocks

const mockPush = vi.fn();

vi.mock("next/navigation", () => ({
  useParams: () => ({ jobId: "job-123" }),
  useRouter: () => ({ push: mockPush }),
}));

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

vi.mock("@/hooks/use-finetuning", () => ({
  useFineTuningJob: vi.fn(),
  useStartFineTuningJob: () => ({ mutate: vi.fn(), isPending: false }),
  useCancelFineTuningJob: () => ({ mutate: vi.fn(), isPending: false }),
  useDeleteFineTuningJob: () => ({ mutate: vi.fn(), isPending: false }),
}));

vi.mock("@/stores/ft-ws-store", () => ({
  useFTWSStore: () => ({
    connect: vi.fn(),
    disconnect: vi.fn(),
    progressPct: 0,
    currentStep: null,
    status: null,
    logs: [],
    error: null,
    outputModel: null,
    durationS: null,
  }),
}));

import { useFineTuningJob } from "@/hooks/use-finetuning";

// Tests

const MOCK_JOB = {
  id: "job-123",
  name: "Test Abliteration Job",
  job_type: "abliterate",
  source_model: "meta-llama/Meta-Llama-3-8B",
  output_model_name: "llama3-abl",
  status: "pending",
  progress_pct: 0,
  current_step: null,
  error_message: null,
  peak_memory_gb: null,
  total_duration_seconds: null,
  config: {},
  created_at: new Date().toISOString(),
};

describe("WorkshopJobDetailPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state", () => {
    vi.mocked(useFineTuningJob).mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useFineTuningJob>);

    render(<WorkshopJobDetailPage />);
    expect(screen.getByTestId("header")).toHaveTextContent("Model Workshop");
  });

  it("renders not found state", () => {
    vi.mocked(useFineTuningJob).mockReturnValue({
      data: undefined,
      isLoading: false,
    } as ReturnType<typeof useFineTuningJob>);

    render(<WorkshopJobDetailPage />);
    expect(screen.getByText("Job not found.")).toBeInTheDocument();
    expect(screen.getByText("Back to Workshop")).toBeInTheDocument();
  });

  it("renders job detail with name and badges", () => {
    vi.mocked(useFineTuningJob).mockReturnValue({
      data: MOCK_JOB,
      isLoading: false,
    } as ReturnType<typeof useFineTuningJob>);

    render(<WorkshopJobDetailPage />);
    expect(screen.getByText("Test Abliteration Job")).toBeInTheDocument();
    expect(screen.getByText("Custom Abliterate")).toBeInTheDocument();
    expect(screen.getByText("pending")).toBeInTheDocument();
  });

  it("renders Start button for pending job", () => {
    vi.mocked(useFineTuningJob).mockReturnValue({
      data: MOCK_JOB,
      isLoading: false,
    } as ReturnType<typeof useFineTuningJob>);

    render(<WorkshopJobDetailPage />);
    expect(screen.getByRole("button", { name: /start/i })).toBeInTheDocument();
  });

  it("renders Delete button for non-running job", () => {
    vi.mocked(useFineTuningJob).mockReturnValue({
      data: MOCK_JOB,
      isLoading: false,
    } as ReturnType<typeof useFineTuningJob>);

    render(<WorkshopJobDetailPage />);
    expect(screen.getByRole("button", { name: /delete/i })).toBeInTheDocument();
  });

  it("renders Job Details card with source and output", () => {
    vi.mocked(useFineTuningJob).mockReturnValue({
      data: MOCK_JOB,
      isLoading: false,
    } as ReturnType<typeof useFineTuningJob>);

    render(<WorkshopJobDetailPage />);
    expect(screen.getByText("Job Details")).toBeInTheDocument();
    expect(screen.getByText("Source Model")).toBeInTheDocument();
    expect(screen.getByText("meta-llama/Meta-Llama-3-8B")).toBeInTheDocument();
    expect(screen.getByText("Output Name")).toBeInTheDocument();
    expect(screen.getByText("llama3-abl")).toBeInTheDocument();
  });

  it("renders Configuration card", () => {
    vi.mocked(useFineTuningJob).mockReturnValue({
      data: MOCK_JOB,
      isLoading: false,
    } as ReturnType<typeof useFineTuningJob>);

    render(<WorkshopJobDetailPage />);
    expect(screen.getByText("Configuration")).toBeInTheDocument();
    expect(screen.getByText("Default configuration")).toBeInTheDocument();
  });

  it("renders completed job with progress bar", () => {
    vi.mocked(useFineTuningJob).mockReturnValue({
      data: {
        ...MOCK_JOB,
        status: "completed",
        progress_pct: 100,
        total_duration_seconds: 120,
      },
      isLoading: false,
    } as ReturnType<typeof useFineTuningJob>);

    render(<WorkshopJobDetailPage />);
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getByText(/Completed/)).toBeInTheDocument();
  });
});
