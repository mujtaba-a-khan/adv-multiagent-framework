import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import WorkshopPage from "@/app/workshop/page";

// Mocks

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) =>
    <a href={href}>{children}</a>,
}));

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title?: string }) => <div data-testid="header">{title}</div>,
}));

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
}));

vi.mock("@/hooks/use-finetuning", () => ({
  useFineTuningJobs: vi.fn(),
  useCreateFineTuningJob: () => ({ mutate: vi.fn(), isPending: false }),
  useStartFineTuningJob: () => ({ mutate: vi.fn() }),
  useCancelFineTuningJob: () => ({ mutate: vi.fn() }),
  useDeleteFineTuningJob: () => ({ mutate: vi.fn() }),
  useDiskStatus: vi.fn(),
  useCleanupOrphans: () => ({ mutate: vi.fn(), isPending: false }),
}));

import { useFineTuningJobs, useDiskStatus } from "@/hooks/use-finetuning";

// Tests

describe("WorkshopPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state", () => {
    vi.mocked(useFineTuningJobs).mockReturnValue({
      data: undefined,
      isLoading: true,
    } as ReturnType<typeof useFineTuningJobs>);
    vi.mocked(useDiskStatus).mockReturnValue({
      data: undefined,
    } as ReturnType<typeof useDiskStatus>);

    render(<WorkshopPage />);
    // "Model Workshop" appears in both the Header mock and the page h2
    expect(screen.getAllByText("Model Workshop")).toHaveLength(2);
    expect(screen.getByText("New Job")).toBeInTheDocument();
  });

  it("renders empty state when no jobs", () => {
    vi.mocked(useFineTuningJobs).mockReturnValue({
      data: { jobs: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useFineTuningJobs>);
    vi.mocked(useDiskStatus).mockReturnValue({
      data: undefined,
    } as ReturnType<typeof useDiskStatus>);

    render(<WorkshopPage />);
    expect(screen.getByText("No jobs yet")).toBeInTheDocument();
    expect(
      screen.getByText("Create your first fine-tuning job to get started."),
    ).toBeInTheDocument();
  });

  it("renders job list with table headers", () => {
    vi.mocked(useFineTuningJobs).mockReturnValue({
      data: {
        jobs: [
          {
            id: "job-1",
            name: "Test Job",
            job_type: "pull_abliterated",
            source_model: "llama3:8b",
            output_model_name: "llama3-abl",
            status: "pending",
            progress_pct: 0,
            created_at: new Date().toISOString(),
            config: {},
          },
        ],
        total: 1,
      },
      isLoading: false,
    } as ReturnType<typeof useFineTuningJobs>);
    vi.mocked(useDiskStatus).mockReturnValue({
      data: undefined,
    } as ReturnType<typeof useDiskStatus>);

    render(<WorkshopPage />);
    expect(screen.getByText("Test Job")).toBeInTheDocument();
    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getByText("Type")).toBeInTheDocument();
    expect(screen.getByText("Source")).toBeInTheDocument();
    expect(screen.getByText("Output")).toBeInTheDocument();
    expect(screen.getByText("Pull Abliterated")).toBeInTheDocument();
  });

  it("renders disk status card when available", () => {
    vi.mocked(useFineTuningJobs).mockReturnValue({
      data: { jobs: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useFineTuningJobs>);
    vi.mocked(useDiskStatus).mockReturnValue({
      data: {
        disk_total_gb: 500,
        disk_free_gb: 200,
        ollama_storage_gb: 50,
        orphan_count: 0,
        orphan_gb: 0,
      },
    } as ReturnType<typeof useDiskStatus>);

    render(<WorkshopPage />);
    expect(screen.getByText("200GB free")).toBeInTheDocument();
    expect(screen.getByText("/ 500GB")).toBeInTheDocument();
    expect(screen.getByText("50GB")).toBeInTheDocument();
  });

  it("renders search input", () => {
    vi.mocked(useFineTuningJobs).mockReturnValue({
      data: { jobs: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useFineTuningJobs>);
    vi.mocked(useDiskStatus).mockReturnValue({
      data: undefined,
    } as ReturnType<typeof useDiskStatus>);

    render(<WorkshopPage />);
    expect(screen.getByPlaceholderText("Search jobs...")).toBeInTheDocument();
  });

  it("renders page description", () => {
    vi.mocked(useFineTuningJobs).mockReturnValue({
      data: { jobs: [], total: 0 },
      isLoading: false,
    } as ReturnType<typeof useFineTuningJobs>);
    vi.mocked(useDiskStatus).mockReturnValue({
      data: undefined,
    } as ReturnType<typeof useDiskStatus>);

    render(<WorkshopPage />);
    expect(
      screen.getByText(/Fine-tune and abliterate models/),
    ).toBeInTheDocument();
  });
});
