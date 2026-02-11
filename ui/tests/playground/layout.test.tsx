import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import PlaygroundLayout from "@/app/playground/layout";

// ── Mocks ────────────────────────────────────────────────────

const mockPush = vi.fn();
const mockPathname = vi.fn(() => "/playground");

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
  usePathname: () => mockPathname(),
}));

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...rest
  }: {
    children: React.ReactNode;
    href: string;
    className?: string;
  }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

const mockConversations = [
  {
    id: "conv-1",
    title: "Testing llama3",
    target_model: "llama3:8b",
    target_provider: "ollama",
    system_prompt: null,
    analyzer_model: "phi4-reasoning:14b",
    active_defenses: [],
    total_messages: 12,
    total_jailbreaks: 3,
    total_blocked: 1,
    total_target_tokens: 1000,
    total_analyzer_tokens: 500,
    created_at: "2025-01-01T00:00:00Z",
    updated_at: "2025-01-02T00:00:00Z",
  },
  {
    id: "conv-2",
    title: "GPT-4o probe",
    target_model: "gpt-4o",
    target_provider: "openai",
    system_prompt: "You are helpful.",
    analyzer_model: "phi4-reasoning:14b",
    active_defenses: [{ name: "rule_based" }],
    total_messages: 5,
    total_jailbreaks: 0,
    total_blocked: 0,
    total_target_tokens: 300,
    total_analyzer_tokens: 200,
    created_at: "2025-01-01T00:00:00Z",
    updated_at: "2025-01-01T12:00:00Z",
  },
];

const mockUsePlaygroundConversations = vi.fn(() => ({
  data: { conversations: mockConversations, total: 2 },
  isLoading: false,
}));

const mockDeleteMutateAsync = vi.fn();
const mockUseDeletePlaygroundConversation = vi.fn(() => ({
  mutateAsync: mockDeleteMutateAsync,
}));

vi.mock("@/hooks/use-playground", () => ({
  usePlaygroundConversations: () => mockUsePlaygroundConversations(),
  useDeletePlaygroundConversation: () =>
    mockUseDeletePlaygroundConversation(),
}));

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title: string }) => (
    <div data-testid="header">{title}</div>
  ),
}));

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

// ── Tests ────────────────────────────────────────────────────

describe("PlaygroundLayout", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockPathname.mockReturnValue("/playground");
    // Reset mock to default data (mockReturnValue persists through clearAllMocks)
    mockUsePlaygroundConversations.mockReturnValue({
      data: { conversations: mockConversations, total: 2 },
      isLoading: false,
    });
  });

  it("renders the header with title", () => {
    render(
      <PlaygroundLayout>
        <div>child content</div>
      </PlaygroundLayout>,
    );
    expect(screen.getByTestId("header")).toHaveTextContent("Playground");
  });

  it("renders conversation list from API data", () => {
    render(
      <PlaygroundLayout>
        <div>child</div>
      </PlaygroundLayout>,
    );
    expect(screen.getByText("Testing llama3")).toBeInTheDocument();
    expect(screen.getByText("GPT-4o probe")).toBeInTheDocument();
  });

  it("renders children in the main area", () => {
    render(
      <PlaygroundLayout>
        <div data-testid="child-content">My chat content</div>
      </PlaygroundLayout>,
    );
    expect(screen.getByTestId("child-content")).toBeInTheDocument();
  });

  it("shows model badges for each conversation", () => {
    render(
      <PlaygroundLayout>
        <div>child</div>
      </PlaygroundLayout>,
    );
    expect(screen.getByText("llama3:8b")).toBeInTheDocument();
    expect(screen.getByText("gpt-4o")).toBeInTheDocument();
  });

  it("shows jailbreak badge when count > 0", () => {
    render(
      <PlaygroundLayout>
        <div>child</div>
      </PlaygroundLayout>,
    );
    expect(screen.getByText("3 jailbreaks")).toBeInTheDocument();
  });

  it("filters conversations by search query", async () => {
    const user = userEvent.setup();
    render(
      <PlaygroundLayout>
        <div>child</div>
      </PlaygroundLayout>,
    );

    const searchInput = screen.getByPlaceholderText("Search...");
    await user.type(searchInput, "llama");

    expect(screen.getByText("Testing llama3")).toBeInTheDocument();
    expect(screen.queryByText("GPT-4o probe")).not.toBeInTheDocument();
  });

  it("filters by model name", async () => {
    const user = userEvent.setup();
    render(
      <PlaygroundLayout>
        <div>child</div>
      </PlaygroundLayout>,
    );

    const searchInput = screen.getByPlaceholderText("Search...");
    await user.type(searchInput, "gpt-4o");

    expect(screen.queryByText("Testing llama3")).not.toBeInTheDocument();
    expect(screen.getByText("GPT-4o probe")).toBeInTheDocument();
  });

  it("highlights active conversation", () => {
    mockPathname.mockReturnValue("/playground/conv-1");
    render(
      <PlaygroundLayout>
        <div>child</div>
      </PlaygroundLayout>,
    );

    // Find the conv-1 link and check its container has active styling
    const conv1Link = screen.getByText("Testing llama3").closest("div");
    expect(conv1Link?.className).toContain("border-primary/40");
  });

  it("does not highlight inactive conversations", () => {
    mockPathname.mockReturnValue("/playground/conv-1");
    render(
      <PlaygroundLayout>
        <div>child</div>
      </PlaygroundLayout>,
    );

    const conv2Link = screen.getByText("GPT-4o probe").closest("div");
    expect(conv2Link?.className).toContain("border-transparent");
  });

  it("renders New Chat button with link to /playground", () => {
    render(
      <PlaygroundLayout>
        <div>child</div>
      </PlaygroundLayout>,
    );

    const newChatLink = screen.getByText("New Chat").closest("a");
    expect(newChatLink).toHaveAttribute("href", "/playground");
  });

  it("shows loading skeletons when data is loading", () => {
    mockUsePlaygroundConversations.mockReturnValue({
      data: undefined as unknown as { conversations: typeof mockConversations; total: number },
      isLoading: true,
    });

    render(
      <PlaygroundLayout>
        <div>child</div>
      </PlaygroundLayout>,
    );

    // Should not show any conversation titles
    expect(screen.queryByText("Testing llama3")).not.toBeInTheDocument();
  });

  it("shows empty state when no conversations exist", () => {
    mockUsePlaygroundConversations.mockReturnValue({
      data: { conversations: [], total: 0 },
      isLoading: false,
    });

    render(
      <PlaygroundLayout>
        <div>child</div>
      </PlaygroundLayout>,
    );

    expect(screen.getByText("No conversations yet")).toBeInTheDocument();
  });

  it("calls delete mutation when delete button clicked", async () => {
    const user = userEvent.setup();
    mockDeleteMutateAsync.mockResolvedValue(undefined);

    render(
      <PlaygroundLayout>
        <div>child</div>
      </PlaygroundLayout>,
    );

    // Find delete buttons (they have Trash2 icons)
    const conv1Container = screen
      .getByText("Testing llama3")
      .closest(".group");
    const deleteBtn = conv1Container?.querySelector("button");
    expect(deleteBtn).toBeTruthy();

    await user.click(deleteBtn!);
    expect(mockDeleteMutateAsync).toHaveBeenCalledWith("conv-1");
  });

  it("navigates to /playground on deleting the active conversation", async () => {
    const user = userEvent.setup();
    mockPathname.mockReturnValue("/playground/conv-1");
    mockDeleteMutateAsync.mockResolvedValue(undefined);

    render(
      <PlaygroundLayout>
        <div>child</div>
      </PlaygroundLayout>,
    );

    const conv1Container = screen
      .getByText("Testing llama3")
      .closest(".group");
    const deleteBtn = conv1Container?.querySelector("button");

    await user.click(deleteBtn!);
    expect(mockPush).toHaveBeenCalledWith("/playground");
  });
});
