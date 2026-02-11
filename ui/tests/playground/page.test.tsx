import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import PlaygroundNewChatPage from "@/app/playground/page";

// ── Mocks ────────────────────────────────────────────────────

const mockPush = vi.fn();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: mockPush }),
}));

const mockCreateMutateAsync = vi.fn();
const mockSendMutateAsync = vi.fn();

vi.mock("@/hooks/use-playground", () => ({
  useCreatePlaygroundConversation: () => ({
    mutateAsync: mockCreateMutateAsync,
    isPending: false,
  }),
  useSendPlaygroundMessage: () => ({
    mutateAsync: mockSendMutateAsync,
    isPending: false,
    variables: null,
  }),
}));

vi.mock("@/hooks/use-targets", () => ({
  useModels: () => ({
    data: { models: ["llama3:8b", "gpt-4o", "mistral:7b"] },
  }),
}));

vi.mock("@/hooks/use-defenses", () => ({
  useDefenses: () => ({
    data: {
      defenses: [
        { name: "rule_based", description: "Pattern matching" },
        { name: "llm_judge", description: "LLM-based screening" },
      ],
    },
  }),
}));

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

// ── Tests ────────────────────────────────────────────────────

describe("PlaygroundNewChatPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders empty state with welcome message", () => {
    render(<PlaygroundNewChatPage />);
    expect(screen.getByText("Playground")).toBeInTheDocument();
    expect(
      screen.getByText(/Test LLM defenses with manual red-teaming/),
    ).toBeInTheDocument();
  });

  it("renders target model selector", () => {
    render(<PlaygroundNewChatPage />);
    expect(
      screen.getByText("Select target model..."),
    ).toBeInTheDocument();
  });

  it("renders analyzer model selector", () => {
    render(<PlaygroundNewChatPage />);
    expect(
      screen.getByText("Analyzer (default)"),
    ).toBeInTheDocument();
  });

  it("renders defenses button", () => {
    render(<PlaygroundNewChatPage />);
    expect(screen.getByText("Defenses")).toBeInTheDocument();
  });

  it("renders system prompt button", () => {
    render(<PlaygroundNewChatPage />);
    expect(screen.getByText("System")).toBeInTheDocument();
  });

  it("disables textarea when no model selected", () => {
    render(<PlaygroundNewChatPage />);
    const textarea = screen.getByPlaceholderText(
      "Select a target model first...",
    );
    expect(textarea).toBeDisabled();
  });

  it("send button disabled when prompt is empty", () => {
    render(<PlaygroundNewChatPage />);
    // The send button (icon only) should be disabled when no model + no prompt
    const buttons = screen.getAllByRole("button");
    const sendButton = buttons.at(-1)!;
    expect(sendButton).toBeDisabled();
  });

  it("creates conversation and sends message on first send", async () => {
    const user = userEvent.setup();

    mockCreateMutateAsync.mockResolvedValue({
      id: "new-conv-id",
      title: "New Chat",
      target_model: "llama3:8b",
    });
    mockSendMutateAsync.mockResolvedValue({
      id: "msg-1",
      message_number: 1,
    });

    render(<PlaygroundNewChatPage />);

    // Select model — use fireEvent since Radix Select uses pointer-events:none
    // First combobox = target model, second = analyzer model
    const selectTriggers = screen.getAllByRole("combobox");
    fireEvent.click(selectTriggers[0]);

    // Select llama3:8b from dropdown
    const option = await screen.findByText("llama3:8b");
    fireEvent.click(option);

    // Type prompt
    const textarea = screen.getByPlaceholderText(
      "Type your attack prompt...",
    );
    await user.type(textarea, "Hello, can you help me?");

    // Click send
    const buttons = screen.getAllByRole("button");
    const sendButton = buttons.at(-1)!;
    await user.click(sendButton);

    // Should have created conversation with auto-generated title
    expect(mockCreateMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Hello, can you help me?",
        target_model: "llama3:8b",
      }),
    );

    // Should navigate to the new conversation (redirect before send)
    expect(mockPush).toHaveBeenCalledWith("/playground/new-conv-id");

    // Should have sent message
    expect(mockSendMutateAsync).toHaveBeenCalledWith({
      conversationId: "new-conv-id",
      prompt: "Hello, can you help me?",
    });
  });
});
