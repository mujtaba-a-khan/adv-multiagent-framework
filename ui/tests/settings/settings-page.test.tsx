import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import SettingsPage from "@/app/settings/page";

// Mocks

vi.mock("@/components/layout/header", () => ({
  Header: ({ title }: { title?: string }) => (
    <div data-testid="header">{title}</div>
  ),
}));

const mockToast = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
}));
vi.mock("sonner", () => ({
  toast: mockToast,
}));

// Tests

describe("SettingsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders header and title", () => {
    render(<SettingsPage />);
    expect(screen.getByTestId("header")).toHaveTextContent("Settings");
    expect(screen.getAllByText("Settings").length).toBeGreaterThan(0);
    expect(
      screen.getByText("Configure providers, API keys, and defaults."),
    ).toBeInTheDocument();
  });

  it("renders ollama configuration section", () => {
    render(<SettingsPage />);
    expect(screen.getByText("Ollama Configuration")).toBeInTheDocument();
    expect(screen.getByLabelText("Ollama Base URL")).toBeInTheDocument();
    expect(screen.getByDisplayValue("http://localhost:11434")).toBeInTheDocument();
  });

  it("renders API keys section", () => {
    render(<SettingsPage />);
    expect(screen.getByText("API Keys")).toBeInTheDocument();
    expect(screen.getByText("OpenAI API Key")).toBeInTheDocument();
    expect(screen.getByText("Anthropic API Key")).toBeInTheDocument();
    expect(screen.getByText("Google AI API Key")).toBeInTheDocument();
  });

  it("renders defaults section with dark mode toggle", () => {
    render(<SettingsPage />);
    expect(screen.getByText("Defaults")).toBeInTheDocument();
    expect(screen.getByText("Dark Mode")).toBeInTheDocument();
  });

  it("renders save button", () => {
    render(<SettingsPage />);
    expect(screen.getByText("Save Settings")).toBeInTheDocument();
  });

  it("calls toast on save", async () => {
    const user = userEvent.setup();
    render(<SettingsPage />);

    const saveButton = screen.getByText("Save Settings");
    await user.click(saveButton);

    expect(mockToast.success).toHaveBeenCalledWith("Settings saved");
  });

  it("renders show/hide toggle for API keys", () => {
    render(<SettingsPage />);
    expect(screen.getByText("Show")).toBeInTheDocument();
  });

  it("renders API key inputs as password by default", () => {
    render(<SettingsPage />);
    const passwordInputs = screen.getAllByPlaceholderText(/sk-|AI/);
    for (const input of passwordInputs) {
      expect(input).toHaveAttribute("type", "password");
    }
  });
});
