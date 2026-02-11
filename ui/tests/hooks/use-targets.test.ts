import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

vi.mock("@/lib/api-client", () => ({
  listModels: vi.fn().mockResolvedValue({
    models: ["llama3:8b", "phi4-reasoning:14b"],
    provider: "ollama",
  }),
  checkHealth: vi.fn().mockResolvedValue({
    provider: "ollama",
    healthy: true,
  }),
}));

import { useModels, useHealthCheck } from "@/hooks/use-targets";

function createWrapper() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const Wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: qc }, children);
  Wrapper.displayName = "TestQueryWrapper";
  return Wrapper;
}

describe("useModels", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches available models", async () => {
    const { result } = renderHook(() => useModels(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({
      models: ["llama3:8b", "phi4-reasoning:14b"],
      provider: "ollama",
    });
  });
});

describe("useHealthCheck", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches health status", async () => {
    const { result } = renderHook(() => useHealthCheck(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({
      provider: "ollama",
      healthy: true,
    });
  });
});
