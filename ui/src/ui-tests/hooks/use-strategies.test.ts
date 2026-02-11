import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

vi.mock("@/lib/api-client", () => ({
  listStrategies: vi.fn().mockResolvedValue({ strategies: [], total: 0 }),
  getStrategy: vi.fn().mockResolvedValue({
    name: "roleplay",
    display_name: "Roleplay",
    category: "prompt_level",
    description: "Uses persona-based jailbreaking",
    estimated_asr: "Medium",
    min_turns: 1,
    max_turns: null,
    requires_white_box: false,
    supports_multi_turn: true,
    parameters: {},
    references: [],
  }),
}));

import { useStrategies, useStrategy } from "@/hooks/use-strategies";

function createWrapper() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const Wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: qc }, children);
  Wrapper.displayName = "TestQueryWrapper";
  return Wrapper;
}

describe("useStrategies", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches strategies list", async () => {
    const { result } = renderHook(() => useStrategies(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ strategies: [], total: 0 });
  });
});

describe("useStrategy", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches single strategy by name", async () => {
    const { result } = renderHook(() => useStrategy("roleplay"), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.name).toBe("roleplay");
    expect(result.current.data?.display_name).toBe("Roleplay");
  });

  it("does not fetch when name is empty", async () => {
    const { result } = renderHook(() => useStrategy(""), {
      wrapper: createWrapper(),
    });
    // Should not trigger fetch — stays idle
    expect(result.current.isFetching).toBe(false);
  });
});
