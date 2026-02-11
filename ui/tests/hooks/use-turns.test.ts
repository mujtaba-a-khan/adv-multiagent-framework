import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

vi.mock("@/lib/api-client", () => ({
  listTurns: vi.fn().mockResolvedValue({ turns: [], total: 0 }),
  getLatestTurn: vi.fn().mockResolvedValue({
    id: "t1",
    session_id: "s1",
    turn_number: 1,
    strategy_name: "roleplay",
    strategy_params: {},
    attack_prompt: "test prompt",
    attacker_reasoning: null,
    target_response: "I cannot help with that",
    raw_target_response: null,
    target_blocked: false,
    judge_verdict: "refused",
    judge_confidence: 0.95,
    severity_score: null,
    specificity_score: null,
    coherence_score: null,
    vulnerability_category: null,
    attack_technique: null,
    attacker_tokens: 100,
    target_tokens: 50,
    analyzer_tokens: 80,
    created_at: new Date().toISOString(),
  }),
}));

import { useTurns, useLatestTurn } from "@/hooks/use-turns";

function createWrapper() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const Wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: qc }, children);
  Wrapper.displayName = "TestQueryWrapper";
  return Wrapper;
}

describe("useTurns", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches turns for a session", async () => {
    const { result } = renderHook(() => useTurns("s1"), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ turns: [], total: 0 });
  });

  it("does not fetch when sessionId is empty", async () => {
    const { result } = renderHook(() => useTurns(""), {
      wrapper: createWrapper(),
    });
    expect(result.current.isFetching).toBe(false);
  });
});

describe("useLatestTurn", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches latest turn for a session", async () => {
    const { result } = renderHook(() => useLatestTurn("s1"), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.id).toBe("t1");
    expect(result.current.data?.turn_number).toBe(1);
  });

  it("does not fetch when sessionId is empty", async () => {
    const { result } = renderHook(() => useLatestTurn(""), {
      wrapper: createWrapper(),
    });
    expect(result.current.isFetching).toBe(false);
  });
});
