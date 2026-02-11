import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

vi.mock("@/lib/api-client", () => ({
  getSessionReport: vi.fn().mockResolvedValue({ session_id: "s1", turns: [] }),
  listExperiments: vi.fn().mockResolvedValue({ experiments: [{ id: "e1" }] }),
  listSessions: vi.fn().mockResolvedValue({
    sessions: [{ id: "s1", status: "completed" }],
  }),
}));

import { useSessionReport, useCompletedSessions } from "@/hooks/use-reports";

function createWrapper() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const Wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: qc }, children);
  Wrapper.displayName = "TestQueryWrapper";
  return Wrapper;
}

describe("useSessionReport", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches session report", async () => {
    const { result } = renderHook(
      () => useSessionReport("e1", "s1"),
      { wrapper: createWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.session_id).toBe("s1");
  });

  it("disabled when ids are empty", () => {
    const { result } = renderHook(
      () => useSessionReport("", ""),
      { wrapper: createWrapper() },
    );
    expect(result.current.fetchStatus).toBe("idle");
  });
});

describe("useCompletedSessions", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches and flattens completed sessions", async () => {
    const { result } = renderHook(
      () => useCompletedSessions(),
      { wrapper: createWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toHaveLength(1);
    expect(result.current.data?.[0]).toHaveProperty("experiment");
  });
});
