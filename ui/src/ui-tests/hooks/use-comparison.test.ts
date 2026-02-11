import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

vi.mock("@/lib/api-client", () => ({
  compareSessions: vi.fn().mockResolvedValue({
    attack_session: { id: "a1" },
    defense_session: { id: "d1" },
  }),
}));

import { useComparison } from "@/hooks/use-comparison";

function createWrapper() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const Wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: qc }, children);
  Wrapper.displayName = "TestQueryWrapper";
  return Wrapper;
}

describe("useComparison", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches comparison data", async () => {
    const { result } = renderHook(
      () => useComparison("e1", "a1", "d1"),
      { wrapper: createWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.attack_session?.id).toBe("a1");
  });

  it("disabled when any id is empty", () => {
    const { result } = renderHook(
      () => useComparison("", "a1", "d1"),
      { wrapper: createWrapper() },
    );
    expect(result.current.fetchStatus).toBe("idle");
  });
});
