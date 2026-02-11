import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

vi.mock("@/lib/api-client", () => ({
  listDefenses: vi.fn().mockResolvedValue({
    defenses: [
      {
        name: "input_classifier",
        description: "Classifies input as malicious or benign",
        defense_type: "input",
      },
      {
        name: "llm_judge",
        description: "LLM-powered input analysis",
        defense_type: "input",
      },
    ],
    total: 2,
  }),
}));

import { useDefenses } from "@/hooks/use-defenses";

function createWrapper() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const Wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: qc }, children);
  Wrapper.displayName = "TestQueryWrapper";
  return Wrapper;
}

describe("useDefenses", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches defense list", async () => {
    const { result } = renderHook(() => useDefenses(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.defenses).toHaveLength(2);
    expect(result.current.data?.defenses[0].name).toBe("input_classifier");
    expect(result.current.data?.total).toBe(2);
  });

  it("returns loading state initially", () => {
    const { result } = renderHook(() => useDefenses(), {
      wrapper: createWrapper(),
    });
    expect(result.current.isLoading).toBe(true);
  });
});
