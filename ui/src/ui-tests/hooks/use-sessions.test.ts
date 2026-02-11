import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

vi.mock("@/lib/api-client", () => ({
  listSessions: vi.fn().mockResolvedValue({ sessions: [], total: 0 }),
  getSession: vi.fn().mockResolvedValue({ id: "s1", status: "completed" }),
  createSession: vi.fn().mockResolvedValue({ id: "s1" }),
  startSession: vi.fn().mockResolvedValue({ id: "s1", status: "running" }),
  deleteSession: vi.fn().mockResolvedValue(undefined),
}));

import {
  useSessions,
  useSession,
  useCreateSession,
  useStartSession,
  useDeleteSession,
} from "@/hooks/use-sessions";
import {
  createSession,
  startSession,
  deleteSession,
} from "@/lib/api-client";

function createWrapper() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const Wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: qc }, children);
  Wrapper.displayName = "TestQueryWrapper";
  return Wrapper;
}

describe("useSessions", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches sessions for experiment", async () => {
    const { result } = renderHook(() => useSessions("e1"), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ sessions: [], total: 0 });
  });

  it("disabled when no experimentId", () => {
    const { result } = renderHook(() => useSessions(""), {
      wrapper: createWrapper(),
    });
    expect(result.current.fetchStatus).toBe("idle");
  });
});

describe("useSession", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches single session", async () => {
    const { result } = renderHook(() => useSession("e1", "s1"), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.id).toBe("s1");
  });
});

describe("useCreateSession", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useCreateSession("e1"), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync(undefined);
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(createSession).toHaveBeenCalledTimes(1);
  });
});

describe("useStartSession", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useStartSession("e1"), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync("s1");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(startSession).toHaveBeenCalledWith("e1", "s1");
  });
});

describe("useDeleteSession", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useDeleteSession("e1"), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync("s1");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(deleteSession).toHaveBeenCalledWith("e1", "s1");
  });
});
