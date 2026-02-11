import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

vi.mock("@/lib/api-client", () => ({
  listPlaygroundConversations: vi
    .fn()
    .mockResolvedValue({ items: [], total: 0 }),
  getPlaygroundConversation: vi
    .fn()
    .mockResolvedValue({ id: "c1", title: "Test" }),
  createPlaygroundConversation: vi.fn().mockResolvedValue({ id: "c1" }),
  updatePlaygroundConversation: vi.fn().mockResolvedValue({ id: "c1" }),
  deletePlaygroundConversation: vi.fn().mockResolvedValue(undefined),
  listPlaygroundMessages: vi
    .fn()
    .mockResolvedValue({ items: [], total: 0 }),
  sendPlaygroundMessage: vi.fn().mockResolvedValue({ id: "m1" }),
}));

import {
  usePlaygroundConversations,
  usePlaygroundConversation,
  useCreatePlaygroundConversation,
  useUpdatePlaygroundConversation,
  useDeletePlaygroundConversation,
  usePlaygroundMessages,
  useSendPlaygroundMessage,
} from "@/hooks/use-playground";
import {
  createPlaygroundConversation,
  updatePlaygroundConversation,
  deletePlaygroundConversation,
  sendPlaygroundMessage,
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

describe("usePlaygroundConversations", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches conversations", async () => {
    const { result } = renderHook(() => usePlaygroundConversations(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ items: [], total: 0 });
  });
});

describe("usePlaygroundConversation", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches single conversation", async () => {
    const { result } = renderHook(
      () => usePlaygroundConversation("c1"),
      { wrapper: createWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.id).toBe("c1");
  });

  it("disabled when id is empty", () => {
    const { result } = renderHook(
      () => usePlaygroundConversation(""),
      { wrapper: createWrapper() },
    );
    expect(result.current.fetchStatus).toBe("idle");
  });
});

describe("usePlaygroundMessages", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches messages for conversation", async () => {
    const { result } = renderHook(
      () => usePlaygroundMessages("c1"),
      { wrapper: createWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ items: [], total: 0 });
  });
});

describe("useCreatePlaygroundConversation", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(
      () => useCreatePlaygroundConversation(),
      { wrapper: createWrapper() },
    );
    await result.current.mutateAsync({
      target_model: "m",
    } as Parameters<typeof createPlaygroundConversation>[0]);
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(createPlaygroundConversation).toHaveBeenCalledTimes(1);
  });
});

describe("useUpdatePlaygroundConversation", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(
      () => useUpdatePlaygroundConversation(),
      { wrapper: createWrapper() },
    );
    await result.current.mutateAsync({
      id: "c1",
      data: {} as Parameters<typeof updatePlaygroundConversation>[1],
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(updatePlaygroundConversation).toHaveBeenCalledTimes(1);
  });
});

describe("useDeletePlaygroundConversation", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(
      () => useDeletePlaygroundConversation(),
      { wrapper: createWrapper() },
    );
    await result.current.mutateAsync("c1");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(deletePlaygroundConversation).toHaveBeenCalledWith("c1");
  });
});

describe("useSendPlaygroundMessage", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(
      () => useSendPlaygroundMessage(),
      { wrapper: createWrapper() },
    );
    await result.current.mutateAsync({
      conversationId: "c1",
      prompt: "Hello",
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(sendPlaygroundMessage).toHaveBeenCalledWith("c1", "Hello");
  });
});
