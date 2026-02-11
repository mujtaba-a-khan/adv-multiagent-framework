import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

vi.mock("@/lib/api-client", () => ({
  listDatasetPrompts: vi
    .fn()
    .mockResolvedValue({ items: [], total: 0 }),
  getDatasetStats: vi
    .fn()
    .mockResolvedValue({ total: 100, harmful: 50, harmless: 50 }),
  addDatasetPrompt: vi.fn().mockResolvedValue({ id: "p1" }),
  uploadDatasetFile: vi.fn().mockResolvedValue({ items: [] }),
  updateDatasetPrompt: vi.fn().mockResolvedValue({ id: "p1" }),
  deleteDatasetPrompt: vi.fn().mockResolvedValue(undefined),
  generateHarmlessCounterpart: vi.fn().mockResolvedValue({ id: "p2" }),
  listDatasetSuggestions: vi
    .fn()
    .mockResolvedValue({ suggestions: [] }),
  confirmDatasetSuggestion: vi.fn().mockResolvedValue({ id: "p1" }),
  dismissDatasetSuggestion: vi.fn().mockResolvedValue(undefined),
  loadDatasetModel: vi.fn().mockResolvedValue(undefined),
  unloadDatasetModel: vi.fn().mockResolvedValue(undefined),
  getModelStatus: vi.fn().mockResolvedValue({ models: [] }),
}));

import {
  useDatasetPrompts,
  useDatasetStats,
  useAddDatasetPrompt,
  useUploadDataset,
  useUpdateDatasetPrompt,
  useDeleteDatasetPrompt,
  useGenerateHarmless,
  useDatasetSuggestions,
  useConfirmSuggestion,
  useDismissSuggestion,
  useLoadModel,
  useUnloadModel,
  useModelStatus,
} from "@/hooks/use-dataset";
import {
  addDatasetPrompt,
  uploadDatasetFile,
  updateDatasetPrompt,
  deleteDatasetPrompt,
  generateHarmlessCounterpart,
  confirmDatasetSuggestion,
  dismissDatasetSuggestion,
  loadDatasetModel,
  unloadDatasetModel,
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

describe("useDatasetPrompts", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches dataset prompts", async () => {
    const { result } = renderHook(() => useDatasetPrompts(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ items: [], total: 0 });
  });

  it("passes category and source filters", async () => {
    const { result } = renderHook(
      () => useDatasetPrompts("harmful", "manual"),
      { wrapper: createWrapper() },
    );
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });
});

describe("useDatasetStats", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches dataset stats", async () => {
    const { result } = renderHook(() => useDatasetStats(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.total).toBe(100);
  });
});

describe("useDatasetSuggestions", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches suggestions", async () => {
    const { result } = renderHook(() => useDatasetSuggestions(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
  });
});

describe("useModelStatus", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches model status", async () => {
    const { result } = renderHook(() => useModelStatus(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ models: [] });
  });
});

describe("useAddDatasetPrompt", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useAddDatasetPrompt(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync({
      data: { text: "test", category: "harmful" } as Parameters<
        typeof addDatasetPrompt
      >[0],
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(addDatasetPrompt).toHaveBeenCalledTimes(1);
  });
});

describe("useUploadDataset", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useUploadDataset(), {
      wrapper: createWrapper(),
    });
    const file = new File(["content"], "test.jsonl");
    await result.current.mutateAsync(file);
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(uploadDatasetFile).toHaveBeenCalledTimes(1);
  });
});

describe("useUpdateDatasetPrompt", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useUpdateDatasetPrompt(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync({
      id: "p1",
      data: { text: "updated" } as Parameters<
        typeof updateDatasetPrompt
      >[1],
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(updateDatasetPrompt).toHaveBeenCalledTimes(1);
  });
});

describe("useDeleteDatasetPrompt", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useDeleteDatasetPrompt(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync("p1");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(deleteDatasetPrompt).toHaveBeenCalledWith("p1");
  });
});

describe("useGenerateHarmless", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useGenerateHarmless(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync({
      data: { prompt_id: "p1" } as Parameters<
        typeof generateHarmlessCounterpart
      >[0],
      model: "llama3:8b",
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(generateHarmlessCounterpart).toHaveBeenCalledTimes(1);
  });
});

describe("useConfirmSuggestion", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useConfirmSuggestion(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync({
      id: "s1",
      autoGenerateCounterpart: true,
      model: "llama3:8b",
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(confirmDatasetSuggestion).toHaveBeenCalledTimes(1);
  });
});

describe("useDismissSuggestion", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useDismissSuggestion(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync("s1");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(dismissDatasetSuggestion).toHaveBeenCalledWith("s1");
  });
});

describe("useLoadModel", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useLoadModel(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync("llama3:8b");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(loadDatasetModel).toHaveBeenCalledWith("llama3:8b");
  });
});

describe("useUnloadModel", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useUnloadModel(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync("llama3:8b");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(unloadDatasetModel).toHaveBeenCalledWith("llama3:8b");
  });
});
