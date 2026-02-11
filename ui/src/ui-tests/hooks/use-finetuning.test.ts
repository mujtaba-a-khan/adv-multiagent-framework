import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

vi.mock("@/lib/api-client", () => ({
  listFineTuningJobs: vi.fn().mockResolvedValue({ items: [], total: 0 }),
  getFineTuningJob: vi
    .fn()
    .mockResolvedValue({ id: "j1", status: "pending" }),
  createFineTuningJob: vi.fn().mockResolvedValue({ id: "j1" }),
  startFineTuningJob: vi.fn().mockResolvedValue({ id: "j1" }),
  cancelFineTuningJob: vi
    .fn()
    .mockResolvedValue({ status: "cancelled" }),
  deleteFineTuningJob: vi.fn().mockResolvedValue(undefined),
  listCustomModels: vi.fn().mockResolvedValue({ models: [] }),
  deleteOllamaModel: vi.fn().mockResolvedValue(undefined),
  getDiskStatus: vi
    .fn()
    .mockResolvedValue({ disk_total_gb: 100, disk_free_gb: 50 }),
  cleanupOrphans: vi.fn().mockResolvedValue({ freed_gb: 1.5 }),
}));

import {
  useFineTuningJobs,
  useFineTuningJob,
  useCreateFineTuningJob,
  useStartFineTuningJob,
  useCancelFineTuningJob,
  useDeleteFineTuningJob,
  useCustomModels,
  useDeleteOllamaModel,
  useDiskStatus,
  useCleanupOrphans,
} from "@/hooks/use-finetuning";
import {
  createFineTuningJob,
  startFineTuningJob,
  cancelFineTuningJob,
  deleteFineTuningJob,
  deleteOllamaModel,
  cleanupOrphans,
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

describe("useFineTuningJobs", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches jobs", async () => {
    const { result } = renderHook(() => useFineTuningJobs(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ items: [], total: 0 });
  });
});

describe("useFineTuningJob", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches single job", async () => {
    const { result } = renderHook(() => useFineTuningJob("j1"), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.id).toBe("j1");
  });
});

describe("useCreateFineTuningJob", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useCreateFineTuningJob(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync({
      job_type: "abliterate",
      base_model: "m",
    } as Parameters<typeof createFineTuningJob>[0]);
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(createFineTuningJob).toHaveBeenCalledTimes(1);
  });
});

describe("useStartFineTuningJob", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useStartFineTuningJob(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync("j1");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(startFineTuningJob).toHaveBeenCalledWith("j1");
  });
});

describe("useCancelFineTuningJob", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useCancelFineTuningJob(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync("j1");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(cancelFineTuningJob).toHaveBeenCalledWith("j1");
  });
});

describe("useDeleteFineTuningJob", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useDeleteFineTuningJob(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync("j1");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(deleteFineTuningJob).toHaveBeenCalledWith("j1");
  });
});

describe("useDeleteOllamaModel", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useDeleteOllamaModel(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync("custom-model");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(deleteOllamaModel).toHaveBeenCalledWith("custom-model");
  });
});

describe("useCleanupOrphans", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useCleanupOrphans(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync();
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(cleanupOrphans).toHaveBeenCalledTimes(1);
  });
});

describe("useCustomModels", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches custom models", async () => {
    const { result } = renderHook(() => useCustomModels(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ models: [] });
  });
});

describe("useDiskStatus", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches disk status", async () => {
    const { result } = renderHook(() => useDiskStatus(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.disk_total_gb).toBe(100);
  });
});
