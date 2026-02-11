import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

vi.mock("@/lib/api-client", () => ({
  listExperiments: vi.fn().mockResolvedValue({ experiments: [], total: 0 }),
  getExperiment: vi.fn().mockResolvedValue({ id: "e1", name: "Test" }),
  createExperiment: vi.fn().mockResolvedValue({ id: "e1" }),
  updateExperiment: vi.fn().mockResolvedValue({ id: "e1" }),
  deleteExperiment: vi.fn().mockResolvedValue(undefined),
}));

import {
  useExperiments,
  useExperiment,
  useCreateExperiment,
  useUpdateExperiment,
  useDeleteExperiment,
} from "@/hooks/use-experiments";
import {
  createExperiment,
  updateExperiment,
  deleteExperiment,
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

describe("useExperiments", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches experiments", async () => {
    const { result } = renderHook(() => useExperiments(), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ experiments: [], total: 0 });
  });
});

describe("useExperiment", () => {
  beforeEach(() => vi.clearAllMocks());

  it("fetches single experiment", async () => {
    const { result } = renderHook(() => useExperiment("e1"), {
      wrapper: createWrapper(),
    });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual({ id: "e1", name: "Test" });
  });
});

describe("useCreateExperiment", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useCreateExperiment(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync({
      name: "test",
      target_model: "m",
      attack_objective: "o",
    } as Parameters<typeof createExperiment>[0]);
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(createExperiment).toHaveBeenCalledTimes(1);
  });
});

describe("useUpdateExperiment", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useUpdateExperiment("e1"), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync({
      name: "updated",
    } as Parameters<typeof updateExperiment>[1]);
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(updateExperiment).toHaveBeenCalledTimes(1);
  });
});

describe("useDeleteExperiment", () => {
  beforeEach(() => vi.clearAllMocks());

  it("calls API and invalidates on success", async () => {
    const { result } = renderHook(() => useDeleteExperiment(), {
      wrapper: createWrapper(),
    });
    await result.current.mutateAsync("e1");
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(deleteExperiment).toHaveBeenCalledWith("e1");
  });
});
