import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listFineTuningJobs,
  getFineTuningJob,
  createFineTuningJob,
  startFineTuningJob,
  cancelFineTuningJob,
  deleteFineTuningJob,
  listCustomModels,
  deleteOllamaModel,
  getDiskStatus,
  cleanupOrphans,
} from "@/lib/api-client";
import type { CreateFineTuningJobRequest } from "@/lib/types";

export function useFineTuningJobs(status?: string) {
  return useQuery({
    queryKey: ["finetuning-jobs", status],
    queryFn: () => listFineTuningJobs(0, 100, status),
    refetchInterval: 10_000,
  });
}

export function useFineTuningJob(jobId: string) {
  return useQuery({
    queryKey: ["finetuning-job", jobId],
    queryFn: () => getFineTuningJob(jobId),
    enabled: !!jobId,
    refetchInterval: 5_000,
  });
}

export function useCreateFineTuningJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateFineTuningJobRequest) =>
      createFineTuningJob(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["finetuning-jobs"] });
    },
  });
}

export function useStartFineTuningJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => startFineTuningJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["finetuning-jobs"] });
    },
  });
}

export function useCancelFineTuningJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => cancelFineTuningJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["finetuning-jobs"] });
    },
  });
}

export function useDeleteFineTuningJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteFineTuningJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["finetuning-jobs"] });
    },
  });
}

export function useCustomModels() {
  return useQuery({
    queryKey: ["custom-models"],
    queryFn: () => listCustomModels(),
    refetchInterval: 30_000,
  });
}

export function useDeleteOllamaModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => deleteOllamaModel(name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["custom-models"] });
    },
  });
}

export function useDiskStatus() {
  return useQuery({
    queryKey: ["disk-status"],
    queryFn: () => getDiskStatus(),
    refetchInterval: 15_000,
  });
}

export function useCleanupOrphans() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => cleanupOrphans(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["disk-status"] });
    },
  });
}
