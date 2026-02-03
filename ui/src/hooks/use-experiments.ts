import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listExperiments,
  getExperiment,
  createExperiment,
  updateExperiment,
  deleteExperiment,
} from "@/lib/api-client";
import type {
  CreateExperimentRequest,
  UpdateExperimentRequest,
} from "@/lib/types";

export function useExperiments(offset = 0, limit = 50) {
  return useQuery({
    queryKey: ["experiments", offset, limit],
    queryFn: () => listExperiments(offset, limit),
  });
}

export function useExperiment(id: string) {
  return useQuery({
    queryKey: ["experiment", id],
    queryFn: () => getExperiment(id),
    enabled: !!id,
  });
}

export function useCreateExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateExperimentRequest) => createExperiment(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}

export function useUpdateExperiment(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: UpdateExperimentRequest) => updateExperiment(id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["experiments"] });
      qc.invalidateQueries({ queryKey: ["experiment", id] });
    },
  });
}

export function useDeleteExperiment() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => deleteExperiment(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}
