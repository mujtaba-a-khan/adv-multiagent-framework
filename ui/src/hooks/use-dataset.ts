import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listDatasetPrompts,
  getDatasetStats,
  addDatasetPrompt,
  uploadDatasetFile,
  updateDatasetPrompt,
  deleteDatasetPrompt,
  generateHarmlessCounterpart,
  listDatasetSuggestions,
  confirmDatasetSuggestion,
  dismissDatasetSuggestion,
  loadDatasetModel,
  unloadDatasetModel,
  getModelStatus,
} from "@/lib/api-client";
import type {
  CreateAbliterationPromptRequest,
  UpdateAbliterationPromptRequest,
  GenerateHarmlessRequest,
} from "@/lib/types";

export function useDatasetPrompts(category?: string, source?: string) {
  return useQuery({
    queryKey: ["dataset-prompts", category, source],
    queryFn: () => listDatasetPrompts(0, 200, category, source),
    refetchInterval: 30_000,
  });
}

export function useDatasetStats() {
  return useQuery({
    queryKey: ["dataset-stats"],
    queryFn: () => getDatasetStats(),
    refetchInterval: 30_000,
  });
}

export function useAddDatasetPrompt() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      data,
      generationModel,
    }: {
      data: CreateAbliterationPromptRequest;
      generationModel?: string;
    }) => addDatasetPrompt(data, generationModel),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-prompts"] });
      qc.invalidateQueries({ queryKey: ["dataset-stats"] });
    },
  });
}

export function useUploadDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => uploadDatasetFile(file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-prompts"] });
      qc.invalidateQueries({ queryKey: ["dataset-stats"] });
    },
  });
}

export function useUpdateDatasetPrompt() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      data,
    }: {
      id: string;
      data: UpdateAbliterationPromptRequest;
    }) => updateDatasetPrompt(id, data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-prompts"] });
      qc.invalidateQueries({ queryKey: ["dataset-stats"] });
    },
  });
}

export function useDeleteDatasetPrompt() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => deleteDatasetPrompt(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-prompts"] });
      qc.invalidateQueries({ queryKey: ["dataset-stats"] });
    },
  });
}

export function useGenerateHarmless() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      data,
      model,
    }: {
      data: GenerateHarmlessRequest;
      model?: string;
    }) => generateHarmlessCounterpart(data, model),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-prompts"] });
      qc.invalidateQueries({ queryKey: ["dataset-stats"] });
    },
  });
}

export function useDatasetSuggestions() {
  return useQuery({
    queryKey: ["dataset-suggestions"],
    queryFn: () => listDatasetSuggestions(),
    refetchInterval: 30_000,
  });
}

export function useConfirmSuggestion() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      autoGenerateCounterpart,
      model,
    }: {
      id: string;
      autoGenerateCounterpart?: boolean;
      model?: string;
    }) => confirmDatasetSuggestion(id, autoGenerateCounterpart, model),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-prompts"] });
      qc.invalidateQueries({ queryKey: ["dataset-stats"] });
      qc.invalidateQueries({ queryKey: ["dataset-suggestions"] });
    },
  });
}

export function useDismissSuggestion() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => dismissDatasetSuggestion(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-suggestions"] });
    },
  });
}

// ── Model Lifecycle ──────────────────────────────────────────

export function useLoadModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (model: string) => loadDatasetModel(model),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-model-status"] });
    },
  });
}

export function useUnloadModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (model: string) => unloadDatasetModel(model),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["dataset-model-status"] });
    },
  });
}

export function useModelStatus() {
  return useQuery({
    queryKey: ["dataset-model-status"],
    queryFn: () => getModelStatus(),
    refetchInterval: 10_000,
  });
}
