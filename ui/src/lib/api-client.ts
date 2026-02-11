import { API_V1 } from "./constants";
import type {
  AbliterationDatasetStats,
  AbliterationPrompt,
  AbliterationPromptListResponse,
  CreateAbliterationPromptRequest,
  CreateExperimentRequest,
  CreateFineTuningJobRequest,
  CreatePlaygroundConversationRequest,
  DatasetSuggestionResponse,
  DefenseListResponse,
  Experiment,
  ExperimentListResponse,
  FineTuningJob,
  FineTuningJobListResponse,
  GenerateHarmlessRequest,
  HealthResponse,
  ModelListResponse,
  PlaygroundConversation,
  PlaygroundConversationListResponse,
  PlaygroundMessage,
  PlaygroundMessageListResponse,
  ReportData,
  Session,
  SessionComparisonResponse,
  SessionListResponse,
  StartSessionRequest,
  Strategy,
  StrategyListResponse,
  Turn,
  TurnListResponse,
  UpdateAbliterationPromptRequest,
  UpdateExperimentRequest,
  UpdatePlaygroundConversationRequest,
} from "./types";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${API_V1}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "Unknown error");
    throw new ApiError(res.status, body);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// Experiments

export async function createExperiment(
  data: CreateExperimentRequest,
): Promise<Experiment> {
  return request<Experiment>("/experiments", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function listExperiments(
  offset = 0,
  limit = 50,
): Promise<ExperimentListResponse> {
  return request<ExperimentListResponse>(
    `/experiments?offset=${offset}&limit=${limit}`,
  );
}

export async function getExperiment(id: string): Promise<Experiment> {
  return request<Experiment>(`/experiments/${id}`);
}

export async function updateExperiment(
  id: string,
  data: UpdateExperimentRequest,
): Promise<Experiment> {
  return request<Experiment>(`/experiments/${id}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}

export async function deleteExperiment(id: string): Promise<void> {
  return request<void>(`/experiments/${id}`, { method: "DELETE" });
}

// Sessions

export async function createSession(
  experimentId: string,
  data?: StartSessionRequest,
): Promise<Session> {
  return request<Session>(`/experiments/${experimentId}/sessions`, {
    method: "POST",
    body: data ? JSON.stringify(data) : undefined,
  });
}

export async function listSessions(
  experimentId: string,
  offset = 0,
  limit = 50,
): Promise<SessionListResponse> {
  return request<SessionListResponse>(
    `/experiments/${experimentId}/sessions?offset=${offset}&limit=${limit}`,
  );
}

export async function getSession(
  experimentId: string,
  sessionId: string,
): Promise<Session> {
  return request<Session>(
    `/experiments/${experimentId}/sessions/${sessionId}`,
  );
}

export async function startSession(
  experimentId: string,
  sessionId: string,
): Promise<Session> {
  return request<Session>(
    `/experiments/${experimentId}/sessions/${sessionId}/start`,
    { method: "POST" },
  );
}

export async function deleteSession(
  experimentId: string,
  sessionId: string,
): Promise<void> {
  return request<void>(
    `/experiments/${experimentId}/sessions/${sessionId}`,
    { method: "DELETE" },
  );
}

// Turns

export async function listTurns(
  sessionId: string,
  offset = 0,
  limit = 100,
): Promise<TurnListResponse> {
  return request<TurnListResponse>(
    `/sessions/${sessionId}/turns?offset=${offset}&limit=${limit}`,
  );
}

export async function getLatestTurn(sessionId: string): Promise<Turn> {
  return request<Turn>(`/sessions/${sessionId}/turns/latest`);
}

// Strategies

export async function listStrategies(): Promise<StrategyListResponse> {
  return request<StrategyListResponse>("/strategies");
}

export async function getStrategy(name: string): Promise<Strategy> {
  return request<Strategy>(`/strategies/${name}`);
}

// Targets

export async function listModels(): Promise<ModelListResponse> {
  return request<ModelListResponse>("/targets/models");
}

export async function checkHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/targets/health");
}

// Defenses

export async function listDefenses(): Promise<DefenseListResponse> {
  return request<DefenseListResponse>("/defenses");
}

// Comparisons

export async function compareSessions(
  experimentId: string,
  attackSessionId: string,
  defenseSessionId: string,
): Promise<SessionComparisonResponse> {
  return request<SessionComparisonResponse>(
    `/experiments/${experimentId}/compare?attack_session_id=${attackSessionId}&defense_session_id=${defenseSessionId}`,
  );
}

// Reports   

export async function getSessionReport(
  experimentId: string,
  sessionId: string,
): Promise<ReportData> {
  return request<ReportData>(
    `/reports/${experimentId}/sessions/${sessionId}`,
  );
}

export async function exportReportJson(
  experimentId: string,
  sessionId: string,
): Promise<ReportData> {
  return request<ReportData>(
    `/reports/${experimentId}/sessions/${sessionId}/export?format=json`,
  );
}

// Fine-Tuning / Model Workshop

export async function createFineTuningJob(
  data: CreateFineTuningJobRequest,
): Promise<FineTuningJob> {
  return request<FineTuningJob>("/finetuning/jobs", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function listFineTuningJobs(
  offset = 0,
  limit = 50,
  status?: string,
): Promise<FineTuningJobListResponse> {
  let url = `/finetuning/jobs?offset=${offset}&limit=${limit}`;
  if (status) url += `&status=${status}`;
  return request<FineTuningJobListResponse>(url);
}

export async function getFineTuningJob(
  jobId: string,
): Promise<FineTuningJob> {
  return request<FineTuningJob>(`/finetuning/jobs/${jobId}`);
}

export async function startFineTuningJob(
  jobId: string,
): Promise<FineTuningJob> {
  return request<FineTuningJob>(`/finetuning/jobs/${jobId}/start`, {
    method: "POST",
  });
}

export async function cancelFineTuningJob(
  jobId: string,
): Promise<{ status: string }> {
  return request<{ status: string }>(`/finetuning/jobs/${jobId}/cancel`, {
    method: "POST",
  });
}

export async function deleteFineTuningJob(
  jobId: string,
): Promise<void> {
  return request<void>(`/finetuning/jobs/${jobId}`, { method: "DELETE" });
}

export async function listCustomModels(): Promise<{
  models: Array<{ name: string; size: number; modified_at: string }>;
}> {
  return request(`/finetuning/models`);
}

export async function deleteOllamaModel(
  name: string,
): Promise<void> {
  return request<void>(`/finetuning/models/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
}

// Disk Management

export interface DiskStatus {
  disk_total_gb: number;
  disk_free_gb: number;
  ollama_storage_gb: number;
  orphan_count: number;
  orphan_gb: number;
}

export interface CleanupResult {
  orphan_count: number;
  freed_bytes: number;
  freed_gb: number;
}

export async function getDiskStatus(): Promise<DiskStatus> {
  return request<DiskStatus>("/finetuning/disk-status");
}

export async function cleanupOrphans(): Promise<CleanupResult> {
  return request<CleanupResult>("/finetuning/cleanup-orphans", {
    method: "POST",
  });
}

// Abliteration Dataset

export async function listDatasetPrompts(
  offset = 0,
  limit = 50,
  category?: string,
  source?: string,
): Promise<AbliterationPromptListResponse> {
  let url = `/finetuning/dataset/prompts?offset=${offset}&limit=${limit}`;
  if (category) url += `&category=${category}`;
  if (source) url += `&source=${source}`;
  return request<AbliterationPromptListResponse>(url);
}

export async function getDatasetStats(): Promise<AbliterationDatasetStats> {
  return request<AbliterationDatasetStats>(
    "/finetuning/dataset/prompts/stats",
  );
}

export async function addDatasetPrompt(
  data: CreateAbliterationPromptRequest,
  generationModel?: string,
): Promise<AbliterationPrompt> {
  let url = "/finetuning/dataset/prompts";
  if (generationModel)
    url += `?generation_model=${encodeURIComponent(generationModel)}`;
  return request<AbliterationPrompt>(url, {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function uploadDatasetFile(
  file: File,
): Promise<AbliterationPromptListResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const url = `${API_V1}/finetuning/dataset/prompts/upload`;
  const res = await fetch(url, { method: "POST", body: formData });
  if (!res.ok) {
    const body = await res.text().catch(() => "Unknown error");
    throw new ApiError(res.status, body);
  }
  return res.json() as Promise<AbliterationPromptListResponse>;
}

export async function updateDatasetPrompt(
  id: string,
  data: UpdateAbliterationPromptRequest,
): Promise<AbliterationPrompt> {
  return request<AbliterationPrompt>(
    `/finetuning/dataset/prompts/${id}`,
    { method: "PUT", body: JSON.stringify(data) },
  );
}

export async function deleteDatasetPrompt(id: string): Promise<void> {
  return request<void>(`/finetuning/dataset/prompts/${id}`, {
    method: "DELETE",
  });
}

export async function generateHarmlessCounterpart(
  data: GenerateHarmlessRequest,
  model?: string,
): Promise<AbliterationPrompt> {
  let url = "/finetuning/dataset/prompts/generate-harmless";
  if (model) url += `?model=${encodeURIComponent(model)}`;
  return request<AbliterationPrompt>(url, {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function listDatasetSuggestions(): Promise<DatasetSuggestionResponse> {
  return request<DatasetSuggestionResponse>(
    "/finetuning/dataset/prompts/suggestions",
  );
}

export async function confirmDatasetSuggestion(
  id: string,
  autoGenerateCounterpart = true,
  model?: string,
): Promise<AbliterationPrompt> {
  let url = `/finetuning/dataset/prompts/${id}/confirm?auto_generate_counterpart=${autoGenerateCounterpart}`;
  if (model) url += `&model=${encodeURIComponent(model)}`;
  return request<AbliterationPrompt>(url, { method: "POST" });
}

export async function dismissDatasetSuggestion(
  id: string,
): Promise<void> {
  return request<void>(`/finetuning/dataset/prompts/${id}/dismiss`, {
    method: "POST",
  });
}

// Dataset Model Lifecycle

export async function loadDatasetModel(model: string): Promise<void> {
  return request<void>("/finetuning/dataset/model/load", {
    method: "POST",
    body: JSON.stringify({ model }),
  });
}

export async function unloadDatasetModel(model: string): Promise<void> {
  return request<void>("/finetuning/dataset/model/unload", {
    method: "POST",
    body: JSON.stringify({ model }),
  });
}

export async function getModelStatus(): Promise<{
  models: Array<{ name: string; size: number }>;
}> {
  return request("/finetuning/dataset/model/status");
}

// Playground

export async function createPlaygroundConversation(
  data: CreatePlaygroundConversationRequest,
): Promise<PlaygroundConversation> {
  return request<PlaygroundConversation>("/playground/conversations", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function listPlaygroundConversations(
  offset = 0,
  limit = 50,
): Promise<PlaygroundConversationListResponse> {
  return request<PlaygroundConversationListResponse>(
    `/playground/conversations?offset=${offset}&limit=${limit}`,
  );
}

export async function getPlaygroundConversation(
  id: string,
): Promise<PlaygroundConversation> {
  return request<PlaygroundConversation>(`/playground/conversations/${id}`);
}

export async function updatePlaygroundConversation(
  id: string,
  data: UpdatePlaygroundConversationRequest,
): Promise<PlaygroundConversation> {
  return request<PlaygroundConversation>(
    `/playground/conversations/${id}`,
    { method: "PATCH", body: JSON.stringify(data) },
  );
}

export async function deletePlaygroundConversation(
  id: string,
): Promise<void> {
  return request<void>(`/playground/conversations/${id}`, {
    method: "DELETE",
  });
}

export async function listPlaygroundMessages(
  conversationId: string,
  offset = 0,
  limit = 100,
): Promise<PlaygroundMessageListResponse> {
  return request<PlaygroundMessageListResponse>(
    `/playground/conversations/${conversationId}/messages?offset=${offset}&limit=${limit}`,
  );
}

export async function sendPlaygroundMessage(
  conversationId: string,
  prompt: string,
): Promise<PlaygroundMessage> {
  return request<PlaygroundMessage>(
    `/playground/conversations/${conversationId}/messages`,
    { method: "POST", body: JSON.stringify({ prompt }) },
  );
}

export { ApiError };
