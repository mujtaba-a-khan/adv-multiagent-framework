// Domain types matching backend API schemas

export interface Experiment {
  id: string;
  name: string;
  description: string | null;
  target_model: string;
  target_provider: string;
  target_system_prompt: string | null;
  attacker_model: string;
  analyzer_model: string;
  defender_model: string;
  attack_objective: string;
  strategy_name: string;
  strategy_params: Record<string, unknown>;
  max_turns: number;
  max_cost_usd: number;
  created_at: string;
  updated_at: string;
}

export interface ExperimentListResponse {
  experiments: Experiment[];
  total: number;
}

export interface CreateExperimentRequest {
  name: string;
  description?: string | null;
  target_model: string;
  target_provider?: string;
  target_system_prompt?: string | null;
  attacker_model?: string;
  analyzer_model?: string;
  defender_model?: string;
  attack_objective: string;
}

export interface UpdateExperimentRequest {
  name?: string | null;
  description?: string | null;
  target_system_prompt?: string | null;
}

export type SessionMode = "attack" | "defense";

export interface DefenseSelectionItem {
  name: string;
  params?: Record<string, unknown>;
}

export interface Session {
  id: string;
  experiment_id: string;
  status: SessionStatus;
  session_mode: SessionMode;
  initial_defenses: DefenseSelectionItem[];
  strategy_name: string;
  strategy_params: Record<string, unknown>;
  max_turns: number;
  max_cost_usd: number;
  total_turns: number;
  total_jailbreaks: number;
  total_refused: number;
  total_borderline: number;
  total_blocked: number;
  asr: number | null;
  total_attacker_tokens: number;
  total_target_tokens: number;
  total_analyzer_tokens: number;
  total_defender_tokens: number;
  estimated_cost_usd: number;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface SessionListResponse {
  sessions: Session[];
  total: number;
}

export interface StartSessionRequest {
  session_mode?: SessionMode;
  initial_defenses?: DefenseSelectionItem[];
  strategy_name: string;
  strategy_params?: Record<string, unknown>;
  max_turns: number;
  max_cost_usd: number;
  separate_reasoning?: boolean;
}

export interface Turn {
  id: string;
  session_id: string;
  turn_number: number;
  strategy_name: string;
  strategy_params: Record<string, unknown>;
  attack_prompt: string;
  attacker_reasoning: string | null;
  target_response: string;
  raw_target_response: string | null;
  target_blocked: boolean;
  judge_verdict: JudgeVerdict;
  judge_confidence: number;
  severity_score: number | null;
  specificity_score: number | null;
  coherence_score: number | null;
  vulnerability_category: string | null;
  attack_technique: string | null;
  attacker_tokens: number;
  target_tokens: number;
  analyzer_tokens: number;
  created_at: string;
}

export interface TurnListResponse {
  turns: Turn[];
  total: number;
}

export interface Strategy {
  name: string;
  display_name: string;
  category: StrategyCategory;
  description: string;
  estimated_asr: string;
  min_turns: number;
  max_turns: number | null;
  requires_white_box: boolean;
  supports_multi_turn: boolean;
  parameters: Record<string, unknown>;
  references: string[];
}

export interface StrategyListResponse {
  strategies: Strategy[];
  total: number;
}

export interface ModelListResponse {
  models: string[];
  provider: string;
}

export interface HealthResponse {
  provider: string;
  healthy: boolean;
}

export interface WSMessage {
  type: WSMessageType;
  session_id: string;
  turn_number: number | null;
  data: Record<string, unknown>;
}

// Enums

export type SessionStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type JudgeVerdict =
  | "jailbreak"
  | "borderline"
  | "refused"
  | "error";

export type StrategyCategory =
  | "prompt_level"
  | "optimization"
  | "multi_turn"
  | "advanced"
  | "composite";

export type WSMessageType =
  | "turn_start"
  | "attack_generated"
  | "target_responded"
  | "turn_complete"
  | "session_complete"
  | "error";

// Report types

export interface ReportFinding {
  turn_number: number;
  strategy_name: string;
  vulnerability_category: string | null;
  severity: number;
  specificity: number;
  attack_prompt_preview: string;
  target_response_preview: string;
}

export interface ReportMetrics {
  total_turns: number;
  total_jailbreaks: number;
  total_refused: number;
  total_borderline: number;
  total_blocked: number;
  asr: number;
  avg_severity: number;
  avg_specificity: number;
  avg_coherence: number;
  total_cost_usd: number;
}

export interface ReportData {
  experiment_id: string;
  experiment_name: string;
  session_id: string;
  target_model: string;
  attack_objective: string;
  strategy_name: string;
  metrics: ReportMetrics;
  findings: ReportFinding[];
  recommendations: string[];
  generated_at: string;
}

// Defenses

export interface DefenseInfo {
  name: string;
  description: string;
  defense_type: string;
}

export interface DefenseListResponse {
  defenses: DefenseInfo[];
  total: number;
}

// Comparisons

export interface SessionComparisonResponse {
  attack_session: Session;
  defense_session: Session;
  attack_turns: Turn[];
  defense_turns: Turn[];
}

// Dashboard aggregate types (computed client-side)

export interface DashboardStats {
  totalExperiments: number;
  activeSessionsCount: number;
  avgASR: number;
  totalTurns: number;
}

// Fine-Tuning / Model Workshop

export type FineTuningJobType = "pull_abliterated" | "abliterate" | "sft";

export type FineTuningJobStatus =
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface FineTuningJob {
  id: string;
  name: string;
  job_type: FineTuningJobType;
  source_model: string;
  output_model_name: string;
  config: Record<string, unknown>;
  status: FineTuningJobStatus;
  progress_pct: number;
  current_step: string | null;
  logs: Array<Record<string, unknown>>;
  error_message: string | null;
  peak_memory_gb: number | null;
  total_duration_seconds: number | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface FineTuningJobListResponse {
  jobs: FineTuningJob[];
  total: number;
}

export interface CreateFineTuningJobRequest {
  name: string;
  job_type: FineTuningJobType;
  source_model: string;
  output_model_name: string;
  config?: Record<string, unknown>;
}

// Abliteration Dataset

export type PromptCategory = "harmful" | "harmless";
export type PromptSource = "manual" | "upload" | "session";
export type PromptStatus = "active" | "suggested";

export interface AbliterationPrompt {
  id: string;
  text: string;
  category: PromptCategory;
  source: PromptSource;
  status: PromptStatus;
  experiment_id: string | null;
  session_id: string | null;
  pair_id: string | null;
  created_at: string;
}

export interface AbliterationPromptListResponse {
  prompts: AbliterationPrompt[];
  total: number;
}

export interface AbliterationDatasetStats {
  harmful_count: number;
  harmless_count: number;
  total: number;
  min_recommended: number;
  warning: string | null;
}

export interface DatasetSuggestionResponse {
  suggestions: AbliterationPrompt[];
  total: number;
}

export interface CreateAbliterationPromptRequest {
  text: string;
  category: PromptCategory;
  source?: PromptSource;
  experiment_id?: string;
  session_id?: string;
  auto_generate_counterpart?: boolean;
}

export interface UpdateAbliterationPromptRequest {
  text?: string;
  category?: PromptCategory;
}

export interface GenerateHarmlessRequest {
  harmful_prompt: string;
  pair_id?: string;
  experiment_id?: string;
  session_id?: string;
}

export interface RunningModelInfo {
  name: string;
  size: number;
}

export interface ModelStatusResponse {
  models: RunningModelInfo[];
}

export type FTWSMessageType =
  | "ft_started"
  | "ft_progress"
  | "ft_log"
  | "ft_completed"
  | "ft_failed"
  | "ft_cancelled";

export interface FTWSMessage {
  type: FTWSMessageType;
  job_id: string;
  data: Record<string, unknown>;
}

// Playground

export interface PlaygroundConversation {
  id: string;
  title: string;
  target_model: string;
  target_provider: string;
  system_prompt: string | null;
  analyzer_model: string;
  active_defenses: DefenseSelectionItem[];
  total_messages: number;
  total_jailbreaks: number;
  total_blocked: number;
  total_target_tokens: number;
  total_analyzer_tokens: number;
  created_at: string;
  updated_at: string;
}

export interface PlaygroundConversationListResponse {
  conversations: PlaygroundConversation[];
  total: number;
}

export interface CreatePlaygroundConversationRequest {
  title: string;
  target_model: string;
  target_provider?: string;
  system_prompt?: string | null;
  analyzer_model?: string;
  active_defenses?: DefenseSelectionItem[];
}

export interface UpdatePlaygroundConversationRequest {
  title?: string;
  system_prompt?: string | null;
  active_defenses?: DefenseSelectionItem[];
}

export interface PlaygroundMessage {
  id: string;
  conversation_id: string;
  message_number: number;
  user_prompt: string;
  target_response: string;
  raw_target_response: string | null;
  target_blocked: boolean;
  blocked_by_defense: string | null;
  block_reason: string | null;
  defenses_applied: DefenseSelectionItem[] | null;
  judge_verdict: JudgeVerdict;
  judge_confidence: number;
  severity_score: number | null;
  specificity_score: number | null;
  vulnerability_category: string | null;
  attack_technique: string | null;
  target_tokens: number;
  analyzer_tokens: number;
  target_latency_ms: number;
  analyzer_latency_ms: number;
  created_at: string;
}

export interface PlaygroundMessageListResponse {
  messages: PlaygroundMessage[];
  total: number;
}

export type PGWSPhase =
  | "started"
  | "target_calling"
  | "target_responded"
  | "analyzing";

export type PGWSMessageType = "pg_processing" | "pg_message_complete";

export interface PGWSMessage {
  type: PGWSMessageType;
  conversation_id: string;
  message_number: number;
  data: Record<string, unknown>;
}
