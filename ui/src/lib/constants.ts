export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const API_V1 = `${API_BASE_URL}/api/v1`;

export const WS_BASE_URL =
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000";

export const ROUTES = {
  dashboard: "/",
  experiments: {
    list: "/experiments",
    new: "/experiments/new",
    detail: (id: string) => `/experiments/${id}`,
    live: (id: string, sessionId: string) =>
      `/experiments/${id}/live?session=${sessionId}`,
    compare: (id: string, attackId: string, defenseId: string) =>
      `/experiments/${id}/compare?attack=${attackId}&defense=${defenseId}`,
  },
  reports: {
    list: "/reports",
    detail: (experimentId: string, sessionId: string) =>
      `/reports/${experimentId}/${sessionId}`,
  },
  strategies: "/strategies",
  targets: "/targets",
  settings: "/settings",
  workshop: {
    list: "/workshop",
    detail: (jobId: string) => `/workshop/${jobId}`,
    dataset: "/workshop/dataset",
  },
  playground: {
    list: "/playground",
    detail: (id: string) => `/playground/${id}`,
  },
} as const;

export const VERDICT_COLORS: Record<string, string> = {
  jailbreak: "text-red-500",
  borderline: "text-amber-500",
  refused: "text-emerald-500",
  error: "text-zinc-400",
};

export const VERDICT_BG: Record<string, string> = {
  jailbreak: "bg-red-500/10 text-red-500 border-red-500/20",
  borderline: "bg-amber-500/10 text-amber-500 border-amber-500/20",
  refused: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20",
  error: "bg-zinc-500/10 text-zinc-400 border-zinc-500/20",
};

export const STATUS_COLORS: Record<string, string> = {
  pending: "bg-zinc-500/10 text-zinc-400 border-zinc-500/20",
  running: "bg-blue-500/10 text-blue-500 border-blue-500/20",
  completed: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20",
  failed: "bg-red-500/10 text-red-500 border-red-500/20",
  cancelled: "bg-zinc-500/10 text-zinc-400 border-zinc-500/20",
};

export const CATEGORY_LABELS: Record<string, string> = {
  prompt_level: "Prompt-Level",
  optimization: "Optimization",
  multi_turn: "Multi-Turn",
  advanced: "Advanced",
  composite: "Composite",
};

export const CATEGORY_COLORS: Record<string, string> = {
  prompt_level: "bg-violet-500/10 text-violet-500 border-violet-500/20",
  optimization: "bg-sky-500/10 text-sky-500 border-sky-500/20",
  multi_turn: "bg-amber-500/10 text-amber-500 border-amber-500/20",
  advanced: "bg-rose-500/10 text-rose-500 border-rose-500/20",
  composite: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20",
};

export const JOB_TYPE_LABELS: Record<string, string> = {
  pull_abliterated: "Pull Abliterated",
  abliterate: "Custom Abliterate",
  sft: "Fine-Tune (SFT)",
};

export const JOB_TYPE_COLORS: Record<string, string> = {
  pull_abliterated: "bg-sky-500/10 text-sky-500 border-sky-500/20",
  abliterate: "bg-violet-500/10 text-violet-500 border-violet-500/20",
  sft: "bg-amber-500/10 text-amber-500 border-amber-500/20",
};

export const PROMPT_CATEGORY_COLORS: Record<string, string> = {
  harmful: "bg-red-500/10 text-red-500 border-red-500/20",
  harmless: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20",
};

export const PROMPT_SOURCE_LABELS: Record<string, string> = {
  manual: "Manual",
  upload: "Upload",
  session: "Session",
};
