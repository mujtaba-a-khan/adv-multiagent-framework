import { create } from "zustand";
import type { FTWSMessage, FineTuningJobStatus } from "@/lib/types";
import { FTSocket } from "@/lib/ft-ws-client";

export interface FTLogEntry {
  level: string;
  message: string;
  timestamp: string;
}

interface FTWSStore {
  socket: FTSocket | null;
  connected: boolean;
  status: FineTuningJobStatus | null;
  progressPct: number;
  currentStep: string | null;
  logs: FTLogEntry[];
  error: string | null;
  outputModel: string | null;
  durationS: number | null;
  lastMessage: FTWSMessage | null;

  connect: (jobId: string) => void;
  disconnect: () => void;
  reset: () => void;
}

export const useFTWSStore = create<FTWSStore>((set, get) => ({
  socket: null,
  connected: false,
  status: null,
  progressPct: 0,
  currentStep: null,
  logs: [],
  error: null,
  outputModel: null,
  durationS: null,
  lastMessage: null,

  connect: (jobId: string) => {
    const existing = get().socket;
    if (existing) existing.disconnect();

    const socket = new FTSocket(jobId);

    socket.subscribe((msg) => {
      set({ lastMessage: msg });

      switch (msg.type) {
        case "ft_started":
          set({ status: "running", progressPct: 0, error: null });
          break;

        case "ft_progress":
          set({
            progressPct: (msg.data?.progress_pct as number) ?? 0,
            currentStep: (msg.data?.current_step as string) ?? null,
          });
          break;

        case "ft_log":
          set((state) => ({
            logs: [
              ...state.logs,
              {
                level: (msg.data?.level as string) ?? "info",
                message: (msg.data?.message as string) ?? "",
                timestamp:
                  (msg.data?.timestamp as string) ??
                  new Date().toISOString(),
              },
            ],
          }));
          break;

        case "ft_completed":
          set({
            status: "completed",
            progressPct: 100,
            currentStep: "Complete",
            outputModel: (msg.data?.output_model as string) ?? null,
            durationS: (msg.data?.duration_s as number) ?? null,
          });
          break;

        case "ft_failed":
          set({
            status: "failed",
            error: (msg.data?.error as string) ?? "Unknown error",
          });
          break;

        case "ft_cancelled":
          set({ status: "cancelled" });
          break;
      }
    });

    socket.connect();
    set({
      socket,
      connected: true,
      logs: [],
      error: null,
      progressPct: 0,
      currentStep: null,
      outputModel: null,
      durationS: null,
    });
  },

  disconnect: () => {
    const s = get().socket;
    if (s) s.disconnect();
    set({ socket: null, connected: false });
  },

  reset: () =>
    set({
      status: null,
      progressPct: 0,
      currentStep: null,
      logs: [],
      error: null,
      outputModel: null,
      durationS: null,
      lastMessage: null,
    }),
}));
