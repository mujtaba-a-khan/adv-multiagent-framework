import { create } from "zustand";
import type { Turn, WSMessage } from "@/lib/types";
import { BattleSocket } from "@/lib/ws-client";

/** Partial turn being built up progressively via WebSocket events. */
export interface PendingTurn {
  turn_number: number;
  strategy_name?: string;
  attack_prompt?: string;
  attacker_reasoning?: string | null;
  target_response?: string;
  raw_target_response?: string | null;
  target_blocked?: boolean;
  is_baseline?: boolean;
  /** Attack objective text sent with turn_start so the UI can preview it. */
  attack_objective?: string;
  /** Session mode broadcast with the first turn_start event. */
  session_mode?: string;
}

interface WSStore {
  socket: BattleSocket | null;
  connected: boolean;
  liveTurns: Turn[];
  pendingTurn: PendingTurn | null;
  lastMessage: WSMessage | null;

  connect: (sessionId: string) => void;
  disconnect: () => void;
  clearTurns: () => void;
}

export const useWSStore = create<WSStore>((set, get) => ({
  socket: null,
  connected: false,
  liveTurns: [],
  pendingTurn: null,
  lastMessage: null,

  connect: (sessionId: string) => {
    const existing = get().socket;
    if (existing) existing.disconnect();

    const socket = new BattleSocket(sessionId);

    socket.subscribe((msg) => {
      set({ lastMessage: msg });

      switch (msg.type) {
        case "turn_start":
          set({
            pendingTurn: {
              turn_number: msg.turn_number ?? 0,
              attack_objective: (msg.data?.attack_objective as string) || undefined,
              session_mode: (msg.data?.session_mode as string) || undefined,
            },
          });
          break;

        case "attack_generated":
          set((state) => ({
            pendingTurn: state.pendingTurn
              ? {
                  ...state.pendingTurn,
                  attack_prompt: (msg.data?.attack_prompt as string) ?? "",
                  attacker_reasoning:
                    (msg.data?.attacker_reasoning as string | null) ?? null,
                  strategy_name: (msg.data?.strategy_name as string) ?? "",
                  is_baseline: (msg.data?.is_baseline as boolean) ?? false,
                }
              : null,
          }));
          break;

        case "target_responded":
          set((state) => ({
            pendingTurn: state.pendingTurn
              ? {
                  ...state.pendingTurn,
                  target_response: (msg.data?.target_response as string) ?? "",
                  raw_target_response: (msg.data?.raw_target_response as string | null) ?? null,
                  target_blocked:
                    (msg.data?.target_blocked as boolean) ?? false,
                }
              : null,
          }));
          break;

        case "turn_complete":
          if (msg.data) {
            const turn = msg.data as unknown as Turn;
            set((state) => ({
              liveTurns: [...state.liveTurns, turn],
              pendingTurn: null,
            }));
          }
          break;

        case "session_complete":
          set({ pendingTurn: null });
          break;
      }
    });

    socket.connect();
    set({ socket, connected: true, liveTurns: [], pendingTurn: null });
  },

  disconnect: () => {
    const s = get().socket;
    if (s) s.disconnect();
    set({ socket: null, connected: false });
  },

  clearTurns: () =>
    set({ liveTurns: [], pendingTurn: null, lastMessage: null }),
}));
