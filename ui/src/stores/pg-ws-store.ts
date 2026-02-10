import { create } from "zustand";
import type { PGWSMessage, PGWSPhase } from "@/lib/types";
import { PGSocket } from "@/lib/pg-ws-client";

interface PGWSStore {
  socket: PGSocket | null;
  connected: boolean;
  phase: PGWSPhase | null;
  targetResponse: string | null;
  targetBlocked: boolean;
  lastMessage: PGWSMessage | null;

  connect: (conversationId: string) => void;
  disconnect: () => void;
  reset: () => void;
}

export const usePGWSStore = create<PGWSStore>((set, get) => ({
  socket: null,
  connected: false,
  phase: null,
  targetResponse: null,
  targetBlocked: false,
  lastMessage: null,

  connect: (conversationId: string) => {
    const existing = get().socket;
    if (existing) existing.disconnect();

    const socket = new PGSocket(conversationId);

    socket.subscribe((msg) => {
      set({ lastMessage: msg });

      switch (msg.type) {
        case "pg_processing":
          set({
            phase: (msg.data?.phase as PGWSPhase) ?? null,
            targetResponse:
              (msg.data?.target_response as string) ?? get().targetResponse,
            targetBlocked:
              (msg.data?.target_blocked as boolean) ?? get().targetBlocked,
          });
          break;

        case "pg_message_complete":
          set({
            phase: null,
            targetResponse: null,
            targetBlocked: false,
          });
          break;
      }
    });

    socket.connect();
    set({
      socket,
      connected: true,
      phase: null,
      targetResponse: null,
      targetBlocked: false,
    });
  },

  disconnect: () => {
    const s = get().socket;
    if (s) s.disconnect();
    set({ socket: null, connected: false });
  },

  reset: () =>
    set({
      phase: null,
      targetResponse: null,
      targetBlocked: false,
      lastMessage: null,
    }),
}));
