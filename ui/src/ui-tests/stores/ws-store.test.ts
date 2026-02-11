import { describe, it, expect, vi, beforeEach } from "vitest";
import type { WSMessage } from "@/lib/types";

// Mock ws-client before importing the store
const mockConnect = vi.fn();
const mockDisconnect = vi.fn();
const mockSubscribe = vi.fn();

vi.mock("@/lib/ws-client", () => {
  return {
    BattleSocket: class {
      connect = mockConnect;
      disconnect = mockDisconnect;
      subscribe = mockSubscribe;
    },
  };
});

const { useWSStore } = await import("@/stores/ws-store");

describe("useWSStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset store to initial state
    useWSStore.setState({
      socket: null,
      connected: false,
      liveTurns: [],
      pendingTurn: null,
      lastMessage: null,
    });
  });

  it("has correct initial state", () => {
    const state = useWSStore.getState();
    expect(state.socket).toBeNull();
    expect(state.connected).toBe(false);
    expect(state.liveTurns).toEqual([]);
    expect(state.pendingTurn).toBeNull();
    expect(state.lastMessage).toBeNull();
  });

  it("connect creates socket and sets connected", () => {
    useWSStore.getState().connect("session-1");
    const state = useWSStore.getState();
    expect(state.connected).toBe(true);
    expect(state.socket).not.toBeNull();
    expect(mockConnect).toHaveBeenCalled();
    expect(mockSubscribe).toHaveBeenCalled();
  });

  it("connect disconnects existing socket", () => {
    // First connect
    useWSStore.getState().connect("session-1");
    // Second connect should disconnect the first
    useWSStore.getState().connect("session-2");
    expect(mockDisconnect).toHaveBeenCalled();
  });

  it("disconnect cleans up socket", () => {
    useWSStore.getState().connect("session-1");
    useWSStore.getState().disconnect();
    const state = useWSStore.getState();
    expect(state.socket).toBeNull();
    expect(state.connected).toBe(false);
  });

  it("disconnect does nothing when no socket", () => {
    useWSStore.getState().disconnect();
    expect(mockDisconnect).not.toHaveBeenCalled();
  });

  it("clearTurns resets turns and messages", () => {
    useWSStore.setState({
      liveTurns: [{ turn_number: 1 } as never],
      pendingTurn: { turn_number: 1 },
      lastMessage: { type: "test" } as WSMessage,
    });
    useWSStore.getState().clearTurns();
    const state = useWSStore.getState();
    expect(state.liveTurns).toEqual([]);
    expect(state.pendingTurn).toBeNull();
    expect(state.lastMessage).toBeNull();
  });

  describe("message handling", () => {
    let messageHandler: (msg: WSMessage) => void;

    beforeEach(() => {
      mockSubscribe.mockImplementation((handler: (msg: WSMessage) => void) => {
        messageHandler = handler;
      });
      useWSStore.getState().connect("session-1");
    });

    it("turn_start creates pending turn", () => {
      messageHandler({
        type: "turn_start",
        turn_number: 1,
        data: { attack_objective: "Test objective", session_mode: "attack" },
      });
      const state = useWSStore.getState();
      expect(state.pendingTurn).not.toBeNull();
      expect(state.pendingTurn?.turn_number).toBe(1);
      expect(state.pendingTurn?.attack_objective).toBe("Test objective");
      expect(state.pendingTurn?.session_mode).toBe("attack");
    });

    it("attack_generated updates pending turn", () => {
      // First create a pending turn
      messageHandler({ type: "turn_start", turn_number: 1, data: {} });
      messageHandler({
        type: "attack_generated",
        data: {
          attack_prompt: "test prompt",
          attacker_reasoning: "reasoning",
          strategy_name: "pair",
          is_baseline: false,
        },
      });
      const state = useWSStore.getState();
      expect(state.pendingTurn?.attack_prompt).toBe("test prompt");
      expect(state.pendingTurn?.attacker_reasoning).toBe("reasoning");
      expect(state.pendingTurn?.strategy_name).toBe("pair");
    });

    it("attack_generated returns null when no pending turn", () => {
      useWSStore.setState({ pendingTurn: null });
      messageHandler({
        type: "attack_generated",
        data: { attack_prompt: "test" },
      });
      expect(useWSStore.getState().pendingTurn).toBeNull();
    });

    it("target_responded updates pending turn", () => {
      messageHandler({ type: "turn_start", turn_number: 1, data: {} });
      messageHandler({
        type: "target_responded",
        data: {
          target_response: "response text",
          raw_target_response: "raw response",
          target_blocked: true,
        },
      });
      const state = useWSStore.getState();
      expect(state.pendingTurn?.target_response).toBe("response text");
      expect(state.pendingTurn?.raw_target_response).toBe("raw response");
      expect(state.pendingTurn?.target_blocked).toBe(true);
    });

    it("turn_complete adds to liveTurns and clears pending", () => {
      messageHandler({ type: "turn_start", turn_number: 1, data: {} });
      const turnData = { turn_number: 1, verdict: "refused" };
      messageHandler({ type: "turn_complete", data: turnData });
      const state = useWSStore.getState();
      expect(state.liveTurns).toHaveLength(1);
      expect(state.pendingTurn).toBeNull();
    });

    it("turn_complete ignores missing data", () => {
      messageHandler({ type: "turn_complete" });
      expect(useWSStore.getState().liveTurns).toHaveLength(0);
    });

    it("session_complete clears pending turn", () => {
      messageHandler({ type: "turn_start", turn_number: 1, data: {} });
      messageHandler({ type: "session_complete" });
      expect(useWSStore.getState().pendingTurn).toBeNull();
    });

    it("all messages update lastMessage", () => {
      const msg: WSMessage = { type: "turn_start", turn_number: 1, data: {} };
      messageHandler(msg);
      expect(useWSStore.getState().lastMessage).toEqual(msg);
    });
  });
});
