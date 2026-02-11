import { describe, it, expect, vi, beforeEach } from "vitest";
import type { FTWSMessage } from "@/lib/types";

const mockConnect = vi.fn();
const mockDisconnect = vi.fn();
const mockSubscribe = vi.fn();

vi.mock("@/lib/ft-ws-client", () => {
  return {
    FTSocket: class {
      connect = mockConnect;
      disconnect = mockDisconnect;
      subscribe = mockSubscribe;
    },
  };
});

const { useFTWSStore } = await import("@/stores/ft-ws-store");

const INITIAL_STATE = {
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
};

describe("useFTWSStore", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useFTWSStore.setState(INITIAL_STATE);
  });

  it("has correct initial state", () => {
    const state = useFTWSStore.getState();
    expect(state.connected).toBe(false);
    expect(state.status).toBeNull();
    expect(state.progressPct).toBe(0);
    expect(state.logs).toEqual([]);
  });

  it("connect creates socket and sets connected", () => {
    useFTWSStore.getState().connect("job-1");
    expect(useFTWSStore.getState().connected).toBe(true);
    expect(mockConnect).toHaveBeenCalled();
  });

  it("connect disconnects existing socket", () => {
    useFTWSStore.getState().connect("job-1");
    useFTWSStore.getState().connect("job-2");
    expect(mockDisconnect).toHaveBeenCalled();
  });

  it("disconnect cleans up", () => {
    useFTWSStore.getState().connect("job-1");
    useFTWSStore.getState().disconnect();
    expect(useFTWSStore.getState().connected).toBe(false);
    expect(useFTWSStore.getState().socket).toBeNull();
  });

  it("reset clears all state except socket", () => {
    useFTWSStore.setState({
      status: "running",
      progressPct: 50,
      logs: [{ level: "info", message: "test", timestamp: "t" }],
      error: "err",
    });
    useFTWSStore.getState().reset();
    const state = useFTWSStore.getState();
    expect(state.status).toBeNull();
    expect(state.progressPct).toBe(0);
    expect(state.logs).toEqual([]);
    expect(state.error).toBeNull();
  });

  describe("message handling", () => {
    let messageHandler: (msg: FTWSMessage) => void;

    beforeEach(() => {
      mockSubscribe.mockImplementation((handler: (msg: FTWSMessage) => void) => {
        messageHandler = handler;
      });
      useFTWSStore.getState().connect("job-1");
    });

    it("ft_started sets running status", () => {
      messageHandler({ type: "ft_started", data: {} });
      expect(useFTWSStore.getState().status).toBe("running");
      expect(useFTWSStore.getState().error).toBeNull();
    });

    it("ft_progress updates progress", () => {
      messageHandler({
        type: "ft_progress",
        data: { progress_pct: 45, current_step: "Training epoch 2" },
      });
      expect(useFTWSStore.getState().progressPct).toBe(45);
      expect(useFTWSStore.getState().currentStep).toBe("Training epoch 2");
    });

    it("ft_log appends to logs", () => {
      messageHandler({
        type: "ft_log",
        data: { level: "info", message: "Starting", timestamp: "2024-01-01T00:00:00Z" },
      });
      messageHandler({
        type: "ft_log",
        data: { level: "warn", message: "Low memory" },
      });
      expect(useFTWSStore.getState().logs).toHaveLength(2);
      expect(useFTWSStore.getState().logs[0].message).toBe("Starting");
      expect(useFTWSStore.getState().logs[1].level).toBe("warn");
    });

    it("ft_completed sets final state", () => {
      messageHandler({
        type: "ft_completed",
        data: { output_model: "my-model:latest", duration_s: 120 },
      });
      const state = useFTWSStore.getState();
      expect(state.status).toBe("completed");
      expect(state.progressPct).toBe(100);
      expect(state.currentStep).toBe("Complete");
      expect(state.outputModel).toBe("my-model:latest");
      expect(state.durationS).toBe(120);
    });

    it("ft_failed sets error state", () => {
      messageHandler({
        type: "ft_failed",
        data: { error: "Out of memory" },
      });
      expect(useFTWSStore.getState().status).toBe("failed");
      expect(useFTWSStore.getState().error).toBe("Out of memory");
    });

    it("ft_cancelled sets cancelled status", () => {
      messageHandler({ type: "ft_cancelled", data: {} });
      expect(useFTWSStore.getState().status).toBe("cancelled");
    });

    it("all messages update lastMessage", () => {
      const msg: FTWSMessage = { type: "ft_started", data: {} };
      messageHandler(msg);
      expect(useFTWSStore.getState().lastMessage).toEqual(msg);
    });
  });
});
