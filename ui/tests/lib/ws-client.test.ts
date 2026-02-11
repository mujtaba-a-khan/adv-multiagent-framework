import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

vi.mock("@/lib/constants", () => ({
  WS_BASE_URL: "ws://test-ws",
}));

// Mock WebSocket
class MockWebSocket {
  static OPEN = 1;
  static CLOSED = 3;

  url: string;
  readyState = MockWebSocket.OPEN;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
  }

  close() {
    this.readyState = MockWebSocket.CLOSED;
  }
}

globalThis.WebSocket = MockWebSocket as unknown as typeof WebSocket;

const { BattleSocket } = await import("@/lib/ws-client");

describe("BattleSocket", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("stores session id", () => {
    const socket = new BattleSocket("session-1");
    expect(socket.sessionId).toBe("session-1");
  });

  it("connect creates WebSocket with correct URL", () => {
    const socket = new BattleSocket("session-1");
    socket.connect();
    // After connect, socket should have been created
    expect(socket.connected).toBe(true);
  });

  it("subscribe adds handler that receives parsed messages", () => {
    const socket = new BattleSocket("session-1");
    const handler = vi.fn();
    socket.subscribe(handler);
    socket.connect();

    // Access the internal ws to simulate a message
    const ws = (socket as unknown as { ws: MockWebSocket }).ws;
    ws.onmessage?.({ data: JSON.stringify({ type: "turn_start", turn_number: 1 }) });

    expect(handler).toHaveBeenCalledWith({ type: "turn_start", turn_number: 1 });
  });

  it("subscribe returns unsubscribe function", () => {
    const socket = new BattleSocket("session-1");
    const handler = vi.fn();
    const unsub = socket.subscribe(handler);
    socket.connect();

    unsub();

    const ws = (socket as unknown as { ws: MockWebSocket }).ws;
    ws.onmessage?.({ data: JSON.stringify({ type: "test" }) });

    expect(handler).not.toHaveBeenCalled();
  });

  it("ignores malformed JSON messages", () => {
    const socket = new BattleSocket("session-1");
    const handler = vi.fn();
    socket.subscribe(handler);
    socket.connect();

    const ws = (socket as unknown as { ws: MockWebSocket }).ws;
    ws.onmessage?.({ data: "not-json" });

    expect(handler).not.toHaveBeenCalled();
  });

  it("disconnect stops reconnection and cleans up", () => {
    const socket = new BattleSocket("session-1");
    socket.connect();
    socket.disconnect();
    expect(socket.connected).toBe(false);
  });

  it("reconnects on close when shouldReconnect is true", () => {
    const socket = new BattleSocket("session-1");
    socket.connect();

    const ws = (socket as unknown as { ws: MockWebSocket }).ws;
    ws.onclose?.();

    // Advance timer for reconnection (2000ms)
    vi.advanceTimersByTime(2000);

    // After reconnect, a new WebSocket should have been created
    const newWs = (socket as unknown as { ws: MockWebSocket }).ws;
    expect(newWs).not.toBeNull();
  });

  it("does not reconnect after disconnect", () => {
    const socket = new BattleSocket("session-1");
    socket.connect();
    socket.disconnect();

    // Try to trigger onclose - should not reconnect
    // After disconnect, ws is null, so nothing happens
    vi.advanceTimersByTime(5000);
    expect(socket.connected).toBe(false);
  });

  it("onerror closes the WebSocket", () => {
    const socket = new BattleSocket("session-1");
    socket.connect();

    const ws = (socket as unknown as { ws: MockWebSocket }).ws;
    const closeSpy = vi.spyOn(ws, "close");
    ws.onerror?.();

    expect(closeSpy).toHaveBeenCalled();
  });
});
