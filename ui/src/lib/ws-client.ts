import { WS_BASE_URL } from "./constants";
import type { WSMessage } from "./types";

export type WSEventHandler = (msg: WSMessage) => void;

export class BattleSocket {
  private ws: WebSocket | null = null;
  private readonly handlers = new Set<WSEventHandler>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private readonly _sessionId: string;
  private _shouldReconnect = true;

  constructor(sessionId: string) {
    this._sessionId = sessionId;
  }

  get sessionId() {
    return this._sessionId;
  }

  get connected() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  connect() {
    this.cleanup();
    this._shouldReconnect = true;

    const url = `${WS_BASE_URL}/api/v1/ws/${this._sessionId}`;
    this.ws = new WebSocket(url);

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as WSMessage;
        this.handlers.forEach((fn) => fn(msg));
      } catch {
        // ignore malformed messages
      }
    };

    this.ws.onclose = () => {
      if (this._shouldReconnect) {
        this.reconnectTimer = setTimeout(() => this.connect(), 2000);
      }
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  subscribe(handler: WSEventHandler) {
    this.handlers.add(handler);
    return () => {
      this.handlers.delete(handler);
    };
  }

  disconnect() {
    this._shouldReconnect = false;
    this.cleanup();
  }

  private cleanup() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.onmessage = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      this.ws.close();
      this.ws = null;
    }
  }
}
