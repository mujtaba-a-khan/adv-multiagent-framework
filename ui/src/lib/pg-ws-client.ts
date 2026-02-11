import { WS_BASE_URL } from "./constants";
import type { PGWSMessage } from "./types";

export type PGEventHandler = (msg: PGWSMessage) => void;

export class PGSocket {
  private ws: WebSocket | null = null;
  private handlers = new Set<PGEventHandler>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _conversationId: string;
  private _shouldReconnect = true;

  constructor(conversationId: string) {
    this._conversationId = conversationId;
  }

  get conversationId() {
    return this._conversationId;
  }

  get connected() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  connect() {
    this.cleanup();
    this._shouldReconnect = true;

    const url = `${WS_BASE_URL}/api/v1/ws/playground/${this._conversationId}`;
    this.ws = new WebSocket(url);

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as PGWSMessage;
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

  subscribe(handler: PGEventHandler) {
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
