import { WS_BASE_URL } from "./constants";
import type { FTWSMessage } from "./types";

export type FTEventHandler = (msg: FTWSMessage) => void;

export class FTSocket {
  private ws: WebSocket | null = null;
  private handlers = new Set<FTEventHandler>();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _jobId: string;
  private _shouldReconnect = true;

  constructor(jobId: string) {
    this._jobId = jobId;
  }

  get jobId() {
    return this._jobId;
  }

  get connected() {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  connect() {
    this.cleanup();
    this._shouldReconnect = true;

    const url = `${WS_BASE_URL}/api/v1/ws/finetuning/${this._jobId}`;
    this.ws = new WebSocket(url);

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data) as FTWSMessage;
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

  subscribe(handler: FTEventHandler) {
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
