import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const RAW_WS_URL = (import.meta.env.VITE_WS_URL as string | undefined)?.trim();

function trimTrailingSlashes(value: string): string {
  return value.replace(/\/+$/, "");
}

function normalizePath(path: string): string {
  return path.startsWith("/") ? path : `/${path}`;
}

function normalizeWsBase(value: string): string {
  const trimmed = trimTrailingSlashes(value);
  if (trimmed.startsWith("http://")) {
    return `ws://${trimmed.slice("http://".length)}`;
  }
  if (trimmed.startsWith("https://")) {
    return `wss://${trimmed.slice("https://".length)}`;
  }
  return trimmed;
}

function defaultWsBase(): string {
  if (typeof window === "undefined") {
    return "ws://localhost:8000";
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}`;
}

const WS_URL = RAW_WS_URL ? normalizeWsBase(RAW_WS_URL) : defaultWsBase();

interface WebSocketOptions<TOut> {
  path: string;
  onMessage: (payload: TOut) => void;
  enabled?: boolean;
}

export function useWebSocket<TIn extends object, TOut>({
  path,
  onMessage,
  enabled = true
}: WebSocketOptions<TOut>): { connected: boolean; send: (payload: TIn) => void } {
  const socketRef = useRef<WebSocket | null>(null);
  const onMessageRef = useRef(onMessage);
  const [connected, setConnected] = useState(false);
  const [retryNonce, setRetryNonce] = useState(0);
  const retryCountRef = useRef(0);

  const url = useMemo(() => `${WS_URL}/api/v1${normalizePath(path)}`, [path]);

  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    let reconnectTimer: ReturnType<typeof window.setTimeout> | null = null;
    let connectTimer: ReturnType<typeof window.setTimeout> | null = null;
    let disposed = false;
    let socket: WebSocket | null = null;
    const connectDelayMs = import.meta.env.DEV ? 30 : 0;

    const scheduleReconnect = (): void => {
      if (!enabled || reconnectTimer !== null || disposed) {
        return;
      }
      const attempt = retryCountRef.current + 1;
      retryCountRef.current = attempt;
      const backoffMs = Math.min(1000 * 2 ** Math.min(attempt - 1, 4), 8000);
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        setRetryNonce((value) => value + 1);
      }, backoffMs);
    };

    if (!enabled) {
      socketRef.current?.close();
      socketRef.current = null;
      setConnected(false);
      retryCountRef.current = 0;
      return;
    }

    const connect = (): void => {
      if (disposed || !enabled) {
        return;
      }

      socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => {
        retryCountRef.current = 0;
        setConnected(true);
      };

      socket.onclose = () => {
        setConnected(false);
        if (socketRef.current === socket) {
          socketRef.current = null;
        }
        scheduleReconnect();
      };

      socket.onerror = () => {
        setConnected(false);
        // Let "close" drive reconnects to avoid churn loops in dev.
      };

      socket.onmessage = (event) => {
        try {
          onMessageRef.current(JSON.parse(event.data) as TOut);
        } catch {
          // Ignore malformed payloads.
        }
      };
    };

    connectTimer = window.setTimeout(connect, connectDelayMs);

    return () => {
      disposed = true;
      if (connectTimer !== null) {
        window.clearTimeout(connectTimer);
      }
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
      }
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
      } else if (socket) {
        socket.onopen = null;
        socket.onclose = null;
        socket.onerror = null;
        socket.onmessage = null;
      }
      if (socketRef.current === socket) {
        socketRef.current = null;
      }
    };
  }, [enabled, retryNonce, url]);

  const send = useCallback((payload: TIn): void => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    socket.send(JSON.stringify(payload));
  }, []);

  return { connected, send };
}
