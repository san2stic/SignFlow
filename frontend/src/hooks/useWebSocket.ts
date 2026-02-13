import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000";

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

  const url = useMemo(() => `${WS_URL}/api/v1${path}`, [path]);

  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    if (!enabled) {
      socketRef.current?.close();
      socketRef.current = null;
      setConnected(false);
      return;
    }

    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onerror = () => setConnected(false);
    socket.onmessage = (event) => {
      try {
        onMessageRef.current(JSON.parse(event.data) as TOut);
      } catch {
        // Ignore malformed payloads.
      }
    };

    return () => {
      socket.close();
      if (socketRef.current === socket) {
        socketRef.current = null;
      }
    };
  }, [enabled, url]);

  const send = useCallback((payload: TIn): void => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    socket.send(JSON.stringify(payload));
  }, []);

  return { connected, send };
}
