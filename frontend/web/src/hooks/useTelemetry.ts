import { useState, useEffect, useRef } from 'react';
import type { TelemetryData } from '../types';

// Derive the WS endpoint from the browser's own host so the hook works:
//   - in dev:    Vite proxies /ws → localhost:8000
//   - in Docker: Nginx proxies /ws → backend:8000
function resolveWsUrl(path: string): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${path}`;
}

export const useTelemetry = (path: string = '/ws/telemetry') => {
  const url = resolveWsUrl(path);
  const [data, setData] = useState<TelemetryData | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const [reconnectTrigger, setReconnectTrigger] = useState(0);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout>;

    try {
      socket = new WebSocket(url);
      socketRef.current = socket;

      socket.onopen = () => {
        setConnected(true);
        setError(null);
        console.log('Connected to telemetry server');
      };

      socket.onmessage = (event) => {
        try {
          const telemetry: TelemetryData = JSON.parse(event.data);
          setData(telemetry);
        } catch (err) {
          console.error('Failed to parse telemetry data', err);
        }
      };

      socket.onclose = () => {
        setConnected(false);
        console.log('Disconnected from telemetry server, retrying in 3s…');
        reconnectTimeout = setTimeout(() => {
          setReconnectTrigger(prev => prev + 1);
        }, 3000);
      };

      socket.onerror = (event) => {
        setError('WebSocket error occurred');
        console.error('WebSocket error', event);
      };
    } catch (err) {
      setTimeout(() => setError('Failed to establish connection'), 0);
      console.error(err);
    }

    return () => {
      if (socket) socket.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, [url, reconnectTrigger]);

  return { data, connected, error };
};
