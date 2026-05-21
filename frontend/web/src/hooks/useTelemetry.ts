import { useState, useEffect, useCallback, useRef } from 'react';
import type { TelemetryData } from '../types';

export const useTelemetry = (url: string = 'ws://localhost:8000/ws/telemetry') => {
  const [data, setData] = useState<TelemetryData | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    try {
      const socket = new WebSocket(url);
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
        console.log('Disconnected from telemetry server');
        // Reconnect after a delay
        setTimeout(connect, 3000);
      };

      socket.onerror = (event) => {
        setError('WebSocket error occurred');
        console.error('WebSocket error', event);
      };
    } catch (err) {
      setError('Failed to establish connection');
      console.error(err);
    }
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [connect]);

  return { data, connected, error };
};
