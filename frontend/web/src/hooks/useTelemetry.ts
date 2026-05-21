import { useState, useEffect, useRef } from 'react';
import type { TelemetryData } from '../types';

export const useTelemetry = (url: string = 'ws://localhost:8000/ws/telemetry') => {
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
        console.log('Disconnected from telemetry server');
        // Reconnect after a delay by triggering the effect
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
      if (socket) {
        socket.close();
      }
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
    };
  }, [url, reconnectTrigger]);

  return { data, connected, error };
};
