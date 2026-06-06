import { createContext, useContext, type ReactNode } from 'react';
import { useTelemetry } from '../hooks/useTelemetry';
import type { TelemetryData } from '../types';

interface TelemetryContextValue {
  data: TelemetryData | null;
  connected: boolean;
  error: string | null;
}

const TelemetryContext = createContext<TelemetryContextValue>({
  data: null,
  connected: false,
  error: null,
});

export const TelemetryProvider = ({ children }: { children: ReactNode }) => {
  const value = useTelemetry();
  return (
    <TelemetryContext.Provider value={value}>
      {children}
    </TelemetryContext.Provider>
  );
};

export const useTelemetryContext = () => useContext(TelemetryContext);
