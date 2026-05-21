export interface TelemetryData {
  speed: number;
  lane_offset: number;
  collision: boolean;
  reward: number;
  action: number[];
}

export interface ConnectionStatus {
  connected: boolean;
  error: string | null;
}
