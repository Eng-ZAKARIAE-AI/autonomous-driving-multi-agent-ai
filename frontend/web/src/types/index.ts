export interface TelemetryData {
  speed: number;
  lane_offset: number;
  collision: boolean;
  reward: number;
  action: number[];
  episode?: number;
  detections?: Detection[];
}

export interface Detection {
  class_name: string;
  class_id: number;
  confidence: number;
  bbox: [number, number, number, number];
}

export interface ConnectionStatus {
  connected: boolean;
  error: string | null;
}

export interface TrainingStatus {
  status: 'idle' | 'training' | 'evaluating' | 'error';
  algorithm: string | null;
  mode: string | null;
  episode: number;
  total_episodes: number;
  reward: number;
  avg_speed: number;
  lane_deviation: number;
  collision: boolean;
  error: string | null;
  elapsed: number;
}

export interface TrainingStartRequest {
  algorithm: 'ppo' | 'sac' | 'baseline';
  mode: 'train' | 'evaluate' | 'infer';
  episodes: number;
  model_path?: string;
  config_path?: string;
}
