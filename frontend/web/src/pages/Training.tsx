import { useState, useEffect, useRef } from 'react';
import { PageHeader } from '../components/PageHeader';
import { RewardChart } from '../components/RewardChart';
import { Play, Square, Settings2, BarChart3, AlertCircle } from 'lucide-react';

const API = 'http://localhost:8000';

interface TrainingStatus {
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

const DEFAULT_STATUS: TrainingStatus = {
  status: 'idle',
  algorithm: null,
  mode: null,
  episode: 0,
  total_episodes: 0,
  reward: 0,
  avg_speed: 0,
  lane_deviation: 0,
  collision: false,
  error: null,
  elapsed: 0,
};

export const Training = () => {
  const [learningRate, setLearningRate] = useState(0.0003);
  const [episodes, setEpisodes] = useState(200);
  const [discountFactor, setDiscountFactor] = useState(0.99);
  const [algorithm, setAlgorithm] = useState('ppo');
  const [mode, setMode] = useState('train');
  const [status, setStatus] = useState<TrainingStatus>(DEFAULT_STATUS);
  const [rewardHistory, setRewardHistory] = useState<{ step: number; reward: number }[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll training status while running
  useEffect(() => {
    const poll = () => {
      fetch(`${API}/api/training/status`)
        .then(r => r.json())
        .then((s: TrainingStatus) => {
          setStatus(s);
          if (s.reward !== 0) {
            setRewardHistory(prev => {
              const next = [...prev, { step: prev.length + 1, reward: s.reward }];
              return next.slice(-50);
            });
          }
        })
        .catch(() => {});
    };

    pollRef.current = setInterval(poll, 2000);
    poll();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const isRunning = status.status === 'training' || status.status === 'evaluating';
  const progress = status.total_episodes > 0
    ? Math.round((status.episode / status.total_episodes) * 100)
    : 0;

  const handleStart = async () => {
    setErrorMsg(null);
    try {
      const res = await fetch(`${API}/api/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm, mode, episodes }),
      });
      const data = await res.json();
      if (!data.success) setErrorMsg(data.message);
    } catch {
      setErrorMsg('Cannot reach backend server (localhost:8000).');
    }
  };

  const handleStop = async () => {
    await fetch(`${API}/api/training/stop`, { method: 'POST' }).catch(() => {});
  };

  const statusColor: Record<string, string> = {
    idle: 'badge-idle',
    training: 'badge-training',
    evaluating: 'badge-info',
    error: 'badge-error',
  };

  return (
    <>
      <PageHeader
        title="Training"
        description="Configure and launch reinforcement learning sessions"
        actions={
          <span className={`badge ${statusColor[status.status] ?? 'badge-idle'}`}>
            {status.status.toUpperCase()}
          </span>
        }
      />

      {errorMsg && (
        <div className="flex items-center gap-2 bg-red/10 border border-red/30 rounded-md p-3 text-sm-custom text-red">
          <AlertCircle size={14} />
          {errorMsg}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* Config */}
        <section className="bg-bg3 border border-border rounded-lg p-[14px_16px]">
          <div className="flex items-center gap-2 mb-4 text-accent">
            <Settings2 size={14} />
            <h3 className="text-sm-custom font-medium text-text uppercase tracking-wider">Configuration</h3>
          </div>

          <form className="space-y-4" onSubmit={e => e.preventDefault()}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Learning Rate</label>
                <input
                  type="number" step="0.0001" value={learningRate}
                  onChange={e => setLearningRate(parseFloat(e.target.value))}
                  disabled={isRunning}
                  className="w-full bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent transition-colors font-mono disabled:opacity-50"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Episodes</label>
                <input
                  type="number" value={episodes}
                  onChange={e => setEpisodes(parseInt(e.target.value))}
                  disabled={isRunning}
                  className="w-full bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent transition-colors font-mono disabled:opacity-50"
                />
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Discount Factor (γ)</label>
                <span className="text-sm-custom font-mono text-accent">{discountFactor}</span>
              </div>
              <input
                type="range" min="0.9" max="1.0" step="0.001" value={discountFactor}
                onChange={e => setDiscountFactor(parseFloat(e.target.value))}
                disabled={isRunning}
                className="w-full h-1 bg-bg4 rounded-lg appearance-none cursor-pointer accent-accent disabled:opacity-50"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Algorithm</label>
                <select
                  value={algorithm}
                  onChange={e => setAlgorithm(e.target.value)}
                  disabled={isRunning}
                  className="w-full bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent transition-colors disabled:opacity-50"
                >
                  <option value="ppo">PPO (Proximal Policy Optimization)</option>
                  <option value="sac">SAC (Soft Actor-Critic)</option>
                  <option value="baseline">Baseline (TransFuser / InterFuser)</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Mode</label>
                <select
                  value={mode}
                  onChange={e => setMode(e.target.value)}
                  disabled={isRunning}
                  className="w-full bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent transition-colors disabled:opacity-50"
                >
                  {algorithm !== 'baseline' && <option value="train">Train</option>}
                  <option value="evaluate">Evaluate</option>
                  <option value="infer">Infer (deterministic)</option>
                </select>
              </div>
            </div>

            <div className="flex gap-2 mt-4">
              <button
                onClick={handleStart}
                disabled={isRunning}
                className="btn btn-primary flex-1 justify-center h-9 disabled:opacity-50"
              >
                <Play size={12} fill="currentColor" strokeWidth={0} />
                {mode === 'train' ? 'Start Training' : 'Start Evaluation'}
              </button>
              {isRunning && (
                <button
                  onClick={handleStop}
                  className="btn h-9 px-4 border-red/40 text-red hover:bg-red/10"
                >
                  <Square size={12} fill="currentColor" strokeWidth={0} />
                  Stop
                </button>
              )}
            </div>
          </form>
        </section>

        {/* Progress */}
        <section className="bg-bg3 border border-border rounded-lg p-[14px_16px]">
          <div className="flex items-center gap-2 mb-4 text-accent">
            <BarChart3 size={14} />
            <h3 className="text-sm-custom font-medium text-text uppercase tracking-wider">Progress</h3>
          </div>

          <div className="space-y-5">
            <div className="space-y-3">
              <div className="flex justify-between items-end">
                <div className="space-y-0.5">
                  <p className="text-xs-custom font-medium text-muted uppercase tracking-wider">Completion</p>
                  <p className="text-[20px] font-semibold text-text font-mono leading-none">{progress}%</p>
                </div>
                <p className="text-sm-custom text-muted font-mono">
                  {status.episode} / {status.total_episodes || episodes} Episodes
                </p>
              </div>
              <div className="w-full h-2 bg-bg4 rounded-full overflow-hidden">
                <div
                  className="h-full bg-accent shadow-[0_0_10px_rgba(88,166,255,0.3)] transition-all duration-700"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <div className="bg-bg border border-border p-3 rounded-md">
                <p className="text-xs-custom font-medium text-muted uppercase tracking-wider mb-1">Episode</p>
                <p className="text-[16px] font-semibold text-text font-mono">{status.episode}</p>
              </div>
              <div className="bg-bg border border-border p-3 rounded-md">
                <p className="text-xs-custom font-medium text-muted uppercase tracking-wider mb-1">Elapsed</p>
                <p className="text-[16px] font-semibold text-text font-mono">{Math.round(status.elapsed)}s</p>
              </div>
              <div className="bg-bg border border-border p-3 rounded-md">
                <p className="text-xs-custom font-medium text-muted uppercase tracking-wider mb-1">Reward</p>
                <p className={`text-[16px] font-semibold font-mono ${status.reward >= 0 ? 'text-green' : 'text-red'}`}>
                  {status.reward >= 0 ? '+' : ''}{status.reward.toFixed(1)}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div className="bg-bg border border-border p-3 rounded-md">
                <p className="text-xs-custom font-medium text-muted uppercase tracking-wider mb-1">Speed (m/s)</p>
                <p className="text-[15px] font-semibold text-text font-mono">{status.avg_speed.toFixed(1)}</p>
              </div>
              <div className="bg-bg border border-border p-3 rounded-md">
                <p className="text-xs-custom font-medium text-muted uppercase tracking-wider mb-1">Lane Dev</p>
                <p className="text-[15px] font-semibold text-text font-mono">{status.lane_deviation.toFixed(2)}m</p>
              </div>
            </div>

            {status.error && (
              <div className="p-3 rounded-md bg-red/10 border border-red/20 text-sm-custom text-red">
                Error: {status.error}
              </div>
            )}

            {rewardHistory.length > 0 && (
              <div className="h-[80px]">
                <p className="text-xs-custom text-muted uppercase tracking-wider mb-1">Live Reward</p>
                <RewardChart data={rewardHistory} />
              </div>
            )}
          </div>
        </section>
      </div>
    </>
  );
};
