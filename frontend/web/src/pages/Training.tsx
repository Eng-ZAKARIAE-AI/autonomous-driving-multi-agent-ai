import { useState } from 'react';
import { PageHeader } from '../components/PageHeader';
import { Play, Settings2, BarChart3 } from 'lucide-react';

export const Training = () => {
  const [learningRate, setLearningRate] = useState(0.0003);
  const [episodes, setEpisodes] = useState(1000);
  const [discountFactor, setDiscountFactor] = useState(0.99);
  const [algorithm, setAlgorithm] = useState('PPO');

  return (
    <>
      <PageHeader 
        title="Training" 
        description="Configure and launch reinforcement learning sessions" 
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* Training Config */}
        <section className="bg-bg3 border border-border rounded-lg p-[14px_16px]">
          <div className="flex items-center gap-2 mb-4 text-accent">
            <Settings2 size={14} />
            <h3 className="text-sm-custom font-medium text-text uppercase tracking-wider">Configuration</h3>
          </div>

          <form className="space-y-4" onSubmit={(e) => e.preventDefault()}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Learning Rate</label>
                <input 
                  type="number" 
                  step="0.0001"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  className="w-full bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent transition-colors font-mono"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Episodes</label>
                <input 
                  type="number" 
                  value={episodes}
                  onChange={(e) => setEpisodes(parseInt(e.target.value))}
                  className="w-full bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent transition-colors font-mono"
                />
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Discount Factor (γ)</label>
                <span className="text-sm-custom font-mono text-accent">{discountFactor}</span>
              </div>
              <input 
                type="range" 
                min="0.9" 
                max="1.0" 
                step="0.001"
                value={discountFactor}
                onChange={(e) => setDiscountFactor(parseFloat(e.target.value))}
                className="w-full h-1 bg-bg4 rounded-lg appearance-none cursor-pointer accent-accent"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Algorithm</label>
              <select 
                value={algorithm}
                onChange={(e) => setAlgorithm(e.target.value)}
                className="w-full bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent transition-colors"
              >
                <option value="PPO">PPO (Proximal Policy Optimization)</option>
                <option value="SAC">SAC (Soft Actor-Critic)</option>
                <option value="DQN">DQN (Deep Q-Network)</option>
              </select>
            </div>

            <button className="btn btn-primary w-full justify-center mt-4 h-9">
              <Play size={12} fill="currentColor" strokeWidth={0} />
              Start Training Session
            </button>
          </form>
        </section>

        {/* Training Progress */}
        <section className="bg-bg3 border border-border rounded-lg p-[14px_16px]">
          <div className="flex items-center gap-2 mb-4 text-accent">
            <BarChart3 size={14} />
            <h3 className="text-sm-custom font-medium text-text uppercase tracking-wider">Progress</h3>
          </div>

          <div className="space-y-6">
            <div className="space-y-3">
              <div className="flex justify-between items-end">
                <div className="space-y-0.5">
                  <p className="text-xs-custom font-medium text-muted uppercase tracking-wider">Completion</p>
                  <p className="text-[20px] font-semibold text-text font-mono leading-none">64%</p>
                </div>
                <p className="text-sm-custom text-muted font-mono">640 / 1000 Episodes</p>
              </div>
              <div className="w-full h-2 bg-bg4 rounded-full overflow-hidden">
                <div className="h-full bg-accent w-[64%] shadow-[0_0_10px_rgba(88,166,255,0.3)]" />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-2">
              <div className="bg-bg border border-border p-3 rounded-md">
                <p className="text-xs-custom font-medium text-muted uppercase tracking-wider mb-1">Episode</p>
                <p className="text-[16px] font-semibold text-text font-mono">642</p>
              </div>
              <div className="bg-bg border border-border p-3 rounded-md">
                <p className="text-xs-custom font-medium text-muted uppercase tracking-wider mb-1">Step</p>
                <p className="text-[16px] font-semibold text-text font-mono">12,402</p>
              </div>
              <div className="bg-bg border border-border p-3 rounded-md">
                <p className="text-xs-custom font-medium text-muted uppercase tracking-wider mb-1">Reward</p>
                <p className="text-[16px] font-semibold text-green font-mono">+124.5</p>
              </div>
            </div>

            <div className="p-3 rounded-md bg-accent2/5 border border-accent2/10">
              <p className="text-sm-custom text-accent/80 leading-relaxed italic">
                "Agent is showing improved lane-keeping stability but continues to struggle with high-speed cornering in wet conditions."
              </p>
            </div>
          </div>
        </section>
      </div>
    </>
  );
};
