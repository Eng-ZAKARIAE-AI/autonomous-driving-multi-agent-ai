import { useState, useEffect } from 'react';
import { Layout } from './components/Layout';
import { MetricCard } from './components/MetricCard';
import { RewardChart } from './components/RewardChart';
import { useTelemetry } from './hooks/useTelemetry';
import { Gauge, Zap, AlertTriangle, Crosshair, Cpu } from 'lucide-react';

function App() {
  const { data, connected } = useTelemetry();
  const [rewardHistory, setRewardHistory] = useState<{ step: number; reward: number }[]>([]);
  const [stepCount, setStepCount] = useState(0);

  useEffect(() => {
    if (data) {
      setStepCount(prev => prev + 1);
      setRewardHistory(prev => {
        const newHistory = [...prev, { step: stepCount, reward: data.reward }];
        return newHistory.slice(-50); // Keep last 50 steps
      });
    }
  }, [data]);

  return (
    <Layout connected={connected}>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard 
          label="Speed" 
          value={data?.speed.toFixed(1) || 0} 
          unit="km/h" 
          icon={Gauge}
        />
        <MetricCard 
          label="Lane Offset" 
          value={data?.lane_offset.toFixed(2) || 0} 
          unit="m" 
          icon={Crosshair}
          status={(data?.lane_offset || 0) > 1.0 ? 'warning' : 'success'}
        />
        <MetricCard 
          label="Cumulative Reward" 
          value={rewardHistory.reduce((acc, curr) => acc + curr.reward, 0).toFixed(1)} 
          icon={Zap}
          status="normal"
        />
        <MetricCard 
          label="Collision" 
          value={data?.collision ? "DETECTED" : "NONE"} 
          icon={AlertTriangle}
          status={data?.collision ? 'error' : 'success'}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Live View Placeholder */}
        <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col shadow-lg">
          <div className="px-4 py-3 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Agent Perception Feed (84x84)</h3>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-safety-green animate-pulse"></span>
              <span className="text-[10px] font-bold text-slate-500 uppercase">Live MJPEG</span>
            </div>
          </div>
          <div className="flex-1 bg-black flex items-center justify-center relative group min-h-[400px]">
             {/* This would be an <img> or <canvas> for the stream */}
             <div className="text-slate-700 flex flex-col items-center gap-4">
                <div className="w-48 h-48 border-2 border-dashed border-slate-800 rounded-lg flex items-center justify-center">
                  <span className="text-xs uppercase tracking-widest opacity-50">No Stream Data</span>
                </div>
                <p className="text-[10px] text-slate-600 max-w-xs text-center uppercase tracking-tighter">
                  Wait for backend to broadcast multi-modal sensor state...
                </p>
             </div>
             
             {/* Overlay Controls */}
             <div className="absolute bottom-4 left-4 right-4 flex justify-between items-end">
                <div className="bg-black/60 backdrop-blur-sm p-3 rounded-lg border border-white/5 flex flex-col gap-2">
                  <div className="flex items-center gap-4">
                    <div className="flex flex-col">
                      <span className="text-[10px] text-slate-500 uppercase font-bold">Steer</span>
                      <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden mt-1">
                        <div 
                          className="h-full bg-primary" 
                          style={{ width: `${Math.abs((data?.action[0] || 0) * 100)}%`, marginLeft: (data?.action[0] || 0) < 0 ? '0' : 'auto' }}
                        ></div>
                      </div>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-[10px] text-slate-500 uppercase font-bold">Throttle</span>
                      <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden mt-1">
                        <div 
                          className="h-full bg-safety-green" 
                          style={{ width: `${(data?.action[1] || 0) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-primary/20 backdrop-blur-sm px-4 py-2 rounded-lg border border-primary/30 flex items-center gap-2">
                  <Cpu size={14} className="text-primary" />
                  <span className="text-xs font-bold text-white uppercase tracking-wider">PPO Agent</span>
                </div>
             </div>
          </div>
        </div>

        {/* Right Sidebar Widgets */}
        <div className="flex flex-col gap-6">
          <RewardChart data={rewardHistory} />
          
          <div className="bg-surface border border-slate-800 rounded-xl p-6 shadow-sm flex-1">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-4">Environment Status</h3>
            <div className="flex flex-col gap-4">
              <div className="flex justify-between items-center py-2 border-b border-slate-800/50">
                <span className="text-sm text-slate-400">Weather</span>
                <span className="text-sm font-semibold text-white">Clear Noon</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-slate-800/50">
                <span className="text-sm text-slate-400">Town</span>
                <span className="text-sm font-semibold text-white">Town03</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-slate-800/50">
                <span className="text-sm text-slate-400">Traffic Density</span>
                <span className="text-sm font-semibold text-white">Medium</span>
              </div>
              <div className="mt-4">
                <button className="w-full bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold py-3 rounded-lg transition-colors uppercase tracking-widest border border-slate-700">
                  Switch to SAC Agent
                </button>
              </div>
              <div className="mt-2">
                <button className="w-full bg-safety-red/10 hover:bg-safety-red/20 text-safety-red text-xs font-bold py-3 rounded-lg transition-colors uppercase tracking-widest border border-safety-red/30">
                  Emergency Stop
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default App;
