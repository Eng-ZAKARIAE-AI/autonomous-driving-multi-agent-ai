import { useEffect, useReducer } from 'react';
import { PageHeader } from '../components/PageHeader';
import { MetricCard } from '../components/MetricCard';
import { RewardChart } from '../components/RewardChart';
import { useTelemetry } from '../hooks/useTelemetry';

type HistoryAction = 
  | { type: 'ADD_DATA'; reward: number }
  | { type: 'INITIALIZE_MOCK' };

function historyReducer(state: { step: number; reward: number }[], action: HistoryAction) {
  switch (action.type) {
    case 'ADD_DATA': {
      const lastStep = state.length > 0 ? state[state.length - 1].step : 0;
      const newHistory = [...state, { step: lastStep + 1, reward: action.reward }];
      return newHistory.slice(-20);
    }
    case 'INITIALIZE_MOCK':
      if (state.length > 0) return state;
      return Array.from({ length: 20 }, (_, i) => ({ step: i, reward: 20 + Math.random() * 60 }));
    default:
      return state;
  }
}

const mockActivities = [
  { id: 1, event: 'Agent PPO-v3 — episode 412 reward', val: '84.1', time: '2m ago', type: 'success' },
  { id: 2, event: 'Simulation launched — Town03, Clear', val: '', time: '8m ago', type: 'info' },
  { id: 3, event: 'Collision detected — SAC-v1, ep 398', val: '', time: '14m ago', type: 'warning' },
  { id: 4, event: 'New best reward — DQN-v2', val: '79.4', time: '22m ago', type: 'success' },
  { id: 5, event: 'Agent SAC-v1 — training error, retrying', val: '', time: '30m ago', type: 'error' },
];

const mockAgents = [
  { name: 'PPO-v3', algo: 'PPO', status: 'training', reward: '84.1' },
  { name: 'SAC-v1', algo: 'SAC', status: 'error', reward: '62.3' },
  { name: 'DQN-v2', algo: 'DQN', status: 'idle', reward: '79.4' },
  { name: 'PPO-v4', algo: 'PPO', status: 'training', reward: '71.8' },
];

export const Dashboard = () => {
  const { data } = useTelemetry();
  const [rewardHistory, dispatch] = useReducer(historyReducer, []);

  useEffect(() => {
    if (data) {
      dispatch({ type: 'ADD_DATA', reward: data.reward });
    } else {
      dispatch({ type: 'INITIALIZE_MOCK' });
    }
  }, [data]);

  return (
    <>
      <PageHeader 
        title="Dashboard" 
        description="Real-time overview — all agents & training sessions" 
      />

      {/* Metric cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard label="Active agents" value="4" delta="+2" trend="up" color="blue" />
        <MetricCard label="Episodes run" value="1 240" delta="↑ 88" trend="up" />
        <MetricCard label="Best reward" value="87.3" delta="+4.1" trend="up" color="green" />
        <MetricCard label="Collisions today" value="12" delta="↑ 3" trend="dn" color="red" />
      </div>

      {/* Chart */}
      <div className="bg-bg3 border border-border rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-base-custom font-medium text-text">Avg reward — last 20 episodes</span>
          <span className="text-sm-custom text-muted font-mono">Agent: PPO-v3 · live</span>
        </div>
        <div className="h-[110px] w-full">
           <RewardChart data={rewardHistory} />
        </div>
      </div>

      {/* Two-col: Activity + Agents */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div className="bg-bg3 border border-border rounded-lg p-[14px_16px]">
          <div className="text-sm-custom font-medium text-text tracking-[0.03em] uppercase mb-2.5">Recent activity</div>
          <div className="divide-y divide-border">
            {mockActivities.map((act) => (
              <div key={act.id} className="flex items-center gap-2.5 py-[7px]">
                <div className={`w-[6px] h-[6px] rounded-full shrink-0 ${
                  act.type === 'success' ? 'bg-green' : 
                  act.type === 'warning' ? 'bg-yellow' : 
                  act.type === 'error' ? 'bg-red' : 'bg-accent'
                }`} />
                <span className="text-base-custom text-[#c9d1d9] flex-1">
                  {act.event} {act.val && <span className="font-mono text-green">{act.val}</span>}
                </span>
                <span className="text-sm-custom text-muted font-mono shrink-0">{act.time}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-bg3 border border-border rounded-lg p-[14px_16px]">
          <div className="text-sm-custom font-medium text-text tracking-[0.03em] uppercase mb-2.5">Agents</div>
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-border">
                <th className="text-xs-custom font-medium text-muted uppercase tracking-[0.06em] text-left p-[0_8px_8px_0]">Name</th>
                <th className="text-xs-custom font-medium text-muted uppercase tracking-[0.06em] text-left p-[0_8px_8px_0]">Algo</th>
                <th className="text-xs-custom font-medium text-muted uppercase tracking-[0.06em] text-left p-[0_8px_8px_0]">Status</th>
                <th className="text-xs-custom font-medium text-muted uppercase tracking-[0.06em] text-right p-[0_8px_8px_0]">Reward</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {mockAgents.map((agent, i) => (
                <tr key={i}>
                  <td className="text-base-custom text-text p-[7px_8px_7px_0] font-mono">{agent.name}</td>
                  <td className="text-base-custom text-muted p-[7px_8px_7px_0]">{agent.algo}</td>
                  <td className="p-[7px_8px_7px_0]">
                    <span className={`badge badge-${agent.status}`}>{agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}</span>
                  </td>
                  <td className="text-base-custom p-[7px_8px_7px_0] font-mono text-right text-green">{agent.reward}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
};
