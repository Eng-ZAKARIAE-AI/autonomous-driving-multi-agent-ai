import { PageHeader } from '../components/PageHeader';
import { Plus, MoreHorizontal, Bot } from 'lucide-react';

const mockAgents = [
  { id: 1, name: 'PPO-v3', algorithm: 'PPO', status: 'training', reward: 84.1, episodes: 412 },
  { id: 2, name: 'SAC-v1', algorithm: 'SAC', status: 'error', reward: 62.3, episodes: 398 },
  { id: 3, name: 'DQN-v2', algorithm: 'DQN', status: 'idle', reward: 79.4, episodes: 850 },
  { id: 4, name: 'PPO-v4', algorithm: 'PPO', status: 'training', reward: 71.8, episodes: 120 },
];

export const Agents = () => {
  return (
    <>
      <PageHeader 
        title="Agents" 
        description="Monitor and manage your autonomous driving agent fleet" 
        actions={
          <button className="btn btn-primary">
            <Plus size={12} strokeWidth={2.5} />
            New Agent
          </button>
        }
      />

      <div className="bg-bg3 border border-border rounded-lg p-[14px_16px]">
        <div className="text-sm-custom font-medium text-text tracking-[0.03em] uppercase mb-4">Agent Fleet</div>
        <table className="w-full border-collapse">
          <thead>
            <tr className="border-b border-border">
              <th className="text-xs-custom font-medium text-muted uppercase tracking-[0.06em] text-left p-[0_8px_8px_0]">Name</th>
              <th className="text-xs-custom font-medium text-muted uppercase tracking-[0.06em] text-left p-[0_8px_8px_0]">Algorithm</th>
              <th className="text-xs-custom font-medium text-muted uppercase tracking-[0.06em] text-left p-[0_8px_8px_0]">Status</th>
              <th className="text-xs-custom font-medium text-muted uppercase tracking-[0.06em] text-right p-[0_8px_8px_0]">Best Reward</th>
              <th className="text-xs-custom font-medium text-muted uppercase tracking-[0.06em] text-right p-[0_8px_8px_0]">Episodes</th>
              <th className="text-xs-custom font-medium text-muted uppercase tracking-[0.06em] text-right p-[0_8px_8px_0]">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {mockAgents.map((agent) => (
              <tr key={agent.id} className="group hover:bg-white/5 transition-colors">
                <td className="p-[10px_8px_10px_0] flex items-center gap-2">
                  <Bot size={14} className="text-muted" />
                  <span className="text-base-custom text-text font-mono">{agent.name}</span>
                </td>
                <td className="text-base-custom text-muted p-[10px_8px_10px_0]">{agent.algorithm}</td>
                <td className="p-[10px_8px_10px_0]">
                  <span className={`badge badge-${agent.status}`}>{agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}</span>
                </td>
                <td className="text-base-custom p-[10px_8px_10px_0] font-mono text-right text-green">{agent.reward.toFixed(1)}</td>
                <td className="text-base-custom p-[10px_8px_10px_0] font-mono text-right text-text">{agent.episodes}</td>
                <td className="p-[10px_8px_10px_0] text-right">
                  <button className="p-1 text-muted hover:text-text transition-colors">
                    <MoreHorizontal size={14} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {mockAgents.map((agent) => (
          <div key={agent.id} className="bg-bg3 border border-border rounded-lg p-[14px_16px] relative overflow-hidden group">
            <div className="absolute top-0 left-0 right-0 h-[2px] bg-accent2 opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="flex justify-between items-start mb-3">
              <div className="text-xs-custom font-medium text-muted uppercase tracking-[0.03em]">{agent.algorithm}</div>
              <span className={`badge badge-${agent.status}`}>{agent.status}</span>
            </div>
            <div className="text-[18px] font-semibold text-text tracking-[-0.01em] mb-1 font-mono">{agent.name}</div>
            <div className="flex justify-between items-center mt-4">
              <div className="text-xs-custom text-muted uppercase">Reward</div>
              <div className="text-base-custom font-mono text-green">{agent.reward}</div>
            </div>
          </div>
        ))}
      </div>
    </>
  );
};
