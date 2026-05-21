import { PageHeader } from '../components/PageHeader';
import { Filter, Download, Search, Calendar } from 'lucide-react';

const mockLogs = [
  { id: 1, timestamp: '14:20:01', agent: 'PPO-v3', event: 'Lane departure detected', value: '1.24m', severity: 'warning' },
  { id: 2, timestamp: '14:20:05', agent: 'PPO-v3', event: 'Correction applied', value: '-0.50', severity: 'training' },
  { id: 3, timestamp: '14:21:12', agent: 'SAC-v1', event: 'Collision imminent', value: 'Dist: 2.1m', severity: 'error' },
  { id: 4, timestamp: '14:21:15', agent: 'SAC-v1', event: 'Emergency braking', value: '100%', severity: 'training' },
  { id: 5, timestamp: '14:22:30', agent: 'PPO-v3', event: 'Checkpoint reached', value: 'CP_04', severity: 'training' },
  { id: 6, timestamp: '14:23:45', agent: 'DQN-v2', event: 'Model sync failure', value: 'Err_504', severity: 'error' },
  { id: 7, timestamp: '14:24:00', agent: 'PPO-v4', event: 'Training episode start', value: '#120', severity: 'training' },
  { id: 8, timestamp: '14:25:10', agent: 'PPO-v3', event: 'High reward peak', value: '12.4', severity: 'training' },
];

export const Telemetry = () => {
  return (
    <>
      <PageHeader 
        title="Telemetry logs" 
        description="Examine detailed execution logs and sensor data history" 
        actions={
          <button className="btn">
            <Download size={12} strokeWidth={2.5} />
            Export CSV
          </button>
        }
      />

      {/* Filters */}
      <div className="bg-bg3 border border-border rounded-lg p-3 flex flex-wrap gap-3 items-center">
        <div className="flex-1 min-w-[200px] relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
          <input 
            type="text" 
            placeholder="Search events or agents..."
            className="w-full bg-bg border border-border rounded-md pl-9 pr-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent"
          />
        </div>
        
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text font-mono">
            <Calendar size={12} className="text-muted" />
            <span>2026-05-21</span>
          </div>
          
          <select className="bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none">
            <option>All Severities</option>
            <option>Info</option>
            <option>Warning</option>
            <option>Error</option>
          </select>

          <button className="btn p-2">
            <Filter size={14} />
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="bg-bg3 border border-border rounded-lg overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-border bg-bg2/50">
              <th className="p-[12px_16px] text-xs-custom font-medium text-muted uppercase tracking-[0.06em]">Timestamp</th>
              <th className="p-[12px_16px] text-xs-custom font-medium text-muted uppercase tracking-[0.06em]">Agent</th>
              <th className="p-[12px_16px] text-xs-custom font-medium text-muted uppercase tracking-[0.06em]">Event</th>
              <th className="p-[12px_16px] text-xs-custom font-medium text-muted uppercase tracking-[0.06em]">Value</th>
              <th className="p-[12px_16px] text-xs-custom font-medium text-muted uppercase tracking-[0.06em] text-right">Severity</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {mockLogs.map((log) => (
              <tr key={log.id} className="hover:bg-white/5 transition-colors group font-mono">
                <td className="p-[10px_16px] text-sm-custom text-muted">{log.timestamp}</td>
                <td className="p-[10px_16px]">
                  <span className="text-sm-custom text-accent">{log.agent}</span>
                </td>
                <td className="p-[10px_16px] text-sm-custom text-text font-sans">{log.event}</td>
                <td className="p-[10px_16px] text-sm-custom text-muted">{log.value}</td>
                <td className="p-[10px_16px] text-right">
                  <span className={`badge badge-${log.severity}`}>{log.severity.toUpperCase()}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="p-[12px_16px] border-t border-border bg-bg2/30 flex justify-between items-center">
          <span className="text-xs-custom text-muted font-medium">Showing 8 of 1,248 logs</span>
          <div className="flex gap-2">
            <button className="btn !p-[3px_10px] !text-xs-custom">PREV</button>
            <button className="btn !p-[3px_10px] !text-xs-custom">NEXT</button>
          </div>
        </div>
      </div>
    </>
  );
};
