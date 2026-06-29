import { useState, useEffect } from 'react';
import { PageHeader } from '../components/PageHeader';
import { useTelemetry } from '../hooks/useTelemetry';
import { Download, Wifi, WifiOff } from 'lucide-react';

interface LogEntry {
  id: number;
  timestamp: string;
  agent: string;
  event: string;
  value: string;
  severity: 'training' | 'warning' | 'error' | 'info';
}

let _logCounter = 0;

function makeLog(telemetry: { speed: number; lane_offset: number; collision: boolean; reward: number; episode?: number }): LogEntry {
  const now = new Date();
  const ts = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
  const ep = telemetry.episode ?? '?';

  if (telemetry.collision) {
    return { id: _logCounter++, timestamp: ts, agent: 'RL-Agent', event: `Collision detected (ep ${ep})`, value: '', severity: 'error' };
  }
  if (Math.abs(telemetry.lane_offset) > 1.5) {
    return { id: _logCounter++, timestamp: ts, agent: 'RL-Agent', event: 'Lane departure', value: `${telemetry.lane_offset.toFixed(2)}m`, severity: 'warning' };
  }
  if (telemetry.reward > 5) {
    return { id: _logCounter++, timestamp: ts, agent: 'RL-Agent', event: `High reward (ep ${ep})`, value: `+${telemetry.reward.toFixed(1)}`, severity: 'training' };
  }
  return { id: _logCounter++, timestamp: ts, agent: 'RL-Agent', event: `Step — speed ${telemetry.speed.toFixed(1)} m/s`, value: `R=${telemetry.reward.toFixed(2)}`, severity: 'info' };
}

const SEV_COLOR: Record<string, string> = {
  training: 'badge-training',
  warning: 'badge-warning',
  error: 'badge-error',
  info: 'badge-info',
};

export const Telemetry = () => {
  const { data, connected } = useTelemetry();
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [search, setSearch] = useState('');
  const [severity, setSeverity] = useState('all');

  useEffect(() => {
    if (!data) return;
    setLogs(prev => [makeLog(data as Parameters<typeof makeLog>[0]), ...prev].slice(0, 200));
  }, [data]);

  const filtered = logs.filter(l => {
    const matchSearch = !search || l.event.toLowerCase().includes(search.toLowerCase()) || l.agent.toLowerCase().includes(search.toLowerCase());
    const matchSev = severity === 'all' || l.severity === severity;
    return matchSearch && matchSev;
  });

  const handleExport = () => {
    const csv = ['timestamp,agent,event,value,severity', ...logs.map(l => `${l.timestamp},${l.agent},"${l.event}",${l.value},${l.severity}`)].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = 'telemetry.csv'; a.click();
  };

  return (
    <>
      <PageHeader
        title="Telemetry logs"
        description="Live sensor data and event stream from the CARLA environment"
        actions={
          <div className="flex items-center gap-2">
            <span className={`badge ${connected ? 'badge-training' : 'badge-error'} flex items-center gap-1`}>
              {connected ? <Wifi size={10} /> : <WifiOff size={10} />}
              {connected ? 'LIVE' : 'OFFLINE'}
            </span>
            <button className="btn" onClick={handleExport}>
              <Download size={12} strokeWidth={2.5} />
              Export CSV
            </button>
          </div>
        }
      />

      {/* Live metrics strip */}
      {data && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: 'Speed', value: `${(data.speed ?? 0).toFixed(1)} m/s`, color: 'text-accent' },
            { label: 'Lane Offset', value: `${(data.lane_offset ?? 0).toFixed(2)} m`, color: Math.abs(data.lane_offset ?? 0) > 1 ? 'text-yellow' : 'text-green' },
            { label: 'Reward', value: (data.reward ?? 0) >= 0 ? `+${(data.reward ?? 0).toFixed(2)}` : `${(data.reward ?? 0).toFixed(2)}`, color: (data.reward ?? 0) >= 0 ? 'text-green' : 'text-red' },
            { label: 'Collision', value: (data.collision ?? false) ? 'YES' : 'NO', color: (data.collision ?? false) ? 'text-red' : 'text-green' },
          ].map(m => (
            <div key={m.label} className="bg-bg3 border border-border rounded-lg p-3">
              <p className="text-xs-custom font-medium text-muted uppercase tracking-wider mb-1">{m.label}</p>
              <p className={`text-[18px] font-semibold font-mono ${m.color}`}>{m.value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="bg-bg3 border border-border rounded-lg p-3 flex flex-wrap gap-3 items-center">
        <div className="flex-1 min-w-[200px] relative">
          <input
            type="text" placeholder="Search events or agents..." value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent"
          />
        </div>
        <select
          value={severity} onChange={e => setSeverity(e.target.value)}
          className="bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none"
        >
          <option value="all">All Severities</option>
          <option value="info">Info</option>
          <option value="training">Training</option>
          <option value="warning">Warning</option>
          <option value="error">Error</option>
        </select>
      </div>

      {/* Table */}
      <div className="bg-bg3 border border-border rounded-lg overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-border bg-bg2/50">
              {['Timestamp', 'Agent', 'Event', 'Value', 'Severity'].map(h => (
                <th key={h} className="p-[12px_16px] text-xs-custom font-medium text-muted uppercase tracking-[0.06em]">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {filtered.length === 0 ? (
              <tr>
                <td colSpan={5} className="p-6 text-center text-muted text-sm-custom">
                  {connected ? 'Waiting for telemetry data…' : 'Backend offline — start the server to receive live data.'}
                </td>
              </tr>
            ) : filtered.slice(0, 100).map(log => (
              <tr key={log.id} className="hover:bg-white/5 transition-colors font-mono">
                <td className="p-[10px_16px] text-sm-custom text-muted">{log.timestamp}</td>
                <td className="p-[10px_16px]"><span className="text-sm-custom text-accent">{log.agent}</span></td>
                <td className="p-[10px_16px] text-sm-custom text-text font-sans">{log.event}</td>
                <td className="p-[10px_16px] text-sm-custom text-muted">{log.value}</td>
                <td className="p-[10px_16px]">
                  <span className={`badge ${SEV_COLOR[log.severity] ?? 'badge-info'}`}>{log.severity.toUpperCase()}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="p-[12px_16px] border-t border-border bg-bg2/30 flex justify-between items-center">
          <span className="text-xs-custom text-muted font-medium">
            Showing {Math.min(filtered.length, 100)} of {filtered.length} logs ({logs.length} total)
          </span>
        </div>
      </div>
    </>
  );
};
