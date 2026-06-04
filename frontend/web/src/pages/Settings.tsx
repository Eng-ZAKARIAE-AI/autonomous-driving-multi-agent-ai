import { useState } from 'react';
import { PageHeader } from '../components/PageHeader';
import { Server, Monitor, Bell, Info, ShieldCheck, ExternalLink } from 'lucide-react';

export const Settings = () => {
  const [wsUrl, setWsUrl] = useState('ws://localhost:8000/ws/telemetry');
  const [theme, setTheme] = useState('dark');
  const [notifications, setNotifications] = useState({
    collisions: true,
    trainingComplete: true,
    systemErrors: true,
  });

  return (
    <div className="max-w-3xl space-y-3">
      <PageHeader 
        title="Settings" 
        description="Configure your workspace preferences and backend connections" 
      />

      {/* Backend Configuration */}
      <section className="bg-bg3 border border-border rounded-lg overflow-hidden">
        <div className="p-[14px_16px] border-b border-border flex items-center gap-3">
          <div className="text-accent">
            <Server size={14} />
          </div>
          <h3 className="text-sm-custom font-medium text-text uppercase tracking-wider">Backend</h3>
        </div>
        <div className="p-[14px_16px] space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">WebSocket URL</label>
            <div className="flex gap-2">
              <input 
                type="text" 
                value={wsUrl}
                onChange={(e) => setWsUrl(e.target.value)}
                className="flex-1 bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent font-mono"
              />
              <button className="btn !p-[3px_12px] !text-xs-custom">
                Test
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Interface Settings */}
      <section className="bg-bg3 border border-border rounded-lg overflow-hidden">
        <div className="p-[14px_16px] border-b border-border flex items-center gap-3">
          <div className="text-purple">
            <Monitor size={14} />
          </div>
          <h3 className="text-sm-custom font-medium text-text uppercase tracking-wider">Interface</h3>
        </div>
        <div className="p-[14px_16px] space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <p className="text-base-custom font-medium text-text">Color Theme</p>
              <p className="text-sm-custom text-muted">Switch between light and dark mode.</p>
            </div>
            <div className="flex bg-bg p-1 rounded-md border border-border">
              <button 
                onClick={() => setTheme('light')}
                className={`px-3 py-1 rounded-md text-xs-custom font-bold transition-all ${theme === 'light' ? 'bg-bg4 text-text' : 'text-muted hover:text-text'}`}
              >
                LIGHT
              </button>
              <button 
                onClick={() => setTheme('dark')}
                className={`px-3 py-1 rounded-md text-xs-custom font-bold transition-all ${theme === 'dark' ? 'bg-bg4 text-text' : 'text-muted hover:text-text'}`}
              >
                DARK
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Notifications */}
      <section className="bg-bg3 border border-border rounded-lg overflow-hidden">
        <div className="p-[14px_16px] border-b border-border flex items-center gap-3">
          <div className="text-yellow">
            <Bell size={14} />
          </div>
          <h3 className="text-sm-custom font-medium text-text uppercase tracking-wider">Notifications</h3>
        </div>
        <div className="p-[14px_16px] space-y-4">
          {[
            { id: 'collisions', label: 'Collision Alerts', desc: 'Instant notification on agent collision events.' },
            { id: 'trainingComplete', label: 'Training Completion', desc: 'Get notified when a training session finishes.' },
            { id: 'systemErrors', label: 'System Errors', desc: 'Alerts for backend or simulator connection failures.' },
          ].map((item) => (
            <div key={item.id} className="flex items-center justify-between py-1">
              <div className="space-y-0.5">
                <p className="text-base-custom font-medium text-text">{item.label}</p>
                <p className="text-sm-custom text-muted">{item.desc}</p>
              </div>
              <button 
                onClick={() => setNotifications(prev => ({ ...prev, [item.id]: !prev[item.id as keyof typeof notifications] }))}
                className={`w-8 h-4 rounded-full transition-colors relative ${notifications[item.id as keyof typeof notifications] ? 'bg-accent2' : 'bg-bg4'}`}
              >
                <div className={`absolute top-0.5 w-3 h-3 bg-white rounded-full transition-all ${notifications[item.id as keyof typeof notifications] ? 'left-4.5' : 'left-0.5'}`} />
              </button>
            </div>
          ))}
        </div>
      </section>

      {/* About */}
      <section className="bg-bg3 border border-border rounded-lg overflow-hidden">
        <div className="p-[14px_16px] border-b border-border flex items-center gap-3">
          <div className="text-green">
            <Info size={14} />
          </div>
          <h3 className="text-sm-custom font-medium text-text uppercase tracking-wider">About</h3>
        </div>
        <div className="p-[14px_16px] space-y-3">
          <div className="flex justify-between items-center py-1 border-b border-border/50">
            <span className="text-sm-custom text-muted">Core Version</span>
            <span className="text-sm-custom font-mono text-text">v1.2.4-stable</span>
          </div>
          <div className="flex justify-between items-center py-1 border-b border-border/50">
            <span className="text-sm-custom text-muted">Frontend UI</span>
            <span className="text-sm-custom font-mono text-text">v4.0.0-viva</span>
          </div>
          
          <div className="pt-2 flex gap-4">
            <a href="#" className="text-xs-custom font-medium text-muted hover:text-accent flex items-center gap-1.5 transition-colors uppercase tracking-wider">
              <ExternalLink size={12} />
              Repository
            </a>
            <a href="#" className="text-xs-custom font-medium text-muted hover:text-accent flex items-center gap-1.5 transition-colors uppercase tracking-wider">
              <ShieldCheck size={12} />
              License
            </a>
          </div>
        </div>
      </section>
    </div>
  );
};
