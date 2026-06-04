import { useState } from 'react';
import { PageHeader } from '../components/PageHeader';
import { Globe, Wind, Rocket, Link2, Link2Off } from 'lucide-react';

export const Simulation = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [ip, setIp] = useState('127.0.0.1');
  const [port, setPort] = useState('2000');
  const [town, setTown] = useState('Town03');
  const [weather, setWeather] = useState('Clear');
  const [trafficDensity, setTrafficDensity] = useState(30);

  return (
    <>
      <PageHeader 
        title="Simulation" 
        description="Configure and connect to the CARLA simulator" 
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {/* CARLA Connection */}
        <section className="bg-bg3 border border-border rounded-lg p-[14px_16px]">
          <div className="flex justify-between items-center mb-4">
            <div className="flex items-center gap-2 text-accent">
              <Link2 size={14} />
              <h3 className="text-sm-custom font-medium text-text uppercase tracking-wider">CARLA Connection</h3>
            </div>
            <span className={`badge ${isConnected ? 'badge-training' : 'badge-error'}`}>
              {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </span>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Host IP</label>
                <input 
                  type="text" 
                  value={ip}
                  onChange={(e) => setIp(e.target.value)}
                  className="w-full bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent transition-colors font-mono"
                  placeholder="127.0.0.1"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Port</label>
                <input 
                  type="text" 
                  value={port}
                  onChange={(e) => setPort(e.target.value)}
                  className="w-full bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent transition-colors font-mono"
                  placeholder="2000"
                />
              </div>
            </div>

            <button 
              onClick={() => setIsConnected(!isConnected)}
              className={`btn w-full justify-center h-9 ${isConnected ? '' : 'btn-primary'}`}
            >
              {isConnected ? <Link2Off size={12} /> : <Link2 size={12} />}
              {isConnected ? 'Disconnect from Simulator' : 'Establish Connection'}
            </button>
          </div>
        </section>

        {/* Scenario Config */}
        <section className="bg-bg3 border border-border rounded-lg p-[14px_16px]">
          <div className="flex items-center gap-2 mb-4 text-accent">
            <Globe size={14} />
            <h3 className="text-sm-custom font-medium text-text uppercase tracking-wider">Scenario</h3>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Map / Town</label>
                <select 
                  value={town}
                  onChange={(e) => setTown(e.target.value)}
                  className="w-full bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent transition-colors"
                >
                  <option value="Town01">Town01 (Small)</option>
                  <option value="Town02">Town02 (Urban)</option>
                  <option value="Town03">Town03 (Highway)</option>
                  <option value="Town04">Town04 (Mountain)</option>
                  <option value="Town05">Town05 (Intersection)</option>
                </select>
              </div>
              <div className="space-y-1.5">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider">Weather Condition</label>
                <select 
                  value={weather}
                  onChange={(e) => setWeather(e.target.value)}
                  className="w-full bg-bg border border-border rounded-md px-3 py-1.5 text-base-custom text-text focus:outline-none focus:border-accent transition-colors"
                >
                  <option value="Clear">Clear Noon</option>
                  <option value="Cloudy">Cloudy</option>
                  <option value="Rain">Heavy Rain</option>
                  <option value="Night">Night Clear</option>
                </select>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <label className="text-xs-custom font-medium text-muted uppercase tracking-wider flex items-center gap-2">
                  <Wind size={12} className="text-muted" />
                  Traffic Density
                </label>
                <span className="text-sm-custom font-mono text-accent">{trafficDensity}%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={trafficDensity}
                onChange={(e) => setTrafficDensity(parseInt(e.target.value))}
                className="w-full h-1 bg-bg4 rounded-lg appearance-none cursor-pointer accent-accent"
              />
            </div>

            <button className="btn btn-primary w-full justify-center h-9">
              <Rocket size={12} fill="currentColor" strokeWidth={0} />
              Launch Simulation
            </button>
          </div>
        </section>
      </div>
    </>
  );
};
