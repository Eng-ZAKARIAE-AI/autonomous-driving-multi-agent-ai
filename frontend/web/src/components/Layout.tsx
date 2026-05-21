import React from 'react';
import { LayoutDashboard, Activity, Cpu, Map, FileText, Settings, Radio } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface SidebarItemProps {
  icon: React.ElementType;
  label: string;
  active?: boolean;
}

const SidebarItem = ({ icon: Icon, label, active }: SidebarItemProps) => (
  <div className={cn(
    "flex items-center gap-3 px-4 py-3 cursor-pointer transition-colors duration-200 rounded-lg",
    active ? "bg-primary text-white" : "text-slate-400 hover:bg-slate-800 hover:text-white"
  )}>
    <Icon size={20} />
    <span className="font-medium">{label}</span>
  </div>
);

interface LayoutProps {
  children: React.ReactNode;
  connected: boolean;
}

export const Layout = ({ children, connected }: LayoutProps) => {
  return (
    <div className="flex h-screen bg-background text-slate-200 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col p-4 gap-6">
        <div className="flex items-center gap-2 px-2 py-4">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center font-bold text-white">AD</div>
          <span className="text-xl font-bold text-white tracking-tight">AD-MAI</span>
        </div>
        
        <nav className="flex flex-col gap-2">
          <SidebarItem icon={LayoutDashboard} label="Dashboard" active />
          <SidebarItem icon={Activity} label="Training" />
          <SidebarItem icon={Cpu} label="Agents" />
          <SidebarItem icon={Map} label="Simulation" />
          <SidebarItem icon={FileText} label="Telemetry Logs" />
          <SidebarItem icon={Settings} label="Settings" />
        </nav>

        <div className="mt-auto pt-4 border-t border-slate-800">
          <div className="flex items-center gap-3 px-4 py-3">
            <Radio size={16} className={cn(connected ? "text-safety-green" : "text-safety-red animate-pulse")} />
            <span className="text-sm font-medium">
              {connected ? "Backend: Online" : "Backend: Offline"}
            </span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-16 bg-slate-900/50 backdrop-blur-md border-b border-slate-800 flex items-center justify-between px-8">
          <h1 className="text-lg font-semibold text-white">Mission Control</h1>
          <div className="flex items-center gap-4">
            <div className="px-3 py-1 bg-slate-800 rounded-full text-xs font-semibold text-slate-300 border border-slate-700">
              CARLA: <span className="text-safety-green ml-1">OK</span>
            </div>
            <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-xs font-medium text-white">
              JD
            </div>
          </div>
        </header>

        {/* Viewport */}
        <div className="flex-1 overflow-y-auto p-8">
          {children}
        </div>
      </main>
    </div>
  );
};
