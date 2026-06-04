import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  Activity, 
  Bot, 
  Cpu, 
  FileText, 
  Settings,
  X
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface SidebarProps {
  connected: boolean;
  onClose?: () => void;
}

const navGroups = [
  {
    label: 'Navigation',
    items: [
      { label: 'Dashboard', icon: LayoutDashboard, path: '/' },
      { label: 'Training', icon: Activity, path: '/training', badge: 'Active' },
      { label: 'Agents', icon: Bot, path: '/agents' },
      { label: 'Simulation', icon: Cpu, path: '/simulation' },
      { label: 'Telemetry logs', icon: FileText, path: '/telemetry' },
    ]
  },
  {
    label: 'System',
    items: [
      { label: 'Settings', icon: Settings, path: '/settings' },
    ]
  }
];

export const Sidebar = ({ connected, onClose }: SidebarProps) => {
  return (
    <div className="h-full flex flex-col bg-bg2 border-r border-border w-[210px]">
      {/* Header */}
      <div className="p-[20px_16px_16px] border-b border-border">
        <div className="flex items-center gap-2 mb-[4px]">
          <div className="w-[26px] h-[26px] bg-accent2 rounded-[6px] flex items-center justify-center font-mono text-[11px] font-medium text-white tracking-[-0.02em] shrink-0">
            AD
          </div>
          <span className="text-[13px] font-semibold text-text tracking-[0.06em]">AD-MAI</span>
          {onClose && (
            <button onClick={onClose} className="ml-auto lg:hidden text-muted hover:text-text">
              <X size={20} />
            </button>
          )}
        </div>
        <p className="text-sm-custom text-muted pl-[34px] -mt-[2px]">Autonomous Driving</p>
      </div>

      {/* Nav Groups */}
      <nav className="flex-1 p-[8px_8px] overflow-y-auto">
        {navGroups.map((group, gIdx) => (
          <div key={gIdx} className="mb-4">
            <div className="text-xs-custom font-medium tracking-[0.1em] uppercase color-muted2 p-[14px_16px_6px]">
              {group.label}
            </div>
            <div className="space-y-[1px]">
              {group.items.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  onClick={onClose}
                  className={({ isActive }) => cn(
                    "flex items-center gap-[10px] p-[7px_10px] rounded-[6px] text-base-custom font-normal transition-all duration-120 group select-none",
                    isActive 
                      ? "bg-[#1c2d40] text-text" 
                      : "text-muted hover:bg-white/5 hover:text-text"
                  )}
                >
                  {({ isActive }) => (
                    <>
                      <item.icon 
                        size={15} 
                        className={cn(
                          "shrink-0",
                          isActive ? "text-accent" : "text-muted group-hover:text-text"
                        )} 
                      />
                      <span>{item.label}</span>
                      {item.badge && (
                        <span className="ml-auto text-xs-custom font-medium bg-accent2 text-white p-[1px_6px] rounded-[10px] font-mono">
                          {item.badge}
                        </span>
                      )}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-[12px_14px] border-t border-border flex items-center gap-2">
        <div className="relative flex items-center justify-center">
          <div className={cn(
            "w-[7px] h-[7px] rounded-full shrink-0 relative z-10",
            connected ? "bg-green" : "bg-red"
          )} />
          <div className={cn(
            "absolute inset-[-3px] rounded-full opacity-25 animate-[pulse-custom_2s_ease-in-out_infinite]",
            connected ? "bg-green" : "bg-red"
          )} />
        </div>
        <span className="text-sm-custom text-muted font-normal">
          Backend: {connected ? 'Online' : 'Offline'}
        </span>
      </div>
    </div>
  );
};
