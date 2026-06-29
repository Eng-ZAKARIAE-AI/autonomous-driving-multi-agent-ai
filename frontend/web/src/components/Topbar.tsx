import { Plus } from 'lucide-react';
import { useLocation } from 'react-router-dom';

interface TopbarProps {
  connected: boolean;
}

export const Topbar = ({ connected }: TopbarProps) => {
  const location = useLocation();
  const path = location.pathname.split('/').filter(Boolean);
  const currentPage = path.length > 0 ? path[0].charAt(0).toUpperCase() + path[0].slice(1) : 'Dashboard';

  return (
    <div className="h-[48px] border-b border-border flex items-center justify-between px-5 bg-bg2 shrink-0">
      <div className="flex items-center gap-2">
        <span className="text-xs-custom text-muted">
          AD-MAI / <span className="text-text font-medium">{currentPage}</span>
        </span>
      </div>
      
      <div className="flex items-center gap-[10px]">
        <div className="text-sm-custom text-muted flex items-center gap-[5px]">
          <svg width="8" height="8" viewBox="0 0 8 8">
            <circle cx="4" cy="4" r="4" fill={connected ? "#3fb950" : "#f85149"} />
          </svg>
          ws://localhost:8000
        </div>
        
        <div className="carla-badge">CARLA: OK</div>
        
        <button className="btn btn-primary">
          <Plus size={12} strokeWidth={2.5} />
          New run
        </button>
      </div>
    </div>
  );
};
