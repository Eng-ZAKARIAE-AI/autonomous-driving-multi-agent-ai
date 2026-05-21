import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface MetricCardProps {
  label: string;
  value: string | number;
  delta?: string;
  trend?: 'up' | 'dn';
  color?: 'blue' | 'green' | 'red' | 'purple' | 'default';
}

export const MetricCard = ({ label, value, delta, trend, color = 'default' }: MetricCardProps) => {
  return (
    <div className="bg-bg3 border border-border rounded-lg p-[14px_16px] relative overflow-hidden group">
      {/* Top accent line */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-accent2 opacity-0 transition-opacity duration-200 group-hover:opacity-100" />
      
      <div className="text-xs-custom text-muted font-medium tracking-[0.03em] uppercase mb-[5px]">{label}</div>
      
      <div className={cn(
        "text-[26px] font-semibold font-mono tracking-[-0.02em] leading-none mb-1",
        color === 'blue' && "text-accent",
        color === 'green' && "text-green",
        color === 'red' && "text-red",
        color === 'purple' && "text-purple",
        color === 'default' && "text-text"
      )}>
        {value}
      </div>
      
      {delta && (
        <div className="text-sm-custom text-muted mt-[5px] flex items-center gap-[4px]">
          {trend === 'up' && <span className="text-green">{delta}</span>}
          {trend === 'dn' && <span className="text-red">{delta}</span>}
          {!trend && delta}
          <span className="opacity-70">since last session</span>
        </div>
      )}
    </div>
  );
};
