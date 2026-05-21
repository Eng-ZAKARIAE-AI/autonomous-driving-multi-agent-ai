import React from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  trend?: number;
  icon?: React.ElementType;
  status?: 'normal' | 'success' | 'warning' | 'error';
}

export const MetricCard = ({ label, value, unit, trend, icon: Icon, status = 'normal' }: MetricCardProps) => {
  return (
    <div className="bg-surface border border-slate-800 rounded-xl p-5 flex flex-col gap-1 shadow-sm">
      <div className="flex items-center justify-between text-slate-400">
        <span className="text-xs font-semibold uppercase tracking-wider">{label}</span>
        {Icon && <Icon size={16} />}
      </div>
      <div className="flex items-baseline gap-1 mt-1">
        <span className={cn(
          "text-2xl font-bold text-white",
          status === 'success' && "text-safety-green",
          status === 'error' && "text-safety-red",
          status === 'warning' && "text-telemetry"
        )}>
          {value}
        </span>
        {unit && <span className="text-sm text-slate-500">{unit}</span>}
      </div>
      {trend !== undefined && (
        <div className={cn(
          "text-xs mt-2 flex items-center gap-1",
          trend >= 0 ? "text-safety-green" : "text-safety-red"
        )}>
          <span>{trend >= 0 ? '↑' : '↓'}</span>
          <span>{Math.abs(trend)}% vs last ep</span>
        </div>
      )}
    </div>
  );
};
