import React from 'react';
import { ChevronDown, RefreshCcw } from 'lucide-react';

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: React.ReactNode;
}

export const PageHeader = ({ title, description, actions }: PageHeaderProps) => {
  return (
    <div className="flex items-start justify-between">
      <div>
        <div className="text-[20px] font-semibold text-text tracking-[-0.02em]">{title}</div>
        {description && <div className="text-xs-custom text-muted mt-[3px]">{description}</div>}
      </div>
      <div className="flex gap-2">
        {actions || (
          <>
            <button className="btn">
              <ChevronDown size={12} strokeWidth={2.5} />
              Export
            </button>
            <button className="btn">
              <RefreshCcw size={12} strokeWidth={2.5} />
              Refresh
            </button>
          </>
        )}
      </div>
    </div>
  );
};
