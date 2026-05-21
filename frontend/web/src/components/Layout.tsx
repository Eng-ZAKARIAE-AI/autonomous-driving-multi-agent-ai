import { useState, type ReactNode } from 'react';
import { Menu } from 'lucide-react';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';

interface LayoutProps {
  children: ReactNode;
  connected: boolean;
}

export const Layout = ({ children, connected }: LayoutProps) => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen bg-bg text-text overflow-hidden font-sans">
      {/* Mobile Sidebar Overlay */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* Sidebar - Desktop */}
      <aside className="hidden lg:block shrink-0">
        <Sidebar connected={connected} />
      </aside>

      {/* Sidebar - Mobile */}
      <aside className={`fixed inset-y-0 left-0 z-50 transform ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-300 ease-in-out lg:hidden`}>
        <Sidebar connected={connected} onClose={() => setIsSidebarOpen(false)} />
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Topbar - Mobile Toggle + Desktop Topbar */}
        <header className="lg:hidden h-14 flex items-center justify-between px-4 border-b border-border bg-bg2">
          <button 
            onClick={() => setIsSidebarOpen(true)}
            className="p-2 text-muted hover:text-text"
          >
            <Menu size={20} />
          </button>
          <span className="text-[13px] font-semibold tracking-wider text-text uppercase">AD-MAI</span>
          <div className="w-8" />
        </header>

        <div className="hidden lg:block">
          <Topbar connected={connected} />
        </div>

        {/* Viewport */}
        <main className="flex-1 overflow-y-auto bg-bg">
          <div className="p-[20px_22px] flex flex-col gap-4">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
};
