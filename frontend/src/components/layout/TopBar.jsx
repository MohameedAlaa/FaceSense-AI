import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import Button from '../common/Button';

export default function TopBar({ onToggleMobileNav }) {
  const location = useLocation();

  const getPageTitle = (path) => {
    switch (path) {
      case '/': return 'Overview';
      case '/dashboard': return 'Telemetry Dashboard';
      case '/analyze': return 'Vision Telemetry & Workspace';
      case '/history': return 'Analysis History';
      case '/insights': return 'Insights & Performance';
      case '/settings': return 'System Settings';
      case '/admin': return 'Model Administration';
      default: return 'FaceSense';
    }
  };

  return (
    <header className="h-14 bg-white/95 dark:bg-[#12182B]/95 backdrop-blur-md border-b border-slate-200 dark:border-[#1E294B] flex items-center justify-between px-4 sm:px-6 shrink-0 z-10 select-none">
      {/* Left: Mobile hamburger + Breadcrumbs and Engine Status */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onToggleMobileNav}
          className="lg:hidden p-1.5 rounded-lg text-slate-500 hover:text-slate-900 dark:text-[#94A3B8] dark:hover:text-[#F8FAFC] hover:bg-slate-100 dark:hover:bg-[#171F36]"
          aria-label="Toggle navigation"
        >
          <span className="material-symbols-outlined text-[20px]">menu</span>
        </button>

        <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-[#94A3B8]">
          <Link to="/dashboard" className="hover:text-slate-900 dark:hover:text-[#F8FAFC]">
            Workspace
          </Link>
          <span className="text-slate-400 dark:text-[#64748B]">/</span>
          <span className="text-slate-900 dark:text-[#F8FAFC] font-medium truncate max-w-[150px] sm:max-w-none">
            {getPageTitle(location.pathname)}
          </span>
        </div>

        <div className="hidden sm:flex h-3.5 w-px bg-slate-200 dark:bg-[#1E294B]"></div>

        <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] text-[11px] text-slate-600 dark:text-[#94A3B8]">
          <span className="w-1.5 h-1.5 rounded-full bg-[#22D3EE]"></span>
          <span>Analysis engine idle — ready</span>
        </div>
      </div>

      {/* Right: Hotkeys and Primary Action */}
      <div className="flex items-center gap-2 sm:gap-3">
        <div className="hidden md:flex items-center gap-2 text-xs text-slate-400 dark:text-[#64748B]">
          <span className="flex items-center gap-1">
            <kbd className="font-mono text-[10px] bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] px-1.5 py-0.5 rounded text-slate-500 dark:text-[#94A3B8]">
              N
            </kbd>
            <span className="text-[11px]">New View</span>
          </span>
          <span className="flex items-center gap-1">
            <kbd className="font-mono text-[10px] bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] px-1.5 py-0.5 rounded text-slate-500 dark:text-[#94A3B8]">
              ?
            </kbd>
            <span className="text-[11px]">Help</span>
          </span>
        </div>

        <Link to="/analyze">
          <Button variant="primary" size="sm" icon="add">
            <span className="hidden xs:inline">New Analysis</span>
            <span className="xs:hidden">Analyze</span>
          </Button>
        </Link>

        <Link 
          to="/settings"
          className="w-8 h-8 rounded-lg overflow-hidden border border-slate-200 dark:border-[#1E294B] hover:border-[#6C63FF] transition-colors shrink-0 flex items-center justify-center bg-gradient-to-tr from-[#22D3EE] via-[#6C63FF] to-[#8B5CF6] text-white text-xs font-bold"
          title="User profile"
        >
          ER
        </Link>
      </div>
    </header>
  );
}
