import React from 'react';
import useTheme from '../../hooks/useTheme';

export default function ThemeSwitcher({ className = '' }) {
  const { themeMode, setTheme } = useTheme();

  return (
    <div className={`flex items-center bg-slate-100 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1E294B] p-0.5 rounded-lg text-xs ${className}`}>
      <button
        type="button"
        onClick={() => setTheme('light')}
        className={`flex-1 py-1 text-[11px] rounded flex items-center justify-center gap-1 transition-all ${
          themeMode === 'light'
            ? 'bg-white text-slate-900 font-medium shadow-sm border border-slate-200'
            : 'text-slate-500 hover:text-slate-900 dark:text-[#64748B] dark:hover:text-[#94A3B8]'
        }`}
        title="Light theme"
      >
        <span className="material-symbols-outlined text-[13px]">light_mode</span>
        <span>Light</span>
      </button>

      <button
        type="button"
        onClick={() => setTheme('dark')}
        className={`flex-1 py-1 text-[11px] rounded flex items-center justify-center gap-1 transition-all ${
          themeMode === 'dark'
            ? 'bg-[#171F36] text-[#F8FAFC] font-medium shadow-sm border border-[#1E294B]'
            : 'text-slate-500 hover:text-slate-900 dark:text-[#64748B] dark:hover:text-[#94A3B8]'
        }`}
        title="Dark theme"
      >
        <span className="material-symbols-outlined text-[13px] text-[#6C63FF]">dark_mode</span>
        <span>Dark</span>
      </button>

      <button
        type="button"
        onClick={() => setTheme('system')}
        className={`flex-1 py-1 text-[11px] rounded flex items-center justify-center gap-1 transition-all ${
          themeMode === 'system'
            ? 'bg-white dark:bg-[#171F36] text-slate-900 dark:text-[#F8FAFC] font-medium shadow-sm border border-slate-200 dark:border-[#1E294B]'
            : 'text-slate-500 hover:text-slate-900 dark:text-[#64748B] dark:hover:text-[#94A3B8]'
        }`}
        title="System preference"
      >
        <span className="material-symbols-outlined text-[13px]">desktop_windows</span>
        <span>Auto</span>
      </button>
    </div>
  );
}
