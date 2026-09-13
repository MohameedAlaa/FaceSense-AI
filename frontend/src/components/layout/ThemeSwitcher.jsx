import React from 'react';
import useTheme from '../../hooks/useTheme';

export default function ThemeSwitcher({ className = '' }) {
  const { preference, themeMode, setTheme } = useTheme();
  const activePref = preference || themeMode || 'system';

  return (
    <div
      role="group"
      aria-label="Color theme switcher"
      className={`flex items-center bg-slate-100 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1E294B] p-1 rounded-xl text-xs gap-1 select-none w-full max-w-full ${className}`}
    >
      <button
        type="button"
        id="theme-btn-light"
        onClick={() => setTheme('light')}
        className={`flex-1 py-1.5 px-2 text-[11px] min-h-[40px] sm:min-h-[36px] rounded-lg flex items-center justify-center gap-1.5 font-medium transition-all ${
          activePref === 'light'
            ? 'bg-white text-[#6C63FF] shadow-sm border border-slate-200 font-semibold ring-1 ring-[#6C63FF]/30'
            : 'text-slate-500 hover:text-slate-900 dark:text-[#64748B] dark:hover:text-[#94A3B8] border border-transparent'
        }`}
        title="Force Light theme"
        aria-pressed={activePref === 'light'}
      >
        <span className={`material-symbols-outlined text-[15px] ${activePref === 'light' ? 'text-[#6C63FF]' : ''}`}>
          light_mode
        </span>
        <span>Light</span>
      </button>

      <button
        type="button"
        id="theme-btn-dark"
        onClick={() => setTheme('dark')}
        className={`flex-1 py-1.5 px-2 text-[11px] min-h-[40px] sm:min-h-[36px] rounded-lg flex items-center justify-center gap-1.5 font-medium transition-all ${
          activePref === 'dark'
            ? 'bg-[#171F36] text-[#22D3EE] shadow-sm border border-[#1E294B] font-semibold ring-1 ring-[#22D3EE]/30'
            : 'text-slate-500 hover:text-slate-900 dark:text-[#64748B] dark:hover:text-[#94A3B8] border border-transparent'
        }`}
        title="Force Dark theme"
        aria-pressed={activePref === 'dark'}
      >
        <span className={`material-symbols-outlined text-[15px] ${activePref === 'dark' ? 'text-[#22D3EE]' : ''}`}>
          dark_mode
        </span>
        <span>Dark</span>
      </button>

      <button
        type="button"
        id="theme-btn-auto"
        onClick={() => setTheme('system')}
        className={`flex-1 py-1.5 px-2 text-[11px] min-h-[40px] sm:min-h-[36px] rounded-lg flex items-center justify-center gap-1.5 font-medium transition-all ${
          activePref === 'system'
            ? 'bg-white dark:bg-[#171F36] text-[#6C63FF] dark:text-[#22D3EE] shadow-sm border border-slate-200 dark:border-[#1E294B] font-semibold ring-1 ring-[#6C63FF]/30 dark:ring-[#22D3EE]/30'
            : 'text-slate-500 hover:text-slate-900 dark:text-[#64748B] dark:hover:text-[#94A3B8] border border-transparent'
        }`}
        title="Follow System theme"
        aria-pressed={activePref === 'system'}
      >
        <span className={`material-symbols-outlined text-[15px] ${activePref === 'system' ? 'text-[#6C63FF] dark:text-[#22D3EE]' : ''}`}>
          desktop_windows
        </span>
        <span>Auto</span>
      </button>
    </div>
  );
}
