import React from 'react';
import { NavLink } from 'react-router-dom';

export default function MobileNav() {
  const tabs = [
    { to: '/dashboard', label: 'Home', icon: 'grid_view' },
    { to: '/analyze', label: 'Analyze', icon: 'analytics', highlight: true },
    { to: '/history', label: 'History', icon: 'history' },
    { to: '/insights', label: 'Insights', icon: 'query_stats' },
    { to: '/settings', label: 'Settings', icon: 'tune' },
  ];

  return (
    <nav className="lg:hidden fixed bottom-0 left-0 right-0 h-16 bg-white/95 dark:bg-[#12182B]/95 backdrop-blur-md border-t border-slate-200 dark:border-[#1E294B] flex items-center justify-around px-2 z-40 select-none shadow-lg">
      {tabs.map((tab) => (
        <NavLink
          key={tab.to}
          to={tab.to}
          className={({ isActive }) =>
            `flex flex-col items-center justify-center gap-1 w-14 py-1 rounded-xl transition-all ${
              isActive
                ? 'text-[#6C63FF] font-semibold'
                : 'text-slate-400 dark:text-[#64748B] hover:text-slate-600 dark:hover:text-[#94A3B8]'
            }`
          }
        >
          {({ isActive }) => (
            <>
              <div className="relative flex items-center justify-center">
                <span className={`material-symbols-outlined text-[20px] ${tab.highlight && isActive ? 'text-[#22D3EE]' : ''}`}>
                  {tab.icon}
                </span>
                {isActive && (
                  <span className="absolute -bottom-1 w-1 h-1 rounded-full bg-[#6C63FF]"></span>
                )}
              </div>
              <span className="text-[10px] tracking-tight">{tab.label}</span>
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}
