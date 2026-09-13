import React, { useState, useEffect } from 'react';
import { NavLink, Link, useNavigate } from 'react-router-dom';
import Logo from '../brand/Logo';
import ThemeSwitcher from './ThemeSwitcher';
import { useAuth } from '../../hooks/useAuth';
import { predictApi } from '../../lib/api/predict';

export default function Sidebar({ className = '', onCloseMobile }) {
  const { user, isAdmin, logout } = useAuth();
  const [modelName, setModelName] = useState('ResidualEmotionCNN');
  const navigate = useNavigate();

  useEffect(() => {
    let mounted = true;
    predictApi.getModelInfo()
      .then((info) => {
        if (mounted && info?.model_name) {
          setModelName(info.model_name);
        }
      })
      .catch(() => {
        // Keep fallback
      });
    return () => {
      mounted = false;
    };
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const navItems = [
    { to: '/dashboard', label: 'Home / Overview', icon: 'grid_view' },
    { to: '/analyze', label: 'Analyze', icon: 'analytics', highlight: true },
    { to: '/history', label: 'History', icon: 'history' },
    { to: '/insights', label: 'Insights', icon: 'query_stats' },
  ];

  // Admin link strictly filtered based on actual authenticated role
  const systemItems = [
    ...(isAdmin ? [{ to: '/admin', label: 'Admin', icon: 'admin_panel_settings' }] : []),
    { to: '/settings', label: 'Settings', icon: 'tune' },
  ];

  // User initials
  const userInitials = user?.email
    ? user.email.substring(0, 2).toUpperCase()
    : 'OP';

  return (
    <aside className={`w-64 bg-white dark:bg-[#12182B] border-r border-slate-200 dark:border-[#1E294B] flex flex-col justify-between shrink-0 p-4 z-20 h-full select-none ${className}`}>
      <div className="flex flex-col gap-4">
        {/* Brand header */}
        <div className="flex items-center justify-between px-1 pt-1 pb-2">
          <Link to="/" className="flex items-center gap-2.5" onClick={onCloseMobile}>
            <Logo variant="compact" size="sm" />
            <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] text-slate-500 dark:text-[#94A3B8] truncate max-w-[90px]" title={modelName}>
              Active
            </span>
          </Link>
          {onCloseMobile && (
            <button
              onClick={onCloseMobile}
              className="lg:hidden p-1 text-slate-400 hover:text-slate-600 dark:hover:text-white"
            >
              <span className="material-symbols-outlined text-[20px]">close</span>
            </button>
          )}
        </div>

        {/* Quick Search / Command Pill */}
        <div className="relative">
          <button
            type="button"
            className="w-full flex items-center justify-between px-3 py-1.5 rounded-lg bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] text-slate-500 dark:text-[#94A3B8] hover:border-slate-300 dark:hover:border-[#2E3B66] hover:text-slate-900 dark:hover:text-[#F8FAFC] transition-colors text-xs"
          >
            <span className="flex items-center gap-2">
              <span className="material-symbols-outlined text-[15px] text-slate-400 dark:text-[#64748B]">search</span>
              <span>Search & Jump...</span>
            </span>
            <kbd className="font-mono text-[10px] bg-white dark:bg-[#12182B] px-1.5 py-0.5 rounded text-slate-400 dark:text-[#94A3B8] border border-slate-200 dark:border-[#1E294B]">
              ⌘K
            </kbd>
          </button>
        </div>

        {/* Main Navigation */}
        <div className="flex flex-col gap-1">
          <div className="text-[10px] font-semibold text-slate-400 dark:text-[#64748B] uppercase tracking-wider px-2 mb-1">
            Navigation
          </div>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={onCloseMobile}
              className={({ isActive }) =>
                `flex items-center justify-between px-2.5 py-2 rounded-lg text-xs transition-colors ${
                  isActive
                    ? 'bg-[#6C63FF]/10 text-[#6C63FF] dark:bg-[#171F36] dark:text-[#F8FAFC] font-medium border border-[#6C63FF]/30 shadow-sm'
                    : 'text-slate-600 dark:text-[#94A3B8] hover:bg-slate-100 dark:hover:bg-[#171F36] hover:text-slate-900 dark:hover:text-[#F8FAFC]'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <span className="flex items-center gap-2.5">
                    <span className={`material-symbols-outlined text-[18px] ${isActive ? 'text-[#6C63FF]' : ''}`}>
                      {item.icon}
                    </span>
                    <span>{item.label}</span>
                  </span>
                  {isActive && (
                    <span className="w-1.5 h-1.5 rounded-full bg-[#6C63FF] shadow-[0_0_8px_#6C63FF]"></span>
                  )}
                </>
              )}
            </NavLink>
          ))}
        </div>

        {/* System Section */}
        <div className="flex flex-col gap-1 pt-2 border-t border-slate-200 dark:border-[#1E294B]">
          <div className="flex items-center justify-between px-2 mb-1">
            <span className="text-[10px] font-semibold text-slate-400 dark:text-[#64748B] uppercase tracking-wider">
              System
            </span>
            {isAdmin && (
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-[#6C63FF]/15 border border-[#6C63FF]/30 text-[#6C63FF] dark:text-[#8B5CF6] font-medium">
                Admin
              </span>
            )}
          </div>
          {systemItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={onCloseMobile}
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-xs transition-colors ${
                  isActive
                    ? 'bg-[#6C63FF]/10 text-[#6C63FF] dark:bg-[#171F36] dark:text-[#F8FAFC] font-medium border border-[#6C63FF]/30 shadow-sm'
                    : 'text-slate-600 dark:text-[#94A3B8] hover:bg-slate-100 dark:hover:bg-[#171F36] hover:text-slate-900 dark:hover:text-[#F8FAFC]'
                }`
              }
            >
              <span className="material-symbols-outlined text-[18px]">{item.icon}</span>
              <span>{item.label}</span>
            </NavLink>
          ))}
        </div>
      </div>

      {/* Sidebar Footer */}
      <div className="flex flex-col gap-3 pt-3 border-t border-slate-200 dark:border-[#1E294B]">
        {/* Theme Switcher */}
        <ThemeSwitcher />

        {/* Engine Operational Status Badge */}
        <div className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-slate-50 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1E294B] text-[11px]">
          <div className="flex items-center gap-2 min-w-0">
            <span className="relative flex h-2 w-2 shrink-0">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#22D3EE] opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-[#22D3EE]"></span>
            </span>
            <span className="text-slate-600 dark:text-[#94A3B8] truncate" title={modelName}>
              {modelName}
            </span>
          </div>
          <span className="font-mono text-[10px] text-[#0891B2] dark:text-[#22D3EE] font-medium shrink-0">ONLINE</span>
        </div>

        {/* User Profile Card */}
        <div className="flex items-center justify-between p-2 rounded-xl bg-slate-50 dark:bg-[#0B1020] border border-slate-200 dark:border-[#1E294B]">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-[#22D3EE] via-[#6C63FF] to-[#8B5CF6] flex items-center justify-center text-white font-bold text-xs ring-1 ring-slate-200 dark:ring-[#1E294B] shrink-0">
              {userInitials}
            </div>
            <div className="flex flex-col min-w-0 text-left">
              <span className="text-xs font-semibold text-slate-800 dark:text-[#F8FAFC] truncate" title={user?.email || 'Operator'}>
                {user?.email || 'Operator'}
              </span>
              <span className="text-[11px] text-slate-400 dark:text-[#64748B] capitalize truncate">
                {user?.role || 'Active Session'}
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={handleLogout}
            className="p-1 rounded text-slate-400 hover:text-rose-500 transition-colors"
            title="Sign out"
          >
            <span className="material-symbols-outlined text-[18px]">logout</span>
          </button>
        </div>
      </div>
    </aside>
  );
}
