import React from 'react';

/**
 * Toast notification component
 */
export default function Toast({
  message,
  type = 'info',
  onClose,
}) {
  if (!message) return null;

  const typeConfig = {
    info: { icon: 'info', color: 'border-[#22D3EE] text-[#0891B2] dark:text-[#22D3EE]' },
    success: { icon: 'check_circle', color: 'border-emerald-500 text-emerald-600 dark:text-emerald-400' },
    error: { icon: 'error', color: 'border-rose-500 text-rose-600 dark:text-rose-400' },
  };

  const config = typeConfig[type] || typeConfig.info;

  return (
    <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2.5 px-4 py-3 rounded-xl bg-white dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] shadow-xl text-xs text-slate-800 dark:text-[#F8FAFC]">
      <span className={`material-symbols-outlined text-[18px] ${config.color}`}>
        {config.icon}
      </span>
      <span>{message}</span>
      {onClose && (
        <button 
          onClick={onClose} 
          className="ml-2 text-slate-400 hover:text-slate-600 dark:hover:text-white"
        >
          <span className="material-symbols-outlined text-[14px]">close</span>
        </button>
      )}
    </div>
  );
}
