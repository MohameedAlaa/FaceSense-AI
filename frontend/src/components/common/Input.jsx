import React from 'react';

/**
 * Analytical Input Field adhering to DESIGN.md
 * Features subtle hairline border and Electric Cyan focus ring
 */
export default function Input({
  label,
  error,
  icon,
  className = '',
  id,
  type = 'text',
  ...props
}) {
  const inputId = id || React.useId();

  return (
    <div className="flex flex-col gap-1 w-full text-left">
      {label && (
        <label 
          htmlFor={inputId} 
          className="text-xs font-medium text-slate-700 dark:text-[#94A3B8]"
        >
          {label}
        </label>
      )}
      <div className="relative flex items-center">
        {icon && (
          <span className="material-symbols-outlined text-[16px] text-slate-400 dark:text-[#64748B] absolute left-3 pointer-events-none">
            {icon}
          </span>
        )}
        <input
          id={inputId}
          type={type}
          className={`w-full text-xs rounded-lg bg-white dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] text-slate-900 dark:text-[#F8FAFC] placeholder-slate-400 dark:placeholder-[#64748B] py-2 transition-all outline-none focus:border-[#22D3EE] dark:focus:border-[#22D3EE] focus:ring-1 focus:ring-[#22D3EE] ${
            icon ? 'pl-9 pr-3' : 'px-3'
          } ${error ? 'border-rose-500 focus:border-rose-500 focus:ring-rose-500' : ''} ${className}`}
          {...props}
        />
      </div>
      {error && <span className="text-[11px] text-rose-500">{error}</span>}
    </div>
  );
}
