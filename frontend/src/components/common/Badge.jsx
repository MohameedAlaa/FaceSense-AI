import React from 'react';

/**
 * Analytical Telemetry Badge adhering to DESIGN.md
 * Variants: default | primary | cyan | success | warning | danger
 */
export default function Badge({
  children,
  variant = 'default',
  icon = null,
  className = '',
}) {
  const variantClasses = {
    default: "bg-slate-100 dark:bg-[#171F36] text-slate-600 dark:text-[#94A3B8] border border-slate-200 dark:border-[#1E294B]",
    primary: "bg-[#6C63FF]/10 text-[#6C63FF] border border-[#6C63FF]/30",
    cyan: "bg-[#22D3EE]/10 text-[#0891B2] dark:text-[#22D3EE] border border-[#22D3EE]/30",
    success: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30",
    warning: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/30",
    danger: "bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/30",
  };

  return (
    <span
      className={`inline-flex items-center gap-1 font-mono text-[10px] font-medium px-2 py-0.5 rounded ${variantClasses[variant] || variantClasses.default} ${className}`}
    >
      {icon && <span className="material-symbols-outlined text-[12px]">{icon}</span>}
      {children}
    </span>
  );
}
