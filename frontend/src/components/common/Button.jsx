import React from 'react';

/**
 * Reusable Button component adhering to DESIGN.md
 * Variants: primary | secondary | ghost | destructive | outline
 * Sizes: sm | md | lg
 */
export default function Button({
  children,
  variant = 'primary',
  size = 'md',
  className = '',
  icon = null,
  disabled = false,
  type = 'button',
  onClick,
  ...props
}) {
  const baseClasses = "inline-flex items-center justify-center gap-1.5 font-medium transition-all duration-150 select-none disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.98]";

  const sizeClasses = {
    sm: "px-2.5 py-1 text-xs rounded-md",
    md: "px-3.5 py-1.5 text-xs rounded-lg",
    lg: "px-5 py-2.5 text-sm rounded-lg",
  };

  const variantClasses = {
    primary: "bg-[#6C63FF] hover:bg-[#5B52EE] text-white shadow-sm shadow-[#6C63FF]/25 border border-[#6C63FF]/50",
    secondary: "bg-[#171F36] dark:bg-[#171F36] bg-slate-100 hover:bg-[#1E294B] dark:hover:bg-[#1E294B] hover:bg-slate-200 text-[#0F172A] dark:text-[#F8FAFC] border border-slate-200 dark:border-[#1E294B]",
    ghost: "bg-transparent hover:bg-[#171F36] dark:hover:bg-[#171F36] hover:bg-slate-100 text-[#64748B] dark:text-[#94A3B8] hover:text-[#0F172A] dark:hover:text-[#F8FAFC]",
    outline: "bg-transparent hover:bg-[#6C63FF]/10 text-[#6C63FF] border border-[#6C63FF]/40",
    destructive: "bg-rose-500/10 hover:bg-rose-500/20 text-rose-500 border border-rose-500/30",
  };

  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      className={`${baseClasses} ${sizeClasses[size] || sizeClasses.md} ${variantClasses[variant] || variantClasses.primary} ${className}`}
      {...props}
    >
      {icon && <span className="material-symbols-outlined text-[16px]">{icon}</span>}
      {children}
    </button>
  );
}
