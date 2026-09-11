import React from 'react';

/**
 * Reusable Card adhering to DESIGN.md
 * Elevations: base (L0), container (L1), elevated (L2)
 */
export default function Card({
  children,
  elevation = 'container',
  className = '',
  onClick,
  ...props
}) {
  const elevationMap = {
    base: "bg-[#F7F7FC] dark:bg-[#0B1020] border border-[#E8E8F3] dark:border-[#1E294B]",
    container: "bg-white dark:bg-[#12182B] border border-[#E8E8F3] dark:border-[#1E294B] shadow-sm",
    elevated: "bg-white dark:bg-[#171F36] border border-[#E8E8F3] dark:border-[#1E294B] shadow-md",
  };

  return (
    <div
      onClick={onClick}
      className={`rounded-xl p-4 transition-all duration-150 ${elevationMap[elevation] || elevationMap.container} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}
