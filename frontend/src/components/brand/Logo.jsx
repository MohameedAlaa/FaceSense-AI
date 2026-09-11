import React from 'react';
import LogoSymbol from './LogoSymbol';

/**
 * Approved FaceSense Brand Logo
 * Variants:
 * - 'full': Symbol + 'FaceSense' wordmark + 'PEOPLE IN FOCUS' tagline
 * - 'compact': Symbol + 'FaceSense' wordmark
 * - 'icon': Symbol only
 * - 'monochrome': Black or White depending on container text color
 */
export default function Logo({
  variant = 'compact',
  size = 'md',
  monochrome = false,
  className = '',
  symbolClassName = '',
}) {
  const sizeMap = {
    sm: { symbol: 'w-6 h-6', title: 'text-base', sub: 'text-[8px]' },
    md: { symbol: 'w-8 h-8', title: 'text-lg', sub: 'text-[9px]' },
    lg: { symbol: 'w-10 h-10', title: 'text-2xl', sub: 'text-[11px]' },
    xl: { symbol: 'w-14 h-14', title: 'text-3xl', sub: 'text-[13px]' },
  };

  const selectedSize = sizeMap[size] || sizeMap.md;

  if (variant === 'icon') {
    return (
      <div className={`inline-flex items-center justify-center ${className}`}>
        <LogoSymbol 
          className={symbolClassName || selectedSize.symbol} 
          monochrome={monochrome} 
        />
      </div>
    );
  }

  return (
    <div className={`inline-flex items-center gap-2.5 select-none ${className}`}>
      <LogoSymbol 
        className={symbolClassName || selectedSize.symbol} 
        monochrome={monochrome} 
      />
      <div className="flex flex-col leading-none">
        <div className={`tracking-tight ${selectedSize.title} ${monochrome ? 'text-current' : 'text-[#0F172A] dark:text-[#F8FAFC]'}`}>
          <span className="font-bold">Face</span>
          <span className="font-medium">Sense</span>
        </div>
        {variant === 'full' && (
          <span 
            className={`font-semibold tracking-[0.24em] mt-1 uppercase ${selectedSize.sub} ${
              monochrome ? 'text-current opacity-70' : 'text-[#64748B] dark:text-[#94A3B8]'
            }`}
          >
            People in Focus
          </span>
        )}
      </div>
    </div>
  );
}
