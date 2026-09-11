import React from 'react';

/**
 * Approved FaceSense Symbol
 * Faithfully matches the brand sheet:
 * - Sleek right-facing profile silhouette
 * - Concentric perceptual ear loop
 * - Focal iris center dot
 * - Cyan-to-Purple gradient (#22D3EE -> #6C63FF -> #8B5CF6)
 */
export default function LogoSymbol({ 
  className = "w-8 h-8", 
  monochrome = false,
  variant = "gradient" 
}) {
  const gradientId = React.useId();

  return (
    <svg 
      className={className} 
      viewBox="0 0 100 100" 
      fill="none" 
      xmlns="http://www.w3.org/2000/svg"
    >
      <defs>
        <linearGradient 
          id={`fs-grad-${gradientId}`} 
          x1="10" 
          y1="90" 
          x2="85" 
          y2="10" 
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0%" stopColor="#22D3EE" />
          <stop offset="45%" stopColor="#6C63FF" />
          <stop offset="100%" stopColor="#8B5CF6" />
        </linearGradient>
      </defs>

      {/* Profile Silhouette */}
      <path 
        d="M58 12C63 17 68 24 67 31C66 38 69 41 71 44C73 47 70 51 68 53C67 54 67 56 68 58C69 60 67 65 65 67C63 69 64 73 66 75C66.5 75.8 67 76.5 66 77.5C64 79 59 81 54 80C48 79 46 72 47 65C47.8 59 47 52 46 47C45 37 49 22 58 12Z" 
        fill={monochrome ? "currentColor" : `url(#fs-grad-${gradientId})`} 
      />

      {/* Perceptual Ear / Telemetry Loop Arc */}
      <path 
        d="M48 24C33 25 21 37 21 52C21 67 32 79 47 79C50 79 52 76.5 50 74C48 71.5 45 71 43 70C33 66 28 58 29 48C30 38 37 31 47 31C50 31 52 28.5 51 26C50 24.5 49 24 48 24Z" 
        fill={monochrome ? "currentColor" : `url(#fs-grad-${gradientId})`} 
      />

      {/* Focal Iris / Sensor Core Dot */}
      <circle 
        cx="40" 
        cy="52" 
        r="7.5" 
        fill={monochrome ? "currentColor" : `url(#fs-grad-${gradientId})`} 
      />
    </svg>
  );
}
