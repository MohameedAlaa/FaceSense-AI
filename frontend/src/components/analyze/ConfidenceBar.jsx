import React from 'react';

/**
 * Animated Confidence Bar for Emotion Probabilities
 */
export default function ConfidenceBar({
  emotion,
  probability,
  isPrimary = false,
}) {
  const percentage = (probability * 100).toFixed(1);

  const emotionIcons = {
    happy: "sentiment_very_satisfied",
    neutral: "sentiment_neutral",
    surprise: "sentiment_satisfied",
    sad: "sentiment_dissatisfied",
    fear: "sentiment_stressed",
    angry: "sentiment_very_dissatisfied",
    disgust: "mood_bad",
  };

  return (
    <div className="flex flex-col gap-1 w-full text-xs">
      <div className="flex items-center justify-between">
        <span className={`flex items-center gap-1.5 capitalize ${
          isPrimary 
            ? "text-slate-900 dark:text-[#F8FAFC] font-semibold" 
            : "text-slate-600 dark:text-[#94A3B8]"
        }`}>
          <span className={`material-symbols-outlined text-[14px] ${
            isPrimary ? "text-[#22D3EE]" : "text-slate-400 dark:text-[#64748B]"
          }`}>
            {emotionIcons[emotion] || "sentiment_neutral"}
          </span>
          <span>{emotion}</span>
        </span>
        <span className={`font-mono text-[11px] ${
          isPrimary 
            ? "text-[#0891B2] dark:text-[#22D3EE] font-bold" 
            : "text-slate-500 dark:text-[#94A3B8]"
        }`}>
          {percentage}%
        </span>
      </div>

      <div className="w-full h-1.5 bg-slate-200 dark:bg-[#0B1020] rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            isPrimary
              ? "bg-[#22D3EE] shadow-[0_0_8px_#22D3EE]"
              : "bg-slate-400 dark:bg-[#64748B]/50"
          }`}
          style={{ width: `${Math.max(1, percentage)}%` }}
        />
      </div>
    </div>
  );
}
