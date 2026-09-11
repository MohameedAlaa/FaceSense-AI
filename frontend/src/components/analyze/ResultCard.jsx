import React from 'react';
import ConfidenceBar from './ConfidenceBar';

export default function ResultCard({
  face,
  isSelected,
  onSelect,
}) {
  if (!face) return null;

  const { id, label, sublabel, predicted_emotion, confidence, probabilities, box } = face;

  // 7 supported emotions in FaceSense
  const emotionsList = ['happy', 'neutral', 'surprise', 'sad', 'fear', 'disgust', 'angry'];
  
  // Sort with highest probability first
  const sortedEmotions = [...emotionsList].sort((a, b) => {
    return (probabilities[b] || 0) - (probabilities[a] || 0);
  });

  return (
    <div
      onClick={() => onSelect(id)}
      className={`bg-slate-50 dark:bg-[#171F36] border rounded-xl p-4 flex flex-col gap-3 shadow-sm transition-all duration-150 cursor-pointer ${
        isSelected
          ? "border-[#6C63FF] ring-1 ring-[#6C63FF]/30"
          : "border-slate-200 dark:border-[#1E294B] hover:border-slate-300 dark:hover:border-[#2E3B66]"
      }`}
    >
      {/* Card Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-white dark:bg-[#0B1020] border border-[#6C63FF] flex items-center justify-center text-[#6C63FF] font-mono text-xs font-bold">
            {id}
          </div>
          <div className="flex flex-col text-left">
            <span className="text-xs font-semibold text-slate-800 dark:text-[#F8FAFC]">
              {label} <span className="text-[10px] text-[#6C63FF] dark:text-[#8B5CF6] font-normal">({sublabel})</span>
            </span>
            <span className="text-[11px] text-slate-400 dark:text-[#64748B] font-mono">
              Box: [{box.left}%, {box.top}%, {box.width}%, {box.height}%]
            </span>
          </div>
        </div>

        <span className="px-2 py-0.5 rounded bg-white dark:bg-[#0B1020] border border-[#22D3EE]/40 text-[#0891B2] dark:text-[#22D3EE] text-[11px] font-mono font-semibold">
          {(confidence * 100).toFixed(1)}%
        </span>
      </div>

      {/* Expression Class Distribution (7 Supported Classes) */}
      <div className="flex flex-col gap-2 pt-2 border-t border-slate-200 dark:border-[#1E294B]">
        <div className="flex items-center justify-between text-[11px] font-semibold text-slate-500 dark:text-[#94A3B8] uppercase tracking-wider">
          <span>Expression Classes</span>
          <span className="text-[10px] font-mono lowercase text-slate-400">7 classes</span>
        </div>

        <div className="flex flex-col gap-2">
          {sortedEmotions.map((emotion) => (
            <ConfidenceBar
              key={emotion}
              emotion={emotion}
              probability={probabilities[emotion] || 0}
              isPrimary={emotion === predicted_emotion}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
