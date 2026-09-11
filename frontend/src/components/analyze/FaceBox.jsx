import React from 'react';

/**
 * Interactive Face Bounding Box with precision reticle corner brackets.
 * Highlights in Electric Cyan (#22D3EE) on active/hover.
 * Strictly adheres to FaceSense CV capabilities (NO landmarks).
 */
export default function FaceBox({
  face,
  isSelected,
  onSelect,
  showBoxes = true,
  showConfidence = true,
}) {
  if (!showBoxes) return null;

  const { box, id, predicted_emotion, confidence } = face;
  const isCyan = isSelected || id === "01";

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
    <div
      onClick={(e) => {
        e.stopPropagation();
        onSelect(id);
      }}
      style={{
        top: `${box.top}%`,
        left: `${box.left}%`,
        width: `${box.width}%`,
        height: `${box.height}%`,
      }}
      className={`absolute cursor-pointer transition-all duration-200 z-10 ${
        isCyan
          ? "border-2 border-[#22D3EE] shadow-[0_0_15px_rgba(34,211,238,0.35)]"
          : "border-2 border-slate-400 dark:border-[#64748B] border-dashed shadow-[0_0_10px_rgba(100,116,139,0.2)]"
      }`}
      title={`${face.label}: ${predicted_emotion} (${(confidence * 100).toFixed(1)}%)`}
    >
      {/* Precision corner reticle brackets */}
      <span className="reticle-corner -top-1 -left-1 border-t-2 border-l-2 border-white"></span>
      <span className="reticle-corner -top-1 -right-1 border-t-2 border-r-2 border-white"></span>
      <span className="reticle-corner -bottom-1 -left-1 border-b-2 border-l-2 border-white"></span>
      <span className="reticle-corner -bottom-1 -right-1 border-b-2 border-r-2 border-white"></span>

      {/* Floating Detection Label */}
      {showConfidence && (
        <div
          className={`detection-label absolute -top-7 left-0 flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-mono font-medium shadow-md whitespace-nowrap ${
            isCyan
              ? "bg-[#0B1020] text-[#22D3EE] border border-[#22D3EE]/50"
              : "bg-[#171F36] text-white border border-slate-600"
          }`}
        >
          <span className="material-symbols-outlined text-[13px]">
            {emotionIcons[predicted_emotion] || "face"}
          </span>
          <span className="capitalize">
            [{id}] {predicted_emotion} • {(confidence * 100).toFixed(1)}%
          </span>
        </div>
      )}
    </div>
  );
}
