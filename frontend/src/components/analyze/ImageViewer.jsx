import React from 'react';
import FaceBox from './FaceBox';

export default function ImageViewer({
  imageSrc,
  faces = [],
  selectedFaceId,
  onSelectFace,
  zoom = 100,
  showBoxes = true,
  showConfidence = true,
}) {
  return (
    <div 
      className="relative flex-1 bg-slate-100 dark:bg-[#090D1A] border border-slate-200 dark:border-[#1E294B] rounded-2xl flex flex-col items-center justify-center p-4 sm:p-6 overflow-hidden min-h-[460px] dot-grid select-none"
      onClick={() => onSelectFace(null)}
    >
      {/* Top Left Active Sensor Badge */}
      <div className="absolute top-4 left-4 z-20 flex items-center gap-2 px-3 py-1 rounded-full bg-white/95 dark:bg-[#12182B]/95 border border-slate-200 dark:border-[#1E294B] backdrop-blur-md text-[11px] text-slate-600 dark:text-[#94A3B8] shadow-md">
        <span className="w-2 h-2 rounded-full bg-[#22D3EE] animate-pulse"></span>
        <span>Active Sensor Overlay</span>
        <span className="text-slate-300 dark:text-[#64748B]">•</span>
        <span className="text-[#0891B2] dark:text-[#22D3EE] font-mono">Vision Telemetry</span>
      </div>

      {/* Top Right Reticle Coordinates Readout */}
      <div className="absolute top-4 right-4 z-20 hidden sm:flex items-center gap-2 px-3 py-1 rounded-md bg-white/85 dark:bg-[#12182B]/85 border border-slate-200 dark:border-[#1E294B] text-[11px] font-mono text-slate-400 dark:text-[#64748B] shadow-sm">
        <span>FOV 84.2°</span>
        <span>•</span>
        <span>sRGB Calibrated</span>
        <span>•</span>
        <span className="text-[#6C63FF] font-semibold">{faces.length} Detections</span>
      </div>

      {/* Image Viewport Container */}
      <div
        className="relative max-w-2xl w-full rounded-xl shadow-xl overflow-hidden border border-slate-200 dark:border-[#1E294B] bg-white dark:bg-[#12182B] transition-transform duration-200"
        style={{ transform: `scale(${zoom / 100})` }}
        onClick={(e) => e.stopPropagation()}
      >
        <img
          src={imageSrc}
          alt="Analyzed face telemetry"
          className="w-full h-auto block select-none pointer-events-none"
        />

        {/* Bounding Box Overlays (Strictly face bounding boxes, NO landmarks) */}
        {faces.map((face) => (
          <FaceBox
            key={face.id}
            face={face}
            isSelected={selectedFaceId === face.id}
            onSelect={onSelectFace}
            showBoxes={showBoxes}
            showConfidence={showConfidence}
          />
        ))}
      </div>

      {/* Micro-hint Footer */}
      <div className="mt-4 flex items-center gap-2 text-[11px] text-slate-500 dark:text-[#64748B] font-mono">
        <span className="material-symbols-outlined text-[13px] text-[#22D3EE]">info</span>
        <span>Click any bounding box to inspect 7-class emotion probabilities.</span>
      </div>
    </div>
  );
}
