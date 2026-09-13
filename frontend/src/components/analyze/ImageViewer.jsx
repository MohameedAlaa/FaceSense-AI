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
  onImageLoad,
}) {
  return (
    <div 
      className="relative w-full max-w-full bg-slate-100 dark:bg-[#090D1A] border border-slate-200 dark:border-[#1E294B] rounded-2xl flex flex-col items-center justify-center p-2.5 sm:p-5 overflow-hidden dot-grid select-none min-h-[200px] sm:min-h-[360px] lg:min-h-[420px] lg:flex-1"
      onClick={() => onSelectFace(null)}
    >
      {/* Top Left Active Sensor Badge */}
      <div className="absolute top-2.5 left-2.5 sm:top-4 sm:left-4 z-20 flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-1 rounded-full bg-white/95 dark:bg-[#12182B]/95 border border-slate-200 dark:border-[#1E294B] backdrop-blur-md text-[10px] sm:text-[11px] text-slate-600 dark:text-[#94A3B8] shadow-md max-w-[48%] truncate">
        <span className="w-2 h-2 rounded-full bg-[#22D3EE] animate-pulse shrink-0"></span>
        <span className="truncate">Vision Telemetry</span>
      </div>

      {/* Top Right Detection Counter Readout (strictly real detections) */}
      {faces.length > 0 && (
        <div className="absolute top-2.5 right-2.5 sm:top-4 sm:right-4 z-20 flex items-center gap-1.5 px-2.5 sm:px-3 py-1 rounded-md bg-white/90 dark:bg-[#12182B]/90 border border-slate-200 dark:border-[#1E294B] text-[10px] sm:text-[11px] font-mono text-slate-500 dark:text-[#94A3B8] shadow-sm max-w-[48%] truncate">
          <span className="text-[#6C63FF] font-semibold truncate">
            {faces.length} {faces.length === 1 ? 'Detection' : 'Detections'}
          </span>
        </div>
      )}

      {/* Image Viewport Container */}
      <div
        className="relative max-w-full sm:max-w-2xl w-full rounded-xl shadow-xl overflow-hidden border border-slate-200 dark:border-[#1E294B] bg-white dark:bg-[#12182B] transition-transform duration-200"
        style={{ transform: `scale(${zoom / 100})` }}
        onClick={(e) => e.stopPropagation()}
      >
        <img
          src={imageSrc}
          alt="Facial telemetry"
          onLoad={(e) => {
            if (onImageLoad) {
              onImageLoad(e.target.naturalWidth, e.target.naturalHeight);
            }
          }}
          className="w-full h-auto block select-none pointer-events-none"
        />

        {/* Bounding Box Overlays (Strictly face bounding boxes) */}
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

      {/* Micro-hint Footer (only when faces exist) */}
      {faces.length > 0 && (
        <div className="mt-3 sm:mt-4 flex items-center gap-1.5 text-[10px] sm:text-[11px] text-slate-500 dark:text-[#64748B] font-mono text-center px-2">
          <span className="material-symbols-outlined text-[13px] text-[#22D3EE] shrink-0">info</span>
          <span>Click any bounding box to inspect 7-class emotion probabilities.</span>
        </div>
      )}
    </div>
  );
}
