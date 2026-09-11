import React, { useState, useRef } from 'react';
import Button from '../common/Button';

export default function UploadArea({
  onImageSelect,
  onUseSample,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleFileInput = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file (JPEG, PNG, WebP).');
      return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
      onImageSelect({
        src: event.target.result,
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
      });
    };
    reader.readAsDataURL(file);
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`relative w-full max-w-xl mx-auto rounded-2xl p-8 sm:p-10 border-2 border-dashed transition-all flex flex-col items-center justify-center text-center select-none ${
        isDragging
          ? "border-[#22D3EE] bg-[#22D3EE]/5 scale-[1.01]"
          : "border-slate-300 dark:border-[#1E294B] bg-white/50 dark:bg-[#12182B]/60 hover:border-[#6C63FF] dark:hover:border-[#6C63FF]/60"
      }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={handleFileInput}
        className="hidden"
      />

      {/* Upload Icon */}
      <div className="w-14 h-14 rounded-2xl bg-[#6C63FF]/10 text-[#6C63FF] flex items-center justify-center mb-4 shadow-inner">
        <span className="material-symbols-outlined text-[28px]">add_photo_alternate</span>
      </div>

      <h3 className="text-base font-semibold text-slate-800 dark:text-[#F8FAFC] mb-1">
        Upload Face or Session Image
      </h3>
      <p className="text-xs text-slate-500 dark:text-[#94A3B8] max-w-sm mb-6">
        Drag and drop high-resolution image, or browse local device. Evaluates 7-class facial telemetry.
      </p>

      {/* Buttons */}
      <div className="flex flex-wrap items-center justify-center gap-3">
        <Button
          variant="primary"
          icon="upload_file"
          onClick={() => fileInputRef.current?.click()}
        >
          Choose Image
        </Button>

        <Button
          variant="secondary"
          icon="photo_camera"
          onClick={() => alert('Webcam capture mode available in live feed.')}
        >
          Camera Action
        </Button>
      </div>

      <div className="flex items-center gap-2 my-4 w-full max-w-xs">
        <div className="h-px bg-slate-200 dark:bg-[#1E294B] flex-1"></div>
        <span className="text-[11px] text-slate-400 dark:text-[#64748B] font-mono">OR</span>
        <div className="h-px bg-slate-200 dark:bg-[#1E294B] flex-1"></div>
      </div>

      <button
        type="button"
        onClick={onUseSample}
        className="inline-flex items-center gap-1.5 text-xs text-[#6C63FF] hover:text-[#5B52EE] font-medium"
      >
        <span className="material-symbols-outlined text-[15px]">auto_awesome</span>
        <span>Load editorial benchmark sample</span>
      </button>

      <span className="text-[10px] text-slate-400 dark:text-[#64748B] font-mono mt-4">
        Supports RAW·JPEG, PNG, WebP • Up to 10 MB
      </span>
    </div>
  );
}
