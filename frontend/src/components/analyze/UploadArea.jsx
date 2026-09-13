import React, { useState, useRef } from 'react';
import Button from '../common/Button';

export default function UploadArea({
  onImageSelect,
  onStartCamera,
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
        file,
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
        type="file"
        ref={fileInputRef}
        onChange={handleFileInput}
        accept="image/jpeg,image/png,image/webp"
        className="hidden"
      />

      <div className="w-16 h-16 rounded-2xl bg-[#6C63FF]/10 text-[#6C63FF] flex items-center justify-center mb-4 ring-8 ring-[#6C63FF]/5 shrink-0">
        <span className="material-symbols-outlined text-[32px]">
          add_photo_alternate
        </span>
      </div>

      <h3 className="text-base sm:text-lg font-bold text-slate-800 dark:text-[#F8FAFC]">
        Analyze an image
      </h3>
      <p className="text-xs sm:text-sm text-slate-500 dark:text-[#94A3B8] max-w-sm mt-1 mb-6 leading-relaxed">
        Upload a photo or use live webcam to detect faces and classify facial expressions.
      </p>

      <div className="flex flex-wrap items-center justify-center gap-3 w-full max-w-xs sm:max-w-none">
        <Button
          variant="primary"
          icon="upload"
          onClick={() => fileInputRef.current?.click()}
          id="btn-upload-image"
          className="min-h-[44px] px-5"
        >
          Upload Image
        </Button>

        <Button
          variant="outline"
          icon="photo_camera"
          onClick={onStartCamera}
          id="btn-use-camera"
          className="min-h-[44px] px-5"
        >
          Use Camera
        </Button>
      </div>

      <div className="mt-8 flex flex-col sm:flex-row items-center gap-2 text-[11px] text-slate-400 dark:text-[#64748B] font-mono text-center">
        <span>Supported: JPG, JPEG, PNG, WebP, Camera Capture</span>
        <span className="hidden sm:inline">•</span>
        <span>Max file size: 10MB</span>
      </div>
    </div>
  );
}
