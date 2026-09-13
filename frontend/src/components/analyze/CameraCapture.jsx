import React, { useState, useEffect, useRef } from 'react';
import Button from '../common/Button';

export default function CameraCapture({ onCapture, onCancel, onError }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [cameraStatus, setCameraStatus] = useState('INITIALIZING'); // 'INITIALIZING' | 'ACTIVE' | 'ERROR'
  const [errorMessage, setErrorMessage] = useState(null);
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });

  // Stop all media tracks helper
  const stopTracks = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch {
          // ignore error during cleanup
        }
      });
      streamRef.current = null;
    }
  };

  const startCamera = async () => {
    stopTracks();
    setCameraStatus('INITIALIZING');
    setErrorMessage(null);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const msg = 'Camera access is not supported by your browser environment.';
      setErrorMessage(msg);
      setCameraStatus('ERROR');
      if (onError) onError(msg);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            setVideoDimensions({
              width: videoRef.current.videoWidth,
              height: videoRef.current.videoHeight,
            });
            videoRef.current.play().catch(() => {});
          }
          setCameraStatus('ACTIVE');
        };
      }
    } catch (err) {
      let friendlyMsg = 'Failed to access webcam.';
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        friendlyMsg = 'Camera permission was denied. Please allow camera access in your browser settings to capture images.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        friendlyMsg = 'No camera device was detected on your system.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        friendlyMsg = 'Camera is currently in use by another application or unavailable.';
      } else if (err.name === 'OverconstrainedError') {
        friendlyMsg = 'Requested camera resolution constraints could not be satisfied by your device.';
      } else if (err.message) {
        friendlyMsg = err.message;
      }

      setErrorMessage(friendlyMsg);
      setCameraStatus('ERROR');
      if (onError) onError(friendlyMsg);
    }
  };

  useEffect(() => {
    startCamera();
    return () => {
      stopTracks();
    };
  }, []);

  const handleCapture = () => {
    const video = videoRef.current;
    if (!video || cameraStatus !== 'ACTIVE') return;

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // Draw frame onto canvas
    ctx.drawImage(video, 0, 0, width, height);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.95);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], `camera_capture_${Date.now()}.jpg`, {
          type: 'image/jpeg',
          lastModified: Date.now(),
        });

        // Immediately stop stream tracks
        stopTracks();

        if (onCapture) {
          onCapture(file, dataUrl);
        }
      },
      'image/jpeg',
      0.95
    );
  };

  const handleCancel = () => {
    stopTracks();
    if (onCancel) {
      onCancel();
    }
  };

  return (
    <div className="relative w-full max-w-full bg-slate-100 dark:bg-[#090D1A] border border-slate-200 dark:border-[#1E294B] rounded-2xl flex flex-col items-center justify-center p-3 sm:p-5 overflow-hidden select-none min-h-[360px] lg:min-h-[460px] lg:flex-1">
      {/* Top Left Live Status Badge */}
      <div className="absolute top-2.5 left-2.5 sm:top-4 sm:left-4 z-20 flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-1 rounded-full bg-white/95 dark:bg-[#12182B]/95 border border-slate-200 dark:border-[#1E294B] backdrop-blur-md text-[10px] sm:text-[11px] text-slate-600 dark:text-[#94A3B8] shadow-md">
        <span
          className={`w-2 h-2 rounded-full ${
            cameraStatus === 'ACTIVE'
              ? 'bg-emerald-400 animate-pulse'
              : cameraStatus === 'INITIALIZING'
              ? 'bg-amber-400 animate-ping'
              : 'bg-rose-500'
          }`}
        ></span>
        <span>
          {cameraStatus === 'ACTIVE'
            ? 'Live Video Stream'
            : cameraStatus === 'INITIALIZING'
            ? 'Requesting Camera...'
            : 'Camera Inactive'}
        </span>
      </div>

      {/* Top Right Resolution Readout */}
      {cameraStatus === 'ACTIVE' && videoDimensions.width > 0 && (
        <div className="absolute top-2.5 right-2.5 sm:top-4 sm:right-4 z-20 flex items-center gap-1.5 px-2.5 sm:px-3 py-1 rounded-md bg-white/90 dark:bg-[#12182B]/90 border border-slate-200 dark:border-[#1E294B] text-[10px] sm:text-[11px] font-mono text-slate-500 dark:text-[#94A3B8] shadow-sm">
          <span>{videoDimensions.width}×{videoDimensions.height}</span>
        </div>
      )}

      {/* Camera Video Viewport or Status State */}
      <div className="relative max-w-full sm:max-w-2xl w-full rounded-xl shadow-xl overflow-hidden border border-slate-200 dark:border-[#1E294B] bg-black flex items-center justify-center min-h-[260px] sm:min-h-[380px]">
        {/* The Live Video Element */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-auto max-h-[440px] object-contain ${
            cameraStatus === 'ACTIVE' ? 'block' : 'hidden'
          }`}
        />

        {/* Initializing Spinner */}
        {cameraStatus === 'INITIALIZING' && (
          <div className="flex flex-col items-center justify-center gap-3 p-8 text-center text-slate-300">
            <div className="w-10 h-10 rounded-full border-2 border-[#6C63FF]/30 border-t-[#6C63FF] animate-spin"></div>
            <div className="flex flex-col gap-1">
              <span className="text-sm font-semibold text-white">Opening Camera</span>
              <span className="text-xs text-slate-400">
                Please allow browser video permissions when prompted...
              </span>
            </div>
          </div>
        )}

        {/* Error State */}
        {cameraStatus === 'ERROR' && (
          <div className="flex flex-col items-center justify-center gap-3 p-6 sm:p-8 text-center max-w-md">
            <div className="w-12 h-12 rounded-full bg-rose-500/20 text-rose-400 flex items-center justify-center">
              <span className="material-symbols-outlined text-[28px]">videocam_off</span>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-sm font-bold text-white">Camera Access Error</span>
              <p className="text-xs text-slate-300 leading-relaxed">{errorMessage}</p>
            </div>
            <div className="flex items-center gap-2 mt-2">
              <Button
                variant="primary"
                size="sm"
                icon="refresh"
                onClick={startCamera}
                className="min-h-[44px] px-4"
              >
                Try Again
              </Button>
              <Button
                variant="outline"
                size="sm"
                icon="arrow_back"
                onClick={handleCancel}
                className="min-h-[44px] px-4 text-white border-slate-600 hover:bg-slate-800"
              >
                Back to Upload
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Camera Action Toolbar */}
      <div className="mt-4 flex flex-wrap items-center justify-center gap-3 w-full max-w-md">
        {cameraStatus === 'ACTIVE' && (
          <Button
            variant="primary"
            size="md"
            icon="photo_camera"
            onClick={handleCapture}
            id="btn-capture-frame"
            className="min-h-[44px] px-6 text-sm font-semibold shadow-lg shadow-[#6C63FF]/20"
          >
            Capture Frame
          </Button>
        )}

        <Button
          variant="outline"
          size="md"
          icon="close"
          onClick={handleCancel}
          id="btn-cancel-camera"
          className="min-h-[44px] px-5 text-sm"
        >
          Cancel
        </Button>
      </div>

      <div className="mt-2 text-[10px] text-slate-400 dark:text-[#64748B] font-mono text-center">
        High-resolution webcam frame capture • Feeds directly to vision telemetry
      </div>
    </div>
  );
}
