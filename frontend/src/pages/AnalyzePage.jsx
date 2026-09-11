import React, { useState } from 'react';
import ImageViewer from '../components/analyze/ImageViewer';
import UploadArea from '../components/analyze/UploadArea';
import ResultCard from '../components/analyze/ResultCard';
import FeedbackPanel from '../components/analyze/FeedbackPanel';
import Modal from '../components/common/Modal';
import Toast from '../components/common/Toast';
import { DEMO_SAMPLE_IMAGE, DEMO_FACES } from '../data/demo/analyses';

export default function AnalyzePage() {
  const [image, setImage] = useState({
    src: DEMO_SAMPLE_IMAGE,
    name: "editorial_session_04.jpg",
    specs: "2400×1800 px • 3.2 MB • 4:3 ratio • sRGB Calibrated",
  });
  
  const [faces, setFaces] = useState(DEMO_FACES);
  const [selectedFaceId, setSelectedFaceId] = useState("01");
  const [zoom, setZoom] = useState(100);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);
  const [analysisState, setAnalysisState] = useState('complete'); // 'idle' | 'analyzing' | 'complete'
  const [jsonModalOpen, setJsonModalOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);

  const selectedFace = faces.find((f) => f.id === selectedFaceId) || faces[0];

  const handleZoom = (delta) => {
    setZoom((prev) => Math.max(50, Math.min(200, prev + delta)));
  };

  const handleResetZoom = () => setZoom(100);

  const handleRerun = () => {
    setAnalysisState('analyzing');
    setTimeout(() => {
      setAnalysisState('complete');
      setToastMessage('Analysis refreshed successfully (183ms)');
    }, 1000);
  };

  const handleImageSelect = (newImage) => {
    setImage({
      src: newImage.src,
      name: newImage.name,
      specs: `${newImage.size} • Uploaded via device`,
    });
    setAnalysisState('analyzing');
    setTimeout(() => {
      setAnalysisState('complete');
      setToastMessage('Face telemetry analysis complete (2 faces localized)');
    }, 1200);
  };

  const handleUseSample = () => {
    setImage({
      src: DEMO_SAMPLE_IMAGE,
      name: "editorial_session_04.jpg",
      specs: "2400×1800 px • 3.2 MB • 4:3 ratio • sRGB Calibrated",
    });
    setFaces(DEMO_FACES);
    setSelectedFaceId("01");
    setAnalysisState('complete');
    setToastMessage('Benchmark image loaded');
  };

  return (
    <div className="flex flex-col xl:flex-row h-full min-w-0 overflow-y-auto xl:overflow-hidden bg-[#F7F7FC] dark:bg-[#0B1020]">
      {/* Toast feedback */}
      <Toast message={toastMessage} onClose={() => setToastMessage(null)} />

      {/* CENTER / LEFT CANVAS WORKSPACE */}
      <section className="flex-1 flex flex-col p-4 sm:p-6 overflow-y-auto min-w-0">
        {/* Workspace Control Bar */}
        <div className="bg-white dark:bg-[#12182B] border border-slate-200 dark:border-[#1E294B] rounded-xl p-3 flex flex-wrap items-center justify-between gap-3 mb-4 shrink-0 shadow-sm">
          {/* File details */}
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-8 h-8 rounded-lg bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] flex items-center justify-center text-slate-500 dark:text-[#94A3B8]">
              <span className="material-symbols-outlined text-[18px]">image</span>
            </div>
            <div className="flex flex-col text-left">
              <div className="flex items-center gap-2">
                <span className="font-semibold text-xs text-slate-800 dark:text-[#F8FAFC] truncate max-w-[180px] sm:max-w-xs">
                  {image.name}
                </span>
                <span className="font-mono text-[10px] px-1.5 py-0.2 rounded bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] text-slate-500 dark:text-[#94A3B8]">
                  RAW·JPEG
                </span>
              </div>
              <div className="text-[11px] text-slate-400 dark:text-[#64748B] font-mono">
                {image.specs}
              </div>
            </div>
          </div>

          {/* Tool Toggles (Strictly actual CV capabilities, NO landmarks) */}
          <div className="flex items-center gap-1.5 bg-slate-100 dark:bg-[#0B1020] p-1 rounded-lg border border-slate-200 dark:border-[#1E294B]">
            <button
              type="button"
              onClick={() => setShowBoxes((prev) => !prev)}
              className={`flex items-center gap-1 px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                showBoxes
                  ? "bg-white dark:bg-[#171F36] text-slate-900 dark:text-[#F8FAFC] shadow-sm border border-slate-200 dark:border-[#1E294B]"
                  : "text-slate-500 dark:text-[#94A3B8] hover:text-slate-900 dark:hover:text-[#F8FAFC]"
              }`}
            >
              <span className="material-symbols-outlined text-[14px] text-[#0891B2] dark:text-[#22D3EE]">
                crop_free
              </span>
              <span>Bounding boxes</span>
            </button>

            <button
              type="button"
              onClick={() => setShowConfidence((prev) => !prev)}
              className={`flex items-center gap-1 px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                showConfidence
                  ? "bg-white dark:bg-[#171F36] text-slate-900 dark:text-[#F8FAFC] shadow-sm border border-slate-200 dark:border-[#1E294B]"
                  : "text-slate-500 dark:text-[#94A3B8] hover:text-slate-900 dark:hover:text-[#F8FAFC]"
              }`}
            >
              <span className="material-symbols-outlined text-[14px] text-[#6C63FF]">
                verified
              </span>
              <span>Confidence</span>
            </button>
          </div>

          {/* Zoom & Canvas Actions */}
          <div className="flex items-center gap-1.5">
            <div className="flex items-center bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] rounded-lg p-0.5 text-xs text-slate-600 dark:text-[#94A3B8]">
              <button
                type="button"
                onClick={() => handleZoom(-10)}
                className="p-1 hover:text-slate-900 dark:hover:text-[#F8FAFC]"
                title="Zoom Out"
              >
                <span className="material-symbols-outlined text-[15px]">remove</span>
              </button>
              <span className="px-2 font-mono text-[11px] text-slate-900 dark:text-[#F8FAFC] select-none">
                {zoom}%
              </span>
              <button
                type="button"
                onClick={() => handleZoom(10)}
                className="p-1 hover:text-slate-900 dark:hover:text-[#F8FAFC]"
                title="Zoom In"
              >
                <span className="material-symbols-outlined text-[15px]">add</span>
              </button>
              <button
                type="button"
                onClick={handleResetZoom}
                className="p-1 hover:text-slate-900 dark:hover:text-[#F8FAFC] ml-0.5"
                title="Reset Zoom"
              >
                <span className="material-symbols-outlined text-[15px]">fit_screen</span>
              </button>
            </div>

            <button
              type="button"
              onClick={handleRerun}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-slate-100 dark:bg-[#171F36] hover:bg-slate-200 dark:hover:bg-[#1E294B] border border-slate-200 dark:border-[#1E294B] text-xs text-slate-700 dark:text-[#94A3B8] hover:text-slate-900 dark:hover:text-[#F8FAFC] transition-colors"
              title="Re-run analysis"
            >
              <span className="material-symbols-outlined text-[14px]">refresh</span>
              <span>Re-run</span>
            </button>

            <button
              type="button"
              onClick={() => setJsonModalOpen(true)}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-slate-100 dark:bg-[#171F36] hover:bg-slate-200 dark:hover:bg-[#1E294B] border border-slate-200 dark:border-[#1E294B] text-xs text-slate-700 dark:text-[#94A3B8] hover:text-slate-900 dark:hover:text-[#F8FAFC] transition-colors"
              title="View raw JSON detections"
            >
              <span className="material-symbols-outlined text-[14px]">code</span>
              <span>JSON</span>
            </button>

            <button
              type="button"
              onClick={() => setAnalysisState('idle')}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-[#6C63FF]/10 text-[#6C63FF] border border-[#6C63FF]/30 hover:bg-[#6C63FF]/20 text-xs font-medium transition-colors"
              title="Upload new photo"
            >
              <span className="material-symbols-outlined text-[14px]">upload</span>
              <span>New</span>
            </button>
          </div>
        </div>

        {/* Focal Image Canvas or Upload State */}
        {analysisState === 'idle' ? (
          <div className="flex-1 flex items-center justify-center p-6 bg-slate-50 dark:bg-[#090D1A] border border-slate-200 dark:border-[#1E294B] rounded-2xl">
            <UploadArea
              onImageSelect={handleImageSelect}
              onUseSample={handleUseSample}
            />
          </div>
        ) : analysisState === 'analyzing' ? (
          <div className="flex-1 flex flex-col items-center justify-center p-12 bg-slate-50 dark:bg-[#090D1A] border border-slate-200 dark:border-[#1E294B] rounded-2xl gap-4">
            <div className="relative flex items-center justify-center w-16 h-16">
              <div className="absolute inset-0 rounded-full border-2 border-[#6C63FF]/30 border-t-[#6C63FF] animate-spin"></div>
              <span className="material-symbols-outlined text-[24px] text-[#22D3EE] animate-pulse">
                center_focus_strong
              </span>
            </div>
            <div className="flex flex-col items-center gap-1">
              <span className="text-sm font-semibold text-slate-800 dark:text-[#F8FAFC]">
                Analyzing Face Telemetry...
              </span>
              <span className="text-xs text-slate-400 dark:text-[#64748B] font-mono">
                Executing ResidualEmotionCNN inference pipeline
              </span>
            </div>
          </div>
        ) : (
          <ImageViewer
            imageSrc={image.src}
            faces={faces}
            selectedFaceId={selectedFaceId}
            onSelectFace={setSelectedFaceId}
            zoom={zoom}
            showBoxes={showBoxes}
            showConfidence={showConfidence}
          />
        )}

        {/* Pipeline Diagnostics Bar */}
        <div className="bg-white dark:bg-[#12182B] border border-slate-200 dark:border-[#1E294B] rounded-xl p-3 mt-4 flex flex-col md:flex-row items-center justify-between gap-3 text-xs text-slate-500 dark:text-[#94A3B8] shrink-0">
          <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
            <span className="uppercase tracking-wider font-semibold text-slate-800 dark:text-[#F8FAFC] flex items-center gap-1">
              <span className="material-symbols-outlined text-[14px] text-[#22D3EE]">check_circle</span>
              Pipeline
            </span>
            <div className="h-3 w-px bg-slate-200 dark:bg-[#1E294B]"></div>
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-50 dark:bg-[#171F36] text-slate-700 dark:text-[#F8FAFC] border border-slate-200 dark:border-[#1E294B]">
              <span className="text-[#0891B2] dark:text-[#22D3EE] font-bold">1.</span>
              <span>Validation</span>
              <span className="text-slate-400 dark:text-[#64748B]">12ms</span>
            </div>
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-50 dark:bg-[#171F36] text-slate-700 dark:text-[#F8FAFC] border border-slate-200 dark:border-[#1E294B]">
              <span className="text-[#0891B2] dark:text-[#22D3EE] font-bold">2.</span>
              <span>Localization</span>
              <span className="text-slate-400 dark:text-[#64748B]">48ms</span>
            </div>
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-50 dark:bg-[#171F36] text-slate-700 dark:text-[#F8FAFC] border border-slate-200 dark:border-[#1E294B]">
              <span className="text-[#0891B2] dark:text-[#22D3EE] font-bold">3.</span>
              <span>Extraction</span>
              <span className="text-slate-400 dark:text-[#64748B]">89ms</span>
            </div>
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-50 dark:bg-[#171F36] text-slate-700 dark:text-[#F8FAFC] border border-slate-200 dark:border-[#1E294B]">
              <span className="text-[#0891B2] dark:text-[#22D3EE] font-bold">4.</span>
              <span>Synthesis</span>
              <span className="text-slate-400 dark:text-[#64748B]">34ms</span>
            </div>
          </div>
          <div className="flex items-center gap-2 font-mono text-[11px] text-slate-500 dark:text-[#64748B] shrink-0">
            <span className="text-slate-800 dark:text-[#F8FAFC] font-medium">Total: 183ms</span>
            <span>•</span>
            <span className="px-2 py-0.5 rounded bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] text-slate-600 dark:text-[#94A3B8]">
              FS-Vision-v2.4 Core
            </span>
          </div>
        </div>
      </section>

      {/* RIGHT RESULTS & FEEDBACK DOCK */}
      <aside className="w-full xl:w-96 bg-white dark:bg-[#12182B] border-t xl:border-t-0 xl:border-l border-slate-200 dark:border-[#1E294B] flex flex-col p-4 sm:p-5 overflow-y-auto shrink-0 gap-4">
        {/* Panel Header */}
        <div className="flex items-center justify-between pb-2 border-b border-slate-200 dark:border-[#1E294B]">
          <div className="flex items-center gap-2">
            <h2 className="font-semibold text-sm text-slate-800 dark:text-[#F8FAFC]">
              Detection Results
            </h2>
            <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-[#6C63FF]/10 text-[#6C63FF] border border-[#6C63FF]/30 font-medium">
              {faces.length} faces
            </span>
          </div>
          <span className="material-symbols-outlined text-[18px] text-slate-400">tune</span>
        </div>

        {/* Confidence Summary */}
        <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[16px] text-[#22D3EE]">verified</span>
            <span className="text-slate-600 dark:text-[#94A3B8]">
              High overall quality <strong className="text-slate-900 dark:text-[#F8FAFC]">85.2%</strong>
            </span>
          </div>
          <span className="font-mono text-[10px] text-slate-400 dark:text-[#64748B]">#8921-A</span>
        </div>

        {/* Subject Select Tabs */}
        <div className="grid grid-cols-3 gap-1 bg-slate-100 dark:bg-[#0B1020] p-1 rounded-lg border border-slate-200 dark:border-[#1E294B] text-xs">
          <button
            type="button"
            onClick={() => setSelectedFaceId(null)}
            className={`py-1 rounded text-center transition-colors ${
              selectedFaceId === null
                ? "bg-white dark:bg-[#171F36] text-slate-900 dark:text-[#F8FAFC] font-medium shadow-sm"
                : "text-slate-500 dark:text-[#64748B] hover:text-slate-900 dark:hover:text-[#F8FAFC]"
            }`}
          >
            All ({faces.length})
          </button>
          {faces.map((f) => (
            <button
              key={f.id}
              type="button"
              onClick={() => setSelectedFaceId(f.id)}
              className={`py-1 rounded text-center transition-colors flex items-center justify-center gap-1 ${
                selectedFaceId === f.id
                  ? "bg-white dark:bg-[#171F36] text-slate-900 dark:text-[#F8FAFC] font-medium shadow-sm border border-[#6C63FF]/40"
                  : "text-slate-500 dark:text-[#64748B] hover:text-slate-900 dark:hover:text-[#F8FAFC]"
              }`}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-[#22D3EE]"></span>
              <span>Face {f.id}</span>
            </button>
          ))}
        </div>

        {/* Primary Subject Result Card */}
        {selectedFace && (
          <ResultCard
            face={selectedFace}
            isSelected={true}
            onSelect={setSelectedFaceId}
          />
        )}

        {/* Secondary Subject Cards if 'All' is selected or list view */}
        {selectedFaceId === null && (
          <div className="flex flex-col gap-3">
            {faces.map((f) => (
              <ResultCard
                key={f.id}
                face={f}
                isSelected={false}
                onSelect={setSelectedFaceId}
              />
            ))}
          </div>
        )}

        {/* Accuracy Review & Feedback Module */}
        {selectedFace && (
          <FeedbackPanel
            selectedFace={selectedFace}
            onFeedbackSubmit={(feedback) => {
              setToastMessage(`Feedback saved: ${feedback.state} for Face ${feedback.faceId}`);
            }}
          />
        )}

        {/* Sticky Actions Footer */}
        <div className="mt-auto flex flex-col gap-2 pt-2 border-t border-slate-200 dark:border-[#1E294B]">
          <button
            type="button"
            onClick={() => setToastMessage('Analysis saved to project history')}
            className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-[#6C63FF] hover:bg-[#5B52EE] text-white text-xs font-semibold shadow-sm transition-colors"
          >
            <span className="material-symbols-outlined text-[16px]">bookmark_border</span>
            <span>Save to Project History</span>
          </button>
          <button
            type="button"
            onClick={() => setToastMessage('Analytical report generated')}
            className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-slate-100 dark:bg-[#171F36] hover:bg-slate-200 dark:hover:bg-[#1E294B] border border-slate-200 dark:border-[#1E294B] text-slate-700 dark:text-[#94A3B8] hover:text-slate-900 dark:hover:text-[#F8FAFC] text-xs font-medium transition-colors"
          >
            <span className="material-symbols-outlined text-[16px]">download</span>
            <span>Download Analytical Report (JSON/CSV)</span>
          </button>
        </div>
      </aside>

      {/* Raw JSON Detections Modal */}
      <Modal
        isOpen={jsonModalOpen}
        onClose={() => setJsonModalOpen(false)}
        title="Raw Detection Payload (FastAPI Schema)"
      >
        <div className="flex flex-col gap-3">
          <pre className="p-3 bg-slate-900 text-slate-100 rounded-lg text-[11px] font-mono overflow-x-auto max-h-80">
            {JSON.stringify({
              status: "success",
              image: image.name,
              total_faces: faces.length,
              detections: faces.map((f) => ({
                id: f.id,
                box: f.box,
                predicted_emotion: f.predicted_emotion,
                confidence: f.confidence,
                probabilities: f.probabilities,
              })),
            }, null, 2)}
          </pre>
          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => setJsonModalOpen(false)}
              className="px-4 py-1.5 text-xs font-medium rounded-lg bg-[#6C63FF] text-white"
            >
              Done
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
