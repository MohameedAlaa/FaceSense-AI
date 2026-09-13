import React, { useState, useEffect } from 'react';
import ImageViewer from '../components/analyze/ImageViewer';
import UploadArea from '../components/analyze/UploadArea';
import ResultCard from '../components/analyze/ResultCard';
import FeedbackPanel from '../components/analyze/FeedbackPanel';
import Modal from '../components/common/Modal';
import Toast from '../components/common/Toast';
import Button from '../components/common/Button';
import { predictApi } from '../lib/api/predict';
import { feedbackApi } from '../lib/api/feedback';

/**
 * State machine for AnalyzePage:
 * - 'EMPTY': No image chosen yet. Shows clean UploadArea and neutral 'No analysis yet' right panel.
 * - 'IMAGE_SELECTED': Image loaded in browser, awaiting user action to run analysis.
 * - 'ANALYZING': Inference request in flight to FastAPI backend.
 * - 'RESULTS': Real prediction received with >= 1 detected face.
 * - 'NO_FACE': Real prediction received with 0 detected faces.
 * - 'ERROR': API or image decoding error occurred.
 */
export default function AnalyzePage() {
  const [analysisState, setAnalysisState] = useState('EMPTY');
  const [currentFile, setCurrentFile] = useState(null);
  const [image, setImage] = useState(null);
  const [imgDimensions, setImgDimensions] = useState({ width: 0, height: 0 });
  const [faces, setFaces] = useState([]);
  const [selectedFaceId, setSelectedFaceId] = useState(null);
  const [zoom, setZoom] = useState(100);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showConfidence, setShowConfidence] = useState(true);
  const [loadingStage, setLoadingStage] = useState('Preparing image...');
  const [processingTimeMs, setProcessingTimeMs] = useState(0);
  const [modelInfo, setModelInfo] = useState(null);
  const [rawResponse, setRawResponse] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [jsonModalOpen, setJsonModalOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);
  const [faceFeedback, setFaceFeedback] = useState({});
  const [submittingFaceId, setSubmittingFaceId] = useState(null);

  // Fetch real model metadata on mount
  useEffect(() => {
    let isMounted = true;
    predictApi.getModelInfo()
      .then((info) => {
        if (isMounted && info) {
          setModelInfo(info);
        }
      })
      .catch(() => {
        // Graceful fallback if backend offline
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const selectedFace = faces.find((f) => f.id === selectedFaceId) || null;

  const handleZoom = (delta) => {
    setZoom((prev) => Math.max(50, Math.min(200, prev + delta)));
  };

  const handleResetZoom = () => setZoom(100);

  // Convert pixel bounding boxes to percentage coordinates
  const mapPredictionsToFaces = (predictions, naturalWidth, naturalHeight) => {
    return predictions.map((pred, idx) => {
      const id = String(idx + 1).padStart(2, '0');
      const box = pred.box || { x: 0, y: 0, w: 0, h: 0 };

      const left = naturalWidth > 0 ? (box.x / naturalWidth) * 100 : 0;
      const top = naturalHeight > 0 ? (box.y / naturalHeight) * 100 : 0;
      const width = naturalWidth > 0 ? (box.w / naturalWidth) * 100 : 0;
      const height = naturalHeight > 0 ? (box.h / naturalHeight) * 100 : 0;

      return {
        id,
        label: `Face ${id}`,
        face_index: idx,
        pixelBox: box,
        box: { left, top, width, height },
        predicted_emotion: pred.emotion,
        confidence: pred.confidence,
        probabilities: pred.all_probabilities || {},
      };
    });
  };

  const executePrediction = async (file, naturalWidth = imgDimensions.width, naturalHeight = imgDimensions.height) => {
    if (!file) return;

    setAnalysisState('ANALYZING');
    setErrorMessage(null);
    setFaceFeedback({});
    setSubmittingFaceId(null);
    setLoadingStage('Transmitting to FastAPI vision engine...');

    try {
      setLoadingStage('Localizing faces & classifying expressions...');
      const response = await predictApi.predictImage(file, { allowFallback: true });
      setRawResponse(response);
      setProcessingTimeMs(response.processing_time_ms ? Math.round(response.processing_time_ms) : 120);

      if (response.faces_detected > 0 && response.predictions?.length > 0) {
        const mapped = mapPredictionsToFaces(response.predictions, naturalWidth, naturalHeight);
        setFaces(mapped);
        setSelectedFaceId(mapped[0]?.id || null);
        setAnalysisState('RESULTS');
        setToastMessage(`Analysis complete: ${response.faces_detected} face${response.faces_detected > 1 ? 's' : ''} detected (${Math.round(response.processing_time_ms || 0)}ms)`);
      } else {
        setFaces([]);
        setSelectedFaceId(null);
        setAnalysisState('NO_FACE');
        setToastMessage('No faces localized in image');
      }
    } catch (err) {
      setAnalysisState('ERROR');
      let friendlyError = err.message || 'Vision pipeline inference failed.';
      if (err.status === 422) {
        friendlyError = 'No face detected in the uploaded image. Please try an image with clearer frontal facial orientation and adequate lighting.';
      } else if (err.status === 400) {
        friendlyError = 'Image decode error. Please ensure the file is an uncorrupted JPEG, PNG, or WebP file.';
      } else if (err.status === 413) {
        friendlyError = 'Image exceeds maximum 10MB upload threshold.';
      } else if (err.status === 429) {
        friendlyError = 'Inference rate limit exceeded. Please wait a moment before running another prediction.';
      }
      setErrorMessage(friendlyError);
      setToastMessage(friendlyError);
    }
  };

  const handleImageDimensions = (width, height) => {
    setImgDimensions({ width, height });
    if (rawResponse && rawResponse.predictions?.length > 0) {
      const remapped = mapPredictionsToFaces(rawResponse.predictions, width, height);
      setFaces(remapped);
    }
  };

  const handleImageSelect = (newImage) => {
    setCurrentFile(newImage.file || null);
    setImage({
      src: newImage.src,
      name: newImage.name,
      specs: `${newImage.size} • Selected for analysis`,
    });
    setFaces([]);
    setSelectedFaceId(null);
    setFaceFeedback({});
    setSubmittingFaceId(null);
    setRawResponse(null);
    setErrorMessage(null);
    setAnalysisState('IMAGE_SELECTED');
  };

  const handleUseSample = async () => {
    try {
      setAnalysisState('ANALYZING');
      setLoadingStage('Loading sample image...');
      const res = await fetch('/sample_face.jpg');
      const blob = await res.blob();
      const file = new File([blob], 'sample_face.jpg', { type: 'image/jpeg' });
      const reader = new FileReader();
      reader.onload = async (event) => {
        setImage({
          file,
          src: event.target.result,
          name: 'sample_face.jpg',
          specs: `${(file.size / 1024).toFixed(1)} KB • Sample face image`,
        });
        setCurrentFile(file);
        await executePrediction(file);
      };
      reader.readAsDataURL(file);
    } catch (err) {
      setAnalysisState('ERROR');
      setErrorMessage(err.message || 'Vision pipeline inference failed.');
    }
  };

  const handleUseMultiSample = async () => {
    try {
      setAnalysisState('ANALYZING');
      setLoadingStage('Loading multi-face sample...');
      const res = await fetch('/multi_face_sample.jpg');
      const blob = await res.blob();
      const file = new File([blob], 'multi_face_sample.jpg', { type: 'image/jpeg' });
      const reader = new FileReader();
      reader.onload = async (event) => {
        setImage({
          file,
          src: event.target.result,
          name: 'multi_face_sample.jpg',
          specs: `${(file.size / 1024).toFixed(1)} KB • Multi-face sample image`,
        });
        setCurrentFile(file);
        await executePrediction(file);
      };
      reader.readAsDataURL(file);
    } catch (err) {
      setAnalysisState('ERROR');
      setErrorMessage(err.message || 'Vision pipeline inference failed.');
    }
  };

  const handleResetToEmpty = () => {
    setCurrentFile(null);
    setImage(null);
    setFaces([]);
    setSelectedFaceId(null);
    setFaceFeedback({});
    setSubmittingFaceId(null);
    setRawResponse(null);
    setErrorMessage(null);
    setAnalysisState('EMPTY');
  };

  const handleRerun = () => {
    if (currentFile) {
      executePrediction(currentFile);
    }
  };

  const handleFeedbackSubmit = async (feedbackData) => {
    setSubmittingFaceId(feedbackData.faceId);
    try {
      const payload = {
        feedback_type: feedbackData.feedback_type,
        predicted_emotion: feedbackData.predicted_emotion,
        confidence: feedbackData.confidence,
        face_index: feedbackData.face_index,
        bounding_box: feedbackData.bounding_box,
        model_version: feedbackData.model_version || modelInfo?.model_name || 'ResidualEmotionCNN-Candidate-epoch39',
        notes: `Submitted via FaceSense Web Analyze workspace for ${feedbackData.faceLabel || 'Face'} (index ${feedbackData.face_index})`,
      };

      if (feedbackData.feedback_type === 'incorrect') {
        payload.corrected_emotion = feedbackData.corrected_emotion;
      }

      const response = await feedbackApi.submitFeedback(payload);

      setFaceFeedback((prev) => ({
        ...prev,
        [feedbackData.faceId]: {
          status: 'submitted',
          feedback_type: feedbackData.feedback_type,
          corrected_emotion: feedbackData.corrected_emotion || null,
          record_id: response?.record_id || null,
        },
      }));

      setToastMessage(`Feedback recorded for ${feedbackData.faceLabel || 'Face'}!`);
      return true;
    } catch (err) {
      setToastMessage(err.message || 'Failed to submit feedback to backend.');
      return false;
    } finally {
      setSubmittingFaceId(null);
    }
  };

  const handleResetFaceFeedback = (faceId) => {
    setFaceFeedback((prev) => {
      const next = { ...prev };
      delete next[faceId];
      return next;
    });
  };

  const meanConfidence = faces.length > 0
    ? ((faces.reduce((acc, f) => acc + (f.confidence || 0), 0) / faces.length) * 100).toFixed(1)
    : null;

  // ==========================================
  // SHARED SUB-RENDERERS
  // ==========================================

  const renderControlBar = (isMobile = false) => {
    if (!image && analysisState === 'EMPTY') return null;

    return (
      <div className="bg-white dark:bg-[#12182B] border border-slate-200 dark:border-[#1E294B] rounded-xl p-3 flex flex-wrap items-center justify-between gap-2.5 shrink-0 shadow-sm w-full max-w-full overflow-hidden">
        {/* File details */}
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-8 h-8 rounded-lg bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] flex items-center justify-center text-slate-500 dark:text-[#94A3B8] shrink-0">
            <span className="material-symbols-outlined text-[18px]">
              {image ? 'image' : 'photo_library'}
            </span>
          </div>
          <div className="flex flex-col text-left min-w-0">
            <div className="flex items-center gap-2 min-w-0">
              <span className="font-semibold text-xs text-slate-800 dark:text-[#F8FAFC] truncate max-w-[130px] sm:max-w-xs">
                {image ? image.name : 'Vision Analysis Workspace'}
              </span>
              {image && (
                <span className="font-mono text-[10px] px-1.5 py-0.2 rounded bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] text-slate-500 dark:text-[#94A3B8] shrink-0">
                  {imgDimensions.width > 0 ? `${imgDimensions.width}×${imgDimensions.height}` : 'IMG'}
                </span>
              )}
            </div>
            <div className="text-[11px] text-slate-400 dark:text-[#64748B] font-mono truncate">
              {image ? image.specs : 'Awaiting file selection'}
            </div>
          </div>
        </div>

        {/* Controls: Only visible when an image exists */}
        {image && (
          <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
            {/* Bounding Box / Confidence Toggles (Only in RESULTS state) */}
            {analysisState === 'RESULTS' && (
              <div className="flex items-center gap-1 bg-slate-100 dark:bg-[#0B1020] p-0.5 rounded-lg border border-slate-200 dark:border-[#1E294B]">
                <button
                  type="button"
                  onClick={() => setShowBoxes((prev) => !prev)}
                  className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                    showBoxes
                      ? "bg-white dark:bg-[#171F36] text-slate-900 dark:text-[#F8FAFC] shadow-sm border border-slate-200 dark:border-[#1E294B]"
                      : "text-slate-500 dark:text-[#94A3B8] hover:text-slate-900 dark:hover:text-[#F8FAFC]"
                  }`}
                  title="Toggle face bounding boxes"
                >
                  <span className="material-symbols-outlined text-[14px] text-[#0891B2] dark:text-[#22D3EE]">
                    crop_free
                  </span>
                  <span className="hidden sm:inline">Boxes</span>
                </button>

                <button
                  type="button"
                  onClick={() => setShowConfidence((prev) => !prev)}
                  className={`flex items-center gap-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
                    showConfidence
                      ? "bg-white dark:bg-[#171F36] text-slate-900 dark:text-[#F8FAFC] shadow-sm border border-slate-200 dark:border-[#1E294B]"
                      : "text-slate-500 dark:text-[#94A3B8] hover:text-slate-900 dark:hover:text-[#F8FAFC]"
                  }`}
                  title="Toggle confidence labels"
                >
                  <span className="material-symbols-outlined text-[14px] text-[#6C63FF]">
                    verified
                  </span>
                  <span className="hidden sm:inline">Confidence</span>
                </button>
              </div>
            )}

            {/* Zoom Controls */}
            <div className="flex items-center bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] rounded-lg p-0.5 text-xs text-slate-600 dark:text-[#94A3B8]">
              <button
                type="button"
                onClick={() => handleZoom(-10)}
                className="p-1 hover:text-slate-900 dark:hover:text-[#F8FAFC]"
                title="Zoom Out"
              >
                <span className="material-symbols-outlined text-[15px]">remove</span>
              </button>
              <span className="px-1.5 font-mono text-[11px] text-slate-900 dark:text-[#F8FAFC] select-none">
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

            {/* Action Buttons */}
            {analysisState === 'RESULTS' && (
              <>
                <button
                  type="button"
                  onClick={handleRerun}
                  className="flex items-center gap-1 px-2 py-1.5 rounded-lg bg-slate-100 dark:bg-[#171F36] hover:bg-slate-200 dark:hover:bg-[#1E294B] border border-slate-200 dark:border-[#1E294B] text-xs text-slate-700 dark:text-[#94A3B8] hover:text-slate-900 dark:hover:text-[#F8FAFC] transition-colors"
                  title="Re-run analysis"
                >
                  <span className="material-symbols-outlined text-[14px]">refresh</span>
                  <span className="hidden sm:inline">Re-run</span>
                </button>

                <button
                  type="button"
                  onClick={() => setJsonModalOpen(true)}
                  className="flex items-center gap-1 px-2 py-1.5 rounded-lg bg-slate-100 dark:bg-[#171F36] hover:bg-slate-200 dark:hover:bg-[#1E294B] border border-slate-200 dark:border-[#1E294B] text-xs text-slate-700 dark:text-[#94A3B8] hover:text-slate-900 dark:hover:text-[#F8FAFC] transition-colors"
                  title="View raw JSON detections"
                >
                  <span className="material-symbols-outlined text-[14px]">code</span>
                  <span className="hidden sm:inline">JSON</span>
                </button>
              </>
            )}

            <button
              type="button"
              onClick={handleResetToEmpty}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-[#6C63FF]/10 text-[#6C63FF] border border-[#6C63FF]/30 hover:bg-[#6C63FF]/20 text-xs font-medium transition-colors"
              title="Upload new photo"
            >
              <span className="material-symbols-outlined text-[14px]">upload</span>
              <span>New</span>
            </button>
          </div>
        )}
      </div>
    );
  };

  const renderCanvas = () => {
    if (analysisState === 'EMPTY') {
      return (
        <div className="flex-1 flex items-center justify-center p-4 sm:p-8 bg-slate-50 dark:bg-[#090D1A] border border-slate-200 dark:border-[#1E294B] rounded-2xl w-full max-w-full">
          <UploadArea
            onImageSelect={handleImageSelect}
            onUseSample={handleUseSample}
            onUseMultiSample={handleUseMultiSample}
          />
        </div>
      );
    }

    if (analysisState === 'IMAGE_SELECTED') {
      return (
        <div className="flex-1 flex flex-col items-center justify-center gap-4 p-3 sm:p-6 bg-slate-50 dark:bg-[#090D1A] border border-slate-200 dark:border-[#1E294B] rounded-2xl w-full max-w-full">
          <ImageViewer
            imageSrc={image?.src}
            faces={[]}
            selectedFaceId={null}
            onSelectFace={() => {}}
            zoom={zoom}
            showBoxes={false}
            showConfidence={false}
            onImageLoad={handleImageDimensions}
          />

          {/* Launch Action Bar */}
          <div className="w-full max-w-md p-3.5 sm:p-4 rounded-xl bg-white dark:bg-[#12182B] border border-slate-200 dark:border-[#1E294B] shadow-md flex flex-col sm:flex-row items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs text-slate-600 dark:text-[#94A3B8]">
              <span className="material-symbols-outlined text-[18px] text-[#22D3EE]">check_circle</span>
              <span>Image loaded and ready</span>
            </div>
            <div className="flex items-center gap-2 w-full sm:w-auto">
              <Button
                variant="primary"
                size="md"
                icon="bolt"
                onClick={() => executePrediction(currentFile)}
                className="w-full sm:w-auto min-h-[44px] px-6"
              >
                Analyze Image
              </Button>
            </div>
          </div>
        </div>
      );
    }

    if (analysisState === 'ANALYZING') {
      return (
        <div className="flex-1 flex flex-col items-center justify-center p-8 sm:p-12 bg-slate-50 dark:bg-[#090D1A] border border-slate-200 dark:border-[#1E294B] rounded-2xl gap-4 w-full max-w-full min-h-[300px]">
          <div className="relative flex items-center justify-center w-16 h-16">
            <div className="absolute inset-0 rounded-full border-2 border-[#6C63FF]/30 border-t-[#6C63FF] animate-spin"></div>
            <span className="material-symbols-outlined text-[24px] text-[#22D3EE] animate-pulse">
              center_focus_strong
            </span>
          </div>
          <div className="flex flex-col items-center gap-1 text-center">
            <span className="text-sm font-semibold text-slate-800 dark:text-[#F8FAFC]">
              {loadingStage}
            </span>
            <span className="text-xs text-slate-400 dark:text-[#64748B] font-mono">
              Model: {modelInfo?.model_name || 'ResidualEmotionCNN'}
            </span>
          </div>
        </div>
      );
    }

    if (analysisState === 'NO_FACE') {
      return (
        <div className="flex-1 flex flex-col items-center justify-center gap-4 p-3 sm:p-6 bg-slate-50 dark:bg-[#090D1A] border border-slate-200 dark:border-[#1E294B] rounded-2xl w-full max-w-full">
          {image && (
            <div className="max-w-md w-full opacity-70">
              <ImageViewer
                imageSrc={image.src}
                faces={[]}
                selectedFaceId={null}
                onSelectFace={() => {}}
                zoom={zoom}
                showBoxes={false}
                showConfidence={false}
                onImageLoad={handleImageDimensions}
              />
            </div>
          )}
          <div className="p-6 rounded-xl bg-amber-500/10 border border-amber-500/30 text-center flex flex-col items-center gap-3 max-w-md w-full">
            <div className="w-12 h-12 rounded-full bg-amber-500/20 text-amber-600 dark:text-amber-400 flex items-center justify-center">
              <span className="material-symbols-outlined text-[28px]">sentiment_neutral</span>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-sm font-bold text-amber-700 dark:text-amber-400">
                No Faces Detected
              </span>
              <p className="text-xs text-slate-600 dark:text-[#94A3B8] leading-relaxed">
                FaceSense scanned the image with Haar Cascade localization but found no recognizable facial contours. Try an image with clearer lighting and frontal orientation.
              </p>
            </div>
            <button
              type="button"
              onClick={handleResetToEmpty}
              className="mt-1 px-4 py-2 min-h-[44px] rounded-lg bg-[#6C63FF] text-white text-xs font-medium hover:bg-[#5B52EE] transition-colors"
            >
              Select Another Image
            </button>
          </div>
        </div>
      );
    }

    if (analysisState === 'ERROR') {
      return (
        <div className="flex-1 flex flex-col items-center justify-center p-8 sm:p-12 bg-rose-500/5 border border-rose-500/20 rounded-2xl gap-4 w-full max-w-full min-h-[300px]">
          <div className="w-14 h-14 rounded-full bg-rose-500/10 text-rose-500 flex items-center justify-center">
            <span className="material-symbols-outlined text-[28px]">warning</span>
          </div>
          <div className="flex flex-col items-center gap-1 text-center max-w-md">
            <span className="text-sm font-semibold text-rose-700 dark:text-rose-400">
              Inference Processing Failed
            </span>
            <p className="text-xs text-slate-600 dark:text-[#94A3B8]">
              {errorMessage}
            </p>
          </div>
          <div className="flex flex-wrap items-center justify-center gap-2 mt-2">
            <button
              type="button"
              onClick={handleRerun}
              className="px-4 py-2 min-h-[44px] rounded-lg bg-[#6C63FF] text-white text-xs font-medium hover:bg-[#5B52EE] transition-colors"
            >
              Retry Inference
            </button>
            <button
              type="button"
              onClick={handleResetToEmpty}
              className="px-4 py-2 min-h-[44px] rounded-lg bg-slate-100 dark:bg-[#171F36] text-slate-700 dark:text-[#94A3B8] text-xs font-medium hover:bg-slate-200 transition-colors"
            >
              Upload Different Image
            </button>
          </div>
        </div>
      );
    }

    // RESULTS state
    return (
      <ImageViewer
        imageSrc={image?.src}
        faces={faces}
        selectedFaceId={selectedFaceId}
        onSelectFace={setSelectedFaceId}
        zoom={zoom}
        showBoxes={showBoxes}
        showConfidence={showConfidence}
        onImageLoad={handleImageDimensions}
      />
    );
  };

  const renderDetectionResultsHeader = () => {
    return (
      <div className="flex flex-col gap-3 w-full min-w-0">
        <div className="flex items-center justify-between pb-2 border-b border-slate-200 dark:border-[#1E294B] w-full min-w-0">
          <div className="flex items-center gap-2 min-w-0">
            <h2 className="font-semibold text-sm text-slate-800 dark:text-[#F8FAFC]">
              Detection Results
            </h2>
            {faces.length > 0 && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-[#6C63FF]/10 text-[#6C63FF] border border-[#6C63FF]/30 font-medium shrink-0">
                {faces.length} {faces.length === 1 ? 'face' : 'faces'}
              </span>
            )}
          </div>
          <span className="material-symbols-outlined text-[18px] text-slate-400">tune</span>
        </div>

        {/* Real Confidence Summary */}
        <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] flex items-center justify-between text-xs w-full min-w-0">
          <div className="flex items-center gap-2 min-w-0">
            <span className="material-symbols-outlined text-[16px] text-[#22D3EE] shrink-0">verified</span>
            <span className="text-slate-600 dark:text-[#94A3B8] truncate">
              Mean Confidence:{' '}
              <strong className="text-slate-900 dark:text-[#F8FAFC]">
                {meanConfidence ? `${meanConfidence}%` : 'N/A'}
              </strong>
            </span>
          </div>
          <span className="font-mono text-[10px] text-slate-400 dark:text-[#64748B] shrink-0">
            ACTIVE
          </span>
        </div>
      </div>
    );
  };

  const renderSubjectTabs = () => {
    if (faces.length <= 1) return null;

    const colsCount = Math.min(faces.length + 1, 4);
    const colsClass = colsCount === 2 ? 'grid-cols-2' : colsCount === 3 ? 'grid-cols-3' : 'grid-cols-2 sm:grid-cols-4';

    return (
      <div className={`grid ${colsClass} gap-1 bg-slate-100 dark:bg-[#0B1020] p-1 rounded-lg border border-slate-200 dark:border-[#1E294B] text-xs w-full min-w-0`}>
        <button
          type="button"
          onClick={() => setSelectedFaceId(null)}
          className={`py-1.5 min-h-[36px] rounded text-center transition-colors truncate cursor-pointer ${
            selectedFaceId === null
              ? "bg-white dark:bg-[#171F36] text-slate-900 dark:text-[#F8FAFC] font-medium shadow-sm"
              : "text-slate-500 dark:text-[#64748B] hover:text-slate-900 dark:hover:text-[#F8FAFC]"
          }`}
        >
          All ({faces.length})
        </button>
        {faces.map((f) => {
          const isSubmitted = faceFeedback[f.id]?.status === 'submitted';
          return (
            <button
              key={f.id}
              type="button"
              id={`tab-face-${f.id}`}
              onClick={() => setSelectedFaceId(f.id)}
              className={`py-1.5 min-h-[36px] rounded text-center transition-colors flex items-center justify-center gap-1 min-w-0 truncate cursor-pointer ${
                selectedFaceId === f.id
                  ? "bg-white dark:bg-[#171F36] text-slate-900 dark:text-[#F8FAFC] font-medium shadow-sm border border-[#6C63FF]/40"
                  : "text-slate-500 dark:text-[#64748B] hover:text-slate-900 dark:hover:text-[#F8FAFC]"
              }`}
            >
              {isSubmitted ? (
                <span className="material-symbols-outlined text-[14px] text-emerald-500 shrink-0">check_circle</span>
              ) : (
                <span className="w-1.5 h-1.5 rounded-full bg-[#22D3EE] shrink-0"></span>
              )}
              <span className="truncate">{f.label}</span>
            </button>
          );
        })}
      </div>
    );
  };

  const renderResultCards = () => {
    if (faces.length === 0) return null;

    if (selectedFaceId === null) {
      return (
        <div className="flex flex-col gap-3 w-full min-w-0">
          {faces.map((face) => (
            <ResultCard
              key={face.id}
              face={face}
              isSelected={false}
              onSelect={setSelectedFaceId}
              feedbackState={faceFeedback[face.id]}
            />
          ))}
        </div>
      );
    }

    if (selectedFace) {
      return (
        <div className="w-full min-w-0">
          <ResultCard
            face={selectedFace}
            isSelected={true}
            onSelect={setSelectedFaceId}
            feedbackState={faceFeedback[selectedFace.id]}
          />
        </div>
      );
    }

    return null;
  };

  const renderFeedback = () => {
    if (faces.length === 0) return null;

    // When "All faces" is active, show the selector for independent face review
    if (selectedFaceId === null) {
      return (
        <div className="bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] rounded-xl p-4 flex flex-col gap-2.5 shadow-sm text-left w-full max-w-full overflow-hidden">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold text-slate-500 dark:text-[#94A3B8] uppercase tracking-wider">
              Review Accuracy
            </span>
            <span className="material-symbols-outlined text-[16px] text-slate-400 dark:text-[#64748B]">
              rate_review
            </span>
          </div>
          <p className="text-xs text-slate-600 dark:text-[#94A3B8]">
            Viewing all {faces.length} detected faces. Select an individual face to submit accuracy feedback independently:
          </p>
          <div className="flex flex-wrap gap-2 pt-1">
            {faces.map((f) => {
              const isRecorded = faceFeedback[f.id]?.status === 'submitted';
              return (
                <button
                  key={f.id}
                  type="button"
                  id={`btn-select-face-review-${f.id}`}
                  onClick={() => setSelectedFaceId(f.id)}
                  className={`flex items-center gap-1.5 px-3 py-2 min-h-[44px] rounded-lg border text-xs font-medium transition-colors cursor-pointer ${
                    isRecorded
                      ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                      : 'border-slate-200 dark:border-[#1E294B] bg-white dark:bg-[#0B1020] text-slate-700 dark:text-[#F8FAFC] hover:border-[#6C63FF]'
                  }`}
                >
                  <span className={`material-symbols-outlined text-[16px] ${isRecorded ? 'text-emerald-500' : 'text-[#6C63FF]'}`}>
                    {isRecorded ? 'check_circle' : 'face'}
                  </span>
                  <span>Review {f.label} {isRecorded ? '✓' : ''}</span>
                </button>
              );
            })}
          </div>
        </div>
      );
    }

    const face = faces.find((f) => f.id === selectedFaceId);
    if (!face) return null;

    return (
      <div className="w-full min-w-0">
        <FeedbackPanel
          key={face.id}
          selectedFace={face}
          feedbackState={faceFeedback[face.id]}
          onFeedbackSubmit={handleFeedbackSubmit}
          onResetFeedback={handleResetFaceFeedback}
          isSubmitting={submittingFaceId === face.id}
          modelVersion={modelInfo?.model_name || 'ResidualEmotionCNN-Candidate-epoch39'}
        />
      </div>
    );
  };

  const renderPipelineBar = () => {
    return (
      <div className="bg-white dark:bg-[#12182B] border border-slate-200 dark:border-[#1E294B] rounded-xl p-3 flex flex-col md:flex-row items-start md:items-center justify-between gap-3 text-xs text-slate-500 dark:text-[#94A3B8] shrink-0 w-full max-w-full overflow-hidden">
        <div className="flex flex-wrap items-center gap-2 font-mono text-[11px]">
          <span className="uppercase tracking-wider font-semibold text-slate-800 dark:text-[#F8FAFC] flex items-center gap-1">
            <span className="material-symbols-outlined text-[14px] text-[#22D3EE]">check_circle</span>
            Pipeline
          </span>
          <div className="h-3 w-px bg-slate-200 dark:bg-[#1E294B]"></div>
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-50 dark:bg-[#171F36] text-slate-700 dark:text-[#F8FAFC] border border-slate-200 dark:border-[#1E294B]">
            <span className="text-[#0891B2] dark:text-[#22D3EE] font-bold">1.</span>
            <span>Validation</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-50 dark:bg-[#171F36] text-slate-700 dark:text-[#F8FAFC] border border-slate-200 dark:border-[#1E294B]">
            <span className="text-[#0891B2] dark:text-[#22D3EE] font-bold">2.</span>
            <span>Detection</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-50 dark:bg-[#171F36] text-slate-700 dark:text-[#F8FAFC] border border-slate-200 dark:border-[#1E294B]">
            <span className="text-[#0891B2] dark:text-[#22D3EE] font-bold">3.</span>
            <span>Classification</span>
          </div>
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px] text-slate-500 dark:text-[#64748B] shrink-0">
          {processingTimeMs > 0 && (
            <>
              <span className="text-slate-800 dark:text-[#F8FAFC] font-medium">Latency: ~{processingTimeMs}ms</span>
              <span>•</span>
            </>
          )}
          <span className="px-2 py-0.5 rounded bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] text-slate-600 dark:text-[#94A3B8] truncate max-w-[200px]">
            {modelInfo?.model_name || 'ResidualEmotionCNN'}
          </span>
        </div>
      </div>
    );
  };

  const renderResultsDockContent = () => {
    if (analysisState === 'EMPTY') {
      return (
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center border border-dashed border-slate-200 dark:border-[#1E294B] rounded-xl gap-3 text-slate-400 dark:text-[#64748B]">
          <div className="w-12 h-12 rounded-xl bg-slate-100 dark:bg-[#171F36] flex items-center justify-center text-slate-400 dark:text-[#94A3B8]">
            <span className="material-symbols-outlined text-[24px]">analytics</span>
          </div>
          <div className="flex flex-col gap-1 max-w-xs">
            <span className="text-sm font-semibold text-slate-700 dark:text-[#F8FAFC]">
              No analysis yet
            </span>
            <p className="text-xs leading-relaxed">
              Upload an image and run analysis to see detected faces, expression classification, confidence, and probabilities.
            </p>
          </div>
        </div>
      );
    }

    if (analysisState === 'IMAGE_SELECTED') {
      return (
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center border border-slate-200 dark:border-[#1E294B] rounded-xl gap-3 text-slate-500 dark:text-[#94A3B8]">
          <div className="w-12 h-12 rounded-xl bg-[#6C63FF]/10 flex items-center justify-center text-[#6C63FF]">
            <span className="material-symbols-outlined text-[24px]">pending_actions</span>
          </div>
          <div className="flex flex-col gap-1 max-w-xs">
            <span className="text-sm font-semibold text-slate-800 dark:text-[#F8FAFC]">
              Ready for analysis
            </span>
            <p className="text-xs leading-relaxed">
              Image loaded: <strong className="text-slate-800 dark:text-[#F8FAFC]">{image?.name}</strong>. Click 'Analyze Image' to detect faces and calculate emotion probabilities.
            </p>
          </div>
          <Button
            variant="primary"
            size="sm"
            icon="bolt"
            onClick={() => executePrediction(currentFile)}
            className="mt-2 min-h-[44px] px-4"
          >
            Analyze Image
          </Button>
        </div>
      );
    }

    if (analysisState === 'ANALYZING') {
      return (
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center border border-slate-200 dark:border-[#1E294B] rounded-xl gap-3">
          <div className="w-10 h-10 rounded-full border-2 border-[#6C63FF]/30 border-t-[#6C63FF] animate-spin"></div>
          <span className="text-xs font-mono text-slate-500 dark:text-[#94A3B8]">
            Evaluating facial telemetry...
          </span>
        </div>
      );
    }

    if (analysisState === 'NO_FACE') {
      return (
        <div className="p-6 text-center border border-dashed border-amber-500/30 rounded-xl text-xs text-slate-500 dark:text-[#94A3B8] flex flex-col items-center gap-2">
          <span className="material-symbols-outlined text-[24px] text-amber-500">face_retouching_off</span>
          <span className="font-semibold text-slate-800 dark:text-[#F8FAFC]">No subjects localized</span>
          <p className="text-[11px] leading-relaxed">
            No face bounding boxes were found in this image. Upload an image containing faces to inspect emotion classification telemetry.
          </p>
        </div>
      );
    }

    if (analysisState === 'ERROR') {
      return (
        <div className="p-6 text-center border border-dashed border-rose-500/30 rounded-xl text-xs text-slate-500 dark:text-[#94A3B8] flex flex-col items-center gap-2">
          <span className="material-symbols-outlined text-[24px] text-rose-500">error</span>
          <span className="font-semibold text-slate-800 dark:text-[#F8FAFC]">Analysis Failed</span>
          <p className="text-[11px] leading-relaxed">
            {errorMessage || 'An error occurred during vision inference.'}
          </p>
        </div>
      );
    }

    // RESULTS state
    return (
      <>
        {renderDetectionResultsHeader()}
        {renderSubjectTabs()}
        {renderResultCards()}
        {renderFeedback()}
      </>
    );
  };

  return (
    <div className="w-full max-w-full min-w-0 bg-[#F7F7FC] dark:bg-[#0B1020]">
      {/* Toast notifications */}
      <Toast message={toastMessage} onClose={() => setToastMessage(null)} />

      {/* ========================================================================= */}
      {/* DESKTOP WORKSPACE (lg: and above, >= 1024px)                             */}
      {/* Preserves the exact approved two-pane workspace with separate side dock. */}
      {/* ========================================================================= */}
      <div className="hidden lg:flex lg:flex-row h-full w-full min-w-0 overflow-hidden">
        {/* Left Canvas Pane */}
        <section className="flex-1 flex flex-col p-6 overflow-y-auto min-w-0 gap-4">
          {renderControlBar(false)}
          <div className="flex-1 flex flex-col min-h-0 min-w-0">
            {renderCanvas()}
          </div>
          {renderPipelineBar()}
        </section>

        {/* Right Inspector Dock */}
        <aside className="w-96 bg-white dark:bg-[#12182B] border-l border-slate-200 dark:border-[#1E294B] flex flex-col p-5 overflow-y-auto shrink-0 gap-4 min-w-0">
          {renderResultsDockContent()}
        </aside>
      </div>

      {/* ========================================================================= */}
      {/* MOBILE & TABLET WORKSPACE (< lg, < 1024px: 375px, 390px, 430px, 768px, 834px) */}
      {/* Genuine single-column workflow with natural page scrolling & zero traps.  */}
      {/* ========================================================================= */}
      <div className="lg:hidden flex flex-col w-full min-w-0 p-3 sm:p-5 gap-4 pb-28 sm:pb-32">
        {analysisState === 'RESULTS' ? (
          /* Exact Mobile RESULTS Order:
             1. Image / Preview with Face Bounding Boxes
             2. Analysis controls
             3. Detection Results (Header & Mean Confidence)
             4. Selected Face / Face 01 / Face 02 selector
             5. Result Card
             6. Feedback
             7. Pipeline / processing information
          */
          <>
            {/* 1. Image / Preview with Bounding Boxes */}
            <div className="w-full min-w-0">
              {renderCanvas()}
            </div>

            {/* 2. Analysis controls */}
            <div className="w-full min-w-0">
              {renderControlBar(true)}
            </div>

            {/* 3. Detection Results (Header & Mean Confidence) */}
            <div className="w-full min-w-0 bg-white dark:bg-[#12182B] border border-slate-200 dark:border-[#1E294B] rounded-xl p-4 shadow-sm">
              {renderDetectionResultsHeader()}
            </div>

            {/* 4. Selected Face / Face 01 / Face 02 selector */}
            {faces.length > 1 && (
              <div className="w-full min-w-0">
                {renderSubjectTabs()}
              </div>
            )}

            {/* 5. Result Card */}
            <div className="w-full min-w-0">
              {renderResultCards()}
            </div>

            {/* 6. Feedback */}
            {faces.length > 0 && (
              <div className="w-full min-w-0">
                {renderFeedback()}
              </div>
            )}

            {/* 7. Pipeline / processing information */}
            <div className="w-full min-w-0 mt-1">
              {renderPipelineBar()}
            </div>
          </>
        ) : analysisState === 'EMPTY' ? (
          /* Mobile EMPTY State */
          <div className="flex flex-col gap-4 w-full min-w-0">
            {renderCanvas()}
            <div className="p-6 text-center border border-dashed border-slate-200 dark:border-[#1E294B] rounded-xl flex flex-col items-center gap-2 bg-white dark:bg-[#12182B] text-slate-400 dark:text-[#64748B]">
              <span className="material-symbols-outlined text-[24px]">analytics</span>
              <span className="text-sm font-semibold text-slate-700 dark:text-[#F8FAFC]">No analysis yet</span>
              <p className="text-xs leading-relaxed max-w-xs">
                Upload an image and run analysis to see detected faces, expression classification, confidence, and probabilities.
              </p>
            </div>
          </div>
        ) : analysisState === 'IMAGE_SELECTED' ? (
          /* Mobile IMAGE_SELECTED State */
          <div className="flex flex-col gap-4 w-full min-w-0">
            {renderCanvas()}
            {renderControlBar(true)}
            <div className="p-4 rounded-xl bg-white dark:bg-[#12182B] border border-slate-200 dark:border-[#1E294B] text-center flex flex-col items-center gap-2">
              <span className="text-xs font-semibold text-slate-800 dark:text-[#F8FAFC]">Ready for analysis</span>
              <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
                Image loaded: {image?.name}. Tap 'Analyze Image' to begin inference.
              </p>
              <Button
                variant="primary"
                size="md"
                icon="bolt"
                onClick={() => executePrediction(currentFile)}
                className="w-full min-h-[44px] mt-1"
              >
                Analyze Image
              </Button>
            </div>
          </div>
        ) : analysisState === 'ANALYZING' ? (
          /* Mobile ANALYZING State */
          <div className="flex flex-col gap-4 w-full min-w-0">
            {renderCanvas()}
          </div>
        ) : analysisState === 'NO_FACE' ? (
          /* Mobile NO_FACE State */
          <div className="flex flex-col gap-4 w-full min-w-0">
            {renderCanvas()}
          </div>
        ) : (
          /* Mobile ERROR State */
          <div className="flex flex-col gap-4 w-full min-w-0">
            {renderCanvas()}
          </div>
        )}
      </div>

      {/* Raw JSON Modal */}
      <Modal
        isOpen={jsonModalOpen}
        onClose={() => setJsonModalOpen(false)}
        title="Raw Detection Payload"
      >
        <div className="flex flex-col gap-3 text-left">
          <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
            Direct response payload from FastAPI endpoint <code className="font-mono text-[#6C63FF]">/api/v1/predict/image</code>:
          </p>
          <pre className="p-3 rounded-lg bg-slate-900 text-emerald-400 font-mono text-[11px] overflow-x-auto max-h-96 border border-slate-800">
            {JSON.stringify(rawResponse || { faces_detected: faces.length, predictions: faces }, null, 2)}
          </pre>
        </div>
      </Modal>
    </div>
  );
}
