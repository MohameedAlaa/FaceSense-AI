import React, { useState } from 'react';

const SUPPORTED_EMOTIONS = [
  'angry',
  'disgust',
  'fear',
  'happy',
  'neutral',
  'sad',
  'surprise',
];

export default function FeedbackPanel({
  selectedFace,
  feedbackState,
  onFeedbackSubmit,
  onResetFeedback,
  isSubmitting = false,
  modelVersion = 'ResidualEmotionCNN-Candidate-epoch39',
}) {
  const [isCorrecting, setIsCorrecting] = useState(false);
  const [selectedCorrection, setSelectedCorrection] = useState(null);

  if (!selectedFace) return null;

  const { id, label, face_index, predicted_emotion, confidence, pixelBox } = selectedFace;
  const isSubmitted = feedbackState?.status === 'submitted';

  const handleVote = async (feedbackType) => {
    if (feedbackType === 'incorrect') {
      setIsCorrecting(true);
      setSelectedCorrection(null);
      return;
    }

    if (onFeedbackSubmit) {
      const faceIdx = typeof face_index === 'number' ? face_index : (parseInt(id, 10) - 1);
      await onFeedbackSubmit({
        faceId: id,
        faceLabel: label,
        face_index: faceIdx,
        feedback_type: feedbackType,
        predicted_emotion,
        confidence,
        bounding_box: pixelBox,
        model_version: modelVersion,
      });
    }
  };

  const handleConfirmCorrection = async () => {
    if (!selectedCorrection) return;

    if (onFeedbackSubmit) {
      const faceIdx = typeof face_index === 'number' ? face_index : (parseInt(id, 10) - 1);
      await onFeedbackSubmit({
        faceId: id,
        faceLabel: label,
        face_index: faceIdx,
        feedback_type: 'incorrect',
        predicted_emotion,
        confidence,
        corrected_emotion: selectedCorrection,
        bounding_box: pixelBox,
        model_version: modelVersion,
      });
    }
    setIsCorrecting(false);
  };

  const handleCancelCorrection = () => {
    setIsCorrecting(false);
    setSelectedCorrection(null);
  };

  const handleReset = () => {
    setIsCorrecting(false);
    setSelectedCorrection(null);
    if (onResetFeedback) {
      onResetFeedback(id);
    }
  };

  return (
    <div className="bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] rounded-xl p-4 flex flex-col gap-3 shadow-sm text-left w-full max-w-full overflow-hidden">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold text-slate-500 dark:text-[#94A3B8] uppercase tracking-wider">
          Review Accuracy • {label}
        </span>
        <span className="material-symbols-outlined text-[16px] text-slate-400 dark:text-[#64748B]">
          rate_review
        </span>
      </div>

      {isSubmitted ? (
        <div className="flex flex-col gap-2 p-3.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-xs">
          <div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-medium">
            <span className="material-symbols-outlined text-[18px]">check_circle</span>
            <span className="font-semibold">Feedback recorded for {label}</span>
          </div>
          <p className="text-[11px] text-slate-600 dark:text-[#94A3B8]">
            {feedbackState.feedback_type === 'correct' && (
              <>Verified as accurate (<span className="capitalize font-medium text-slate-800 dark:text-[#F8FAFC]">{predicted_emotion}</span>).</>
            )}
            {feedbackState.feedback_type === 'uncertain' && (
              'Flagged as uncertain for review.'
            )}
            {feedbackState.feedback_type === 'incorrect' && (
              <>Corrected label set to <strong className="capitalize text-slate-800 dark:text-[#F8FAFC]">{feedbackState.corrected_emotion}</strong>.</>
            )}
          </p>
          <button
            type="button"
            onClick={handleReset}
            className="self-start text-[11px] text-[#6C63FF] hover:text-[#5B52EE] underline font-medium mt-1 min-h-[36px] flex items-center"
          >
            Change feedback
          </button>
        </div>
      ) : (
        <>
          <p className="text-xs text-slate-800 dark:text-[#F8FAFC]">
            Was the expression classification accurate for <strong>{label}</strong> (<span className="capitalize">{predicted_emotion}</span> • {(confidence * 100).toFixed(1)}%)?
          </p>

          {!isCorrecting ? (
            /* Vote Buttons: Correct / Incorrect / Uncertain */
            <div className="grid grid-cols-3 gap-2 w-full">
              <button
                type="button"
                id={`btn-feedback-correct-${id}`}
                disabled={isSubmitting}
                onClick={() => handleVote('correct')}
                className="flex items-center justify-center gap-1.5 py-2.5 px-2 min-h-[44px] rounded-lg bg-white dark:bg-[#0B1020] border border-slate-200 dark:border-[#1E294B] text-slate-700 dark:text-[#F8FAFC] text-xs font-medium hover:border-[#22D3EE] hover:text-[#0891B2] dark:hover:text-[#22D3EE] transition-colors cursor-pointer"
              >
                <span className="material-symbols-outlined text-[16px] text-[#22D3EE]">check</span>
                <span>Correct</span>
              </button>

              <button
                type="button"
                id={`btn-feedback-incorrect-${id}`}
                disabled={isSubmitting}
                onClick={() => handleVote('incorrect')}
                className="flex items-center justify-center gap-1.5 py-2.5 px-2 min-h-[44px] rounded-lg bg-white dark:bg-[#0B1020] border border-slate-200 dark:border-[#1E294B] text-slate-700 dark:text-[#F8FAFC] text-xs font-medium hover:border-rose-400 hover:text-rose-500 transition-colors cursor-pointer"
              >
                <span className="material-symbols-outlined text-[16px] text-rose-400">close</span>
                <span>Incorrect</span>
              </button>

              <button
                type="button"
                id={`btn-feedback-uncertain-${id}`}
                disabled={isSubmitting}
                onClick={() => handleVote('uncertain')}
                className="flex items-center justify-center gap-1.5 py-2.5 px-2 min-h-[44px] rounded-lg bg-white dark:bg-[#0B1020] border border-slate-200 dark:border-[#1E294B] text-slate-700 dark:text-[#F8FAFC] text-xs font-medium hover:border-[#8B5CF6] hover:text-[#8B5CF6] transition-colors cursor-pointer"
              >
                <span className="material-symbols-outlined text-[16px] text-[#8B5CF6]">help</span>
                <span>Uncertain</span>
              </button>
            </div>
          ) : (
            /* Correction Flow: Require a corrected emotion before submitting incorrect feedback */
            <div className="flex flex-col gap-2.5 pt-2 border-t border-slate-200 dark:border-[#1E294B] w-full">
              <div className="flex items-center justify-between">
                <span className="text-[11px] text-slate-700 dark:text-[#94A3B8] font-medium">
                  Select corrected emotion for {label}:
                </span>
                <span className="text-[10px] text-rose-500 font-mono">
                  Required
                </span>
              </div>

              {/* 7 Supported Emotions */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 w-full">
                {SUPPORTED_EMOTIONS.map((emotion) => {
                  const isChosen = selectedCorrection === emotion;
                  return (
                    <button
                      key={emotion}
                      type="button"
                      id={`btn-emotion-${emotion}-${id}`}
                      disabled={isSubmitting}
                      onClick={() => setSelectedCorrection(emotion)}
                      className={`min-h-[44px] px-2.5 py-2 rounded-lg text-xs capitalize font-medium transition-colors flex items-center justify-center gap-1 border cursor-pointer ${
                        isChosen
                          ? 'border-[#6C63FF] bg-[#6C63FF] text-white shadow-sm'
                          : 'bg-white dark:bg-[#0B1020] border-slate-200 dark:border-[#1E294B] text-slate-700 dark:text-[#94A3B8] hover:border-[#6C63FF] hover:text-slate-900 dark:hover:text-[#F8FAFC]'
                      }`}
                    >
                      {isChosen && (
                        <span className="material-symbols-outlined text-[14px]">check</span>
                      )}
                      <span>{emotion}</span>
                    </button>
                  );
                })}
              </div>

              {/* Actions: Submit Correction (strictly disabled until emotion selected) & Cancel */}
              <div className="flex items-center gap-2 pt-1">
                <button
                  type="button"
                  id={`btn-submit-correction-${id}`}
                  disabled={!selectedCorrection || isSubmitting}
                  onClick={handleConfirmCorrection}
                  className={`flex-1 min-h-[44px] py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                    selectedCorrection && !isSubmitting
                      ? 'bg-[#6C63FF] text-white hover:bg-[#5B52EE] shadow-sm cursor-pointer'
                      : 'bg-slate-200 dark:bg-[#1E294B] text-slate-400 dark:text-[#64748B] cursor-not-allowed'
                  }`}
                >
                  <span className="material-symbols-outlined text-[16px]">done</span>
                  <span>
                    {selectedCorrection ? `Submit as ${selectedCorrection}` : 'Select an emotion above'}
                  </span>
                </button>

                <button
                  type="button"
                  disabled={isSubmitting}
                  onClick={handleCancelCorrection}
                  className="min-h-[44px] px-3 py-2 rounded-lg border border-slate-200 dark:border-[#1E294B] bg-white dark:bg-[#0B1020] text-slate-600 dark:text-[#94A3B8] text-xs font-medium hover:bg-slate-100 dark:hover:bg-[#171F36] transition-colors cursor-pointer"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          <p className="text-[10px] text-slate-400 dark:text-[#64748B] leading-tight">
            Feedback refines dataset calibration and continuous learning under privacy-first standards.
          </p>
        </>
      )}
    </div>
  );
}
