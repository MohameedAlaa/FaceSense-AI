import React, { useState } from 'react';

export default function FeedbackPanel({
  selectedFace,
  onFeedbackSubmit,
}) {
  const [selectedState, setSelectedState] = useState(null);
  const [groundTruth, setGroundTruth] = useState(null);
  const [submitted, setSubmitted] = useState(false);

  if (!selectedFace) return null;

  const { id, label, predicted_emotion, confidence } = selectedFace;
  const emotions = ['happy', 'neutral', 'surprise', 'sad', 'fear', 'disgust', 'angry'];

  const handleVote = (state) => {
    setSelectedState(state);
    if (state === 'correct') {
      setSubmitted(true);
      if (onFeedbackSubmit) onFeedbackSubmit({ faceId: id, state: 'correct' });
    } else if (state === 'uncertain') {
      setSubmitted(true);
      if (onFeedbackSubmit) onFeedbackSubmit({ faceId: id, state: 'uncertain' });
    }
  };

  const handleGroundTruthSelect = (emotion) => {
    setGroundTruth(emotion);
    setSubmitted(true);
    if (onFeedbackSubmit) {
      onFeedbackSubmit({
        faceId: id,
        state: 'incorrect',
        correctedEmotion: emotion,
      });
    }
  };

  const handleReset = () => {
    setSelectedState(null);
    setGroundTruth(null);
    setSubmitted(false);
  };

  return (
    <div className="bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] rounded-xl p-4 flex flex-col gap-3 shadow-sm text-left">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold text-slate-500 dark:text-[#94A3B8] uppercase tracking-wider">
          Review Accuracy
        </span>
        <span className="material-symbols-outlined text-[14px] text-slate-400 dark:text-[#64748B]">
          rate_review
        </span>
      </div>

      {submitted ? (
        <div className="flex flex-col gap-2 p-3 rounded-lg bg-[#6C63FF]/10 border border-[#6C63FF]/30 text-xs">
          <div className="flex items-center gap-2 text-[#6C63FF] font-medium">
            <span className="material-symbols-outlined text-[16px]">check_circle</span>
            <span>Feedback recorded for {label}!</span>
          </div>
          <p className="text-[11px] text-slate-500 dark:text-[#94A3B8]">
            {selectedState === 'correct' && 'Verified as accurate.'}
            {selectedState === 'uncertain' && 'Flagged as uncertain for review.'}
            {selectedState === 'incorrect' && `Corrected label set to ${groundTruth}.`}
          </p>
          <button
            type="button"
            onClick={handleReset}
            className="self-start text-[11px] text-[#6C63FF] underline hover:text-[#5B52EE] mt-1"
          >
            Change feedback
          </button>
        </div>
      ) : (
        <>
          <p className="text-xs text-slate-800 dark:text-[#F8FAFC]">
            Was the expression classification accurate for <strong>{label}</strong> ({predicted_emotion} • {(confidence * 100).toFixed(1)}%)?
          </p>

          {/* Vote Buttons */}
          <div className="grid grid-cols-3 gap-2">
            <button
              type="button"
              onClick={() => handleVote('correct')}
              className={`flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-white dark:bg-[#0B1020] border text-xs font-medium transition-colors ${
                selectedState === 'correct'
                  ? 'border-[#22D3EE] text-[#0891B2] dark:text-[#22D3EE] bg-[#22D3EE]/10'
                  : 'border-slate-200 dark:border-[#1E294B] text-slate-700 dark:text-[#F8FAFC] hover:border-[#22D3EE] hover:text-[#0891B2] dark:hover:text-[#22D3EE]'
              }`}
            >
              <span className="material-symbols-outlined text-[14px] text-[#22D3EE]">check</span>
              <span>Yes</span>
            </button>

            <button
              type="button"
              onClick={() => setSelectedState('incorrect')}
              className={`flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-white dark:bg-[#0B1020] border text-xs font-medium transition-colors ${
                selectedState === 'incorrect'
                  ? 'border-rose-400 text-rose-500 bg-rose-500/10'
                  : 'border-slate-200 dark:border-[#1E294B] text-slate-700 dark:text-[#F8FAFC] hover:border-rose-400 hover:text-rose-500'
              }`}
            >
              <span className="material-symbols-outlined text-[14px] text-rose-400">close</span>
              <span>No</span>
            </button>

            <button
              type="button"
              onClick={() => handleVote('uncertain')}
              className={`flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-white dark:bg-[#0B1020] border text-xs font-medium transition-colors ${
                selectedState === 'uncertain'
                  ? 'border-[#8B5CF6] text-[#8B5CF6] bg-[#8B5CF6]/10'
                  : 'border-slate-200 dark:border-[#1E294B] text-slate-700 dark:text-[#F8FAFC] hover:border-[#8B5CF6] hover:text-[#8B5CF6]'
              }`}
            >
              <span className="material-symbols-outlined text-[14px] text-[#8B5CF6]">help</span>
              <span>Uncertain</span>
            </button>
          </div>

          {/* If 'No' clicked, reveal ground-truth assign pills */}
          {selectedState === 'incorrect' && (
            <div className="flex flex-col gap-1.5 pt-2 border-t border-slate-200 dark:border-[#1E294B] animate-fade-in">
              <span className="text-[10px] text-slate-500 dark:text-[#64748B] font-mono">
                What emotion would you select?
              </span>
              <div className="flex flex-wrap gap-1">
                {emotions.map((emotion) => (
                  <button
                    key={emotion}
                    type="button"
                    onClick={() => handleGroundTruthSelect(emotion)}
                    className="px-2 py-0.5 rounded text-[11px] capitalize bg-white dark:bg-[#0B1020] border border-slate-200 dark:border-[#1E294B] text-slate-700 dark:text-[#94A3B8] hover:text-slate-900 dark:hover:text-[#F8FAFC] hover:border-[#6C63FF] transition-colors"
                  >
                    {emotion}
                  </button>
                ))}
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
