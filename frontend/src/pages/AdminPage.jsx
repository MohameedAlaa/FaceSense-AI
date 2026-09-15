import React, { useState, useEffect, useCallback } from 'react';
import Card from '../components/common/Card';
import Badge from '../components/common/Badge';
import Button from '../components/common/Button';
import { useAuth } from '../hooks/useAuth';
import { feedbackApi } from '../lib/api/feedback';
import { predictApi } from '../lib/api/predict';

const FER2013_EMOTIONS = [
  'angry',
  'disgust',
  'fear',
  'happy',
  'neutral',
  'sad',
  'surprise',
];

function AdminFeedbackImage({ imageUrl, alt = 'Face crop' }) {
  const [blobUrl, setBlobUrl] = useState(null);
  const [hasError, setHasError] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!imageUrl) {
      setBlobUrl(null);
      setHasError(false);
      setIsLoading(false);
      return;
    }

    let isMounted = true;
    let objectUrl = null;

    setIsLoading(true);
    setHasError(false);

    feedbackApi
      .getFeedbackImageBlob(imageUrl)
      .then((blob) => {
        if (!isMounted) return;
        objectUrl = URL.createObjectURL(blob);
        setBlobUrl(objectUrl);
        setIsLoading(false);
      })
      .catch(() => {
        if (!isMounted) return;
        setHasError(true);
        setIsLoading(false);
      });

    return () => {
      isMounted = false;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [imageUrl]);

  if (!imageUrl || hasError) {
    return (
      <span className="material-symbols-outlined text-[24px] text-slate-400">face</span>
    );
  }

  if (isLoading && !blobUrl) {
    return (
      <span className="material-symbols-outlined text-[20px] text-slate-400 animate-spin">
        progress_activity
      </span>
    );
  }

  return (
    <img
      src={blobUrl}
      alt={alt}
      className="w-full h-full object-cover"
      onError={() => {
        setHasError(true);
      }}
    />
  );
}

export default function AdminPage() {
  const { user, isAdmin } = useAuth();
  const [stats, setStats] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Feedback Review state
  const [reviewOverview, setReviewOverview] = useState(null);
  const [selectedEmotion, setSelectedEmotion] = useState('happy');
  const [reviewItems, setReviewItems] = useState([]);
  const [reviewStatusFilter, setReviewStatusFilter] = useState('all');
  const [isReviewLoading, setIsReviewLoading] = useState(false);
  const [reviewError, setReviewError] = useState(null);
  const [actionInProgress, setActionInProgress] = useState({});

  const fetchOverview = useCallback(() => {
    return feedbackApi.getReviewOverview()
      .then((data) => setReviewOverview(data))
      .catch((err) => {
        // Keep graceful if not loaded
        console.warn('Could not load review overview:', err);
      });
  }, []);

  const fetchEmotionItems = useCallback((emotion, statusFilter) => {
    setIsReviewLoading(true);
    setReviewError(null);
    return feedbackApi.getReviewByEmotion(emotion, statusFilter === 'all' ? null : statusFilter)
      .then((res) => {
        setReviewItems(res.items || []);
      })
      .catch((err) => {
        setReviewError(err.message || 'Failed to load feedback records for review.');
      })
      .finally(() => {
        setIsReviewLoading(false);
      });
  }, []);

  useEffect(() => {
    let isMounted = true;
    setIsLoading(true);

    Promise.all([
      feedbackApi.getStats().catch((err) => {
        if (err.status === 403) {
          throw new Error('Access forbidden: Administrator authorization required to view feedback statistics.');
        }
        throw err;
      }),
      predictApi.getModelInfo().catch(() => null),
      feedbackApi.getReviewOverview().catch(() => null),
    ])
      .then(([statsData, modelData, overviewData]) => {
        if (isMounted) {
          setStats(statsData);
          setModelInfo(modelData);
          setReviewOverview(overviewData);
          setError(null);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Failed to fetch administrative statistics.');
        }
      })
      .finally(() => {
        if (isMounted) {
          setIsLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (isAdmin && selectedEmotion) {
      fetchEmotionItems(selectedEmotion, reviewStatusFilter);
    }
  }, [isAdmin, selectedEmotion, reviewStatusFilter, fetchEmotionItems]);

  const handleDecision = async (feedbackId, decision, suggestedLabel) => {
    setActionInProgress((prev) => ({ ...prev, [feedbackId]: true }));
    try {
      await feedbackApi.updateReviewDecision(feedbackId, decision, suggestedLabel);
      // Refresh items and overview counts
      await Promise.all([
        fetchEmotionItems(selectedEmotion, reviewStatusFilter),
        fetchOverview(),
      ]);
    } catch (err) {
      alert(`Decision update failed: ${err.message || 'Network error'}`);
    } finally {
      setActionInProgress((prev) => ({ ...prev, [feedbackId]: false }));
    }
  };

  if (!isAdmin) {
    return (
      <div className="p-8 max-w-xl mx-auto my-12 text-center flex flex-col items-center gap-4">
        <div className="w-14 h-14 rounded-2xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-500">
          <span className="material-symbols-outlined text-[32px]">gpp_bad</span>
        </div>
        <h2 className="text-xl font-bold text-slate-900 dark:text-[#F8FAFC]">
          Administrative Access Restricted
        </h2>
        <p className="text-xs text-slate-500 dark:text-[#94A3B8] leading-relaxed">
          The feedback statistics endpoint <code className="font-mono text-[#6C63FF]">/api/v1/feedback/stats</code> requires admin authorization. Your account ({user?.email}) has role <span className="font-mono font-semibold">{user?.role}</span>.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 sm:p-6 lg:p-8 flex flex-col gap-6 max-w-7xl mx-auto text-left">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-[#F8FAFC]">
              Model Administration
            </h1>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#6C63FF]/15 text-[#6C63FF] border border-[#6C63FF]/30 font-semibold">
              Admin Protected (Active)
            </span>
          </div>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-[#94A3B8]">
            Live model performance telemetry, benchmark checkpoints, and verified human feedback gates.
          </p>
        </div>
      </div>

      {/* Safety Notice Banner */}
      <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start gap-3 text-xs text-amber-700 dark:text-amber-300">
        <span className="material-symbols-outlined text-[20px] text-amber-500 shrink-0">
          security
        </span>
        <div className="flex flex-col gap-0.5">
          <span className="font-semibold">Administrative Boundary Active</span>
          <span>
            Model promotion, fine-tuning retraining loops, and dataset rebuilds require authenticated administrative CLI operations and strict evaluation gates.
          </span>
        </div>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-center gap-2 text-xs text-rose-600 dark:text-rose-400">
          <span className="material-symbols-outlined text-[18px]">error</span>
          <span>{error}</span>
        </div>
      )}

      {/* Model Performance Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card elevation="container" className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 dark:text-[#94A3B8]">Production Checkpoint</span>
          <div className="text-sm font-bold font-mono text-[#6C63FF] mt-1 truncate" title={modelInfo?.model_name}>
            {modelInfo?.model_name || 'ResidualEmotionCNN-Candidate-epoch39'}
          </div>
          <span className="text-[11px] text-slate-400 font-mono">
            {modelInfo?.architecture || 'ResidualEmotionCNN'} • {modelInfo?.device || 'CUDA/CPU'}
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 dark:text-[#94A3B8]">Feedback Accuracy Rate</span>
          <div className="text-2xl font-bold font-mono text-emerald-500 mt-1">
            {stats?.accuracy_rate !== null && stats?.accuracy_rate !== undefined
              ? `${(stats.accuracy_rate * 100).toFixed(1)}%`
              : stats?.total_records
              ? `${(((stats.correct || 0) / (stats.total_records || 1)) * 100).toFixed(1)}%`
              : 'N/A'}
          </div>
          <span className="text-[11px] text-slate-400 font-mono">
            From verified feedback records
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 dark:text-[#94A3B8]">Supported Classes</span>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {modelInfo?.num_classes || 7}
          </div>
          <span className="text-[11px] text-[#0891B2] dark:text-[#22D3EE] font-mono">
            Standard categorical expressions
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 dark:text-[#94A3B8]">Total Feedback Records</span>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {stats?.total_records ?? (isLoading ? '...' : 0)}
          </div>
          <span className="text-[11px] text-[#8B5CF6] font-mono">
            PostgreSQL / JSONL combined
          </span>
        </Card>
      </div>

      {/* Human-in-the-Loop Feedback Review Section */}
      <Card elevation="container" className="flex flex-col gap-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100 dark:border-[#1E294B]">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[22px] text-[#6C63FF]">rate_review</span>
            <div>
              <h2 className="text-base font-bold text-slate-900 dark:text-[#F8FAFC]">
                Human-in-the-Loop Feedback Review
              </h2>
              <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
                Review user feedback submissions by emotion. Admin decisions are authoritative for dataset inclusion.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <label htmlFor="review-status-filter" className="text-xs text-slate-500 dark:text-[#94A3B8]">Filter:</label>
            <select
              id="review-status-filter"
              value={reviewStatusFilter}
              onChange={(e) => setReviewStatusFilter(e.target.value)}
              className="text-xs font-mono bg-white dark:bg-[#171F36] text-slate-800 dark:text-[#F8FAFC] border border-slate-200 dark:border-[#1E294B] rounded-lg px-2.5 py-1 focus:outline-none focus:ring-1 focus:ring-[#6C63FF]"
            >
              <option value="all">All Statuses</option>
              <option value="pending">Pending</option>
              <option value="approved">Approved</option>
              <option value="rejected">Rejected</option>
            </select>
          </div>
        </div>

        {/* Emotion Selector Cards with Status Counts */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2.5">
          {FER2013_EMOTIONS.map((emotion) => {
            const counts = reviewOverview?.emotions?.[emotion] || { pending: 0, approved: 0, rejected: 0, total: 0 };
            const isSelected = selectedEmotion === emotion;

            return (
              <button
                key={emotion}
                type="button"
                onClick={() => setSelectedEmotion(emotion)}
                className={`p-3 rounded-xl text-left flex flex-col gap-1.5 transition-all border ${
                  isSelected
                    ? 'bg-[#6C63FF]/10 border-[#6C63FF] shadow-sm shadow-[#6C63FF]/10 ring-1 ring-[#6C63FF]'
                    : 'bg-slate-50 dark:bg-[#171F36] border-slate-200 dark:border-[#1E294B] hover:border-slate-300 dark:hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className={`text-xs font-bold capitalize ${isSelected ? 'text-[#6C63FF]' : 'text-slate-800 dark:text-[#F8FAFC]'}`}>
                    {emotion}
                  </span>
                  {counts.pending > 0 && (
                    <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold font-mono bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30">
                      {counts.pending}
                    </span>
                  )}
                </div>

                <div className="flex flex-col gap-0.5 font-mono text-[10px]">
                  <span className="text-amber-600 dark:text-amber-400 font-medium">
                    {counts.pending} Pending
                  </span>
                  <span className="text-emerald-600 dark:text-emerald-400">
                    {counts.approved} Approved
                  </span>
                  <span className="text-rose-600 dark:text-rose-400">
                    {counts.rejected} Rejected
                  </span>
                </div>
              </button>
            );
          })}
        </div>

        {/* Review Items Container */}
        <div className="flex flex-col gap-3 mt-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold capitalize text-slate-700 dark:text-[#94A3B8]">
                {selectedEmotion} Feedback Submissions
              </span>
              <Badge variant="primary">{reviewItems.length} Records</Badge>
            </div>
          </div>

          {reviewError && (
            <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-xs text-rose-600 dark:text-rose-400">
              {reviewError}
            </div>
          )}

          {isReviewLoading ? (
            <div className="p-12 text-center text-xs text-slate-400 font-mono flex flex-col items-center gap-2">
              <div className="w-6 h-6 border-2 border-[#6C63FF] border-t-transparent rounded-full animate-spin"></div>
              <span>Loading {selectedEmotion} feedback submissions...</span>
            </div>
          ) : reviewItems.length === 0 ? (
            <div className="p-10 rounded-xl bg-slate-50 dark:bg-[#171F36] border border-dashed border-slate-200 dark:border-[#1E294B] text-center flex flex-col items-center gap-2 text-xs text-slate-400">
              <span className="material-symbols-outlined text-[32px] text-slate-400">inbox</span>
              <span>No {selectedEmotion} feedback submissions found matching the current filter.</span>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {reviewItems.map((item) => {
                const isPending = (item.review_status || 'pending').toLowerCase() === 'pending';
                const isApproved = (item.review_status || '').toLowerCase() === 'approved';
                const isRejected = (item.review_status || '').toLowerCase() === 'rejected';

                // Determine suggested final label: user correction if given, else predicted emotion
                const targetApprovedLabel = item.corrected_emotion || item.predicted_emotion;

                return (
                  <div
                    key={item.feedback_id}
                    className={`p-4 rounded-xl border flex flex-col justify-between gap-3 text-xs transition-colors ${
                      isApproved
                        ? 'bg-emerald-500/[0.03] border-emerald-500/30'
                        : isRejected
                        ? 'bg-rose-500/[0.03] border-rose-500/30'
                        : 'bg-slate-50 dark:bg-[#171F36] border-slate-200 dark:border-[#1E294B]'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2 pb-2 border-b border-slate-200/60 dark:border-[#1E294B]/60">
                      <div className="flex flex-col">
                        <span className="font-mono text-[11px] font-bold text-slate-800 dark:text-[#F8FAFC]">
                          {item.feedback_id}
                        </span>
                        {item.created_at && (
                          <span className="text-[10px] text-slate-400">
                            {new Date(item.created_at).toLocaleString()}
                          </span>
                        )}
                      </div>

                      {/* Status Badges */}
                      {isPending && <Badge variant="warning">Pending Review</Badge>}
                      {isApproved && <Badge variant="success">Approved</Badge>}
                      {isRejected && <Badge variant="danger">Rejected</Badge>}
                    </div>

                    <div className="flex items-start gap-3">
                      {/* Image Preview */}
                      <div className="w-16 h-16 rounded-lg bg-slate-200 dark:bg-[#0A0E1A] border border-slate-300 dark:border-[#1E294B] shrink-0 overflow-hidden flex items-center justify-center">
                        <AdminFeedbackImage imageUrl={item.image_url} alt={`Face crop ${item.feedback_id}`} />
                      </div>

                      {/* Prediction & Feedback Metadata */}
                      <div className="flex flex-col gap-1 flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <span className="text-slate-500 dark:text-[#94A3B8]">Model Prediction:</span>
                          <span className="font-bold font-mono capitalize text-slate-800 dark:text-[#F8FAFC]">
                            {item.predicted_emotion} ({((item.confidence || 0) * 100).toFixed(1)}%)
                          </span>
                        </div>

                        <div className="flex items-center justify-between">
                          <span className="text-slate-500 dark:text-[#94A3B8]">User Feedback:</span>
                          <span className="capitalize font-semibold text-[#6C63FF]">
                            {item.state}
                          </span>
                        </div>

                        {item.corrected_emotion && (
                          <div className="flex items-center justify-between">
                            <span className="text-slate-500 dark:text-[#94A3B8]">User Correction:</span>
                            <span className="font-bold font-mono capitalize text-amber-500">
                              {item.corrected_emotion}
                            </span>
                          </div>
                        )}

                        {item.final_label && (
                          <div className="flex items-center justify-between pt-1 border-t border-slate-200/40 dark:border-[#1E294B]/40">
                            <span className="text-slate-500 dark:text-[#94A3B8]">Admin Final Label:</span>
                            <span className="font-bold font-mono capitalize text-emerald-500">
                              {item.final_label}
                            </span>
                          </div>
                        )}

                        {item.notes && (
                          <div className="text-[10px] text-slate-400 truncate italic mt-0.5">
                            "{item.notes}"
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Action Controls */}
                    <div className="flex items-center justify-end gap-2 pt-2 border-t border-slate-200/60 dark:border-[#1E294B]/60">
                      <Button
                        size="sm"
                        variant="destructive"
                        disabled={actionInProgress[item.feedback_id] || isRejected}
                        onClick={() => handleDecision(item.feedback_id, 'reject')}
                        icon="close"
                      >
                        {isRejected ? 'Rejected' : 'Reject'}
                      </Button>

                      <Button
                        size="sm"
                        variant="primary"
                        disabled={actionInProgress[item.feedback_id] || isApproved}
                        onClick={() => handleDecision(item.feedback_id, 'approve', targetApprovedLabel)}
                        icon="check"
                      >
                        {isApproved ? 'Approved' : `Approve as ${targetApprovedLabel}`}
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </Card>

      {/* Feedback Quality Audit Breakdown */}
      <Card elevation="container" className="flex flex-col gap-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-[#1E294B]">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[18px] text-[#6C63FF]">fact_check</span>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
              Human Feedback Dataset Quality
            </h3>
          </div>
          <Badge variant="cyan">{stats?.total_records ?? 0} Total Records</Badge>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 dark:text-[#94A3B8]">Verified Correct</span>
              <span className="material-symbols-outlined text-emerald-500 text-[18px]">check_circle</span>
            </div>
            <span className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
              {stats?.correct ?? 0}
            </span>
            <span className="text-[11px] text-slate-400">Usable training samples</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 dark:text-[#94A3B8]">Ground-truth Corrected</span>
              <span className="material-symbols-outlined text-rose-500 text-[18px]">edit</span>
            </div>
            <span className="text-2xl font-bold font-mono text-rose-600 dark:text-rose-400">
              {stats?.incorrect ?? 0}
            </span>
            <span className="text-[11px] text-slate-400">Relabeled for fine-tuning</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 dark:text-[#94A3B8]">Uncertain (Quarantine)</span>
              <span className="material-symbols-outlined text-amber-500 text-[18px]">help</span>
            </div>
            <span className="text-2xl font-bold font-mono text-amber-600 dark:text-amber-400">
              {stats?.uncertain ?? 0}
            </span>
            <span className="text-[11px] text-slate-400">Excluded from training</span>
          </div>
        </div>

        {/* Emotion Distribution from feedback stats if present */}
        {stats?.emotions && Object.keys(stats.emotions).length > 0 && (
          <div className="mt-2 pt-3 border-t border-slate-100 dark:border-[#1E294B] flex flex-col gap-2">
            <span className="text-xs font-semibold text-slate-700 dark:text-[#94A3B8]">
              Feedback Distribution by Emotion Class
            </span>
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
              {Object.entries(stats.emotions).map(([emotion, count]) => (
                <div
                  key={emotion}
                  className="p-2.5 rounded-lg bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] flex flex-col items-center gap-0.5 text-center"
                >
                  <span className="text-[11px] capitalize text-slate-500 dark:text-[#94A3B8] font-medium">
                    {emotion}
                  </span>
                  <span className="text-base font-bold font-mono text-slate-800 dark:text-[#F8FAFC]">
                    {count}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}

