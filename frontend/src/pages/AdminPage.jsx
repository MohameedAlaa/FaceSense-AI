import React, { useState, useEffect } from 'react';
import Card from '../components/common/Card';
import Badge from '../components/common/Badge';
import { useAuth } from '../hooks/useAuth';
import { feedbackApi } from '../lib/api/feedback';
import { predictApi } from '../lib/api/predict';

export default function AdminPage() {
  const { user, isAdmin } = useAuth();
  const [stats, setStats] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

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
    ])
      .then(([statsData, modelData]) => {
        if (isMounted) {
          setStats(statsData);
          setModelInfo(modelData);
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
