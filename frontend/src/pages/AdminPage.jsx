import React from 'react';
import Card from '../components/common/Card';
import Badge from '../components/common/Badge';
import { DEMO_METRICS } from '../data/demo/metrics';

export default function AdminPage() {
  const { adminOverview } = DEMO_METRICS;

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
              Admin Protected
            </span>
          </div>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-[#94A3B8]">
            Supervised model performance telemetry, benchmark checkpoints, and feedback review gates.
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
            Model promotion, retraining loops, and dataset rebuilds require authenticated administrative CLI operations and strict evaluation gates.
          </span>
        </div>
      </div>

      {/* Model Performance Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card elevation="container" className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 dark:text-[#94A3B8]">Production Checkpoint</span>
          <div className="text-sm font-bold font-mono text-[#6C63FF] mt-1 truncate">
            {adminOverview.modelVersion}
          </div>
          <span className="text-[11px] text-slate-400 font-mono">
            Epoch 39 • FER2013 + Feedback
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 dark:text-[#94A3B8]">Validation Accuracy</span>
          <div className="text-2xl font-bold font-mono text-emerald-500 mt-1">
            {adminOverview.validationAccuracy}%
          </div>
          <span className="text-[11px] text-slate-400 font-mono">
            Balanced 7-class evaluation
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 dark:text-[#94A3B8]">Macro F1 Score</span>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {adminOverview.macroF1}
          </div>
          <span className="text-[11px] text-[#0891B2] dark:text-[#22D3EE] font-mono">
            Unweighted class fairness
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 dark:text-[#94A3B8]">Weighted F1 Score</span>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {adminOverview.weightedF1}
          </div>
          <span className="text-[11px] text-[#8B5CF6] font-mono">
            Frequency-weighted metric
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
          <Badge variant="cyan">{adminOverview.totalFeedbackRecords} Total Records</Badge>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 dark:text-[#94A3B8]">Verified Correct</span>
              <span className="material-symbols-outlined text-emerald-500 text-[18px]">check_circle</span>
            </div>
            <span className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
              {adminOverview.correctFeedback}
            </span>
            <span className="text-[11px] text-slate-400">Usable training samples</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 dark:text-[#94A3B8]">Ground-truth Corrected</span>
              <span className="material-symbols-outlined text-rose-500 text-[18px]">edit</span>
            </div>
            <span className="text-2xl font-bold font-mono text-rose-600 dark:text-rose-400">
              {adminOverview.incorrectFeedback}
            </span>
            <span className="text-[11px] text-slate-400">Relabeled for fine-tuning</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] flex flex-col gap-1">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 dark:text-[#94A3B8]">Uncertain (Quarantine)</span>
              <span className="material-symbols-outlined text-amber-500 text-[18px]">help</span>
            </div>
            <span className="text-2xl font-bold font-mono text-amber-600 dark:text-amber-400">
              {adminOverview.uncertainFeedback}
            </span>
            <span className="text-[11px] text-slate-400">Excluded from training</span>
          </div>
        </div>
      </Card>
    </div>
  );
}
