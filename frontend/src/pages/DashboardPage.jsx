import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Card from '../components/common/Card';
import Badge from '../components/common/Badge';
import Button from '../components/common/Button';
import { useAuth } from '../hooks/useAuth';
import { predictApi } from '../lib/api/predict';
import { feedbackApi } from '../lib/api/feedback';

export default function DashboardPage() {
  const { user, isAdmin } = useAuth();
  const [modelInfo, setModelInfo] = useState(null);
  const [adminStats, setAdminStats] = useState(null);

  useEffect(() => {
    let mounted = true;
    predictApi.getModelInfo()
      .then((info) => {
        if (mounted && info) setModelInfo(info);
      })
      .catch(() => {});

    if (isAdmin) {
      feedbackApi.getStats()
        .then((stats) => {
          if (mounted && stats) setAdminStats(stats);
        })
        .catch(() => {});
    }

    return () => {
      mounted = false;
    };
  }, [isAdmin]);

  return (
    <div className="p-3 sm:p-6 lg:p-8 flex flex-col gap-5 sm:gap-6 max-w-7xl mx-auto text-left w-full max-w-full overflow-hidden">
      {/* Welcome Banner */}
      <div className="bg-gradient-to-r from-white via-white to-slate-50 dark:from-[#12182B] dark:via-[#171F36] dark:to-[#12182B] border border-slate-200 dark:border-[#1E294B] rounded-2xl p-4 sm:p-6 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4 w-full max-w-full min-w-0">
        <div className="flex flex-col gap-1 min-w-0 max-w-full">
          <div className="flex items-center gap-2 flex-wrap min-w-0">
            <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-[#F8FAFC]">
              Telemetry Dashboard
            </h1>
            <Badge variant="cyan">
              {modelInfo?.status || 'Online'}
            </Badge>
          </div>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-[#94A3B8] break-words">
            Operator session active for <strong className="text-slate-800 dark:text-[#F8FAFC] break-all">{user?.email}</strong> ({user?.role}).
          </p>
        </div>

        <Link to="/analyze" className="w-full sm:w-auto shrink-0">
          <Button variant="primary" icon="add" size="md" className="w-full sm:w-auto min-h-[44px]">
            Launch Analysis
          </Button>
        </Link>
      </div>

      {/* 4 Summary Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 w-full max-w-full">
        <Card elevation="container" className="flex flex-col gap-1 min-w-0">
          <div className="flex items-center justify-between text-slate-500 dark:text-[#94A3B8] text-xs">
            <span>Production Model</span>
            <span className="material-symbols-outlined text-[18px] text-[#6C63FF]">memory</span>
          </div>
          <div className="text-base font-bold font-mono text-[#6C63FF] mt-1 truncate" title={modelInfo?.model_name}>
            {modelInfo?.model_name || 'ResidualEmotionCNN'}
          </div>
          <span className="text-[11px] text-slate-400 font-mono truncate">
            {modelInfo?.architecture || 'PyTorch CNN'} • {modelInfo?.device || 'CUDA/CPU'}
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1 min-w-0">
          <div className="flex items-center justify-between text-slate-500 dark:text-[#94A3B8] text-xs">
            <span>Emotion Classes</span>
            <span className="material-symbols-outlined text-[18px] text-[#22D3EE]">face</span>
          </div>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {modelInfo?.num_classes || 7}
          </div>
          <span className="text-[11px] text-slate-400 dark:text-[#64748B]">
            Categorical taxonomy
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1 min-w-0">
          <div className="flex items-center justify-between text-slate-500 dark:text-[#94A3B8] text-xs">
            <span>Feedback Records</span>
            <span className="material-symbols-outlined text-[18px] text-emerald-500">rate_review</span>
          </div>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {adminStats ? adminStats.total_records : (isAdmin ? '...' : 'Admin Only')}
          </div>
          <span className="text-[11px] text-emerald-500 font-medium truncate">
            {adminStats ? `${adminStats.correct} verified correct` : 'Human-in-the-loop'}
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1 min-w-0">
          <div className="flex items-center justify-between text-slate-500 dark:text-[#94A3B8] text-xs">
            <span>Session History</span>
            <span className="material-symbols-outlined text-[18px] text-[#8B5CF6]">history</span>
          </div>
          <div className="text-sm font-semibold font-mono text-slate-600 dark:text-[#94A3B8] mt-2">
            Awaiting History API
          </div>
          <span className="text-[11px] text-slate-400 truncate">
            Database persistence ready
          </span>
        </Card>
      </div>

      {/* Quick Launch Analyze Card */}
      <Card elevation="container" className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 sm:p-6 bg-gradient-to-r from-[#6C63FF]/5 via-transparent to-[#22D3EE]/5 border border-[#6C63FF]/20 w-full max-w-full min-w-0">
        <div className="flex flex-col gap-1 min-w-0">
          <h3 className="text-base font-bold text-slate-900 dark:text-[#F8FAFC]">
            Vision Emotion Telemetry Workspace
          </h3>
          <p className="text-xs text-slate-500 dark:text-[#94A3B8] max-w-xl leading-relaxed">
            Upload images or camera captures for real-time face localization, bounding box mapping, and 7-class emotion probability analysis.
          </p>
        </div>
        <Link to="/analyze" className="w-full sm:w-auto shrink-0">
          <Button variant="primary" icon="arrow_forward" className="w-full sm:w-auto min-h-[44px]">
            Open Analyze Workspace
          </Button>
        </Link>
      </Card>
    </div>
  );
}
