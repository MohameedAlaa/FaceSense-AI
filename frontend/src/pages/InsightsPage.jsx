import React, { useState, useEffect } from 'react';
import Card from '../components/common/Card';
import Badge from '../components/common/Badge';
import { useAuth } from '../hooks/useAuth';
import { predictApi } from '../lib/api/predict';
import { feedbackApi } from '../lib/api/feedback';

export default function InsightsPage() {
  const { isAdmin } = useAuth();
  const [modelInfo, setModelInfo] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    let mounted = true;
    predictApi.getModelInfo()
      .then((info) => {
        if (mounted && info) setModelInfo(info);
      })
      .catch(() => {});

    if (isAdmin) {
      feedbackApi.getStats()
        .then((s) => {
          if (mounted && s) setStats(s);
        })
        .catch(() => {});
    }

    return () => {
      mounted = false;
    };
  }, [isAdmin]);

  const classes = modelInfo?.classes || ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];

  return (
    <div className="p-3 sm:p-6 lg:p-8 flex flex-col gap-5 sm:gap-6 max-w-7xl mx-auto text-left w-full max-w-full overflow-hidden">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-4 min-w-0">
        <div className="min-w-0">
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-[#F8FAFC]">
            Telemetry Insights
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-[#94A3B8]">
            Model distribution metrics and classification taxonomy insights.
          </p>
        </div>
        <Badge variant="cyan" className="max-w-[220px] sm:max-w-xs shrink-0">
          {modelInfo?.model_name || 'Production Core'}
        </Badge>
      </div>

      {/* Model Taxonomy Card */}
      <Card elevation="container" className="flex flex-col gap-4 p-4 sm:p-6 w-full max-w-full min-w-0">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-[#1E294B] min-w-0 gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <span className="material-symbols-outlined text-[18px] text-[#22D3EE] shrink-0">category</span>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC] truncate">
              Supported Classification Taxonomy
            </h3>
          </div>
          <span className="text-xs font-mono text-slate-400 shrink-0">
            {classes.length} Units
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2.5 sm:gap-3 w-full">
          {classes.map((cls) => (
            <div
              key={cls}
              className="p-3 rounded-xl bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] flex flex-col items-center gap-1 text-center min-w-0"
            >
              <span className="text-xs font-semibold capitalize text-slate-800 dark:text-[#F8FAFC] truncate w-full">
                {cls}
              </span>
              <span className="text-[10px] font-mono text-slate-400 dark:text-[#64748B] truncate w-full">
                {stats?.emotions && stats.emotions[cls] !== undefined
                  ? `${stats.emotions[cls]} feedback`
                  : 'Softmax Class'}
              </span>
            </div>
          ))}
        </div>
      </Card>

      {/* Analytics Notice */}
      <Card elevation="container" className="p-6 sm:p-8 text-center flex flex-col items-center gap-3 w-full max-w-full min-w-0">
        <div className="w-12 h-12 rounded-xl bg-[#8B5CF6]/10 text-[#8B5CF6] flex items-center justify-center shrink-0">
          <span className="material-symbols-outlined text-[24px]">query_stats</span>
        </div>
        <h3 className="text-sm font-bold text-slate-900 dark:text-[#F8FAFC]">
          Temporal Drift & Confidence Calibration Insights
        </h3>
        <p className="text-xs text-slate-500 dark:text-[#94A3B8] max-w-md leading-relaxed">
          Aggregated time-series trend analysis and rolling confidence decay tracking will be unlocked when the backend analytics telemetry worker is enabled.
        </p>
      </Card>
    </div>
  );
}
