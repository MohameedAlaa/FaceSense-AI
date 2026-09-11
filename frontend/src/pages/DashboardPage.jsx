import React from 'react';
import { Link } from 'react-router-dom';
import Card from '../components/common/Card';
import Badge from '../components/common/Badge';
import Button from '../components/common/Button';
import { DEMO_METRICS } from '../data/demo/metrics';

export default function DashboardPage() {
  const { totalAnalyses, facesAnalyzed, averageConfidence, feedbackAccuracy, emotionDistribution, recentActivities } = DEMO_METRICS;

  return (
    <div className="p-4 sm:p-6 lg:p-8 flex flex-col gap-6 max-w-7xl mx-auto text-left">
      {/* Welcome Banner */}
      <div className="bg-gradient-to-r from-white via-white to-slate-50 dark:from-[#12182B] dark:via-[#171F36] dark:to-[#12182B] border border-slate-200 dark:border-[#1E294B] rounded-2xl p-6 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-[#F8FAFC]">
              Telemetry Dashboard
            </h1>
            <Badge variant="cyan">v2.4 Active</Badge>
          </div>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-[#94A3B8]">
            Real-time inference overview and facial expression telemetry monitoring.
          </p>
        </div>

        <Link to="/analyze">
          <Button variant="primary" icon="add" size="md">
            Launch New Analysis
          </Button>
        </Link>
      </div>

      {/* 4 Summary Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card elevation="container" className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-slate-500 dark:text-[#94A3B8] text-xs">
            <span>Total Analyses</span>
            <span className="material-symbols-outlined text-[18px] text-[#6C63FF]">analytics</span>
          </div>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {totalAnalyses.toLocaleString()}
          </div>
          <span className="text-[11px] text-emerald-500 font-medium flex items-center gap-0.5">
            <span className="material-symbols-outlined text-[14px]">arrow_upward</span>
            <span>+14.2% this week</span>
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-slate-500 dark:text-[#94A3B8] text-xs">
            <span>Faces Analyzed</span>
            <span className="material-symbols-outlined text-[18px] text-[#22D3EE]">face</span>
          </div>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {facesAnalyzed.toLocaleString()}
          </div>
          <span className="text-[11px] text-slate-400 dark:text-[#64748B]">
            ~2.5 faces per session
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-slate-500 dark:text-[#94A3B8] text-xs">
            <span>Average Confidence</span>
            <span className="material-symbols-outlined text-[18px] text-[#8B5CF6]">verified</span>
          </div>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {averageConfidence}%
          </div>
          <span className="text-[11px] text-emerald-500 font-medium flex items-center gap-0.5">
            <span className="material-symbols-outlined text-[14px]">check</span>
            <span>Optimal confidence range</span>
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1">
          <div className="flex items-center justify-between text-slate-500 dark:text-[#94A3B8] text-xs">
            <span>Feedback Accuracy</span>
            <span className="material-symbols-outlined text-[18px] text-emerald-500">rate_review</span>
          </div>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {feedbackAccuracy}%
          </div>
          <span className="text-[11px] text-slate-400 dark:text-[#64748B]">
            Human-verified ground truth
          </span>
        </Card>
      </div>

      {/* Main Grid: Emotion Distribution + Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left 7 cols: Emotion Distribution */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          <Card elevation="container" className="flex flex-col gap-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-[#1E294B]">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px] text-[#6C63FF]">bar_chart</span>
                <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
                  Expression Class Distribution
                </h3>
              </div>
              <span className="text-[11px] font-mono text-slate-400">7-class benchmark</span>
            </div>

            <div className="flex flex-col gap-3">
              {emotionDistribution.map((item) => (
                <div key={item.emotion} className="flex flex-col gap-1 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="capitalize font-medium text-slate-700 dark:text-[#F8FAFC]">
                      {item.emotion}
                    </span>
                    <span className="font-mono text-[11px] text-slate-400">
                      {item.count} ({item.percentage}%)
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-100 dark:bg-[#0B1020] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${item.percentage}%`,
                        backgroundColor: item.color,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Right 5 cols: Recent Activity Feed */}
        <div className="lg:col-span-5 flex flex-col gap-4">
          <Card elevation="container" className="flex flex-col gap-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-[#1E294B]">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px] text-[#22D3EE]">history</span>
                <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
                  Recent Activity
                </h3>
              </div>
              <Link to="/history" className="text-xs text-[#6C63FF] hover:underline font-medium">
                View all
              </Link>
            </div>

            <div className="flex flex-col divide-y divide-slate-100 dark:divide-[#1E294B]">
              {recentActivities.map((act) => (
                <div key={act.id} className="py-3 first:pt-0 last:pb-0 flex items-start gap-3 text-xs">
                  <div className="w-8 h-8 rounded-lg bg-slate-100 dark:bg-[#171F36] flex items-center justify-center text-slate-500 dark:text-[#94A3B8] shrink-0 mt-0.5">
                    <span className="material-symbols-outlined text-[16px]">
                      {act.type === 'analysis' ? 'image' : act.type === 'feedback' ? 'rate_review' : 'settings'}
                    </span>
                  </div>
                  <div className="flex flex-col flex-1 min-w-0 text-left">
                    <div className="flex items-center justify-between gap-1">
                      <span className="font-semibold text-slate-800 dark:text-[#F8FAFC] truncate">
                        {act.title}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400 dark:text-[#64748B] shrink-0">
                        {act.time}
                      </span>
                    </div>
                    <span className="text-[11px] text-slate-500 dark:text-[#94A3B8] truncate">
                      {act.result}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
