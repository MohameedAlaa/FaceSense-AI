import React from 'react';
import Card from '../components/common/Card';
import Badge from '../components/common/Badge';
import { DEMO_METRICS } from '../data/demo/metrics';

export default function InsightsPage() {
  const { emotionDistribution, processingLatencyMs, averageConfidence, feedbackAccuracy } = DEMO_METRICS;

  return (
    <div className="p-4 sm:p-6 lg:p-8 flex flex-col gap-6 max-w-7xl mx-auto text-left">
      {/* Header */}
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-[#F8FAFC]">
          Model Telemetry Insights
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-[#94A3B8]">
          Aggregated analytical performance, class balance, and confidence metrics across evaluated sessions.
        </p>
      </div>

      {/* Top 3 KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card elevation="container" className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 dark:text-[#94A3B8]">Inference Latency</span>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {processingLatencyMs} ms
          </div>
          <span className="text-[11px] text-[#0891B2] dark:text-[#22D3EE] font-mono">
            Sub-200ms real-time target met
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 dark:text-[#94A3B8]">Mean Confidence</span>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {averageConfidence}%
          </div>
          <span className="text-[11px] text-emerald-500 font-mono">
            High certainty threshold
          </span>
        </Card>

        <Card elevation="container" className="flex flex-col gap-1">
          <span className="text-xs text-slate-500 dark:text-[#94A3B8]">Feedback Verification Rate</span>
          <div className="text-2xl font-bold font-mono text-slate-900 dark:text-[#F8FAFC] mt-1">
            {feedbackAccuracy}%
          </div>
          <span className="text-[11px] text-[#6C63FF] font-mono">
            Human verified agreement
          </span>
        </Card>
      </div>

      {/* Main Charts Area */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Emotion Distribution Frequency Chart */}
        <div className="lg:col-span-8 flex flex-col gap-4">
          <Card elevation="container" className="flex flex-col gap-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-[#1E294B]">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px] text-[#22D3EE]">signal_cellular_alt</span>
                <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
                  Class Distribution Volume (7 Emotions)
                </h3>
              </div>
              <Badge variant="default">All Evaluated Faces</Badge>
            </div>

            {/* Vertical Bar Representation */}
            <div className="h-64 flex items-end justify-between gap-2 sm:gap-4 pt-6 px-2">
              {emotionDistribution.map((item) => (
                <div key={item.emotion} className="flex-1 flex flex-col items-center gap-2 h-full justify-end">
                  <span className="font-mono text-[10px] text-slate-400">
                    {item.percentage}%
                  </span>
                  <div
                    className="w-full max-w-[48px] rounded-t-lg transition-all duration-500 hover:opacity-80"
                    style={{
                      height: `${Math.max(8, item.percentage * 2.2)}%`,
                      backgroundColor: item.color,
                    }}
                    title={`${item.emotion}: ${item.count} samples`}
                  />
                  <span className="capitalize text-[11px] font-medium text-slate-700 dark:text-[#94A3B8] truncate max-w-full">
                    {item.emotion}
                  </span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Pipeline Stage Latency Breakdown */}
        <div className="lg:col-span-4 flex flex-col gap-4">
          <Card elevation="container" className="flex flex-col gap-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-[#1E294B]">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-[18px] text-[#8B5CF6]">timer</span>
                <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
                  Pipeline Latency
                </h3>
              </div>
              <span className="text-[11px] font-mono text-slate-400">183ms total</span>
            </div>

            <div className="flex flex-col gap-3.5 text-xs">
              <div>
                <div className="flex justify-between mb-1 text-slate-600 dark:text-[#94A3B8]">
                  <span>1. Input Validation</span>
                  <span className="font-mono">12 ms</span>
                </div>
                <div className="w-full h-1.5 bg-slate-100 dark:bg-[#0B1020] rounded-full overflow-hidden">
                  <div className="h-full bg-[#22D3EE] rounded-full" style={{ width: '7%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-1 text-slate-600 dark:text-[#94A3B8]">
                  <span>2. Face Localization (Reticle)</span>
                  <span className="font-mono">48 ms</span>
                </div>
                <div className="w-full h-1.5 bg-slate-100 dark:bg-[#0B1020] rounded-full overflow-hidden">
                  <div className="h-full bg-[#6C63FF] rounded-full" style={{ width: '26%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-1 text-slate-600 dark:text-[#94A3B8]">
                  <span>3. Residual Feature Extraction</span>
                  <span className="font-mono">89 ms</span>
                </div>
                <div className="w-full h-1.5 bg-slate-100 dark:bg-[#0B1020] rounded-full overflow-hidden">
                  <div className="h-full bg-[#8B5CF6] rounded-full" style={{ width: '48%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-1 text-slate-600 dark:text-[#94A3B8]">
                  <span>4. Probability Synthesis</span>
                  <span className="font-mono">34 ms</span>
                </div>
                <div className="w-full h-1.5 bg-slate-100 dark:bg-[#0B1020] rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-400 rounded-full" style={{ width: '19%' }} />
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
