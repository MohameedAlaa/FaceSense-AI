import React from 'react';
import Card from '../components/common/Card';
import Badge from '../components/common/Badge';

export default function HistoryPage() {
  return (
    <div className="p-4 sm:p-6 lg:p-8 flex flex-col gap-6 max-w-7xl mx-auto text-left">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-[#F8FAFC]">
            Analysis History
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-[#94A3B8]">
            Browse and inspect past vision telemetry sessions and verified ground-truth records.
          </p>
        </div>
        <Badge variant="outline">Endpoint Pending</Badge>
      </div>

      {/* Integration Status Notice */}
      <Card elevation="container" className="p-8 text-center flex flex-col items-center gap-4 max-w-2xl mx-auto my-8">
        <div className="w-14 h-14 rounded-2xl bg-[#6C63FF]/10 text-[#6C63FF] flex items-center justify-center">
          <span className="material-symbols-outlined text-[28px]">history</span>
        </div>
        <div className="flex flex-col gap-1">
          <h3 className="text-base font-bold text-slate-900 dark:text-[#F8FAFC]">
            Historical Telemetry Logging API Not Connected
          </h3>
          <p className="text-xs text-slate-500 dark:text-[#94A3B8] leading-relaxed max-w-md">
            The FastAPI backend currently exposes inference (<code className="font-mono text-[#6C63FF]">/predict/image</code>) and feedback (<code className="font-mono text-[#6C63FF]">/feedback</code>) endpoints. Session persistence for individual historical analyses will be wired to a dedicated history endpoint in an upcoming phase.
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono text-slate-400 dark:text-[#64748B] pt-4 border-t border-slate-100 dark:border-[#1E294B] w-full justify-center">
          <span className="material-symbols-outlined text-[14px] text-[#22D3EE]">info</span>
          <span>Zero biometric images are stored permanently without explicit consent.</span>
        </div>
      </Card>
    </div>
  );
}
