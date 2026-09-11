import React, { useState } from 'react';
import Card from '../components/common/Card';
import Badge from '../components/common/Badge';
import Input from '../components/common/Input';
import { DEMO_HISTORY_RECORDS } from '../data/demo/analyses';

export default function HistoryPage() {
  const [search, setSearch] = useState('');
  const [selectedEmotion, setSelectedEmotion] = useState('all');

  const emotions = ['all', 'happy', 'neutral', 'surprise', 'sad', 'fear', 'disgust', 'angry'];

  const filteredRecords = DEMO_HISTORY_RECORDS.filter((rec) => {
    const matchesSearch = rec.filename.toLowerCase().includes(search.toLowerCase());
    const matchesEmotion = selectedEmotion === 'all' || rec.primary_emotion === selectedEmotion;
    return matchesSearch && matchesEmotion;
  });

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
        <span className="text-xs font-mono text-slate-400">
          {filteredRecords.length} records found
        </span>
      </div>

      {/* Filters Bar */}
      <Card elevation="container" className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4 p-4">
        <div className="max-w-md w-full">
          <Input
            icon="search"
            placeholder="Filter by session image filename..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        {/* Emotion Filter Chips */}
        <div className="flex flex-wrap items-center gap-1.5 overflow-x-auto pb-1 md:pb-0">
          {emotions.map((em) => (
            <button
              key={em}
              type="button"
              onClick={() => setSelectedEmotion(em)}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium capitalize transition-colors ${
                selectedEmotion === em
                  ? "bg-[#6C63FF] text-white shadow-sm"
                  : "bg-slate-100 dark:bg-[#171F36] text-slate-600 dark:text-[#94A3B8] hover:text-slate-900 dark:hover:text-[#F8FAFC]"
              }`}
            >
              {em}
            </button>
          ))}
        </div>
      </Card>

      {/* Desktop / Tablet Table View */}
      <div className="hidden sm:block bg-white dark:bg-[#12182B] border border-slate-200 dark:border-[#1E294B] rounded-2xl overflow-hidden shadow-sm">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-50 dark:bg-[#171F36] border-b border-slate-200 dark:border-[#1E294B] text-slate-500 dark:text-[#94A3B8] font-mono text-[11px] uppercase">
            <tr>
              <th className="py-3.5 px-4">Session Image</th>
              <th className="py-3.5 px-4">Timestamp</th>
              <th className="py-3.5 px-4">Faces Localized</th>
              <th className="py-3.5 px-4">Primary Expression</th>
              <th className="py-3.5 px-4">Confidence</th>
              <th className="py-3.5 px-4">Feedback State</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-[#1E294B]">
            {filteredRecords.map((rec) => (
              <tr key={rec.id} className="hover:bg-slate-50/80 dark:hover:bg-[#171F36]/50 transition-colors">
                <td className="py-3 px-4 font-medium text-slate-800 dark:text-[#F8FAFC]">
                  <div className="flex items-center gap-2.5">
                    <img 
                      src={rec.thumbnail} 
                      alt="" 
                      className="w-9 h-9 rounded-lg object-cover border border-slate-200 dark:border-[#1E294B]"
                    />
                    <span className="truncate max-w-[200px]">{rec.filename}</span>
                  </div>
                </td>
                <td className="py-3 px-4 text-slate-500 dark:text-[#94A3B8] font-mono text-[11px]">
                  {rec.date}
                </td>
                <td className="py-3 px-4 text-slate-700 dark:text-[#F8FAFC] font-mono">
                  {rec.faces_count} {rec.faces_count === 1 ? 'face' : 'faces'}
                </td>
                <td className="py-3 px-4">
                  <span className="capitalize font-semibold text-slate-800 dark:text-[#F8FAFC]">
                    {rec.primary_emotion}
                  </span>
                </td>
                <td className="py-3 px-4 font-mono text-[11px] text-[#0891B2] dark:text-[#22D3EE] font-bold">
                  {(rec.confidence * 100).toFixed(1)}%
                </td>
                <td className="py-3 px-4">
                  <Badge 
                    variant={
                      rec.feedback_state === 'correct' 
                        ? 'success' 
                        : rec.feedback_state === 'uncertain' 
                        ? 'warning' 
                        : 'danger'
                    }
                  >
                    {rec.feedback_state}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mobile Card-based List View */}
      <div className="sm:hidden flex flex-col gap-3">
        {filteredRecords.map((rec) => (
          <Card key={rec.id} elevation="container" className="flex flex-col gap-2">
            <div className="flex items-center gap-3">
              <img 
                src={rec.thumbnail} 
                alt="" 
                className="w-12 h-12 rounded-lg object-cover border border-slate-200 dark:border-[#1E294B] shrink-0"
              />
              <div className="flex flex-col min-w-0">
                <span className="text-xs font-semibold text-slate-800 dark:text-[#F8FAFC] truncate">
                  {rec.filename}
                </span>
                <span className="text-[10px] text-slate-400 font-mono">
                  {rec.date}
                </span>
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-slate-100 dark:border-[#1E294B] text-xs">
              <span className="capitalize font-medium text-slate-700 dark:text-[#F8FAFC]">
                {rec.primary_emotion} • {(rec.confidence * 100).toFixed(1)}%
              </span>
              <Badge 
                variant={
                  rec.feedback_state === 'correct' 
                    ? 'success' 
                    : rec.feedback_state === 'uncertain' 
                    ? 'warning' 
                    : 'danger'
                }
              >
                {rec.feedback_state}
              </Badge>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
