import React, { useState } from 'react';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import ThemeSwitcher from '../components/layout/ThemeSwitcher';
import Toast from '../components/common/Toast';

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState('appearance');
  const [toastMessage, setToastMessage] = useState(null);

  const tabs = [
    { id: 'appearance', label: 'Appearance', icon: 'palette' },
    { id: 'account', label: 'Account Profile', icon: 'person' },
    { id: 'privacy', label: 'Privacy & Security', icon: 'security' },
    { id: 'notifications', label: 'Notifications', icon: 'notifications' },
  ];

  return (
    <div className="p-4 sm:p-6 lg:p-8 flex flex-col gap-6 max-w-5xl mx-auto text-left">
      <Toast message={toastMessage} onClose={() => setToastMessage(null)} />

      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-[#F8FAFC]">
          System Settings
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 dark:text-[#94A3B8]">
          Configure workspace appearance, operator credentials, and calibration telemetry.
        </p>
      </div>

      {/* Tabs Row */}
      <div className="flex border-b border-slate-200 dark:border-[#1E294B] gap-4 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 pb-3 px-1 text-xs font-medium border-b-2 transition-all whitespace-nowrap ${
              activeTab === tab.id
                ? "border-[#6C63FF] text-[#6C63FF] dark:text-[#F8FAFC]"
                : "border-transparent text-slate-500 hover:text-slate-800 dark:text-[#94A3B8] dark:hover:text-[#F8FAFC]"
            }`}
          >
            <span className="material-symbols-outlined text-[16px]">{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Appearance Settings */}
      {activeTab === 'appearance' && (
        <Card elevation="container" className="flex flex-col gap-6 p-6">
          <div className="flex flex-col gap-1 pb-4 border-b border-slate-100 dark:border-[#1E294B]">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
              Color Theme & Interface
            </h3>
            <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
              Select whether to follow system preferences or pin Light or Dark mode.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <span className="text-xs font-semibold text-slate-800 dark:text-[#F8FAFC] block">
                Active Theme Preference
              </span>
              <span className="text-[11px] text-slate-400 dark:text-[#64748B]">
                Persisted in local browser storage across sessions.
              </span>
            </div>
            <div className="w-full sm:w-64">
              <ThemeSwitcher />
            </div>
          </div>
        </Card>
      )}

      {/* Account Settings */}
      {activeTab === 'account' && (
        <Card elevation="container" className="flex flex-col gap-4 p-6">
          <div className="flex flex-col gap-1 pb-4 border-b border-slate-100 dark:border-[#1E294B]">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
              Operator Profile
            </h3>
            <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
              Identity used for human-in-the-loop verified feedback records.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <Input label="Full Name" defaultValue="Elena Rostova" />
            <Input label="Role Title" defaultValue="Product Lead · Systems Lab" />
            <Input label="Email Address" defaultValue="elena.rostova@facesense.internal" type="email" />
            <Input label="Department" defaultValue="Vision Telemetry Core" />
          </div>

          <div className="flex justify-end pt-4 border-t border-slate-100 dark:border-[#1E294B]">
            <Button variant="primary" size="sm" onClick={() => setToastMessage('Profile settings saved')}>
              Save Profile
            </Button>
          </div>
        </Card>
      )}

      {/* Privacy Settings */}
      {activeTab === 'privacy' && (
        <Card elevation="container" className="flex flex-col gap-4 p-6">
          <div className="flex flex-col gap-1 pb-4 border-b border-slate-100 dark:border-[#1E294B]">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
              Data Privacy & Retention
            </h3>
            <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
              FaceSense operates strictly under zero-storage biometric privacy guidelines.
            </p>
          </div>

          <div className="flex flex-col gap-3 text-xs text-slate-600 dark:text-[#94A3B8]">
            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B]">
              <div>
                <span className="font-semibold text-slate-800 dark:text-[#F8FAFC] block">
                  Anonymous Feedback Logging
                </span>
                <span className="text-[11px] text-slate-400">
                  Strip all personal identifying headers when storing feedback records.
                </span>
              </div>
              <input type="checkbox" defaultChecked className="accent-[#6C63FF] w-4 h-4 rounded" />
            </div>

            <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B]">
              <div>
                <span className="font-semibold text-slate-800 dark:text-[#F8FAFC] block">
                  Face Crop Obfuscation
                </span>
                <span className="text-[11px] text-slate-400">
                  Only store normalized 48×48 grayscale patches for model evaluation.
                </span>
              </div>
              <input type="checkbox" defaultChecked className="accent-[#6C63FF] w-4 h-4 rounded" />
            </div>
          </div>
        </Card>
      )}

      {/* Notifications */}
      {activeTab === 'notifications' && (
        <Card elevation="container" className="flex flex-col gap-4 p-6">
          <div className="flex flex-col gap-1 pb-4 border-b border-slate-100 dark:border-[#1E294B]">
            <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
              Alerts & Webhooks
            </h3>
            <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
              Configure system alerts for low-confidence detections or pipeline anomalies.
            </p>
          </div>

          <div className="flex items-center justify-between p-3 rounded-lg bg-slate-50 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] text-xs">
            <div>
              <span className="font-semibold text-slate-800 dark:text-[#F8FAFC] block">
                Low Confidence Warning Notifications
              </span>
              <span className="text-[11px] text-slate-400">
                Notify when maximum probability falls below 60% threshold.
              </span>
            </div>
            <input type="checkbox" defaultChecked className="accent-[#6C63FF] w-4 h-4 rounded" />
          </div>
        </Card>
      )}
    </div>
  );
}
