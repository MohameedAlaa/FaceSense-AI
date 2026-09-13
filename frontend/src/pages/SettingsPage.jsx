import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Card from '../components/common/Card';
import Button from '../components/common/Button';
import Input from '../components/common/Input';
import ThemeSwitcher from '../components/layout/ThemeSwitcher';
import Toast from '../components/common/Toast';
import { useAuth } from '../hooks/useAuth';

export default function SettingsPage() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('appearance');
  const [toastMessage, setToastMessage] = useState(null);

  const tabs = [
    { id: 'appearance', label: 'Appearance', icon: 'palette' },
    { id: 'account', label: 'Account Profile', icon: 'person' },
    { id: 'privacy', label: 'Privacy & Security', icon: 'security' },
    { id: 'notifications', label: 'Notifications', icon: 'notifications' },
  ];

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

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
              Active credentials validated through FastAPI OAuth2 /auth/me.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="text-xs font-medium text-slate-700 dark:text-[#94A3B8] block mb-1">
                Authenticated Email
              </label>
              <div className="p-2.5 rounded-lg bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] font-mono text-xs text-slate-800 dark:text-[#F8FAFC]">
                {user?.email || 'operator@facesense.internal'}
              </div>
            </div>

            <div>
              <label className="text-xs font-medium text-slate-700 dark:text-[#94A3B8] block mb-1">
                System Role
              </label>
              <div className="p-2.5 rounded-lg bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] font-mono text-xs text-slate-800 dark:text-[#F8FAFC] capitalize">
                {user?.role || 'user'}
              </div>
            </div>

            <div>
              <label className="text-xs font-medium text-slate-700 dark:text-[#94A3B8] block mb-1">
                Account ID
              </label>
              <div className="p-2.5 rounded-lg bg-slate-100 dark:bg-[#171F36] border border-slate-200 dark:border-[#1E294B] font-mono text-xs text-slate-800 dark:text-[#F8FAFC]">
                #{user?.id || 1}
              </div>
            </div>

            <div>
              <label className="text-xs font-medium text-slate-700 dark:text-[#94A3B8] block mb-1">
                Security Status
              </label>
              <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 font-mono text-xs flex items-center gap-1.5">
                <span className="material-symbols-outlined text-[14px]">verified_user</span>
                <span>Active Bearer Token</span>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between pt-4 border-t border-slate-100 dark:border-[#1E294B]">
            <Button variant="danger" size="sm" icon="logout" onClick={handleLogout}>
              Sign Out
            </Button>
            <Button variant="primary" size="sm" onClick={() => setToastMessage('Preferences updated')}>
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
                  Associate feedback with user ID without storing raw camera media.
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
                  Only process normalized 48×48 grayscale patches for inference.
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
