import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import AppShell from '../components/layout/AppShell';

// Pages
import LandingPage from '../pages/LandingPage';
import LoginPage from '../pages/LoginPage';
import RegisterPage from '../pages/RegisterPage';
import DashboardPage from '../pages/DashboardPage';
import AnalyzePage from '../pages/AnalyzePage';
import HistoryPage from '../pages/HistoryPage';
import InsightsPage from '../pages/InsightsPage';
import SettingsPage from '../pages/SettingsPage';
import AdminPage from '../pages/AdminPage';

export default function AppRoutes() {
  return (
    <Routes>
      {/* Public Pages */}
      <Route path="/" element={<LandingPage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      {/* Workspace Pages (Wrapped in AppShell) */}
      <Route
        path="/dashboard"
        element={
          <AppShell>
            <DashboardPage />
          </AppShell>
        }
      />
      <Route
        path="/analyze"
        element={
          <AppShell>
            <AnalyzePage />
          </AppShell>
        }
      />
      <Route
        path="/history"
        element={
          <AppShell>
            <HistoryPage />
          </AppShell>
        }
      />
      <Route
        path="/insights"
        element={
          <AppShell>
            <InsightsPage />
          </AppShell>
        }
      />
      <Route
        path="/settings"
        element={
          <AppShell>
            <SettingsPage />
          </AppShell>
        }
      />
      <Route
        path="/admin"
        element={
          <AppShell>
            <AdminPage />
          </AppShell>
        }
      />

      {/* Fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
