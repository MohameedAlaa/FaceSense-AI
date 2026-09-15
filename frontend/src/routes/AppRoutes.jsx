import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import AppShell from '../components/layout/AppShell';
import { useAuth } from '../hooks/useAuth';

// Eagerly loaded public & primary flow pages
import LandingPage from '../pages/LandingPage';
import LoginPage from '../pages/LoginPage';
import RegisterPage from '../pages/RegisterPage';
import AnalyzePage from '../pages/AnalyzePage';

// Lazy-loaded secondary authenticated & administrative pages
const DashboardPage = lazy(() => import('../pages/DashboardPage'));
const HistoryPage = lazy(() => import('../pages/HistoryPage'));
const InsightsPage = lazy(() => import('../pages/InsightsPage'));
const SettingsPage = lazy(() => import('../pages/SettingsPage'));
const AdminPage = lazy(() => import('../pages/AdminPage'));

/**
 * Lightweight loading fallback for lazy-loaded route chunks.
 * Visually consistent with FaceSense UI, theme-adaptive, accessible, and non-blocking.
 */
function RouteLoadingFallback() {
  return (
    <div
      role="status"
      aria-label="Loading page content"
      className="min-h-[40vh] flex flex-col items-center justify-center p-8 w-full"
    >
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-[#6C63FF]/30 border-t-[#6C63FF] rounded-full animate-spin" />
        <span className="text-xs font-medium text-slate-500 dark:text-[#94A3B8]">
          Loading workspace...
        </span>
      </div>
    </div>
  );
}

/**
 * Route guard requiring active user authentication.
 */
function ProtectedRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#F7F7FC] dark:bg-[#0B1020] flex items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 border-2 border-[#6C63FF]/30 border-t-[#6C63FF] rounded-full animate-spin"></div>
          <span className="text-xs font-mono text-slate-500 dark:text-[#94A3B8]">
            Validating security session...
          </span>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children;
}

/**
 * Route guard strictly enforcing admin authorization.
 */
function AdminRoute({ children }) {
  const { isAuthenticated, isAdmin, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#F7F7FC] dark:bg-[#0B1020] flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-[#6C63FF]/30 border-t-[#6C63FF] rounded-full animate-spin"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (!isAdmin) {
    return (
      <AppShell>
        <div className="p-8 max-w-xl mx-auto my-12 text-center flex flex-col items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-500">
            <span className="material-symbols-outlined text-[32px]">gpp_bad</span>
          </div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-[#F8FAFC]">
            Administrative Access Restricted
          </h2>
          <p className="text-xs text-slate-500 dark:text-[#94A3B8] leading-relaxed">
            Your authenticated account does not have administrative privileges. Model administration, fine-tuning checkpoints, and aggregate feedback telemetry require an authorized operator role.
          </p>
          <a
            href="/dashboard"
            className="px-4 py-2 rounded-lg bg-[#6C63FF] text-white text-xs font-medium hover:bg-[#5B52EE] transition-colors"
          >
            Return to Dashboard
          </a>
        </div>
      </AppShell>
    );
  }

  return children;
}

/**
 * Redirects already authenticated users away from auth pages (/login, /register).
 */
function PublicOnlyRoute({ children }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return null;
  }

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return children;
}

export default function AppRoutes() {
  return (
    <Routes>
      {/* Public Pages */}
      <Route path="/" element={<LandingPage />} />
      <Route
        path="/login"
        element={
          <PublicOnlyRoute>
            <LoginPage />
          </PublicOnlyRoute>
        }
      />
      <Route
        path="/register"
        element={
          <PublicOnlyRoute>
            <RegisterPage />
          </PublicOnlyRoute>
        }
      />

      {/* Authenticated Workspace Pages */}
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <AppShell>
              <Suspense fallback={<RouteLoadingFallback />}>
                <DashboardPage />
              </Suspense>
            </AppShell>
          </ProtectedRoute>
        }
      />
      <Route
        path="/analyze"
        element={
          <ProtectedRoute>
            <AppShell>
              <AnalyzePage />
            </AppShell>
          </ProtectedRoute>
        }
      />
      <Route
        path="/history"
        element={
          <ProtectedRoute>
            <AppShell>
              <Suspense fallback={<RouteLoadingFallback />}>
                <HistoryPage />
              </Suspense>
            </AppShell>
          </ProtectedRoute>
        }
      />
      <Route
        path="/insights"
        element={
          <ProtectedRoute>
            <AppShell>
              <Suspense fallback={<RouteLoadingFallback />}>
                <InsightsPage />
              </Suspense>
            </AppShell>
          </ProtectedRoute>
        }
      />
      <Route
        path="/settings"
        element={
          <ProtectedRoute>
            <AppShell>
              <Suspense fallback={<RouteLoadingFallback />}>
                <SettingsPage />
              </Suspense>
            </AppShell>
          </ProtectedRoute>
        }
      />

      {/* Admin Route */}
      <Route
        path="/admin"
        element={
          <AdminRoute>
            <AppShell>
              <Suspense fallback={<RouteLoadingFallback />}>
                <AdminPage />
              </Suspense>
            </AppShell>
          </AdminRoute>
        }
      />

      {/* Fallback */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
