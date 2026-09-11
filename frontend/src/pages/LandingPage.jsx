import React from 'react';
import { Link } from 'react-router-dom';
import Logo from '../components/brand/Logo';
import ThemeSwitcher from '../components/layout/ThemeSwitcher';
import Button from '../components/common/Button';
import Card from '../components/common/Card';
import Badge from '../components/common/Badge';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#F7F7FC] dark:bg-[#0B1020] text-slate-900 dark:text-[#F8FAFC] transition-colors flex flex-col">
      {/* Top Navbar */}
      <nav className="h-16 px-6 sm:px-12 flex items-center justify-between border-b border-slate-200 dark:border-[#1E294B] bg-white/80 dark:bg-[#0B1020]/80 backdrop-blur-md sticky top-0 z-30">
        <Link to="/">
          <Logo variant="full" size="md" />
        </Link>

        <div className="flex items-center gap-4">
          <ThemeSwitcher />
          <Link to="/login" className="text-xs font-medium text-slate-600 dark:text-[#94A3B8] hover:text-[#6C63FF] dark:hover:text-white transition-colors">
            Sign In
          </Link>
          <Link to="/analyze">
            <Button variant="primary" size="sm">
              Launch Workspace
            </Button>
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center px-4 sm:px-8 py-16 sm:py-24 text-center max-w-5xl mx-auto">
        {/* Release Tag */}
        <Badge variant="cyan" className="mb-6">
          <span className="w-1.5 h-1.5 rounded-full bg-[#22D3EE] animate-ping mr-1"></span>
          FaceSense Vision Telemetry Core v2.4 Online
        </Badge>

        <h1 className="text-3xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-slate-900 dark:text-[#F8FAFC] max-w-4xl mb-6">
          Understand People. <br className="hidden sm:inline" />
          <span className="bg-gradient-to-r from-[#22D3EE] via-[#6C63FF] to-[#8B5CF6] bg-clip-text text-transparent">
            Build Better Experiences.
          </span>
        </h1>

        <p className="text-sm sm:text-lg text-slate-600 dark:text-[#94A3B8] max-w-2xl mb-8 leading-relaxed">
          High-precision facial expression telemetry and emotion intelligence.
          Calibrated for authentic human-computer interaction and privacy-first feedback.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-4 mb-14">
          <Link to="/analyze">
            <Button variant="primary" size="lg" icon="center_focus_strong">
              Open Analyze Workspace
            </Button>
          </Link>
          <Link to="/dashboard">
            <Button variant="secondary" size="lg" icon="dashboard">
              View Telemetry Dashboard
            </Button>
          </Link>
        </div>

        {/* Brand Philosophy Bar */}
        <div className="w-full max-w-2xl py-3 px-6 rounded-2xl bg-white dark:bg-[#12182B] border border-slate-200 dark:border-[#1E294B] shadow-sm flex flex-wrap items-center justify-around gap-2 text-xs font-mono font-medium text-slate-500 dark:text-[#94A3B8] mb-16">
          <span>HUMAN</span>
          <span className="text-slate-300 dark:text-[#1E294B]">|</span>
          <span>INTELLIGENT</span>
          <span className="text-slate-300 dark:text-[#1E294B]">|</span>
          <span>PRECISE</span>
          <span className="text-slate-300 dark:text-[#1E294B]">|</span>
          <span>TRUSTED</span>
        </div>

        {/* 4 Feature Pillars Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 w-full text-left">
          <Card elevation="container" className="flex flex-col gap-2">
            <div className="w-9 h-9 rounded-lg bg-[#22D3EE]/10 text-[#0891B2] dark:text-[#22D3EE] flex items-center justify-center">
              <span className="material-symbols-outlined text-[20px]">crop_free</span>
            </div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
              Precision Bounding Boxes
            </h3>
            <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
              Multi-subject localization and reticle highlighting on calibrated high-resolution viewports.
            </p>
          </Card>

          <Card elevation="container" className="flex flex-col gap-2">
            <div className="w-9 h-9 rounded-lg bg-[#6C63FF]/10 text-[#6C63FF] flex items-center justify-center">
              <span className="material-symbols-outlined text-[20px]">analytics</span>
            </div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
              7-Class Expression Telemetry
            </h3>
            <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
              Deep residual emotion inference across happy, neutral, surprise, sad, fear, disgust, and angry.
            </p>
          </Card>

          <Card elevation="container" className="flex flex-col gap-2">
            <div className="w-9 h-9 rounded-lg bg-[#8B5CF6]/10 text-[#8B5CF6] flex items-center justify-center">
              <span className="material-symbols-outlined text-[20px]">rate_review</span>
            </div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
              Human-in-the-Loop Feedback
            </h3>
            <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
              Interactive ground-truth labeling and calibration benchmarks to continuously refine accuracy.
            </p>
          </Card>

          <Card elevation="container" className="flex flex-col gap-2">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 text-emerald-500 flex items-center justify-center">
              <span className="material-symbols-outlined text-[20px]">security</span>
            </div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-[#F8FAFC]">
              Enterprise Security
            </h3>
            <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
              JWT authentication, Argon2id encryption, upload validation, and rate-limited endpoints.
            </p>
          </Card>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-6 border-t border-slate-200 dark:border-[#1E294B] text-center text-xs text-slate-400 dark:text-[#64748B]">
        <div className="max-w-5xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>© 2026 FaceSense AI. People in Focus.</span>
          <div className="flex items-center gap-4">
            <Link to="/login" className="hover:underline">Login</Link>
            <Link to="/register" className="hover:underline">Register</Link>
            <Link to="/analyze" className="hover:underline">Workspace</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
