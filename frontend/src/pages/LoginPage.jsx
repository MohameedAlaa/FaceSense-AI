import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Logo from '../components/brand/Logo';
import Card from '../components/common/Card';
import Input from '../components/common/Input';
import Button from '../components/common/Button';
import ThemeSwitcher from '../components/layout/ThemeSwitcher';

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    // Frontend foundation phase: Navigate to dashboard
    navigate('/dashboard');
  };

  return (
    <div className="min-h-screen bg-[#F7F7FC] dark:bg-[#0B1020] flex flex-col justify-between p-4 sm:p-6 text-slate-900 dark:text-[#F8FAFC]">
      {/* Top bar with theme toggle */}
      <div className="flex items-center justify-between max-w-5xl w-full mx-auto">
        <Link to="/">
          <Logo variant="compact" size="sm" />
        </Link>
        <ThemeSwitcher />
      </div>

      {/* Centered Login Card */}
      <div className="w-full max-w-md mx-auto my-8">
        <Card elevation="elevated" className="p-8 flex flex-col gap-6 text-left">
          <div className="flex flex-col items-center text-center gap-2">
            <Logo variant="full" size="lg" />
            <h2 className="text-lg font-bold text-slate-900 dark:text-[#F8FAFC] mt-4">
              Sign in to FaceSense
            </h2>
            <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
              Enter your credentials to access the vision telemetry workspace.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <Input
              label="Work Email"
              type="email"
              icon="mail"
              placeholder="operator@facesense.internal"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />

            <Input
              label="Password"
              type="password"
              icon="lock"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />

            <div className="flex items-center justify-between text-xs text-slate-500 dark:text-[#94A3B8]">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" defaultChecked className="accent-[#6C63FF] rounded" />
                <span>Remember session</span>
              </label>
              <a href="#forgot" className="text-[#6C63FF] hover:underline">
                Forgot password?
              </a>
            </div>

            <Button type="submit" variant="primary" size="md" className="w-full mt-2">
              Sign In to Workspace
            </Button>
          </form>

          <div className="pt-4 border-t border-slate-100 dark:border-[#1E294B] text-center text-xs text-slate-500 dark:text-[#94A3B8]">
            <span>Don't have an operator account? </span>
            <Link to="/register" className="text-[#6C63FF] font-semibold hover:underline">
              Create account
            </Link>
          </div>
        </Card>
      </div>

      {/* Footer */}
      <footer className="text-center text-xs text-slate-400 dark:text-[#64748B]">
        FaceSense Vision Telemetry Platform • Enterprise Security Active
      </footer>
    </div>
  );
}
