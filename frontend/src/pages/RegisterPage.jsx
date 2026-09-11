import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Logo from '../components/brand/Logo';
import Card from '../components/common/Card';
import Input from '../components/common/Input';
import Button from '../components/common/Button';
import ThemeSwitcher from '../components/layout/ThemeSwitcher';

export default function RegisterPage() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    navigate('/login');
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

      {/* Centered Register Card */}
      <div className="w-full max-w-md mx-auto my-8">
        <Card elevation="elevated" className="p-8 flex flex-col gap-6 text-left">
          <div className="flex flex-col items-center text-center gap-2">
            <Logo variant="full" size="lg" />
            <h2 className="text-lg font-bold text-slate-900 dark:text-[#F8FAFC] mt-4">
              Create FaceSense Account
            </h2>
            <p className="text-xs text-slate-500 dark:text-[#94A3B8]">
              Register new operator credentials for telemetry evaluation.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <Input
              label="Full Name"
              type="text"
              icon="badge"
              placeholder="Elena Rostova"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />

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
              placeholder="At least 8 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />

            <div className="text-[11px] text-slate-400 dark:text-[#64748B] leading-relaxed">
              By registering, you agree to FaceSense zero-storage biometric privacy standards and security policies.
            </div>

            <Button type="submit" variant="primary" size="md" className="w-full mt-2">
              Create Account
            </Button>
          </form>

          <div className="pt-4 border-t border-slate-100 dark:border-[#1E294B] text-center text-xs text-slate-500 dark:text-[#94A3B8]">
            <span>Already have an account? </span>
            <Link to="/login" className="text-[#6C63FF] font-semibold hover:underline">
              Sign in
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
