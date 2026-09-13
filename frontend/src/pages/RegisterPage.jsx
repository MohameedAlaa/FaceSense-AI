import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Logo from '../components/brand/Logo';
import Card from '../components/common/Card';
import Input from '../components/common/Input';
import Button from '../components/common/Button';
import ThemeSwitcher from '../components/layout/ThemeSwitcher';
import { useAuth } from '../hooks/useAuth';

export default function RegisterPage() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { register, login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (password.length < 8) {
      setError('Password must be at least 8 characters long.');
      return;
    }

    setIsSubmitting(true);

    try {
      // Strictly send only email & password as required by FastAPI UserRegisterRequest schema
      await register(email, password);

      // Auto-authenticate after successful registration
      try {
        await login(email, password);
        navigate('/dashboard');
      } catch {
        // If auto-login fails, redirect to login page
        navigate('/login', { state: { registered: true } });
      }
    } catch (err) {
      setError(err.message || 'Registration failed. An account with this email may already exist.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#F7F7FC] dark:bg-[#0B1020] flex flex-col justify-between p-4 sm:p-6 text-slate-900 dark:text-[#F8FAFC]">
      {/* Top bar with brand and theme toggle */}
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

          {error && (
            <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 flex items-start gap-2.5 text-xs text-rose-600 dark:text-rose-400">
              <span className="material-symbols-outlined text-[18px] shrink-0">error</span>
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <Input
              label="Operator Name (Local Display)"
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
              autoComplete="email"
            />

            <Input
              label="Password"
              type="password"
              icon="lock"
              placeholder="At least 8 characters"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="new-password"
            />

            <div className="text-[11px] text-slate-400 dark:text-[#64748B] flex items-center gap-1.5">
              <span className="material-symbols-outlined text-[14px] text-[#22D3EE]">info</span>
              <span>First registered user receives administrative role automatically.</span>
            </div>

            <Button
              type="submit"
              variant="primary"
              size="lg"
              className="w-full justify-center mt-2"
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <span className="flex items-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                  <span>Creating Account...</span>
                </span>
              ) : (
                'Create Account'
              )}
            </Button>
          </form>

          <div className="text-center text-xs text-slate-500 dark:text-[#94A3B8] pt-4 border-t border-slate-100 dark:border-[#1E294B]">
            Already registered?{' '}
            <Link to="/login" className="text-[#6C63FF] font-semibold hover:underline">
              Sign in
            </Link>
          </div>
        </Card>
      </div>

      {/* Footer info */}
      <div className="text-center text-[11px] text-slate-400 dark:text-[#64748B] font-mono">
        FaceSense AI Vision Core • ResidualEmotionCNN Pipeline
      </div>
    </div>
  );
}
