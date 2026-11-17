import React from 'react';
import { NavLink, Link } from 'react-router-dom';
import { Activity, ShieldCheck, HelpCircle, Layers, AlertCircle } from 'lucide-react';
import { useAnalysis } from '../../context/AnalysisContext';

export const Navbar: React.FC = () => {
  const { health, items } = useAnalysis();

  const completedCount = items.filter((i) => i.status === 'completed').length;

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border bg-white/95 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        
        {/* Brand / Logo */}
        <Link to="/" className="flex items-center gap-2.5 transition-opacity hover:opacity-90">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-zinc-900 text-white shadow-subtle">
            <Activity className="h-5 w-5 text-primary-400" />
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-1.5">
              <span className="text-base font-semibold tracking-tight text-zinc-900">SkinLens</span>
              <span className="rounded bg-zinc-100 px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-zinc-600">
                PROTOTYPE
              </span>
            </div>
            <span className="text-[11px] text-zinc-500">Multimodal Lesion Classifier</span>
          </div>
        </Link>

        {/* Navigation links */}
        <nav className="flex items-center gap-1 sm:gap-2">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-zinc-100 text-zinc-900'
                  : 'text-zinc-600 hover:bg-zinc-50 hover:text-zinc-900'
              }`
            }
          >
            Home
          </NavLink>

          <NavLink
            to="/analyze"
            className={({ isActive }) =>
              `flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-zinc-100 text-zinc-900'
                  : 'text-zinc-600 hover:bg-zinc-50 hover:text-zinc-900'
              }`
            }
          >
            <span>Analyze</span>
            {items.length > 0 && (
              <span className="flex h-5 min-w-[20px] items-center justify-center rounded-full bg-zinc-200 px-1.5 text-[11px] font-semibold text-zinc-800">
                {items.length}
              </span>
            )}
          </NavLink>

          {completedCount > 0 && (
            <NavLink
              to="/results"
              className={({ isActive }) =>
                `flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-zinc-100 text-zinc-900'
                    : 'text-zinc-600 hover:bg-zinc-50 hover:text-zinc-900'
                }`
              }
            >
              <Layers className="h-4 w-4 text-zinc-500" />
              <span>Results</span>
              <span className="flex h-5 min-w-[20px] items-center justify-center rounded-full bg-primary-50 px-1.5 text-[11px] font-semibold text-primary-700">
                {completedCount}
              </span>
            </NavLink>
          )}

          <NavLink
            to="/methodology"
            className={({ isActive }) =>
              `flex items-center gap-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-zinc-100 text-zinc-900'
                  : 'text-zinc-600 hover:bg-zinc-50 hover:text-zinc-900'
              }`
            }
          >
            <HelpCircle className="h-4 w-4 text-zinc-400" />
            <span>Methodology</span>
          </NavLink>
        </nav>

        {/* Backend health pill */}
        <div className="hidden items-center sm:flex">
          <div className="flex items-center gap-2 rounded-full border border-border bg-zinc-50 px-3 py-1 text-xs">
            {health?.model_ready ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500"></span>
                </span>
                <span className="font-medium text-zinc-700">Model Ready</span>
                <span className="text-zinc-400">({health.device.toUpperCase()})</span>
              </>
            ) : (
              <>
                <AlertCircle className="h-3.5 w-3.5 text-amber-500" />
                <span className="font-medium text-zinc-600">Connecting...</span>
              </>
            )}
          </div>
        </div>

      </div>
    </header>
  );
};
