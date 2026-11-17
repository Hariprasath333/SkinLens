import React from 'react';
import { AlertTriangle, ShieldCheck, HeartPulse } from 'lucide-react';
import { Link } from 'react-router-dom';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-border bg-white text-zinc-600">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        
        {/* Strict Medical Disclaimer Box */}
        <div className="mb-6 rounded-lg border border-amber-200/80 bg-amber-50/50 p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 shrink-0 text-amber-700 mt-0.5" />
            <div className="text-xs leading-relaxed text-amber-900">
              <span className="font-semibold text-amber-950">Important Medical Disclaimer: </span>
              SkinLens is an academic research demonstration and educational tool developed for a capstone project.
              It is <strong>not</strong> a clinically certified diagnostic device and must <strong>never</strong> replace formal evaluation,
              biopsy, or clinical consultation by a certified dermatologist or licensed medical physician.
            </div>
          </div>
        </div>

        {/* Footer info & links */}
        <div className="flex flex-col items-center justify-between gap-4 text-xs text-zinc-500 sm:flex-row">
          <div className="flex items-center gap-2">
            <HeartPulse className="h-4 w-4 text-zinc-400" />
            <span>SkinLens Research Prototype — HAM10000 Multimodal Classification</span>
          </div>

          <div className="flex items-center gap-6">
            <Link to="/methodology" className="transition-colors hover:text-zinc-900">
              Methodology & Architecture
            </Link>
            <Link to="/analyze" className="transition-colors hover:text-zinc-900">
              Image Analysis
            </Link>
            <span>Temporary Browser Session (No Storage)</span>
          </div>
        </div>

      </div>
    </footer>
  );
};
