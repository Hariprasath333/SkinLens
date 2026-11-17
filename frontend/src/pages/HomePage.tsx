import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Activity, ShieldCheck, Cpu, UploadCloud, Layers, Sparkles, FileText, CheckCircle2 } from 'lucide-react';
import { CLASS_NAMES, CLASS_DETAILS } from '../types';
import { CategoryBadge } from '../components/common/Badge';

export const HomePage: React.FC = () => {
  return (
    <div className="space-y-16 py-10 sm:py-16">
      
      {/* Hero Section */}
      <section className="mx-auto max-w-5xl px-4 text-center sm:px-6 lg:px-8">
        <div className="inline-flex items-center gap-2 rounded-full border border-border bg-zinc-100/80 px-3 py-1 text-xs font-medium text-zinc-700 mb-6 shadow-2xs">
          <Activity className="h-3.5 w-3.5 text-primary-600" />
          <span>Multimodal Hybrid Deep Learning • HAM10000 Dataset</span>
        </div>

        <h1 className="text-4xl font-extrabold tracking-tight text-zinc-950 sm:text-5xl lg:text-6xl">
          Understand skin lesions through <br className="hidden sm:inline" />
          <span className="text-zinc-700">intelligent image analysis.</span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-base sm:text-lg text-zinc-600 leading-relaxed">
          SkinLens is an educational and research prototype that evaluates dermoscopic images
          using a hybrid deep learning architecture—combining EfficientNetV2-S, Swin Transformer V2,
          and clinical patient metadata.
        </p>

        {/* CTAs */}
        <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3">
          <Link
            to="/analyze"
            className="flex w-full sm:w-auto items-center justify-center gap-2 rounded-lg bg-zinc-900 px-6 py-3 text-sm font-semibold text-white shadow-sm hover:bg-zinc-800 transition-colors"
          >
            <span>Start Analysis</span>
            <ArrowRight className="h-4 w-4" />
          </Link>

          <Link
            to="/methodology"
            className="flex w-full sm:w-auto items-center justify-center gap-2 rounded-lg border border-border bg-white px-6 py-3 text-sm font-semibold text-zinc-700 shadow-2xs hover:bg-zinc-50 hover:text-zinc-900 transition-colors"
          >
            <FileText className="h-4 w-4 text-zinc-400" />
            <span>Explore Methodology</span>
          </Link>
        </div>

        {/* Highlight Metrics */}
        <div className="mt-12 grid grid-cols-2 sm:grid-cols-4 gap-4 border-t border-border pt-8 text-left">
          <div className="p-3">
            <div className="font-mono text-2xl font-bold text-zinc-900">85.45%</div>
            <div className="text-xs text-zinc-500 font-medium mt-0.5">Test Accuracy (N=2,014)</div>
          </div>
          <div className="p-3">
            <div className="font-mono text-2xl font-bold text-zinc-900">76.03%</div>
            <div className="text-xs text-zinc-500 font-medium mt-0.5">Macro-F1 Score</div>
          </div>
          <div className="p-3">
            <div className="font-mono text-2xl font-bold text-zinc-900">T = 1.064</div>
            <div className="text-xs text-zinc-500 font-medium mt-0.5">Temperature Scaled</div>
          </div>
          <div className="p-3">
            <div className="font-mono text-2xl font-bold text-zinc-900">Grad-CAM</div>
            <div className="text-xs text-zinc-500 font-medium mt-0.5">Visual Feature Heatmap</div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-8 text-center sm:text-left">
          <h2 className="text-2xl font-bold tracking-tight text-zinc-950">How SkinLens Works</h2>
          <p className="text-xs text-zinc-500 mt-1">Four-stage multimodal analysis pipeline</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="rounded-xl border border-border bg-white p-5 shadow-card space-y-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-zinc-100 text-zinc-900 font-semibold text-sm">
              01
            </div>
            <h3 className="text-sm font-semibold text-zinc-900">Upload Dermoscopy</h3>
            <p className="text-xs text-zinc-600 leading-relaxed">
              Upload one or multiple dermoscopic lesion images in batch. Images are processed temporarily in memory.
            </p>
          </div>

          <div className="rounded-xl border border-border bg-white p-5 shadow-card space-y-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-zinc-100 text-zinc-900 font-semibold text-sm">
              02
            </div>
            <h3 className="text-sm font-semibold text-zinc-900">Enter Clinical Data</h3>
            <p className="text-xs text-zinc-600 leading-relaxed">
              Optionally provide patient age, sex, and anatomical localization independently for each lesion.
            </p>
          </div>

          <div className="rounded-xl border border-border bg-white p-5 shadow-card space-y-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-zinc-100 text-zinc-900 font-semibold text-sm">
              03
            </div>
            <h3 className="text-sm font-semibold text-zinc-900">Dual-Branch Fusion</h3>
            <p className="text-xs text-zinc-600 leading-relaxed">
              EfficientNetV2-S and Swin Transformer V2 extract visual patterns, fused with the 32-dim metadata MLP representation.
            </p>
          </div>

          <div className="rounded-xl border border-border bg-white p-5 shadow-card space-y-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-zinc-100 text-zinc-900 font-semibold text-sm">
              04
            </div>
            <h3 className="text-sm font-semibold text-zinc-900">Calibrated Results & CAM</h3>
            <p className="text-xs text-zinc-600 leading-relaxed">
              Review calibrated probabilities across all 7 categories with an overlaid Grad-CAM activation heatmap.
            </p>
          </div>
        </div>
      </section>

      {/* Supported Diagnostic Categories */}
      <section className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mb-6 flex flex-col sm:flex-row sm:items-end justify-between gap-2 border-b border-border pb-4">
          <div>
            <h2 className="text-2xl font-bold tracking-tight text-zinc-950">Seven Supported Categories</h2>
            <p className="text-xs text-zinc-500 mt-1">Classified according to the canonical HAM10000 dermoscopy benchmark</p>
          </div>
          <span className="text-xs text-zinc-400">Alphabetical Index 0 to 6</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {CLASS_NAMES.map((cls) => {
            const details = CLASS_DETAILS[cls];
            return (
              <div key={cls} className="rounded-xl border border-border bg-white p-5 shadow-subtle hover:border-zinc-300 transition-colors space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-zinc-900 bg-zinc-100 px-2 py-0.5 rounded">
                    {details.code}
                  </span>
                  <CategoryBadge category={details.category} size="sm" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-zinc-900">{details.name}</h3>
                  <p className="text-xs text-zinc-600 mt-1.5 leading-relaxed">{details.description}</p>
                </div>
                <div className="text-[11px] text-zinc-400 pt-2 border-t border-border/60">
                  {details.clinicalSignificance}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Privacy & Research Context */}
      <section className="mx-auto max-w-5xl px-4 sm:px-6 lg:px-8">
        <div className="rounded-2xl border border-border bg-white p-8 shadow-card space-y-4">
          <div className="flex items-center gap-2.5">
            <ShieldCheck className="h-5 w-5 text-zinc-700" />
            <h2 className="text-lg font-bold text-zinc-900">Privacy & Ephemeral Session Guarantee</h2>
          </div>
          <p className="text-xs sm:text-sm text-zinc-600 leading-relaxed">
            SkinLens does not require user registration or personal accounts. Uploaded dermoscopic images and clinical metadata
            are processed exclusively in ephemeral server memory during inference and are <strong>never stored permanently</strong> to databases
            or file disks. Results exist solely within your active browser memory.
          </p>
          <div className="pt-2">
            <Link
              to="/analyze"
              className="inline-flex items-center gap-2 text-xs font-semibold text-zinc-900 hover:text-zinc-600 transition-colors"
            >
              <span>Ready to test images</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>
        </div>
      </section>

    </div>
  );
};
