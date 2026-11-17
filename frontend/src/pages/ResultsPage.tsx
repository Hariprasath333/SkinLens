import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAnalysis } from '../context/AnalysisContext';
import { CaseSummaryCard } from '../components/results/CaseSummaryCard';
import { ProbabilityDistribution } from '../components/results/ProbabilityDistribution';
import { GradCamViewer } from '../components/results/GradCamViewer';
import { AlertTriangle, ArrowLeft, Trash2, CheckCircle2, AlertCircle, Layers } from 'lucide-react';

export const ResultsPage: React.FC = () => {
  const navigate = useNavigate();
  const { items, clearSession } = useAnalysis();

  const [activeCaseIndex, setActiveCaseIndex] = useState(0);

  // If queue is empty, show empty state with link to /analyze
  if (items.length === 0) {
    return (
      <div className="mx-auto max-w-md px-4 py-24 text-center space-y-4">
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-zinc-100 text-zinc-400">
          <Layers className="h-6 w-6" />
        </div>
        <h2 className="text-lg font-bold text-zinc-900">No Analysis Results Found</h2>
        <p className="text-xs text-zinc-500">
          Upload and analyze dermoscopic images on the analysis page to view predictions and Grad-CAM visualizations.
        </p>
        <Link
          to="/analyze"
          className="inline-flex items-center gap-2 rounded-lg bg-zinc-900 px-4 py-2 text-xs font-semibold text-white hover:bg-zinc-800 transition-colors"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          <span>Go to Image Upload</span>
        </Link>
      </div>
    );
  }

  const activeItem = items[activeCaseIndex] || items[0];
  const activeResult = activeItem.result;

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-8">
      
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-border pb-4 gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Link to="/analyze" className="text-zinc-400 hover:text-zinc-600 transition-colors">
              <ArrowLeft className="h-4 w-4" />
            </Link>
            <h1 className="text-2xl font-bold tracking-tight text-zinc-950">
              Analysis Results
            </h1>
          </div>
          <p className="text-xs text-zinc-500 mt-1">
            Independent evaluations for {items.length} {items.length === 1 ? 'case' : 'cases'} with temperature calibration and feature interpretability.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => {
              clearSession();
              navigate('/analyze');
            }}
            className="flex items-center gap-1.5 rounded-lg border border-border bg-white px-3 py-2 text-xs font-medium text-zinc-600 hover:bg-zinc-50 hover:text-zinc-900 transition-colors"
          >
            <Trash2 className="h-3.5 w-3.5 text-zinc-400" />
            <span>Clear Session</span>
          </button>

          <Link
            to="/analyze"
            className="rounded-lg bg-zinc-900 px-3.5 py-2 text-xs font-semibold text-white hover:bg-zinc-800 transition-colors"
          >
            New Upload
          </Link>
        </div>
      </div>

      {/* Multi-Case Switcher (Tab bar) if more than 1 image */}
      {items.length > 1 && (
        <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-none">
          {items.map((item, idx) => {
            const isActive = idx === activeCaseIndex;
            const isCompleted = item.status === 'completed';
            const isFailed = item.status === 'failed';

            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setActiveCaseIndex(idx)}
                className={`flex shrink-0 items-center gap-2.5 rounded-lg border px-3 py-2 text-xs transition-all ${
                  isActive
                    ? 'border-zinc-900 bg-zinc-900 text-white shadow-2xs'
                    : 'border-border bg-white text-zinc-700 hover:border-zinc-300 hover:bg-zinc-50'
                }`}
              >
                <img
                  src={item.previewUrl}
                  alt={item.file.name}
                  className="h-5 w-5 rounded object-cover"
                />
                <span className="font-medium">Case #{idx + 1}</span>
                {isCompleted && (
                  <span
                    className={`rounded px-1.5 py-0.2 text-[10px] font-mono uppercase ${
                      isActive ? 'bg-white/20 text-white' : 'bg-zinc-100 text-zinc-700'
                    }`}
                  >
                    {item.result?.predicted_class || 'DONE'}
                  </span>
                )}
                {isFailed && (
                  <span className="text-rose-400 font-semibold text-[10px]">FAILED</span>
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* Main Case Result View */}
      {activeResult && activeResult.status === 'completed' ? (
        <div className="space-y-6">
          {/* Top: Summary Card */}
          <CaseSummaryCard
            result={activeResult}
            previewUrl={activeItem.previewUrl}
            index={activeCaseIndex}
          />

          {/* Middle: 2 Columns for Probability Distribution & Grad-CAM */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Probability Bars (5 cols) */}
            <div className="lg:col-span-5">
              <ProbabilityDistribution
                calibratedProbabilities={activeResult.calibrated_probabilities}
                rawProbabilities={activeResult.raw_probabilities}
                predictedClass={activeResult.predicted_class}
              />
            </div>

            {/* Grad-CAM Viewer (7 cols) */}
            <div className="lg:col-span-7">
              <GradCamViewer gradcam={activeResult.gradcam} />
            </div>
          </div>
        </div>
      ) : activeItem.status === 'failed' ? (
        <div className="rounded-xl border border-rose-200 bg-rose-50/50 p-8 text-center shadow-card space-y-3">
          <AlertCircle className="mx-auto h-8 w-8 text-rose-600" />
          <h2 className="text-base font-bold text-rose-950">
            Case #{activeCaseIndex + 1} Analysis Failed
          </h2>
          <p className="text-xs text-rose-800 max-w-md mx-auto">
            {activeItem.error || 'The image could not be processed by the inference engine.'}
          </p>
          <p className="text-[11px] text-zinc-500">
            Other successfully processed cases in this batch remain valid and unaffected.
          </p>
        </div>
      ) : (
        <div className="rounded-xl border border-border bg-white p-8 text-center shadow-card text-xs text-zinc-500">
          This case is currently pending processing.
        </div>
      )}

      {/* Safety Notice Banner */}
      <div className="rounded-xl border border-border bg-white p-4 shadow-subtle flex items-start gap-3">
        <AlertTriangle className="h-4 w-4 shrink-0 text-amber-600 mt-0.5" />
        <p className="text-xs text-zinc-600 leading-relaxed">
          <span className="font-semibold text-zinc-900">Clinical Responsibility Note: </span>
          Predicted probabilities reflect statistical patterns learned on the HAM10000 dataset.
          SkinLens predictions are not medical diagnoses and must not be used as clinical justification to delay,
          alter, or forgo professional dermatological examination.
        </p>
      </div>

    </div>
  );
};
