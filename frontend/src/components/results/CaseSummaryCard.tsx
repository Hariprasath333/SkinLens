import React from 'react';
import { BatchItemResult, CLASS_DETAILS, ClassKey } from '../../types';
import { CategoryBadge } from '../common/Badge';
import { CheckCircle2, AlertTriangle, ShieldAlert } from 'lucide-react';

interface CaseSummaryCardProps {
  result: BatchItemResult;
  previewUrl: string;
  index: number;
}

export const CaseSummaryCard: React.FC<CaseSummaryCardProps> = ({ result, previewUrl, index }) => {
  const predKey = (result.predicted_class || 'nv') as ClassKey;
  const details = CLASS_DETAILS[predKey] || CLASS_DETAILS.nv;

  const calConfidence = result.calibrated_confidence ? (result.calibrated_confidence * 100).toFixed(1) : '0.0';
  const rawConfidence = result.raw_confidence ? (result.raw_confidence * 100).toFixed(1) : '0.0';

  return (
    <div className="rounded-xl border border-border bg-white p-6 shadow-card">
      <div className="flex flex-col md:flex-row items-start gap-6">
        
        {/* Left: Original Image Thumbnail */}
        <div className="relative aspect-square w-32 md:w-36 shrink-0 overflow-hidden rounded-lg border border-border bg-zinc-100 shadow-2xs">
          <img
            src={previewUrl}
            alt={result.id}
            className="h-full w-full object-cover"
          />
          <div className="absolute top-2 left-2 rounded bg-black/60 px-2 py-0.5 text-[10px] font-semibold text-white backdrop-blur-xs">
            Case #{index + 1}
          </div>
        </div>

        {/* Right: Diagnosis and Metadata Details */}
        <div className="flex-1 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-zinc-500 uppercase tracking-wider">Predicted Category</span>
                <CategoryBadge category={details.category} size="sm" />
              </div>
              <h2 className="text-xl font-bold tracking-tight text-zinc-900 mt-0.5">
                {details.name} <span className="text-zinc-400 font-mono text-base font-normal">({details.code})</span>
              </h2>
            </div>

            {/* Confidence Cards */}
            <div className="flex items-center gap-3">
              <div className="rounded-lg border border-border bg-zinc-50 px-3 py-1.5 text-right">
                <div className="text-[10px] uppercase tracking-wider text-zinc-400 font-medium">Calibrated Conf.</div>
                <div className="font-mono text-lg font-bold text-zinc-900">{calConfidence}%</div>
              </div>
              <div className="rounded-lg border border-border/60 bg-white px-3 py-1.5 text-right">
                <div className="text-[10px] uppercase tracking-wider text-zinc-400 font-medium">Raw Conf.</div>
                <div className="font-mono text-sm font-semibold text-zinc-600">{rawConfidence}%</div>
              </div>
            </div>
          </div>

          {/* Description & Clinical Significance */}
          <p className="text-xs text-zinc-600 leading-relaxed">
            {details.description}
          </p>

          {/* Metadata pill summary */}
          <div className="flex flex-wrap items-center gap-2 pt-1 text-xs text-zinc-600">
            <span className="font-medium text-zinc-500">Metadata Used:</span>
            <span className="rounded bg-zinc-100 px-2 py-0.5 text-zinc-700">
              Age: {result.metadata_used.age !== null && result.metadata_used.age !== undefined ? `${result.metadata_used.age} yrs` : 'Unknown (Defaulted 0.5)'}
            </span>
            <span className="rounded bg-zinc-100 px-2 py-0.5 text-zinc-700 capitalize">
              Sex: {result.metadata_used.sex || 'Unknown'}
            </span>
            <span className="rounded bg-zinc-100 px-2 py-0.5 text-zinc-700 capitalize">
              Site: {result.metadata_used.localization || 'Unknown'}
            </span>
          </div>

          {details.category === 'Malignant' && (
            <div className="mt-2 flex items-center gap-2 rounded-md bg-rose-50 border border-rose-200/80 p-2 text-xs text-rose-800">
              <ShieldAlert className="h-4 w-4 shrink-0 text-rose-600" />
              <span>
                <strong>Malignancy Alert:</strong> Immediate clinical review and dermatoscopic examination by a medical professional is strongly advised.
              </span>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};
