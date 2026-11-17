import React from 'react';
import { ClassKey, CLASS_DETAILS } from '../../types';

interface ProbabilityBarProps {
  classKey: ClassKey;
  calibratedProb: number;
  rawProb?: number;
  isHighest: boolean;
}

export const ProbabilityBar: React.FC<ProbabilityBarProps> = ({
  classKey,
  calibratedProb,
  rawProb,
  isHighest,
}) => {
  const details = CLASS_DETAILS[classKey];
  const calPercent = (calibratedProb * 100).toFixed(1);
  const rawPercent = rawProb !== undefined ? (rawProb * 100).toFixed(1) : null;

  // Determine bar fill color based on clinical category
  const barColor =
    details.category === 'Malignant'
      ? isHighest
        ? 'bg-rose-600'
        : 'bg-rose-300'
      : details.category === 'Precancerous / In-situ'
      ? isHighest
        ? 'bg-amber-600'
        : 'bg-amber-300'
      : isHighest
      ? 'bg-primary-600'
      : 'bg-zinc-300';

  return (
    <div className={`rounded-lg p-2.5 transition-colors ${isHighest ? 'bg-zinc-50 border border-zinc-200' : 'hover:bg-zinc-50/50'}`}>
      <div className="mb-1.5 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <span className={`font-mono font-semibold text-xs px-1.5 py-0.5 rounded ${isHighest ? 'bg-zinc-900 text-white' : 'bg-zinc-200 text-zinc-700'}`}>
            {details.code}
          </span>
          <span className="font-medium text-zinc-800">{details.name}</span>
        </div>
        <div className="flex items-center gap-2">
          {rawPercent && rawPercent !== calPercent && (
            <span className="text-[11px] text-zinc-400" title="Uncalibrated model output">
              raw {rawPercent}%
            </span>
          )}
          <span className={`font-mono font-semibold text-xs ${isHighest ? 'text-zinc-900' : 'text-zinc-700'}`}>
            {calPercent}%
          </span>
        </div>
      </div>

      {/* Progress Track */}
      <div className="h-2 w-full overflow-hidden rounded-full bg-zinc-100">
        <div
          className={`h-full rounded-full transition-all duration-500 ease-out ${barColor}`}
          style={{ width: `${Math.max(calibratedProb * 100, 0.5)}%` }}
        />
      </div>
    </div>
  );
};
