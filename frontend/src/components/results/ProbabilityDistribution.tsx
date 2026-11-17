import React from 'react';
import { CLASS_NAMES, ClassKey } from '../../types';
import { ProbabilityBar } from '../common/ProbabilityBar';
import { BarChart3 } from 'lucide-react';

interface ProbabilityDistributionProps {
  calibratedProbabilities?: Record<string, number>;
  rawProbabilities?: Record<string, number>;
  predictedClass?: string;
}

export const ProbabilityDistribution: React.FC<ProbabilityDistributionProps> = ({
  calibratedProbabilities = {},
  rawProbabilities = {},
  predictedClass,
}) => {
  return (
    <div className="rounded-xl border border-border bg-white p-6 shadow-card space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-border pb-3 gap-2">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-zinc-500" />
          <h3 className="text-sm font-semibold text-zinc-900">
            Seven-Class Probability Distribution
          </h3>
        </div>
        <div className="text-xs text-zinc-500">
          Temperature-Scaled (<code className="font-mono text-zinc-700">T = 1.064</code>)
        </div>
      </div>

      <div className="space-y-2">
        {CLASS_NAMES.map((cls) => {
          const calProb = calibratedProbabilities[cls] || 0.0;
          const rawProb = rawProbabilities[cls];
          const isHighest = cls === predictedClass;

          return (
            <ProbabilityBar
              key={cls}
              classKey={cls as ClassKey}
              calibratedProb={calProb}
              rawProb={rawProb}
              isHighest={isHighest}
            />
          );
        })}
      </div>

      <div className="border-t border-border pt-3 flex items-center justify-between text-[11px] text-zinc-400">
        <span>Sum of calibrated probabilities: 100.0%</span>
        <span>Canonical HAM10000 Class Ordering (Alphabetical 0–6)</span>
      </div>
    </div>
  );
};
