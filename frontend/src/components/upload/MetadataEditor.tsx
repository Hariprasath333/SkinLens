import React from 'react';
import { QueueItem, Sex, ANATOMICAL_SITES } from '../../types';
import { User, MapPin, Calendar, Copy, Info } from 'lucide-react';

interface MetadataEditorProps {
  item: QueueItem;
  itemIndex: number;
  totalItems: number;
  onUpdate: (id: string, updates: Partial<{ age: string; sex: Sex; localization: string }>) => void;
  onApplyToAll: (metadata: { age: string; sex: Sex; localization: string }) => void;
  disabled?: boolean;
}

export const MetadataEditor: React.FC<MetadataEditorProps> = ({
  item,
  itemIndex,
  totalItems,
  onUpdate,
  onApplyToAll,
  disabled = false,
}) => {
  return (
    <div className="rounded-xl border border-border bg-white p-5 shadow-card">
      <div className="mb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-border pb-3 gap-2">
        <div className="flex items-center gap-3">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-zinc-900 text-xs font-semibold text-white">
            #{itemIndex + 1}
          </div>
          <div>
            <h3 className="text-sm font-semibold text-zinc-900">
              Clinical Metadata Editor
            </h3>
            <p className="text-xs text-zinc-500 truncate max-w-sm" title={item.file.name}>
              {item.file.name} (Case {itemIndex + 1} of {totalItems})
            </p>
          </div>
        </div>

        {totalItems > 1 && (
          <button
            type="button"
            disabled={disabled}
            onClick={() =>
              onApplyToAll({
                age: item.age,
                sex: item.sex,
                localization: item.localization,
              })
            }
            className="flex items-center gap-1.5 rounded-md border border-zinc-300 bg-white px-2.5 py-1 text-xs font-medium text-zinc-700 shadow-2xs hover:bg-zinc-50 hover:text-zinc-900 transition-colors disabled:opacity-50"
            title="Convenience action: copies current Age, Sex, and Site to all uploaded images"
          >
            <Copy className="h-3.5 w-3.5 text-zinc-500" />
            <span>Apply this metadata to all</span>
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Age Field */}
        <div>
          <label className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-zinc-700">
            <Calendar className="h-3.5 w-3.5 text-zinc-400" />
            <span>Patient Age</span>
            <span className="text-[11px] font-normal text-zinc-400">(Optional)</span>
          </label>
          <input
            type="number"
            min="0"
            max="115"
            step="1"
            placeholder="e.g. 45"
            value={item.age}
            disabled={disabled}
            onChange={(e) => onUpdate(item.id, { age: e.target.value })}
            className="w-full rounded-md border border-zinc-300 px-3 py-1.5 text-xs text-zinc-900 placeholder:text-zinc-400 focus:border-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900 transition-colors"
          />
          <p className="mt-1 text-[11px] text-zinc-400">
            Leave blank if unknown. Normalized against 90 years.
          </p>
        </div>

        {/* Sex Field */}
        <div>
          <label className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-zinc-700">
            <User className="h-3.5 w-3.5 text-zinc-400" />
            <span>Patient Sex</span>
          </label>
          <div className="grid grid-cols-3 gap-1">
            {(['female', 'male', 'unknown'] as Sex[]).map((s) => (
              <button
                key={s}
                type="button"
                disabled={disabled}
                onClick={() => onUpdate(item.id, { sex: s })}
                className={`rounded-md py-1.5 text-xs font-medium capitalize transition-all border ${
                  item.sex === s
                    ? 'border-zinc-900 bg-zinc-900 text-white shadow-2xs'
                    : 'border-zinc-200 bg-white text-zinc-700 hover:border-zinc-300 hover:bg-zinc-50'
                }`}
              >
                {s}
              </button>
            ))}
          </div>
          <p className="mt-1 text-[11px] text-zinc-400">
            Mapped to HAM10000 3-way one-hot vector.
          </p>
        </div>

        {/* Anatomical Localization Field */}
        <div>
          <label className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-zinc-700">
            <MapPin className="h-3.5 w-3.5 text-zinc-400" />
            <span>Anatomical Localization</span>
          </label>
          <select
            value={item.localization}
            disabled={disabled}
            onChange={(e) => onUpdate(item.id, { localization: e.target.value })}
            className="w-full rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-xs text-zinc-900 focus:border-zinc-900 focus:outline-none focus:ring-1 focus:ring-zinc-900 transition-colors"
          >
            {ANATOMICAL_SITES.map((site) => (
              <option key={site} value={site} className="capitalize">
                {site}
              </option>
            ))}
          </select>
          <p className="mt-1 text-[11px] text-zinc-400">
            Mapped to 16 canonical site categories.
          </p>
        </div>
      </div>

      <div className="mt-4 flex items-center gap-2 rounded-md bg-zinc-50 p-2.5 text-[11px] text-zinc-600 border border-zinc-200/60">
        <Info className="h-3.5 w-3.5 shrink-0 text-zinc-400" />
        <span>
          Metadata directly modulates feature fusion in the trained MLP branch (21 features → 32 latent representations).
          Missing values are seamlessly handled using the exact training-time fallback encoding.
        </span>
      </div>
    </div>
  );
};
