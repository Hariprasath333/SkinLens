import React, { useRef } from 'react';
import { QueueItem } from '../../types';
import { X, RefreshCw, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react';

interface ImageQueueProps {
  items: QueueItem[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onRemove: (id: string) => void;
  onReplace: (id: string, newFile: File) => void;
  disabled?: boolean;
}

export const ImageQueue: React.FC<ImageQueueProps> = ({
  items,
  selectedId,
  onSelect,
  onRemove,
  onReplace,
  disabled = false,
}) => {
  const replaceInputRefs = useRef<Record<string, HTMLInputElement | null>>({});

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-zinc-500">
        <span className="font-medium text-zinc-700">Image Queue ({items.length})</span>
        <span>Click thumbnail to edit metadata</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
        {items.map((item, idx) => {
          const isSelected = item.id === selectedId;

          return (
            <div
              key={item.id}
              onClick={() => onSelect(item.id)}
              className={`group relative flex flex-col overflow-hidden rounded-lg border bg-white transition-all cursor-pointer ${
                isSelected
                  ? 'border-zinc-900 ring-1 ring-zinc-900 shadow-sm'
                  : 'border-border hover:border-zinc-300'
              }`}
            >
              {/* Thumbnail */}
              <div className="relative aspect-square w-full overflow-hidden bg-zinc-100">
                <img
                  src={item.previewUrl}
                  alt={item.file.name}
                  className="h-full w-full object-cover transition-transform group-hover:scale-105"
                />

                {/* Case badge */}
                <div className="absolute top-1.5 left-1.5 rounded bg-black/60 px-1.5 py-0.5 text-[10px] font-medium text-white backdrop-blur-xs">
                  #{idx + 1}
                </div>

                {/* Status Indicator */}
                <div className="absolute top-1.5 right-1.5">
                  {item.status === 'processing' && (
                    <div className="rounded-full bg-white/90 p-1 shadow-xs text-primary-600 animate-spin">
                      <Loader2 className="h-3 w-3" />
                    </div>
                  )}
                  {item.status === 'completed' && (
                    <div className="rounded-full bg-emerald-500 p-0.5 shadow-xs text-white">
                      <CheckCircle2 className="h-3.5 w-3.5" />
                    </div>
                  )}
                  {item.status === 'failed' && (
                    <div className="rounded-full bg-rose-500 p-0.5 shadow-xs text-white">
                      <AlertCircle className="h-3.5 w-3.5" />
                    </div>
                  )}
                </div>

                {/* Hover overlay actions */}
                {!disabled && (
                  <div className="absolute inset-0 bg-black/40 opacity-0 transition-opacity group-hover:opacity-100 flex items-center justify-center gap-2">
                    <button
                      type="button"
                      title="Replace image"
                      onClick={(e) => {
                        e.stopPropagation();
                        replaceInputRefs.current[item.id]?.click();
                      }}
                      className="rounded bg-white/90 p-1.5 text-zinc-700 hover:bg-white hover:text-zinc-900"
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      title="Remove image"
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemove(item.id);
                      }}
                      className="rounded bg-white/90 p-1.5 text-rose-600 hover:bg-white hover:text-rose-700"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )}

                {/* Hidden input for replace */}
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="hidden"
                  ref={(el) => {
                    replaceInputRefs.current[item.id] = el;
                  }}
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      onReplace(item.id, e.target.files[0]);
                    }
                  }}
                />
              </div>

              {/* Caption */}
              <div className="p-2 text-left">
                <p className="truncate text-xs font-medium text-zinc-900" title={item.file.name}>
                  {item.file.name}
                </p>
                <div className="mt-0.5 flex items-center justify-between text-[11px] text-zinc-500">
                  <span>{(item.file.size / 1024).toFixed(0)} KB</span>
                  <span className="capitalize">{item.sex}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
