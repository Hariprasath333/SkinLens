import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAnalysis } from '../context/AnalysisContext';
import { Dropzone } from '../components/upload/Dropzone';
import { ImageQueue } from '../components/upload/ImageQueue';
import { MetadataEditor } from '../components/upload/MetadataEditor';
import { Play, Trash2, ArrowRight, AlertCircle, Loader2, CheckCircle2 } from 'lucide-react';

export const AnalyzePage: React.FC = () => {
  const navigate = useNavigate();
  const {
    items,
    selectedItemId,
    selectedItem,
    isAnalyzing,
    batchError,
    addFiles,
    removeItem,
    replaceFile,
    updateMetadata,
    applyMetadataToAll,
    runBatchAnalysis,
    clearSession,
    selectItem,
  } = useAnalysis();

  const [notification, setNotification] = useState<string | null>(null);

  const handleApplyToAll = (metadata: { age: string; sex: any; localization: string }) => {
    applyMetadataToAll(metadata);
    setNotification('Metadata profile copied to all uploaded images.');
    setTimeout(() => setNotification(null), 3500);
  };

  const handleStartAnalysis = async () => {
    const success = await runBatchAnalysis();
    if (success) {
      navigate('/results');
    }
  };

  const selectedIndex = items.findIndex((i) => i.id === selectedItemId);
  const completedCount = items.filter((i) => i.status === 'completed').length;

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-8">
      
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-border pb-4 gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-950">
            Dermoscopic Image Analysis
          </h1>
          <p className="text-xs text-zinc-500 mt-1">
            Upload single or multiple skin lesion images with independent clinical metadata.
          </p>
        </div>

        {items.length > 0 && (
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={isAnalyzing}
              onClick={clearSession}
              className="flex items-center gap-1.5 rounded-lg border border-border bg-white px-3 py-2 text-xs font-medium text-zinc-600 hover:bg-zinc-50 hover:text-zinc-900 transition-colors disabled:opacity-50"
            >
              <Trash2 className="h-3.5 w-3.5 text-zinc-400" />
              <span>Clear Queue</span>
            </button>

            {completedCount > 0 && (
              <button
                type="button"
                onClick={() => navigate('/results')}
                className="flex items-center gap-1.5 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-xs font-semibold text-zinc-900 hover:bg-zinc-50 transition-colors"
              >
                <span>View Results ({completedCount})</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        )}
      </div>

      {/* Notifications / Errors */}
      {batchError && (
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-xs text-rose-800 flex items-start gap-3">
          <AlertCircle className="h-4 w-4 shrink-0 text-rose-600 mt-0.5" />
          <div>
            <span className="font-semibold text-rose-950">Analysis Error: </span>
            {batchError}
          </div>
        </div>
      )}

      {notification && (
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-800 flex items-center gap-2 animate-fade-in">
          <CheckCircle2 className="h-4 w-4 text-emerald-600" />
          <span>{notification}</span>
        </div>
      )}

      {/* Upload Zone */}
      <Dropzone onFilesSelected={addFiles} disabled={isAnalyzing} />

      {/* Queue & Metadata Workspace */}
      {items.length > 0 && (
        <div className="space-y-6">
          {/* Top: Image Queue Grid */}
          <ImageQueue
            items={items}
            selectedId={selectedItemId}
            onSelect={selectItem}
            onRemove={removeItem}
            onReplace={replaceFile}
            disabled={isAnalyzing}
          />

          {/* Bottom: Active Item Metadata Editor */}
          {selectedItem && (
            <MetadataEditor
              item={selectedItem}
              itemIndex={selectedIndex >= 0 ? selectedIndex : 0}
              totalItems={items.length}
              onUpdate={updateMetadata}
              onApplyToAll={handleApplyToAll}
              disabled={isAnalyzing}
            />
          )}

          {/* Batch Run Action Bar */}
          <div className="flex flex-col sm:flex-row items-center justify-between rounded-xl border border-border bg-white p-4 shadow-card gap-4">
            <div className="text-xs text-zinc-500">
              <span className="font-semibold text-zinc-900">{items.length}</span> {items.length === 1 ? 'image' : 'images'} ready for sequential analysis.
              <span className="block sm:inline sm:ml-2 text-zinc-400">
                Grad-CAM visual heatmap enabled for EfficientNetV2-S branch.
              </span>
            </div>

            <button
              type="button"
              disabled={isAnalyzing}
              onClick={handleStartAnalysis}
              className="flex w-full sm:w-auto items-center justify-center gap-2 rounded-lg bg-zinc-900 px-6 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-zinc-800 transition-colors disabled:opacity-60"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin text-white" />
                  <span>Processing Images...</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-white" />
                  <span>Analyze {items.length > 1 ? `${items.length} Images` : 'Image'}</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

    </div>
  );
};
