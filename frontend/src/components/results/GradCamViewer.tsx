import React, { useState } from 'react';
import { GradCamResponse } from '../../types';
import { Eye, Layers, Info, HelpCircle } from 'lucide-react';

interface GradCamViewerProps {
  gradcam?: GradCamResponse;
}

export const GradCamViewer: React.FC<GradCamViewerProps> = ({ gradcam }) => {
  const [activeTab, setActiveTab] = useState<'comparison' | 'overlay' | 'heatmap'>('comparison');

  if (!gradcam || !gradcam.available) {
    return (
      <div className="rounded-xl border border-border bg-white p-6 text-center shadow-card">
        <Info className="mx-auto h-8 w-8 text-zinc-400 mb-2" />
        <h3 className="text-sm font-semibold text-zinc-900">Grad-CAM Explanation Unavailable</h3>
        <p className="mt-1 text-xs text-zinc-500 max-w-md mx-auto">
          {gradcam?.error || 'Visual explanation could not be generated for this case.'}
        </p>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-white p-6 shadow-card space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-border pb-3 gap-2">
        <div>
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-zinc-500" />
            <h3 className="text-sm font-semibold text-zinc-900">
              Visual Interpretability (Grad-CAM)
            </h3>
          </div>
          <p className="text-xs text-zinc-500 mt-0.5">
            Feature localization targeting convolutional head <code className="font-mono text-[11px] bg-zinc-100 px-1 py-0.5 rounded text-zinc-700">{gradcam.target_layer || 'model.effnet.conv_head'}</code>
          </p>
        </div>

        {/* View Mode Switcher */}
        <div className="flex items-center rounded-lg border border-border bg-zinc-100 p-0.5 text-xs">
          <button
            type="button"
            onClick={() => setActiveTab('comparison')}
            className={`rounded-md px-2.5 py-1 font-medium transition-all ${
              activeTab === 'comparison'
                ? 'bg-white text-zinc-900 shadow-2xs'
                : 'text-zinc-600 hover:text-zinc-900'
            }`}
          >
            3-Panel Side-by-Side
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('overlay')}
            className={`rounded-md px-2.5 py-1 font-medium transition-all ${
              activeTab === 'overlay'
                ? 'bg-white text-zinc-900 shadow-2xs'
                : 'text-zinc-600 hover:text-zinc-900'
            }`}
          >
            Overlay Only
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('heatmap')}
            className={`rounded-md px-2.5 py-1 font-medium transition-all ${
              activeTab === 'heatmap'
                ? 'bg-white text-zinc-900 shadow-2xs'
                : 'text-zinc-600 hover:text-zinc-900'
            }`}
          >
            Raw Heatmap
          </button>
        </div>
      </div>

      {/* Panels */}
      {activeTab === 'comparison' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* 1. Original */}
          <div className="space-y-1.5 text-center">
            <span className="text-xs font-semibold text-zinc-700">1. Original Image (384×384)</span>
            <div className="aspect-square w-full overflow-hidden rounded-lg border border-border bg-zinc-100">
              <img
                src={gradcam.original_image}
                alt="Original dermoscopy"
                className="h-full w-full object-cover"
              />
            </div>
            <p className="text-[11px] text-zinc-400">Preprocessed RGB dermoscopic input</p>
          </div>

          {/* 2. Heatmap */}
          <div className="space-y-1.5 text-center">
            <span className="text-xs font-semibold text-zinc-700">2. Grad-CAM Activation Heatmap</span>
            <div className="aspect-square w-full overflow-hidden rounded-lg border border-border bg-zinc-100">
              <img
                src={gradcam.heatmap}
                alt="Activation heatmap"
                className="h-full w-full object-cover"
              />
            </div>
            <p className="text-[11px] text-zinc-400">OpenCV JET colormap (warmer = higher activation)</p>
          </div>

          {/* 3. Composite Overlay */}
          <div className="space-y-1.5 text-center">
            <span className="text-xs font-semibold text-zinc-700">3. Composite Overlay</span>
            <div className="aspect-square w-full overflow-hidden rounded-lg border border-border bg-zinc-100">
              <img
                src={gradcam.overlay}
                alt="Composite overlay"
                className="h-full w-full object-cover"
              />
            </div>
            <p className="text-[11px] text-zinc-400">Weighted blend (0.6× Image + 0.4× Heatmap)</p>
          </div>
        </div>
      )}

      {activeTab === 'overlay' && (
        <div className="flex flex-col items-center justify-center p-4">
          <div className="aspect-square max-w-md w-full overflow-hidden rounded-xl border border-border bg-zinc-100 shadow-sm">
            <img
              src={gradcam.overlay}
              alt="Composite overlay"
              className="h-full w-full object-cover"
            />
          </div>
          <span className="mt-2 text-xs font-medium text-zinc-600">Composite Overlay (0.6x Original + 0.4x Heatmap)</span>
        </div>
      )}

      {activeTab === 'heatmap' && (
        <div className="flex flex-col items-center justify-center p-4">
          <div className="aspect-square max-w-md w-full overflow-hidden rounded-xl border border-border bg-zinc-100 shadow-sm">
            <img
              src={gradcam.heatmap}
              alt="Raw heatmap"
              className="h-full w-full object-cover"
            />
          </div>
          <span className="mt-2 text-xs font-medium text-zinc-600">Raw Grad-CAM Activation Map</span>
        </div>
      )}

      {/* Scientific Interpretability Scope Notice */}
      <div className="rounded-lg bg-zinc-50 border border-zinc-200/70 p-3.5 text-xs text-zinc-600 flex items-start gap-2.5">
        <HelpCircle className="h-4 w-4 shrink-0 text-zinc-400 mt-0.5" />
        <div className="leading-relaxed">
          <span className="font-semibold text-zinc-900">Scientific Scope & Limitations: </span>
          Grad-CAM visualizes spatial regions in the input image that most strongly influenced the <strong>EfficientNetV2-S convolutional feature branch</strong>.
          It does <strong>not</strong> explain self-attention mechanisms in the Swin Transformer V2 branch or the clinical metadata fusion branch.
          This visualization is an interpretability aid for research and education, not proof of clinical reasoning.
        </div>
      </div>
    </div>
  );
};
