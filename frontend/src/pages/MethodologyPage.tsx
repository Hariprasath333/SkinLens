import React from 'react';
import { Cpu, Database, Compass, Layers, ShieldAlert, BarChart2, CheckCircle2 } from 'lucide-react';
import { CLASS_NAMES, CLASS_DETAILS } from '../types';

export const MethodologyPage: React.FC = () => {
  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 lg:px-8 space-y-12">
      
      {/* Title */}
      <div className="border-b border-border pb-6 space-y-2">
        <div className="inline-flex items-center gap-1.5 rounded-md bg-zinc-100 px-2 py-0.5 text-xs font-semibold text-zinc-700">
          <Database className="h-3.5 w-3.5 text-zinc-500" />
          <span>Research Methodology & Technical Architecture</span>
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-zinc-950 sm:text-4xl">
          Multimodal Deep Learning Architecture
        </h1>
        <p className="text-sm text-zinc-600 leading-relaxed max-w-3xl">
          SkinLens combines dual high-capacity vision backbones with clinical metadata feature fusion,
          post-hoc temperature scaling, and layer-specific gradient-weighted class activation mapping (Grad-CAM).
        </p>
      </div>

      {/* 1. Dataset Section */}
      <section className="rounded-xl border border-border bg-white p-6 shadow-card space-y-4">
        <h2 className="text-lg font-bold text-zinc-900 flex items-center gap-2">
          <Database className="h-4 w-4 text-zinc-500" />
          <span>1. The HAM10000 Benchmark Dataset</span>
        </h2>
        <p className="text-xs sm:text-sm text-zinc-600 leading-relaxed">
          The <strong>HAM10000</strong> ("Human Against Machine with 10000 training images") dataset is a benchmark
          collection of 10,015 dermatoscopic images acquired from the Department of Dermatology at the Medical University of Vienna
          and the skin cancer practice of Cliff Rosendahl in Queensland, Australia. Cases represent a diverse range of skin pigmentation
          and lesion types, validated via histopathology (53.3%), expert follow-up (33.2%), or in-vivo confocal microscopy.
        </p>

        {/* 7 Class Table */}
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-left text-xs">
            <thead className="bg-zinc-50 border-b border-border text-zinc-500 font-medium">
              <tr>
                <th className="py-2.5 px-3">Idx</th>
                <th className="py-2.5 px-3">Class</th>
                <th className="py-2.5 px-3">Category Name</th>
                <th className="py-2.5 px-3">Pathological Type</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border text-zinc-700">
              {CLASS_NAMES.map((cls, idx) => {
                const det = CLASS_DETAILS[cls];
                return (
                  <tr key={cls} className="hover:bg-zinc-50/50">
                    <td className="py-2 px-3 font-mono text-zinc-400">{idx}</td>
                    <td className="py-2 px-3 font-mono font-semibold text-zinc-900 uppercase">{det.code}</td>
                    <td className="py-2 px-3 font-medium text-zinc-900">{det.name}</td>
                    <td className="py-2 px-3 text-zinc-500">{det.category}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* 2. Architecture Diagram Section */}
      <section className="rounded-xl border border-border bg-white p-6 shadow-card space-y-6">
        <div>
          <h2 className="text-lg font-bold text-zinc-900 flex items-center gap-2">
            <Cpu className="h-4 w-4 text-zinc-500" />
            <span>2. Dual-Backbone Multimodal Architecture</span>
          </h2>
          <p className="text-xs sm:text-sm text-zinc-600 mt-1">
            SkinLens implements an ensemble-style feature fusion pipeline connecting convolutional features, hierarchical vision transformer tokens, and a tabular metadata embedding network.
          </p>
        </div>

        {/* Visual Architecture Representation */}
        <div className="rounded-lg border border-zinc-200 bg-zinc-50/60 p-5 font-mono text-xs space-y-4">
          <div className="text-center font-sans font-semibold text-zinc-800 text-sm">
            SkinLensHybrid Architecture Flow
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-center">
            {/* Branch 1 */}
            <div className="rounded-md border border-zinc-200 bg-white p-3 shadow-2xs space-y-1">
              <div className="font-semibold text-zinc-900 font-sans">Vision Branch A</div>
              <div className="text-[11px] text-zinc-500">EfficientNetV2-S</div>
              <div className="text-[10px] text-zinc-400">tf_efficientnetv2_s.in21k_ft_in1k</div>
              <div className="mt-2 text-primary-600 font-bold">1,280 features</div>
            </div>

            {/* Branch 2 */}
            <div className="rounded-md border border-zinc-200 bg-white p-3 shadow-2xs space-y-1">
              <div className="font-semibold text-zinc-900 font-sans">Vision Branch B</div>
              <div className="text-[11px] text-zinc-500">Swin Transformer V2</div>
              <div className="text-[10px] text-zinc-400">swinv2_base_window12to24_192to384</div>
              <div className="mt-2 text-primary-600 font-bold">1,024 features</div>
            </div>

            {/* Branch 3 */}
            <div className="rounded-md border border-zinc-200 bg-white p-3 shadow-2xs space-y-1">
              <div className="font-semibold text-zinc-900 font-sans">Metadata Branch</div>
              <div className="text-[11px] text-zinc-500">MLP (Age + Sex + Site)</div>
              <div className="text-[10px] text-zinc-400">21-dim vector → Linear(21, 64) → Linear(64, 32)</div>
              <div className="mt-2 text-primary-600 font-bold">32 features</div>
            </div>
          </div>

          <div className="text-center text-zinc-400 text-sm">↓ Feature Concatenation</div>

          <div className="rounded-md border border-zinc-300 bg-white p-3 text-center shadow-2xs">
            <div className="font-sans font-semibold text-zinc-900">
              Multimodal Fused Representation Vector
            </div>
            <div className="text-zinc-600 mt-0.5">
              1,280 (EffNet) + 1,024 (SwinV2) + 32 (Metadata) = <span className="font-bold text-zinc-900">2,336 features</span>
            </div>
          </div>

          <div className="text-center text-zinc-400 text-sm">↓ Classification Head</div>

          <div className="rounded-md border border-zinc-200 bg-white p-3 text-center shadow-2xs">
            <div className="font-sans font-semibold text-zinc-900">
              Linear(2,336 → 512) → LayerNorm → GELU → Dropout(0.4) → Linear(512 → 7)
            </div>
            <div className="text-[11px] text-zinc-500 mt-1">
              Output Logits: 7-class logits [akiec, bcc, bkl, df, mel, nv, vasc]
            </div>
          </div>
        </div>
      </section>

      {/* 3. Calibration & Grad-CAM */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Temperature Scaling */}
        <div className="rounded-xl border border-border bg-white p-6 shadow-card space-y-3">
          <h2 className="text-base font-bold text-zinc-900 flex items-center gap-2">
            <Compass className="h-4 w-4 text-zinc-500" />
            <span>3. Temperature Calibration</span>
          </h2>
          <p className="text-xs text-zinc-600 leading-relaxed">
            Deep neural networks frequently suffer from overconfidence. SkinLens applies post-hoc <strong>Global Temperature Scaling</strong>,
            multiplying logit denominators by fitted temperature scalar <code className="font-mono text-zinc-800">T</code>:
          </p>
          <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3 text-center font-mono text-xs text-zinc-800">
            P_i = exp(z_i / T) / Σ_j exp(z_j / T)
          </div>
          <p className="text-xs text-zinc-600 leading-relaxed">
            Fitted strictly on validation set log-likelihood: <code className="font-mono text-zinc-900 font-semibold">T = 1.064053</code>.
            Lowers the Expected Calibration Error (ECE) from <strong>0.1587</strong> to <strong>0.1411</strong> on the held-out test set.
          </p>
        </div>

        {/* Grad-CAM */}
        <div className="rounded-xl border border-border bg-white p-6 shadow-card space-y-3">
          <h2 className="text-base font-bold text-zinc-900 flex items-center gap-2">
            <Layers className="h-4 w-4 text-zinc-500" />
            <span>4. Grad-CAM Interpretability</span>
          </h2>
          <p className="text-xs text-zinc-600 leading-relaxed">
            Grad-CAM calculates the gradient of the predicted class score with respect to feature activation maps of
            layer <code className="font-mono text-zinc-800 text-[11px]">model.effnet.conv_head</code> (1,280 channels):
          </p>
          <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3 text-center font-mono text-xs text-zinc-800">
            α_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)
          </div>
          <p className="text-xs text-zinc-600 leading-relaxed">
            Spatially explains the <strong>EfficientNetV2-S convolutional branch only</strong>. It does not reflect attention in Swin Transformer V2
            or metadata features, acting as an interpretability aid rather than complete proof of model reasoning.
          </p>
        </div>
      </section>

      {/* 4. Documented Research Evaluation Results */}
      <section className="rounded-xl border border-border bg-white p-6 shadow-card space-y-4">
        <div>
          <h2 className="text-lg font-bold text-zinc-900 flex items-center gap-2">
            <BarChart2 className="h-4 w-4 text-zinc-500" />
            <span>5. Documented Research Test Evaluation</span>
          </h2>
          <p className="text-xs text-zinc-500 mt-0.5">
            Evaluated on held-out test split (2,014 samples) from the verified project training session.
          </p>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="rounded-lg border border-border bg-zinc-50 p-3 text-center">
            <div className="font-mono text-xl font-bold text-zinc-900">85.45%</div>
            <div className="text-[11px] text-zinc-500 font-medium">Test Accuracy</div>
          </div>
          <div className="rounded-lg border border-border bg-zinc-50 p-3 text-center">
            <div className="font-mono text-xl font-bold text-zinc-900">76.03%</div>
            <div className="text-[11px] text-zinc-500 font-medium">Macro-F1 Score</div>
          </div>
          <div className="rounded-lg border border-border bg-zinc-50 p-3 text-center">
            <div className="font-mono text-xl font-bold text-zinc-900">78.49%</div>
            <div className="text-[11px] text-zinc-500 font-medium">Macro-Precision</div>
          </div>
          <div className="rounded-lg border border-border bg-zinc-50 p-3 text-center">
            <div className="font-mono text-xl font-bold text-zinc-900">76.47%</div>
            <div className="text-[11px] text-zinc-500 font-medium">Macro-Recall</div>
          </div>
        </div>

        {/* Per-class Metrics Breakdown Table */}
        <div className="overflow-x-auto rounded-lg border border-border mt-3">
          <table className="w-full text-left text-xs">
            <thead className="bg-zinc-50 border-b border-border text-zinc-500 font-medium">
              <tr>
                <th className="py-2 px-3">Class</th>
                <th className="py-2 px-3">Precision</th>
                <th className="py-2 px-3">Recall</th>
                <th className="py-2 px-3">F1-Score</th>
                <th className="py-2 px-3">Test Support</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border text-zinc-700 font-mono">
              <tr><td className="py-2 px-3 font-semibold text-zinc-900">akiec</td><td>0.71</td><td>0.81</td><td>0.76</td><td>68</td></tr>
              <tr><td className="py-2 px-3 font-semibold text-zinc-900">bcc</td><td>0.80</td><td>0.86</td><td>0.83</td><td>100</td></tr>
              <tr><td className="py-2 px-3 font-semibold text-zinc-900">bkl</td><td>0.76</td><td>0.81</td><td>0.78</td><td>205</td></tr>
              <tr><td className="py-2 px-3 font-semibold text-zinc-900">df</td><td>0.82</td><td>0.41</td><td>0.55</td><td>22</td></tr>
              <tr><td className="py-2 px-3 font-semibold text-zinc-900">mel</td><td>0.54</td><td>0.79</td><td>0.64</td><td>222</td></tr>
              <tr><td className="py-2 px-3 font-semibold text-zinc-900">nv</td><td>0.97</td><td>0.88</td><td>0.92</td><td>1363</td></tr>
              <tr><td className="py-2 px-3 font-semibold text-zinc-900">vasc</td><td>0.90</td><td>0.79</td><td>0.84</td><td>34</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 5. Scientific Limitations */}
      <section className="rounded-xl border border-amber-200 bg-amber-50/40 p-6 space-y-3">
        <h2 className="text-base font-bold text-amber-950 flex items-center gap-2">
          <ShieldAlert className="h-4 w-4 text-amber-700" />
          <span>6. Model Limitations & Ethical Considerations</span>
        </h2>
        <ul className="list-disc pl-5 space-y-1.5 text-xs text-amber-900 leading-relaxed">
          <li>
            <strong>Class Imbalance:</strong> The HAM10000 dataset has pronounced class distribution disparity (melanocytic nevi comprise over 67% of cases), introducing potential sensitivity imbalances across rare lesion categories like dermatofibroma.
          </li>
          <li>
            <strong>Demographic Generalization:</strong> The dataset predominantly reflects fair-skinned European and Australian patient cohorts (Fitzpatrick skin types I–III). Performance on darker skin types (Fitzpatrick types IV–VI) remains unestablished.
          </li>
          <li>
            <strong>Artifact Vulnerability:</strong> Dermoscopic artifacts such as surgical skin markers, dermatoscope air bubbles, and dense hair follicles can alter feature activations.
          </li>
          <li>
            <strong>No Clinical Approval:</strong> This model is not certified as Software as a Medical Device (SaMD) by the FDA, CE, or CDSCO.
          </li>
        </ul>
      </section>

    </div>
  );
};
