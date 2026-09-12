import React from 'react';
import { Award, CheckCircle2, FileCheck, BarChart3, HelpCircle } from 'lucide-react';

export const EvaluationSection: React.FC = () => {
  return (
    <section className="py-20 px-4 lg:px-8 border-b border-white/10 relative">
      <div className="max-w-7xl mx-auto space-y-12">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300 text-xs font-semibold uppercase tracking-wider">
            <Award className="w-3.5 h-3.5" />
            <span>Empirical Validation</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            Authoritative Benchmark Metrics
          </h2>
          <p className="text-slate-300 text-base">
            All system components have been empirically evaluated against human-gold datasets with strict test set isolation and reproducibility protocols.
          </p>
        </div>

        {/* Metric Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Card 1: Intent Classifier */}
          <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <h3 className="font-bold text-white text-base">Intent Classification</h3>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">N=100 Test</span>
            </div>
            <div className="space-y-3">
              <div>
                <span className="text-xs text-slate-400">Untouched Test Accuracy</span>
                <p className="text-3xl font-extrabold text-purple-400">57.0%</p>
                <p className="text-[11px] text-slate-400">Macro F1: 0.5393 | Weighted F1: 0.5667</p>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1 text-xs">
                <span className="font-semibold text-slate-200">High-Confidence Errors (≥ 0.70):</span>
                <p className="text-emerald-400 font-bold">13 errors (down 65.8% from 38)</p>
              </div>
            </div>
          </div>

          {/* Card 2: Hybrid Retrieval */}
          <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <h3 className="font-bold text-white text-base">Hybrid RAG Retrieval</h3>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300">N=200 Benchmark</span>
            </div>
            <div className="space-y-3">
              <div>
                <span className="text-xs text-slate-400">Top-1 Recall (R@1)</span>
                <p className="text-3xl font-extrabold text-cyan-400">52.5% <span className="text-xs text-slate-400 font-normal">(Phase 9: 53.5%)</span></p>
                <p className="text-[11px] text-slate-400">R@5: 61.0% | R@10: 65.0% | MRR: 0.5570</p>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1 text-xs">
                <span className="font-semibold text-slate-200">Retrieval Engine:</span>
                <p className="text-cyan-300 font-medium">Dense FAISS + BM25 RRF (k=60) + Intent Boost</p>
              </div>
            </div>
          </div>

          {/* Card 3: Gemini 1.5 Flash Response */}
          <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <h3 className="font-bold text-white text-base">Response Generation</h3>
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-pink-500/20 text-pink-300">N=50 Gemini</span>
            </div>
            <div className="space-y-3">
              <div>
                <span className="text-xs text-slate-400">Overall LLM Judge Quality</span>
                <p className="text-3xl font-extrabold text-pink-400">4.72 <span className="text-xs text-slate-400 font-normal">/ 5.00</span></p>
                <p className="text-[11px] text-slate-400">Grounding Pass: 98.0% | Safety Pass: 100%</p>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1 text-xs">
                <span className="font-semibold text-slate-200">LLM Model:</span>
                <p className="text-pink-300 font-medium">Gemini 1.5 Flash (temp=0.2, max_tokens=500)</p>
              </div>
            </div>
          </div>
        </div>

        {/* Diagnostic vs Untouched Protocol Note */}
        <div className="p-4 rounded-xl bg-purple-500/10 border border-purple-500/20 text-xs text-slate-300 flex items-start gap-3">
          <HelpCircle className="w-5 h-5 text-purple-400 shrink-0 mt-0.5" />
          <div>
            <strong className="text-white font-semibold">Metric Protocol Note:</strong> The 57.0% intent accuracy metric is measured strictly on N=100 untouched test examples with zero dev-label leakage. Full Gold diagnostic accuracy across all 200 cases is 69.0% (Macro F1: 0.6930).
          </div>
        </div>
      </div>
    </section>
  );
};
