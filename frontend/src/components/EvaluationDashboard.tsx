import React, { useEffect, useState } from 'react';
import {
  Award,
  BarChart3,
  TrendingUp,
  ShieldCheck,
  Zap,
  XCircle,
  AlertTriangle,
  FileText,
  CheckCircle2,
  HelpCircle,
  ArrowRight,
  RefreshCw,
  Sliders,
  Scale
} from 'lucide-react';
import { fetchProjectStats } from '../api/client';
import { StatsResponse } from '../types/api';

export const EvaluationDashboard: React.FC = () => {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [loadingStats, setLoadingStats] = useState<boolean>(true);

  useEffect(() => {
    let isMounted = true;
    fetchProjectStats()
      .then((data) => {
        if (isMounted) setStats(data);
      })
      .catch(() => {
        // Silently handle if backend stats call fails, static validated numbers are rendered as fallback
      })
      .finally(() => {
        if (isMounted) setLoadingStats(false);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const retrievalMethods = [
    { name: 'Dense (all-MiniLM-L6-v2)', r1: 28.0, isWinner: false },
    { name: 'Intent-Aware Dense', r1: 51.5, isWinner: false },
    { name: 'BM25 Lexical', r1: 28.5, isWinner: false },
    { name: 'Hybrid RRF (k=60)', r1: 36.0, isWinner: false },
    { name: 'Hybrid + Intent (Selected Engine)', r1: 53.5, isWinner: true },
  ];

  return (
    <div className="py-12 px-4 lg:px-8 max-w-7xl mx-auto space-y-14">
      {/* Evaluation Hero */}
      <div className="text-center max-w-4xl mx-auto space-y-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300 text-xs font-semibold uppercase tracking-wider">
          <Award className="w-3.5 h-3.5" />
          <span>Empirical Validation & Benchmark Dashboard</span>
        </div>
        <h1 className="text-3xl sm:text-5xl font-extrabold text-white tracking-tight">
          Evaluation, Not Just Demos
        </h1>
        <p className="text-slate-300 text-base sm:text-lg max-w-3xl mx-auto leading-relaxed">
          Human-labeled validation, controlled retrieval benchmarks, real Gemini evaluation, and risk-aware escalation testing.
        </p>

        {stats && (
          <div className="pt-2 flex flex-wrap justify-center gap-3 text-xs">
            <span className="px-3 py-1 rounded-full bg-slate-900 border border-white/10 text-slate-300 font-mono">
              Corpus: <strong className="text-cyan-300">{stats.canonical_corpus_size.toLocaleString()} cases</strong>
            </span>
            <span className="px-3 py-1 rounded-full bg-slate-900 border border-white/10 text-slate-300 font-mono">
              Classifier: <strong className="text-purple-300">{stats.active_intent_classifier}</strong>
            </span>
            <span className="px-3 py-1 rounded-full bg-slate-900 border border-white/10 text-slate-300 font-mono">
              LLM: <strong className="text-pink-300">{stats.active_llm_model}</strong>
            </span>
          </div>
        )}
      </div>

      {/* SECTION 1 — INTENT CLASSIFICATION */}
      <section className="space-y-6">
        <div className="flex items-center gap-3 border-b border-white/10 pb-3">
          <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400 border border-purple-500/30">
            <BarChart3 className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-white">1. Intent Classification Evaluation</h2>
            <p className="text-xs text-slate-400">Strictly separated untouched human-gold test vs full-gold diagnostic</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Headline Headline Card */}
          <div className="glass-panel p-6 rounded-2xl border border-purple-500/30 space-y-4 relative overflow-hidden">
            <div className="absolute top-0 right-0 px-3 py-1 bg-purple-600 text-white font-bold text-[10px] uppercase tracking-wider rounded-bl-xl">
              Headline Generalization Benchmark
            </div>

            <div className="space-y-1">
              <span className="text-xs font-semibold text-purple-300 uppercase tracking-wider">Untouched Gold Test (N=100)</span>
              <div className="flex items-baseline gap-3">
                <span className="text-4xl sm:text-5xl font-black text-white">57.0%</span>
                <span className="text-sm font-semibold text-purple-300">Accuracy</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 pt-2 text-xs font-mono">
              <div className="p-3 rounded-xl bg-slate-950 border border-white/10">
                <span className="text-slate-400 block text-[10px]">Macro F1</span>
                <strong className="text-purple-300 text-base">53.93%</strong>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 border border-white/10">
                <span className="text-slate-400 block text-[10px]">Weighted F1</span>
                <strong className="text-purple-300 text-base">56.67%</strong>
              </div>
            </div>

            <p className="text-xs text-slate-300 bg-purple-500/10 p-3 rounded-xl border border-purple-500/20 leading-relaxed">
              "The untouched test set was separated from the development set and was not used for classifier fitting or tuning."
            </p>
          </div>

          {/* Full Gold Diagnostic Card */}
          <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-4">
            <div className="space-y-1">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Full Gold Diagnostic (N=200)</span>
              <div className="flex items-baseline gap-3">
                <span className="text-4xl sm:text-5xl font-black text-slate-200">69.0%</span>
                <span className="text-sm font-semibold text-slate-400">Diagnostic Accuracy</span>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-white/10 text-xs font-mono">
              <span className="text-slate-400 block text-[10px]">Diagnostic Macro F1</span>
              <strong className="text-slate-200 text-base">0.6930</strong>
            </div>

            <p className="text-xs text-amber-300/90 bg-amber-500/10 p-3 rounded-xl border border-amber-500/20 leading-relaxed">
              <strong className="text-amber-200 block mb-0.5">Metric Policy Note:</strong>
              Full-gold diagnostic; not the headline generalization result.
            </p>
          </div>
        </div>

        {/* High Confidence Error Result Subcard */}
        <div className="glass-panel p-6 rounded-2xl border border-emerald-500/30 bg-gradient-to-r from-slate-950 via-purple-950/20 to-slate-950 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/10 pb-3">
            <div>
              <h3 className="font-bold text-white text-base">High-Confidence Error Reduction</h3>
              <p className="text-xs text-slate-400">Confidence threshold &ge; 0.70 misclassifications</p>
            </div>
            <span className="px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 font-bold text-xs border border-emerald-500/30 self-start sm:self-auto">
              65.8% Error Reduction
            </span>
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-around gap-4 py-2">
            <div className="text-center p-4 rounded-xl bg-slate-950 border border-rose-500/30 w-full sm:w-auto min-w-[140px]">
              <span className="text-xs text-rose-400 font-semibold uppercase block">Baseline Errors</span>
              <span className="text-3xl font-black text-rose-400">38</span>
            </div>
            <div className="flex items-center justify-center">
              <ArrowRight className="w-6 h-6 text-purple-400 rotate-90 sm:rotate-0" />
            </div>
            <div className="text-center p-4 rounded-xl bg-slate-950 border border-emerald-500/30 w-full sm:w-auto min-w-[140px]">
              <span className="text-xs text-emerald-400 font-semibold uppercase block">Calibrated Errors</span>
              <span className="text-3xl font-black text-emerald-400">13</span>
            </div>
          </div>

          <p className="text-xs text-slate-300 bg-slate-950/80 p-3 rounded-xl border border-white/10">
            "This reduction reflects fewer confidently incorrect predictions in the evaluated classifier configuration."
          </p>
        </div>
      </section>

      {/* SECTION 2 — RETRIEVAL EVALUATION */}
      <section className="space-y-6">
        <div className="flex items-center gap-3 border-b border-white/10 pb-3">
          <div className="p-2 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
            <TrendingUp className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-white">2. Retrieval & RAG Evaluation</h2>
            <p className="text-xs text-slate-400">Authoritative Phase 9 benchmark and Day 6/7 regression reproduction</p>
          </div>
        </div>

        {/* Retrieval Benchmarks Side-by-Side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Phase 9 Benchmark */}
          <div className="glass-panel p-6 rounded-2xl border border-cyan-500/30 space-y-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <h3 className="font-bold text-white text-sm">Phase 9 Human-Gold Benchmark</h3>
              <span className="text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                Authoritative
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="p-2.5 rounded-xl bg-slate-950 border border-white/10">
                <span className="text-[10px] text-slate-400 block font-mono">R@1</span>
                <strong className="text-lg font-bold text-cyan-300">53.5%</strong>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950 border border-white/10">
                <span className="text-[10px] text-slate-400 block font-mono">R@3</span>
                <strong className="text-lg font-bold text-cyan-300">55.5%</strong>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950 border border-white/10">
                <span className="text-[10px] text-slate-400 block font-mono">R@5</span>
                <strong className="text-lg font-bold text-cyan-300">58.0%</strong>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 text-center">
              <div className="p-2.5 rounded-xl bg-slate-950 border border-white/10">
                <span className="text-[10px] text-slate-400 block font-mono">R@10</span>
                <strong className="text-lg font-bold text-cyan-300">65.5%</strong>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950 border border-white/10">
                <span className="text-[10px] text-slate-400 block font-mono">MRR</span>
                <strong className="text-lg font-bold text-cyan-300">~0.559</strong>
              </div>
            </div>
          </div>

          {/* Day 6/7 Reproduction */}
          <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-4">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <h3 className="font-bold text-slate-200 text-sm">Day 6/7 Regression Reproduction</h3>
              <span className="text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-white/10">
                Reproduction
              </span>
            </div>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="p-2.5 rounded-xl bg-slate-950 border border-white/10">
                <span className="text-[10px] text-slate-400 block font-mono">R@1</span>
                <strong className="text-lg font-bold text-slate-200">52.5%</strong>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950 border border-white/10">
                <span className="text-[10px] text-slate-400 block font-mono">R@3</span>
                <strong className="text-lg font-bold text-slate-200">56.0%</strong>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950 border border-white/10">
                <span className="text-[10px] text-slate-400 block font-mono">R@5</span>
                <strong className="text-lg font-bold text-slate-200">61.0%</strong>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 text-center">
              <div className="p-2.5 rounded-xl bg-slate-950 border border-white/10">
                <span className="text-[10px] text-slate-400 block font-mono">R@10</span>
                <strong className="text-lg font-bold text-slate-200">65.0%</strong>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-950 border border-white/10">
                <span className="text-[10px] text-slate-400 block font-mono">MRR</span>
                <strong className="text-lg font-bold text-slate-200">0.5570</strong>
              </div>
            </div>
          </div>
        </div>

        {/* Retrieval Method Comparison Bar Graph Visual */}
        <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-4">
          <h3 className="font-bold text-white text-base">Retrieval Method Ablation Comparison (Top-1 Recall / R@1)</h3>
          <div className="space-y-3 pt-2">
            {retrievalMethods.map((m, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className={`font-medium ${m.isWinner ? 'text-cyan-300 font-bold' : 'text-slate-300'}`}>
                    {m.name} {m.isWinner && '★ (Winning Configuration)'}
                  </span>
                  <span className="font-mono font-bold text-cyan-400">{m.r1.toFixed(1)}%</span>
                </div>
                <div className="h-3 w-full bg-slate-950 rounded-full overflow-hidden border border-white/5">
                  <div
                    className={`h-full rounded-full transition-all duration-1000 ${
                      m.isWinner
                        ? 'bg-gradient-to-r from-cyan-500 to-blue-500 shadow-glow-cyan'
                        : 'bg-slate-700/60'
                    }`}
                    style={{ width: `${(m.r1 / 60) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-300 bg-slate-950 p-3 rounded-xl border border-white/10">
            "Intent-aware compatibility ranking substantially improved top-1 retrieval on the evaluated human-gold benchmark."
          </p>
        </div>
      </section>

      {/* SECTION 3 & 4 — GENERATION & ESCALATION */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* SECTION 3 — GENERATION */}
        <section className="glass-panel p-6 rounded-2xl border border-pink-500/30 space-y-5">
          <div className="flex items-center gap-3 border-b border-white/10 pb-3">
            <div className="p-2 rounded-lg bg-pink-500/20 text-pink-400 border border-pink-500/30">
              <Zap className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-extrabold text-white">3. Response Generation (Gemini 1.5)</h2>
              <p className="text-xs text-slate-400">LLM judge evaluation on N=50 deterministic sample</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="p-4 rounded-xl bg-slate-950 border border-white/10 text-center">
              <span className="text-[10px] text-slate-400 uppercase block font-semibold">LLM Judge Score</span>
              <span className="text-3xl font-black text-pink-400">4.72 <span className="text-xs text-slate-400 font-normal">/ 5</span></span>
            </div>
            <div className="p-4 rounded-xl bg-slate-950 border border-white/10 text-center">
              <span className="text-[10px] text-slate-400 uppercase block font-semibold">Safety Score</span>
              <span className="text-3xl font-black text-emerald-400">100%</span>
            </div>
          </div>

          <p className="text-xs text-slate-300 bg-slate-950 p-3.5 rounded-xl border border-white/10 leading-relaxed">
            "Responses were evaluated using a real Gemini-based judge on a deterministic 50-case sample."
          </p>
        </section>

        {/* SECTION 4 — ESCALATION TRADE-OFF */}
        <section className="glass-panel p-6 rounded-2xl border border-amber-500/30 space-y-5">
          <div className="flex items-center gap-3 border-b border-white/10 pb-3">
            <div className="p-2 rounded-lg bg-amber-500/20 text-amber-400 border border-amber-500/30">
              <Scale className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-extrabold text-white">4. Risk Policy & Escalation Trade-off</h2>
              <p className="text-xs text-slate-400">N=50 human agreement and risk-aware policy threshold 0.35</p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
            <div className="p-3 rounded-xl bg-slate-950 border border-white/10">
              <span className="text-[10px] text-slate-400 block">Agreement</span>
              <strong className="text-emerald-400 text-lg">74%</strong>
            </div>
            <div className="p-3 rounded-xl bg-slate-950 border border-white/10">
              <span className="text-[10px] text-slate-400 block">False Auto</span>
              <strong className="text-emerald-400 text-lg">6%</strong>
            </div>
            <div className="p-3 rounded-xl bg-slate-950 border border-white/10">
              <span className="text-[10px] text-slate-400 block">False Escalation</span>
              <strong className="text-amber-400 text-lg">20%</strong>
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-950 border border-white/10 text-xs space-y-2">
            <div className="flex items-center justify-between font-bold">
              <span className="text-emerald-400">SAFER AUTOMATION:</span>
              <span className="text-slate-300">False auto: 12% &rarr; 6%</span>
            </div>
            <div className="flex items-center justify-between font-bold">
              <span className="text-amber-400">TRADE-OFF COST:</span>
              <span className="text-slate-300">False escalation: 14% &rarr; 20%</span>
            </div>
          </div>

          <p className="text-xs text-slate-300 bg-slate-950 p-3 rounded-xl border border-white/10 leading-relaxed">
            "The final classifier/policy configuration became more conservative: unsafe auto-handling decreased while unnecessary escalation increased."
          </p>
        </section>
      </div>

      {/* SECTION 5 — WHAT FAILED (EXPERIMENTS WE REJECTED) */}
      <section className="space-y-4">
        <div className="flex items-center gap-3 border-b border-white/10 pb-3">
          <div className="p-2 rounded-lg bg-rose-500/20 text-rose-400 border border-rose-500/30">
            <XCircle className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-white">5. Experiments We Rejected</h2>
            <p className="text-xs text-slate-400">Negative results and rejected architectural candidates</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-5 rounded-2xl glass-panel border border-rose-500/30 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-rose-400 uppercase">Hierarchical Classifier</span>
              <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 text-[10px] font-bold border border-rose-500/30">REJECTED</span>
            </div>
            <p className="text-2xl font-black text-white">26.0% <span className="text-xs text-slate-400 font-normal">Accuracy</span></p>
            <p className="text-xs text-slate-400 leading-relaxed">
              Collapsed under class imbalance on dev set; stage-1 errors cascaded to child predictions.
            </p>
          </div>

          <div className="p-5 rounded-2xl glass-panel border border-rose-500/30 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-rose-400 uppercase">MS-MARCO Cross-Encoder</span>
              <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 text-[10px] font-bold border border-rose-500/30">REJECTED</span>
            </div>
            <p className="text-2xl font-black text-white">35.5% <span className="text-xs text-slate-400 font-normal">R@1</span></p>
            <p className="text-xs text-slate-400 leading-relaxed">
              Underperformed intent-aware hybrid ranking on domain-specific e-commerce queries.
            </p>
          </div>

          <div className="p-5 rounded-2xl glass-panel border border-rose-500/30 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-rose-400 uppercase">SentenceTransformer Reranker</span>
              <span className="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 text-[10px] font-bold border border-rose-500/30">REJECTED</span>
            </div>
            <p className="text-2xl font-black text-white">47.0% <span className="text-xs text-slate-400 font-normal">R@1</span></p>
            <p className="text-xs text-slate-400 leading-relaxed">
              Added latency without beating the 53.5% R@1 achieved by hybrid RRF + intent boost.
            </p>
          </div>
        </div>

        <p className="text-xs text-slate-300 bg-slate-950 p-4 rounded-xl border border-white/10 leading-relaxed">
          "More sophisticated components were not automatically better. Components were retained only when they improved the relevant evaluation objective."
        </p>
      </section>

      {/* SECTION 6 — REAL SYSTEM LIMITATIONS */}
      <section className="space-y-4">
        <div className="flex items-center gap-3 border-b border-white/10 pb-3">
          <div className="p-2 rounded-lg bg-amber-500/20 text-amber-400 border border-amber-500/30">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-white">6. System Limitations</h2>
            <p className="text-xs text-slate-400">Transparent technical caveats and operational bounds</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-xl glass-panel border border-white/10 space-y-1">
            <strong className="text-amber-300 font-bold block text-sm">1. Taxonomy Ambiguity</strong>
            <p className="text-slate-300">
              Taxonomy ambiguity remains a major source of classification errors (e.g. distinguishing refund inquiries vs order cancellations).
            </p>
          </div>

          <div className="p-4 rounded-xl glass-panel border border-white/10 space-y-1">
            <strong className="text-amber-300 font-bold block text-sm">2. Untouched Gold Test N=100</strong>
            <p className="text-slate-300">
              Sample size N=100 provides strict test isolation but has higher variance across rare intent categories.
            </p>
          </div>

          <div className="p-4 rounded-xl glass-panel border border-white/10 space-y-1">
            <strong className="text-amber-300 font-bold block text-sm">3. False Escalation Rate (20%)</strong>
            <p className="text-slate-300">
              Prioritizing safety leads to 20% unnecessary human escalation on borderline cases.
            </p>
          </div>

          <div className="p-4 rounded-xl glass-panel border border-white/10 space-y-1">
            <strong className="text-amber-300 font-bold block text-sm">4. Retrieval Relevance Proxy</strong>
            <p className="text-slate-300">
              Retrieval relevance is proxy-based rather than fully manually judged for all 15,316 corpus documents.
            </p>
          </div>

          <div className="p-4 rounded-xl glass-panel border border-white/10 space-y-1 md:col-span-2">
            <strong className="text-amber-300 font-bold block text-sm">5. Embedding Cold-Start Latency</strong>
            <p className="text-slate-300">
              First-query execution incurs a minor initialization delay while FAISS index and MiniLM model load into memory.
            </p>
          </div>
        </div>
      </section>

      {/* SECTION 7 — METHODOLOGY PIPELINE */}
      <section className="space-y-4">
        <div className="flex items-center gap-3 border-b border-white/10 pb-3">
          <div className="p-2 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
            <FileText className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-white">7. Evaluation Methodology</h2>
            <p className="text-xs text-slate-400">End-to-end disciplined validation workflow</p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-6 gap-3 text-center text-xs">
          <div className="p-3 rounded-xl bg-slate-950 border border-purple-500/30 space-y-1">
            <span className="font-bold text-purple-300 block text-[11px]">1. Gold Labels</span>
            <p className="text-[10px] text-slate-400">Human annotations</p>
          </div>
          <div className="p-3 rounded-xl bg-slate-950 border border-cyan-500/30 space-y-1">
            <span className="font-bold text-cyan-300 block text-[11px]">2. Dev/Test Split</span>
            <p className="text-[10px] text-slate-400">Strict separation</p>
          </div>
          <div className="p-3 rounded-xl bg-slate-950 border border-pink-500/30 space-y-1">
            <span className="font-bold text-pink-300 block text-[11px]">3. Benchmarks</span>
            <p className="text-[10px] text-slate-400">R@1, R@5, MRR</p>
          </div>
          <div className="p-3 rounded-xl bg-slate-950 border border-emerald-500/30 space-y-1">
            <span className="font-bold text-emerald-300 block text-[11px]">4. Gemini Judge</span>
            <p className="text-[10px] text-slate-400">5-point quality scale</p>
          </div>
          <div className="p-3 rounded-xl bg-slate-950 border border-amber-500/30 space-y-1">
            <span className="font-bold text-amber-300 block text-[11px]">5. Escalation</span>
            <p className="text-[10px] text-slate-400">Human policy labels</p>
          </div>
          <div className="p-3 rounded-xl bg-slate-950 border border-blue-500/30 space-y-1">
            <span className="font-bold text-blue-300 block text-[11px]">6. Regression</span>
            <p className="text-[10px] text-slate-400">Automated test suite</p>
          </div>
        </div>
      </section>

      {/* SECTION 8 — WHAT IS MISLEADING ABOUT MY HEADLINE NUMBER? */}
      <section className="space-y-4">
        <div className="flex items-center gap-3 border-b border-white/10 pb-3">
          <div className="p-2 rounded-lg bg-purple-500/20 text-purple-400 border border-purple-500/30">
            <HelpCircle className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-white">8. What is Misleading About My Headline Number?</h2>
            <p className="text-xs text-slate-400">Critical evaluation reflection on weak-label inflation vs human-gold truth</p>
          </div>
        </div>

        <div className="p-6 rounded-2xl glass-panel border border-purple-500/30 space-y-4 text-xs leading-relaxed text-slate-200">
          <p>
            During initial development, weak-label validation on heuristically annotated datasets produced a seemingly impressive <strong className="text-purple-300 font-bold">~87% classifier accuracy</strong>. However, those weak labels were generated using keyword heuristics that shared identical patterns with the training rules, creating artificial evaluation inflation.
          </p>
          <p>
            When evaluated against strict, double-annotated <strong className="text-white font-bold">Human Gold Test cases (N=100)</strong>, accuracy dropped to <strong className="text-purple-300 font-bold">57.0% (Macro F1: 53.93%)</strong>. 
          </p>
          <div className="p-4 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-200 font-medium">
            <strong className="block text-white mb-1">Key Engineering Takeaway:</strong>
            "Headline generalization metrics must be measured on strictly separated, untouched human-gold benchmark data. Claiming ~87% accuracy based on weak heuristic labels is misleading because it measures rule memorization rather than real-world customer query generalization."
          </div>
        </div>
      </section>

      {/* SECTION 9 — ONE-WEEK NEXT STEPS */}
      <section className="space-y-4">
        <div className="flex items-center gap-3 border-b border-white/10 pb-3">
          <div className="p-2 rounded-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
            <Sliders className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-extrabold text-white">9. One-Week Next Steps</h2>
            <p className="text-xs text-slate-400">Prioritized engineering roadmap for system enhancement</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-xl glass-panel border border-white/10 space-y-1.5">
            <strong className="text-emerald-300 font-bold block text-sm">1. Expand Human Gold Set</strong>
            <p className="text-slate-300">
              Expand human-gold intent labels from N=200 to N=1,000 to reduce test sample variance across rare categories.
            </p>
          </div>

          <div className="p-4 rounded-xl glass-panel border border-white/10 space-y-1.5">
            <strong className="text-emerald-300 font-bold block text-sm">2. Refine Taxonomy Boundaries</strong>
            <p className="text-slate-300">
              Publish formal annotation guidelines to resolve intent confusion between refund requests vs order cancellations.
            </p>
          </div>

          <div className="p-4 rounded-xl glass-panel border border-white/10 space-y-1.5">
            <strong className="text-emerald-300 font-bold block text-sm">3. Pairwise Relevance Labels</strong>
            <p className="text-slate-300">
              Collect human pairwise relevance judgments for retrieval rather than relying on intent-matching proxies.
            </p>
          </div>

          <div className="p-4 rounded-xl glass-panel border border-white/10 space-y-1.5">
            <strong className="text-emerald-300 font-bold block text-sm">4. Fine-Tuned Reranker</strong>
            <p className="text-slate-300">
              Train a domain-specific cross-encoder reranker directly on e-commerce customer support resolution pairs.
            </p>
          </div>

          <div className="p-4 rounded-xl glass-panel border border-white/10 space-y-1.5">
            <strong className="text-emerald-300 font-bold block text-sm">5. Multi-Annotator Policy Tuning</strong>
            <p className="text-slate-300">
              Calibrate RiskAwarePolicy thresholds using multi-agent consensus scoring to optimize the escalation trade-off.
            </p>
          </div>

          <div className="p-4 rounded-xl glass-panel border border-white/10 space-y-1.5">
            <strong className="text-emerald-300 font-bold block text-sm">6. Multi-Turn Dialogue State</strong>
            <p className="text-slate-300">
              Add persistent conversation memory to handle multi-turn customer dialogues and clarifying question flows.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
