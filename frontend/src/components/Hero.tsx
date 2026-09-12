import React from 'react';
import { Sparkles, ArrowRight, ShieldCheck, Cpu, Database, Layers, CheckCircle2, Bot } from 'lucide-react';

interface HeroProps {
  onStartDemo: () => void;
  onExploreArchitecture: () => void;
}

export const Hero: React.FC<HeroProps> = ({ onStartDemo, onExploreArchitecture }) => {
  return (
    <section className="relative overflow-hidden pt-10 pb-16 px-4 lg:px-8 bg-grid-pattern">
      {/* Ambient Lighting & Glow Orbs */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[450px] bg-purple-600/15 rounded-full blur-[150px] pointer-events-none animate-pulse-glow" />
      <div className="absolute top-1/3 right-10 w-[450px] h-[350px] bg-cyan-600/10 rounded-full blur-[130px] pointer-events-none" />

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-12 items-center relative z-10">
        {/* Left Hero Column */}
        <div className="lg:col-span-7 space-y-8 text-center lg:text-left">
          {/* Pill Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300 text-xs font-semibold uppercase tracking-wider shadow-sm">
            <Sparkles className="w-3.5 h-3.5 text-purple-400" />
            <span>Hiver SDE Take-Home Challenge • Portfolio Product</span>
          </div>

          {/* Main Headline */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight text-white leading-[1.1]">
            Understand. Resolve.{' '}
            <span className="gradient-text-purple block mt-1.5">Delight.</span>
          </h1>

          {/* Subheadline */}
          <p className="text-base sm:text-lg text-slate-300 font-normal leading-relaxed max-w-2xl mx-auto lg:mx-0">
            An AI-powered customer support agent built for <strong className="text-purple-300 font-semibold">AmazonHelp</strong>. Grounded in historical support conversations via hybrid vector-sparse RAG, entity safety verification, and risk-aware human escalation.
          </p>

          {/* Action CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4 pt-1">
            <button
              onClick={onStartDemo}
              className="w-full sm:w-auto px-7 py-4 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-bold text-sm shadow-glow-violet hover:shadow-purple-500/50 transition-all flex items-center justify-center gap-3 group"
            >
              <span>Try Live Demo</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>

            <button
              onClick={onExploreArchitecture}
              className="w-full sm:w-auto px-7 py-4 rounded-xl glass-panel glass-panel-hover text-slate-200 font-semibold text-sm flex items-center justify-center gap-2 border border-white/10"
            >
              <Layers className="w-4 h-4 text-cyan-400" />
              <span>Explore Architecture</span>
            </button>
          </div>

          {/* Capability Badges */}
          <div className="pt-2 flex flex-wrap items-center justify-center lg:justify-start gap-6 text-xs font-medium text-slate-400">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>17 Support Intents</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Hybrid Dense + BM25 RAG</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Gemini 1.5 Flash</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Risk Threshold = 0.35</span>
            </div>
          </div>
        </div>

        {/* Right Hero Visual Console Card */}
        <div className="lg:col-span-5">
          <div className="relative glass-panel rounded-2xl p-6 border border-white/10 shadow-2xl space-y-6 animate-float">
            {/* Top Glowing Bar */}
            <div className="absolute top-0 left-8 right-8 h-1 bg-gradient-to-r from-purple-500 via-pink-500 to-cyan-500 rounded-b-full" />

            {/* Visual Header */}
            <div className="flex items-center justify-between border-b border-white/10 pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-xl bg-purple-500/20 text-purple-400 border border-purple-500/30">
                  <Bot className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-white text-sm">Hiver Agent Console</h3>
                  <p className="text-xs text-slate-400">Active RAG Engine • AmazonHelp</p>
                </div>
              </div>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-semibold border border-emerald-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                Live Engine
              </span>
            </div>

            {/* Model Badges Grid */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-slate-950/80 border border-white/5 space-y-1">
                <span className="text-slate-400 text-[10px] uppercase font-semibold">Classifier</span>
                <p className="font-bold text-purple-300 truncate">Calibrated Hybrid</p>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/80 border border-white/5 space-y-1">
                <span className="text-slate-400 text-[10px] uppercase font-semibold">Retrieval</span>
                <p className="font-bold text-cyan-300 truncate">Dense + BM25 RRF</p>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/80 border border-white/5 space-y-1">
                <span className="text-slate-400 text-[10px] uppercase font-semibold">LLM Generator</span>
                <p className="font-bold text-pink-300 truncate">Gemini 1.5 Flash</p>
              </div>
              <div className="p-3 rounded-xl bg-slate-950/80 border border-white/5 space-y-1">
                <span className="text-slate-400 text-[10px] uppercase font-semibold">Escalation Policy</span>
                <p className="font-bold text-emerald-300 truncate">RiskAware (th=0.35)</p>
              </div>
            </div>

            {/* Live Processing Preview Box */}
            <div className="p-4 rounded-xl bg-slate-950/90 border border-white/10 space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span className="font-semibold text-slate-300">Live Query Monitor</span>
                <span className="text-purple-400 font-mono">case_id: LIVE-8012</span>
              </div>
              <p className="text-xs font-medium text-slate-200 italic">
                "Where is my package? The tracking info shows delivered but I haven't received anything."
              </p>
              <div className="flex items-center justify-between pt-2 border-t border-white/10 text-xs">
                <span className="text-emerald-400 font-bold flex items-center gap-1">
                  <ShieldCheck className="w-4 h-4" />
                  AUTO_HANDLE
                </span>
                <span className="text-slate-400 font-mono text-[11px]">Risk: 0.1170 (&lt; 0.35)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
