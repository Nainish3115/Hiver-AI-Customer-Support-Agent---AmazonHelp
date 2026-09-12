import React, { useState } from 'react';
import {
  Sparkles,
  Send,
  Target,
  Search,
  Bot,
  ShieldCheck,
  Scale,
  AlertCircle,
  RefreshCw,
  Clock,
  CheckCircle2,
  AlertTriangle,
  FileText,
  Layers,
  ChevronRight
} from 'lucide-react';
import { processSupportMessage } from '../api/client';
import { SupportProcessResponse } from '../types/api';

export const LiveDemo: React.FC = () => {
  const [query, setQuery] = useState<string>('Where is my package? Order ID: 403-1234567-8901234');
  const [caseId, setCaseId] = useState<string>('DEMO-WEB-101');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<SupportProcessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const presets = [
    {
      label: '📦 Order Tracking',
      text: 'Where is my package? Order ID: 403-1234567-8901234',
      cid: 'DEMO-TRACK-101',
    },
    {
      label: '⚡ Damaged Item',
      text: 'Item arrived broken and defective, I want a replacement immediately!',
      cid: 'DEMO-DAMAGE-202',
    },
    {
      label: '💳 Billing Charge (High Risk)',
      text: 'Double charged on my credit card for prime membership!',
      cid: 'DEMO-BILLING-303',
    },
    {
      label: '🔄 Return & Refund',
      text: 'How do I return an item for a full refund?',
      cid: 'DEMO-REFUND-404',
    },
    {
      label: '🔒 Account Access',
      text: 'Locked out of my account and cannot log in to place an order.',
      cid: 'DEMO-ACCESS-505',
    },
  ];

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const res = await processSupportMessage({
        customer_message: query.trim(),
        case_id: caseId.trim() || undefined,
        policy_type: 'risk_aware',
      });
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Failed to communicate with the FastAPI backend.');
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const selectPreset = (p: { text: string; cid: string }) => {
    setQuery(p.text);
    setCaseId(p.cid);
  };

  return (
    <section className="py-12 px-4 lg:px-8 max-w-7xl mx-auto space-y-10">
      {/* Page Title & Eyebrow */}
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/30 text-purple-300 text-xs font-semibold uppercase tracking-wider">
          <Sparkles className="w-3.5 h-3.5 text-purple-400" />
          <span>Interactive AI Sandbox</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Live Customer Support RAG Workspace
        </h1>
        <p className="text-slate-300 text-sm sm:text-base">
          Submit customer inquiries below to trigger end-to-end intent classification, RAG retrieval, Gemini generation, safety checking, and risk-aware policy decisioning in real-time.
        </p>
      </div>

      {/* Preset Scenario Query Chips */}
      <div className="space-y-2">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
          Select Preset Inquiry Scenario:
        </span>
        <div className="flex flex-wrap gap-2">
          {presets.map((p, idx) => (
            <button
              key={idx}
              onClick={() => selectPreset(p)}
              className="px-3 py-1.5 rounded-lg glass-panel glass-panel-hover text-xs font-medium text-slate-200 hover:text-white flex items-center gap-1.5 border border-white/10"
            >
              <span>{p.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Input Form Card */}
      <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
            <label className="text-sm font-bold text-white flex items-center gap-2">
              <FileText className="w-4 h-4 text-purple-400" />
              Customer Inquiry Message
            </label>
            <div className="flex items-center gap-2 w-full sm:w-auto">
              <span className="text-xs text-slate-400 font-mono">Case Reference:</span>
              <input
                type="text"
                value={caseId}
                onChange={(e) => setCaseId(e.target.value)}
                placeholder="Optional Case ID"
                className="px-3 py-1 rounded-lg bg-slate-950 border border-white/10 text-xs font-mono text-purple-300 focus:outline-none focus:border-purple-500 w-36"
              />
            </div>
          </div>

          <div className="relative">
            <textarea
              rows={3}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
              placeholder="Enter customer message here..."
              className="w-full p-4 rounded-xl bg-slate-950/80 border border-white/10 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-purple-500/60 focus:ring-1 focus:ring-purple-500/40 text-sm resize-none disabled:opacity-60"
            />
          </div>

          <div className="flex items-center justify-between pt-1">
            <span className="text-xs text-slate-400">
              {query.length} / 2000 characters
            </span>
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:opacity-50 text-white font-bold text-sm shadow-glow-violet transition-all flex items-center gap-2 cursor-pointer disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Processing RAG Pipeline...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Analyze Inquiry</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Error Alert Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs space-y-2 flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <strong className="font-bold text-rose-200">API Execution Error</strong>
            <p>{error}</p>
            <p className="text-slate-400 font-mono text-[11px]">
              Make sure FastAPI backend is running: <code className="text-purple-300">uvicorn hiver_agent.api.app:app --reload</code>
            </p>
          </div>
        </div>
      )}

      {/* Real API Results Display */}
      {result && (
        <div className="space-y-6">
          {/* Status Progress Header */}
          <div className="flex items-center justify-between px-2 text-xs text-slate-400">
            <span className="flex items-center gap-2 text-purple-300 font-semibold">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              Pipeline Execution Complete
            </span>
            <div className="flex items-center gap-4 font-mono">
              {result.request_id && (
                <span className="text-[11px] text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-white/10">
                  ID: <strong className="text-purple-300">{result.request_id}</strong>
                </span>
              )}
              <span className="flex items-center gap-1">
                <Clock className="w-3.5 h-3.5 text-cyan-400" />
                Latency: {(result.latency_ms / 1000).toFixed(2)}s ({result.latency_ms} ms)
              </span>
            </div>
          </div>

          {/* Results Grid Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Intent & Retrieval */}
            <div className="lg:col-span-5 space-y-6">
              {/* Intent Card */}
              <div className="glass-panel p-5 rounded-2xl border border-white/10 space-y-3">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <span className="text-xs font-bold text-slate-300 flex items-center gap-2">
                    <Target className="w-4 h-4 text-purple-400" />
                    1. Predicted Intent
                  </span>
                  <span className="text-xs font-mono font-bold text-purple-300 bg-purple-500/20 px-2.5 py-0.5 rounded-full border border-purple-500/30">
                    {(result.intent_confidence * 100).toFixed(1)}% Conf
                  </span>
                </div>
                <div>
                  <p className="text-base font-extrabold text-white font-mono">
                    {result.predicted_intent}
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Calibrated Hybrid Classifier (Rule + Centroid Prototype)
                  </p>
                </div>
              </div>

              {/* Retrieval Card */}
              <div className="glass-panel p-5 rounded-2xl border border-white/10 space-y-3">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <span className="text-xs font-bold text-slate-300 flex items-center gap-2">
                    <Search className="w-4 h-4 text-cyan-400" />
                    2. Hybrid Retrieval (RAG)
                  </span>
                  <span className="text-xs font-mono text-cyan-300 bg-cyan-500/20 px-2 py-0.5 rounded-full">
                    {result.selected_evidence_count} Cases Formatted
                  </span>
                </div>
                <div className="space-y-2">
                  <span className="text-xs text-slate-400 block">Retrieved Historical Case IDs:</span>
                  <div className="flex flex-wrap gap-1.5">
                    {result.retrieved_case_ids.map((cid, i) => (
                      <span
                        key={i}
                        className="px-2.5 py-1 rounded-md bg-slate-900 border border-white/10 text-xs font-mono text-cyan-300"
                      >
                        {cid} ({result.retrieved_scores[i] ? result.retrieved_scores[i].toFixed(3) : 'RRF'})
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Safety & Grounding Card */}
              <div className="glass-panel p-5 rounded-2xl border border-white/10 space-y-3">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <span className="text-xs font-bold text-slate-300 flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                    3. Safety & Grounding Validation
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                    <span className="text-slate-400 text-[10px]">Grounding Status</span>
                    <p className="font-bold text-emerald-400 uppercase tracking-wide">
                      {result.grounding_status}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                    <span className="text-slate-400 text-[10px]">Safety Checker</span>
                    <p className="font-bold text-emerald-400 uppercase tracking-wide">
                      {result.safety_check_passed ? 'PASS' : 'FAIL'}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column: AI Draft Response & Decision */}
            <div className="lg:col-span-7 space-y-6">
              {/* Decision Banner Card */}
              <div
                className={`p-6 rounded-2xl border shadow-xl space-y-4 ${
                  result.decision === 'AUTO_HANDLE'
                    ? 'bg-emerald-500/10 border-emerald-500/40 shadow-glow-emerald'
                    : 'bg-amber-500/10 border-amber-500/40 shadow-glow-amber'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {result.decision === 'AUTO_HANDLE' ? (
                      <CheckCircle2 className="w-7 h-7 text-emerald-400" />
                    ) : (
                      <AlertTriangle className="w-7 h-7 text-amber-400" />
                    )}
                    <div>
                      <span className="text-xs uppercase font-extrabold tracking-wider text-slate-300">
                        Final Escalation Decision
                      </span>
                      <h3
                        className={`text-2xl font-black ${
                          result.decision === 'AUTO_HANDLE' ? 'text-emerald-400' : 'text-amber-400'
                        }`}
                      >
                        {result.decision}
                      </h3>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-[10px] uppercase text-slate-400 font-semibold block">Reason Code</span>
                    <span className="font-mono text-xs font-bold text-white bg-slate-900/80 px-2.5 py-1 rounded-lg border border-white/10">
                      {result.reason_code}
                    </span>
                  </div>
                </div>

                {/* Risk Score Meter */}
                <div className="space-y-1.5 pt-2 border-t border-white/10">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-slate-300">Composite Risk Score:</span>
                    <span className="font-bold text-white">
                      {result.risk_score.toFixed(4)}{' '}
                      <span className="text-slate-400 font-normal">(Threshold: 0.3500)</span>
                    </span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-900 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        result.risk_score < 0.35 ? 'bg-emerald-500' : 'bg-amber-500'
                      }`}
                      style={{ width: `${Math.min(100, (result.risk_score / 1.0) * 100)}%` }}
                    />
                  </div>
                </div>

                <p className="text-xs text-slate-200 leading-relaxed italic">
                  "{result.decision_reason}"
                </p>
              </div>

              {/* AI Draft Response Card */}
              <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-4">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <span className="text-sm font-bold text-white flex items-center gap-2">
                    <Bot className="w-5 h-5 text-pink-400" />
                    Generated Customer Support Response
                  </span>
                  <span className="text-xs font-mono font-semibold px-2.5 py-0.5 rounded-full bg-pink-500/20 text-pink-300 border border-pink-500/30">
                    {result.generation_metadata?.provider === 'gemini' && result.generation_metadata?.mode === 'real'
                      ? 'Gemini 1.5 Flash (Real LLM)'
                      : result.generation_metadata?.provider === 'gemini' && result.generation_metadata?.mode === 'mock'
                      ? 'Gemini Mock Fallback (Offline Mode)'
                      : 'Deterministic Resolution Template'}
                  </span>
                </div>

                <div className="p-4 rounded-xl bg-slate-950/90 border border-white/10 text-slate-100 text-sm leading-relaxed whitespace-pre-wrap font-sans">
                  {result.draft_response}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
