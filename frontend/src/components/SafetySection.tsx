import React from 'react';
import { Scale, CheckCircle2, AlertTriangle, ShieldCheck, HelpCircle } from 'lucide-react';

export const SafetySection: React.FC = () => {
  return (
    <section className="py-20 px-4 lg:px-8 bg-slate-950/80 border-b border-white/10">
      <div className="max-w-7xl mx-auto space-y-12">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-4">
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-semibold uppercase tracking-wider">
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Safety & Risk Gating</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            AI Should Know When Not To Act
          </h2>
          <p className="text-slate-300 text-base">
            The <strong className="text-emerald-400 font-semibold">RiskAwarePolicy</strong> evaluates multiple real-time uncertainty signals before permitting automated response handling.
          </p>
        </div>

        {/* Risk Calculation & Decision Split */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Signal Weights Column */}
          <div className="lg:col-span-6 space-y-4">
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Scale className="w-5 h-5 text-purple-400" />
              Composite Risk Components
            </h3>

            <div className="glass-panel p-4 rounded-xl border border-white/10 space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-semibold text-purple-300">1. Intent Uncertainty Risk</span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-purple-500/20 text-purple-300">Weight: 40%</span>
              </div>
              <p className="text-xs text-slate-300">
                Measures ambiguity in predicted intent. Low classifier confidence or high entropy raises overall risk.
              </p>
            </div>

            <div className="glass-panel p-4 rounded-xl border border-white/10 space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-semibold text-cyan-300">2. Category Sensitivity Risk</span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300">Weight: 30%</span>
              </div>
              <p className="text-xs text-slate-300">
                High-risk intents (e.g. <code className="text-purple-300">payment_billing_issue</code>, <code className="text-purple-300">account_access_login</code>) carry explicit risk multipliers.
              </p>
            </div>

            <div className="glass-panel p-4 rounded-xl border border-white/10 space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="font-semibold text-pink-300">3. Retrieval Quality Risk</span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-pink-500/20 text-pink-300">Weight: 30%</span>
              </div>
              <p className="text-xs text-slate-300">
                Evaluates top RRF similarity scores. Weak historical evidence context increases human escalation probability.
              </p>
            </div>
          </div>

          {/* Decision Branch Visual */}
          <div className="lg:col-span-6 space-y-6">
            <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-6 relative overflow-hidden">
              <div className="flex items-center justify-between border-b border-white/10 pb-4">
                <div>
                  <h4 className="font-bold text-white text-base">Policy Risk Threshold</h4>
                  <p className="text-xs text-slate-400">Fixed Production Threshold: <code className="text-purple-300">0.3500</code></p>
                </div>
                <span className="px-3 py-1 rounded-full bg-purple-500/20 text-purple-300 text-xs font-bold font-mono">
                  th = 0.35
                </span>
              </div>

              {/* Threshold Branch 1: Auto Handle */}
              <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2 font-bold text-emerald-400 text-sm">
                    <CheckCircle2 className="w-5 h-5" />
                    Risk &lt; 0.35 → AUTO_HANDLE
                  </span>
                  <span className="text-xs font-bold text-emerald-400 bg-emerald-500/20 px-2.5 py-0.5 rounded-full">
                    Safe Auto
                  </span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Sufficient evidence, low intent uncertainty, and verified safety check pass. System sends automated grounded response.
                </p>
              </div>

              {/* Threshold Branch 2: Human Escalation */}
              <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="flex items-center gap-2 font-bold text-amber-400 text-sm">
                    <AlertTriangle className="w-5 h-5" />
                    Risk ≥ 0.35 → ESCALATE_TO_HUMAN
                  </span>
                  <span className="text-xs font-bold text-amber-400 bg-amber-500/20 px-2.5 py-0.5 rounded-full">
                    Human Oversight
                  </span>
                </div>
                <p className="text-xs text-slate-300 leading-relaxed">
                  High category risk, weak evidence quality, or ungrounded generation flags. Ticket routes immediately to human support.
                </p>
              </div>

              {/* Safety Trade-off Explanation Box */}
              <div className="p-3.5 rounded-xl bg-slate-950/90 border border-white/10 text-xs text-slate-300 space-y-1">
                <span className="font-bold text-white flex items-center gap-1.5">
                  <HelpCircle className="w-4 h-4 text-purple-400" />
                  Validated Escalation Safety Trade-off (N=50):
                </span>
                <p className="text-slate-400">
                  Substituting the Day 4 integrated model under the unchanged policy halved high-risk <strong className="text-emerald-400">false auto-handle errors from 12.0% to 6.0%</strong>, while conservative false escalations increased from 14.0% to 20.0% (Overall agreement remained stable at 74.0%).
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
