import React, { useState } from 'react';
import { Target, Search, Layers, FileText, Bot, ShieldCheck, Scale, ArrowRight, CheckCircle2, ChevronRight, X } from 'lucide-react';

export const ArchitectureFlow: React.FC = () => {
  const [selectedStage, setSelectedStage] = useState<number | null>(null);

  const stages = [
    {
      id: 1,
      title: 'Customer Message',
      shortTitle: '1. Message',
      icon: FileText,
      color: 'from-blue-500 to-cyan-500',
      badge: 'Input',
      description: 'Raw customer inquiry text received via API.',
      details: [
        'Supports freeform customer text input',
        'Extracts optional case_id reference',
        'Validates message length & non-empty content'
      ]
    },
    {
      id: 2,
      title: 'Intent Classifier',
      shortTitle: '2. Intent',
      icon: Target,
      color: 'from-purple-500 to-indigo-500',
      badge: 'Classifier',
      description: 'CalibratedRulePrototypeHybrid predicts 17 intents.',
      details: [
        'High-specificity keyword rule matching',
        'Centroid prototype semantic similarity',
        'High-confidence errors reduced 38 -> 13'
      ]
    },
    {
      id: 3,
      title: 'Hybrid Retrieval',
      shortTitle: '3. Retrieval',
      icon: Search,
      color: 'from-indigo-500 to-cyan-500',
      badge: 'RAG Search',
      description: 'Dual dense-sparse search across 15,316 documents.',
      details: [
        'FAISS Dense Embeddings (all-MiniLM-L6-v2)',
        'Sparse BM25 Keyword Matching',
        'Reciprocal Rank Fusion (RRF k=60)'
      ]
    },
    {
      id: 4,
      title: 'Intent Ranking',
      shortTitle: '4. Ranking',
      icon: Layers,
      color: 'from-cyan-500 to-teal-500',
      badge: 'Boost',
      description: 'Boosts cases matching predicted intent taxonomy.',
      details: [
        'Intent compatibility factor (0.15 boost)',
        'Filters out-of-domain historical cases',
        'Phase 9 Benchmark: R@1 = 53.5% / R@10 = 65.5%'
      ]
    },
    {
      id: 5,
      title: 'Evidence Selection',
      shortTitle: '5. Evidence',
      icon: FileText,
      color: 'from-teal-500 to-emerald-500',
      badge: 'Formatting',
      description: 'Formats top-3 evidence cases for Gemini context.',
      details: [
        'Enforces top-3 evidence item cap',
        'Excludes target self-case (leakage control)',
        'Formats structured customer/agent turns'
      ]
    },
    {
      id: 6,
      title: 'Gemini 1.5 Flash',
      shortTitle: '6. Gemini LLM',
      icon: Bot,
      color: 'from-pink-500 to-purple-500',
      badge: 'Generation',
      description: 'Generates grounded support draft response.',
      details: [
        'Strict evidence grounding system instructions',
        'Temperature = 0.2, max_tokens = 500',
        '4.72 / 5.00 Overall LLM Judge Quality score'
      ]
    },
    {
      id: 7,
      title: 'Safety Checker',
      shortTitle: '7. Safety',
      icon: ShieldCheck,
      color: 'from-emerald-500 to-green-500',
      badge: 'Validation',
      description: 'GroundingSafetyChecker verifies response safety.',
      details: [
        'Entity & order ID consistency verification',
        'Hallucinated refund/policy detection',
        '100% Safety Pass Rate in N=50 test'
      ]
    },
    {
      id: 8,
      title: 'RiskAware Decision',
      shortTitle: '8. Decision',
      icon: Scale,
      color: 'from-purple-600 to-pink-600',
      badge: 'Escalation',
      description: 'RiskAwarePolicy evaluates threshold (0.35).',
      details: [
        'Composite risk = Uncertainty + Category + Retrieval',
        'Risk < 0.35 -> AUTO_HANDLE',
        'Risk >= 0.35 -> ESCALATE_TO_HUMAN'
      ]
    }
  ];

  const activeModalStage = selectedStage !== null ? stages.find(s => s.id === selectedStage) : null;

  return (
    <section className="py-20 px-4 lg:px-8 border-b border-white/10 relative">
      <div className="max-w-7xl mx-auto space-y-12">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-4">
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-semibold uppercase tracking-wider">
            <Layers className="w-3.5 h-3.5" />
            <span>Architecture Flowchart</span>
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            7-Stage RAG & Escalation Pipeline
          </h2>
          <p className="text-slate-300 text-base">
            Click any stage node below to inspect its exact mechanics, ML algorithms, and safety parameters.
          </p>
        </div>

        {/* Desktop Pipeline Flow Nodes (2 rows of 4 connected nodes) */}
        <div className="hidden lg:block space-y-8">
          {/* Row 1: Stages 1 to 4 */}
          <div className="grid grid-cols-4 gap-4 items-center relative">
            {stages.slice(0, 4).map((st, idx) => {
              const Icon = st.icon;
              return (
                <div key={st.id} className="relative flex items-center">
                  <div
                    onClick={() => setSelectedStage(st.id)}
                    className="w-full glass-panel p-5 rounded-2xl border border-white/10 glass-panel-hover cursor-pointer space-y-3 relative"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-mono font-bold text-slate-400">STAGE 0{st.id}</span>
                      <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-gradient-to-r ${st.color} text-white`}>
                        {st.badge}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-xl bg-gradient-to-r ${st.color} text-white shadow-md`}>
                        <Icon className="w-4 h-4" />
                      </div>
                      <h4 className="font-bold text-white text-sm truncate">{st.title}</h4>
                    </div>
                    <p className="text-xs text-slate-300 line-clamp-2">{st.description}</p>
                  </div>
                  {idx < 3 && (
                    <ArrowRight className="w-5 h-5 text-purple-400/60 shrink-0 mx-1 z-10" />
                  )}
                </div>
              );
            })}
          </div>

          {/* Connection Divider Arrow */}
          <div className="flex justify-end pr-12">
            <div className="flex items-center gap-2 text-xs font-mono text-purple-400 bg-purple-500/10 px-3 py-1 rounded-full border border-purple-500/20">
              <span>Pipeline Flow</span>
              <ArrowRight className="w-4 h-4 rotate-90" />
            </div>
          </div>

          {/* Row 2: Stages 5 to 8 */}
          <div className="grid grid-cols-4 gap-4 items-center relative">
            {stages.slice(4, 8).map((st, idx) => {
              const Icon = st.icon;
              return (
                <div key={st.id} className="relative flex items-center">
                  <div
                    onClick={() => setSelectedStage(st.id)}
                    className="w-full glass-panel p-5 rounded-2xl border border-white/10 glass-panel-hover cursor-pointer space-y-3 relative"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[11px] font-mono font-bold text-slate-400">STAGE 0{st.id}</span>
                      <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-gradient-to-r ${st.color} text-white`}>
                        {st.badge}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-xl bg-gradient-to-r ${st.color} text-white shadow-md`}>
                        <Icon className="w-4 h-4" />
                      </div>
                      <h4 className="font-bold text-white text-sm truncate">{st.title}</h4>
                    </div>
                    <p className="text-xs text-slate-300 line-clamp-2">{st.description}</p>
                  </div>
                  {idx < 3 && (
                    <ArrowRight className="w-5 h-5 text-purple-400/60 shrink-0 mx-1 z-10" />
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Mobile / Tablet Vertical Stack */}
        <div className="lg:hidden space-y-4">
          {stages.map((st) => {
            const Icon = st.icon;
            return (
              <div
                key={st.id}
                onClick={() => setSelectedStage(st.id)}
                className="glass-panel p-4 rounded-xl border border-white/10 flex items-center justify-between cursor-pointer"
              >
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-lg bg-gradient-to-r ${st.color} text-white`}>
                    <Icon className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="font-bold text-white text-sm">{st.shortTitle}</h4>
                    <p className="text-xs text-slate-400 truncate max-w-[200px]">{st.description}</p>
                  </div>
                </div>
                <ChevronRight className="w-4 h-4 text-slate-400" />
              </div>
            );
          })}
        </div>

        {/* Stage Inspection Modal */}
        {activeModalStage && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md">
            <div className="glass-panel p-6 rounded-2xl border border-purple-500/40 max-w-lg w-full space-y-5 shadow-2xl relative">
              <button
                onClick={() => setSelectedStage(null)}
                className="absolute top-4 right-4 p-1.5 rounded-lg bg-slate-900 border border-white/10 text-slate-400 hover:text-white"
              >
                <X className="w-4 h-4" />
              </button>

              <div className="flex items-center gap-3">
                <div className={`p-3 rounded-xl bg-gradient-to-r ${activeModalStage.color} text-white`}>
                  {React.createElement(activeModalStage.icon, { className: 'w-6 h-6' })}
                </div>
                <div>
                  <span className="text-xs font-mono font-bold text-purple-400">STAGE 0{activeModalStage.id}</span>
                  <h3 className="text-xl font-bold text-white">{activeModalStage.title}</h3>
                </div>
              </div>

              <p className="text-sm text-slate-300 leading-relaxed">
                {activeModalStage.description}
              </p>

              <div className="space-y-2 pt-3 border-t border-white/10">
                <h5 className="text-xs font-bold uppercase tracking-wider text-slate-400">Technical Details:</h5>
                {activeModalStage.details.map((d, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs text-slate-200">
                    <CheckCircle2 className="w-4 h-4 text-purple-400 shrink-0 mt-0.5" />
                    <span>{d}</span>
                  </div>
                ))}
              </div>

              <div className="pt-2 flex justify-end">
                <button
                  onClick={() => setSelectedStage(null)}
                  className="px-4 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-bold text-xs"
                >
                  Close Inspection
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
