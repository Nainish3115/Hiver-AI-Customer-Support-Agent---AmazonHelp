import React from 'react';

export const MetricsBar: React.FC = () => {
  const metrics = [
    {
      label: 'Raw Conversation Turns',
      value: '81,186',
      subtext: 'AmazonHelp corpus scale',
      color: 'text-purple-400',
    },
    {
      label: 'Indexed RAG Corpus',
      value: '15,316',
      subtext: 'Canonical problem documents',
      color: 'text-cyan-400',
    },
    {
      label: 'Support Intents',
      value: '17',
      subtext: 'Taxonomy categories',
      color: 'text-pink-400',
    },
    {
      label: 'Untouched Test Accuracy',
      value: '57.0%',
      subtext: 'Human-gold evaluation (N=100)',
      color: 'text-purple-300',
    },
    {
      label: 'False Auto Rate',
      value: '6.0%',
      subtext: 'Reduced 50% (down from 12%)',
      color: 'text-emerald-400',
    },
    {
      label: 'Safety Pass Rate',
      value: '100%',
      subtext: 'Phase 6.5 Gemini test (N=50)',
      color: 'text-emerald-300',
    },
  ];

  return (
    <section className="py-10 px-4 lg:px-8 border-y border-white/10 bg-slate-950/60">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-5">
          {metrics.map((m, idx) => (
            <div
              key={idx}
              className="glass-panel glass-panel-hover p-4 rounded-xl text-center space-y-1.5 border border-white/10"
            >
              <p className={`text-2xl lg:text-3xl font-black tracking-tight ${m.color}`}>
                {m.value}
              </p>
              <h4 className="text-xs font-bold text-white">{m.label}</h4>
              <p className="text-[11px] text-slate-400 leading-tight">{m.subtext}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
