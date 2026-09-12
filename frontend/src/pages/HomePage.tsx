import React, { useState } from 'react';
import { Navbar } from '../components/Navbar';
import { Hero } from '../components/Hero';
import { MetricsBar } from '../components/MetricsBar';
import { ArchitectureFlow } from '../components/ArchitectureFlow';
import { SafetySection } from '../components/SafetySection';
import { EvaluationSection } from '../components/EvaluationSection';
import { EvaluationDashboard } from '../components/EvaluationDashboard';
import { RetrievalExplorer } from '../components/RetrievalExplorer';
import { LiveDemo } from '../components/LiveDemo';
import { Footer } from '../components/Footer';

export const HomePage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'home' | 'demo' | 'retrieval' | 'evaluations' | 'architecture' | 'safety'>('home');

  return (
    <div className="min-h-screen flex flex-col bg-[#0B0F19] text-slate-100 selection:bg-purple-500 selection:text-white">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1">
        {activeTab === 'home' && (
          <>
            <Hero
              onStartDemo={() => setActiveTab('demo')}
              onExploreArchitecture={() => setActiveTab('architecture')}
            />
            <MetricsBar />
            <ArchitectureFlow />
            <SafetySection />
            <EvaluationSection />
            <div className="py-16 text-center border-t border-white/10 bg-gradient-to-b from-transparent to-purple-950/20">
              <h2 className="text-3xl font-extrabold text-white mb-4">Ready to Experience Live AI Support?</h2>
              <button
                onClick={() => setActiveTab('demo')}
                className="px-8 py-4 rounded-xl bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white font-bold text-base shadow-glow-violet transition-all"
              >
                Launch Interactive Demo Workspace
              </button>
            </div>
          </>
        )}

        {activeTab === 'demo' && <LiveDemo />}
        {activeTab === 'retrieval' && <RetrievalExplorer />}
        {activeTab === 'evaluations' && <EvaluationDashboard />}
        {activeTab === 'architecture' && <ArchitectureFlow />}
        {activeTab === 'safety' && <SafetySection />}
      </main>

      <Footer />
    </div>
  );
};
