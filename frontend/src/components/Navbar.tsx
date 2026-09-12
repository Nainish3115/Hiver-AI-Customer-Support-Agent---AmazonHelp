import React, { useEffect, useState } from 'react';
import { Bot, Sparkles, Menu, X, CheckCircle2, AlertTriangle } from 'lucide-react';
import { checkHealth } from '../api/client';

interface NavbarProps {
  activeTab: 'home' | 'demo' | 'retrieval' | 'evaluations' | 'architecture' | 'safety';
  setActiveTab: (tab: 'home' | 'demo' | 'retrieval' | 'evaluations' | 'architecture' | 'safety') => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  useEffect(() => {
    let mounted = true;
    const verifyHealth = async () => {
      try {
        const res = await checkHealth();
        if (mounted) setIsBackendHealthy(res.pipeline_ready);
      } catch {
        if (mounted) setIsBackendHealthy(false);
      }
    };
    verifyHealth();
    const interval = setInterval(verifyHealth, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const navItems: Array<{ id: 'home' | 'demo' | 'retrieval' | 'evaluations'; label: string }> = [
    { id: 'home', label: 'Home' },
    { id: 'demo', label: 'Live Demo' },
    { id: 'retrieval', label: 'Retrieval' },
    { id: 'evaluations', label: 'Evaluation' },
  ];

  const handleNavClick = (tab: 'home' | 'demo' | 'retrieval' | 'evaluations' | 'architecture' | 'safety') => {
    setActiveTab(tab);
    setMobileMenuOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-white/10 px-4 lg:px-8 py-3.5">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Brand Logo & Title */}
        <div
          onClick={() => handleNavClick('home')}
          className="flex items-center gap-3 cursor-pointer group"
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && handleNavClick('home')}
        >
          <div className="relative p-2.5 rounded-xl bg-gradient-to-tr from-purple-600 via-purple-500 to-pink-500 text-white shadow-glow-violet group-hover:scale-105 transition-transform">
            <Bot className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-xl tracking-tight gradient-text-purple">
                HIVER AI
              </span>
              <span className="text-[10px] font-bold uppercase px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
                v0.1.0
              </span>
            </div>
            <p className="text-xs text-slate-400 font-medium hidden sm:block">Customer Support Agent • AmazonHelp</p>
          </div>
        </div>

        {/* Desktop Navigation Links */}
        <nav className="hidden md:flex items-center gap-1 bg-slate-900/80 p-1.5 rounded-xl border border-white/10">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => handleNavClick(item.id)}
              className={`px-4 py-2 rounded-lg text-xs font-semibold transition-all relative ${
                activeTab === item.id
                  ? 'bg-purple-600/30 text-purple-200 border border-purple-500/40 shadow-sm'
                  : 'text-slate-300 hover:text-white hover:bg-white/5'
              }`}
            >
              {item.label}
              {activeTab === item.id && (
                <span className="absolute bottom-0 left-3 right-3 h-0.5 bg-purple-400 rounded-full" />
              )}
            </button>
          ))}
        </nav>

        {/* Status Indicator & Live Demo CTA */}
        <div className="flex items-center gap-3">
          {/* Status Dot */}
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-950/80 border border-white/10 text-xs">
            {isBackendHealthy === null ? (
              <span className="w-2 h-2 rounded-full bg-slate-400 animate-pulse" />
            ) : isBackendHealthy ? (
              <span className="flex items-center gap-1.5 text-emerald-400 font-semibold">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                API Ready
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-amber-400 font-semibold">
                <AlertTriangle className="w-3.5 h-3.5" />
                API Offline
              </span>
            )}
          </div>

          {/* Live Demo CTA */}
          <button
            onClick={() => handleNavClick('demo')}
            className={`px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition-all shadow-md ${
              activeTab === 'demo'
                ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-glow-violet ring-2 ring-purple-400/50'
                : 'bg-gradient-to-r from-purple-600/90 to-pink-600/90 hover:from-purple-600 hover:to-pink-600 text-white shadow-glow-violet'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-purple-200" />
            <span>Try Live Demo</span>
          </button>

          {/* Mobile Menu Toggle */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-lg bg-slate-900 border border-white/10 text-slate-300 hover:text-white"
            aria-label="Toggle mobile menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden pt-4 pb-2 px-2 border-t border-white/10 mt-3 space-y-2">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => handleNavClick(item.id)}
              className={`w-full text-left px-4 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                activeTab === item.id
                  ? 'bg-purple-600/30 text-purple-200 border border-purple-500/40'
                  : 'text-slate-300 hover:bg-white/5'
              }`}
            >
              {item.label}
            </button>
          ))}
          <div className="pt-2 border-t border-white/10 flex items-center justify-between px-4 text-xs">
            <span className="text-slate-400">API Status:</span>
            {isBackendHealthy ? (
              <span className="text-emerald-400 font-semibold">Ready</span>
            ) : (
              <span className="text-amber-400 font-semibold">Offline</span>
            )}
          </div>
        </div>
      )}
    </header>
  );
};
