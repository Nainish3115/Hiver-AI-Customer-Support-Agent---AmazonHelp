import React from 'react';
import { Bot, Github, Heart, ShieldCheck } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-white/10 py-12 px-4 lg:px-8 bg-slate-950/80 text-xs text-slate-400">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-purple-500/20 text-purple-400 border border-purple-500/30">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <p className="font-bold text-white text-sm">Hiver AI Customer Support Agent</p>
            <p className="text-[11px] text-slate-400">AmazonHelp Customer Support Automation RAG & Escalation System</p>
          </div>
        </div>

        <div className="flex items-center gap-6 text-slate-400">
          <span>Hiver SDE Take-Home Challenge</span>
          <span>•</span>
          <span className="flex items-center gap-1">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            Safety & Risk Gated
          </span>
        </div>
      </div>
    </footer>
  );
};
