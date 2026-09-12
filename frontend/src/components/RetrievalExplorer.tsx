import React, { useState, useEffect } from 'react';
import { Search, Database, Layers, ArrowRight, X, Clock, HelpCircle, CheckCircle2, AlertCircle, FileText, Sparkles, Filter } from 'lucide-react';
import { searchRetrievalCorpus } from '../api/client';
import { RetrievalSearchResult } from '../types/api';

export const RetrievalExplorer: React.FC = () => {
  const [query, setQuery] = useState<string>('My package says delivered but I never got it');
  const [topK, setTopK] = useState<number>(10);
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<RetrievalSearchResult[]>([]);
  const [totalRetrieved, setTotalRetrieved] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [error, setError] = useState<string | null>(null);
  const [selectedCase, setSelectedCase] = useState<RetrievalSearchResult | null>(null);
  const [hasSearched, setHasSearched] = useState<boolean>(false);

  const presets = [
    { label: '📦 Missing Package', text: 'My package says delivered but I never got it' },
    { label: '💳 Double Charge', text: 'I was charged twice on my credit card' },
    { label: '❌ Cancel Order', text: 'How do I cancel my order before it ships?' },
    { label: '⚡ Damaged Item', text: 'My item arrived damaged and defective' },
    { label: '🔒 Login Issue', text: 'I cannot access my account and need help logging in' },
    { label: '💰 Refund Inquiry', text: 'Can I get a full refund for my return?' },
  ];

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      const res = await searchRetrievalCorpus(query.trim(), topK);
      setResults(res.results);
      setTotalRetrieved(res.total_retrieved);
      setLatencyMs(res.latency_ms);
    } catch (err: any) {
      setError(err.message || 'Failed to execute retrieval search with backend API.');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Initial load with default query
    handleSearch();
  }, []);

  return (
    <section className="py-12 px-4 lg:px-8 max-w-7xl mx-auto space-y-10">
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-semibold uppercase tracking-wider">
          <Database className="w-3.5 h-3.5" />
          <span>RAG Evidence Inspector</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Retrieval & Evidence Explorer
        </h1>
        <p className="text-slate-300 text-sm sm:text-base">
          Search across <strong className="text-cyan-300 font-semibold">15,316 indexed historical support cases</strong>. Inspect how dense embeddings, BM25 lexical search, and RRF rank fusion select evidence for Gemini generation.
        </p>
      </div>

      {/* RRF Architecture Explanation Box */}
      <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-4">
        <div className="flex items-center justify-between border-b border-white/10 pb-3">
          <h3 className="font-bold text-white text-sm flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            Hybrid Retrieval Architecture (RRF k=60)
          </h3>
          <span className="text-xs font-mono text-cyan-300 bg-cyan-500/20 px-2.5 py-0.5 rounded-full border border-cyan-500/30">
            Phase 9 Benchmark: 53.5% R@1
          </span>
        </div>

        {/* Visual Pipeline Bar */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs text-center">
          <div className="p-3 rounded-xl bg-slate-950/80 border border-purple-500/30 space-y-1">
            <span className="font-bold text-purple-300 block">1. Dense FAISS</span>
            <p className="text-[11px] text-slate-400">all-MiniLM-L6-v2 (384d semantic search)</p>
          </div>
          <div className="p-3 rounded-xl bg-slate-950/80 border border-cyan-500/30 space-y-1">
            <span className="font-bold text-cyan-300 block">2. Sparse BM25</span>
            <p className="text-[11px] text-slate-400">rank_bm25 exact keyword matching</p>
          </div>
          <div className="p-3 rounded-xl bg-slate-950/80 border border-pink-500/30 space-y-1">
            <span className="font-bold text-pink-300 block">3. RRF Fusion</span>
            <p className="text-[11px] text-slate-400">Reciprocal Rank Fusion (k=60)</p>
          </div>
          <div className="p-3 rounded-xl bg-slate-950/80 border border-emerald-500/30 space-y-1">
            <span className="font-bold text-emerald-300 block">4. Intent Boost</span>
            <p className="text-[11px] text-slate-400">Intent compatibility factor (0.15)</p>
          </div>
        </div>
      </div>

      {/* Presets Chips */}
      <div className="space-y-2">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider block">
          Preset RAG Search Queries:
        </span>
        <div className="flex flex-wrap gap-2">
          {presets.map((p, idx) => (
            <button
              key={idx}
              onClick={() => {
                setQuery(p.text);
              }}
              className="px-3 py-1.5 rounded-lg glass-panel glass-panel-hover text-xs font-medium text-slate-200 hover:text-white flex items-center gap-1.5 border border-white/10"
            >
              <span>{p.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Search Input Card */}
      <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-4">
        <form onSubmit={handleSearch} className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
            <label className="text-sm font-bold text-white flex items-center gap-2">
              <Search className="w-4 h-4 text-cyan-400" />
              Corpus Search Query
            </label>
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400 font-mono flex items-center gap-1">
                <Filter className="w-3.5 h-3.5" /> Top K Results:
              </span>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="px-3 py-1 rounded-lg bg-slate-950 border border-white/10 text-xs font-mono text-cyan-300 focus:outline-none focus:border-cyan-500 cursor-pointer"
              >
                <option value={5}>Top 5</option>
                <option value={10}>Top 10</option>
                <option value={20}>Top 20</option>
                <option value={50}>Top 50</option>
              </select>
            </div>
          </div>

          <div className="relative flex gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search historical support cases..."
              className="w-full p-4 rounded-xl bg-slate-950/90 border border-white/10 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-500/60 text-sm font-sans"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-6 py-4 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white font-bold text-sm shadow-glow-cyan transition-all shrink-0 flex items-center gap-2"
            >
              <Search className="w-4 h-4" />
              <span>Search</span>
            </button>
          </div>
        </form>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400" />
          <div>
            <strong className="font-bold">Retrieval Error:</strong> {error}
          </div>
        </div>
      )}

      {/* Results Header */}
      {hasSearched && !loading && !error && (
        <div className="flex items-center justify-between text-xs text-slate-400 px-1">
          <span>
            Found <strong className="text-white font-bold">{totalRetrieved}</strong> matching historical cases
          </span>
          <span className="font-mono text-cyan-400 flex items-center gap-1">
            <Clock className="w-3.5 h-3.5" />
            Search Latency: {latencyMs} ms
          </span>
        </div>
      )}

      {/* Results Cards List */}
      {loading ? (
        <div className="glass-panel p-12 rounded-2xl text-center space-y-3 border border-white/10">
          <div className="w-8 h-8 rounded-full border-2 border-cyan-500 border-t-transparent animate-spin mx-auto" />
          <p className="text-sm font-semibold text-slate-300">Searching 15,316 historical support cases...</p>
        </div>
      ) : results.length > 0 ? (
        <div className="space-y-4">
          {results.map((res, idx) => (
            <div
              key={idx}
              onClick={() => setSelectedCase(res)}
              className="glass-panel glass-panel-hover p-5 rounded-2xl border border-white/10 cursor-pointer space-y-3 relative group"
            >
              {/* Card Top Row */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="w-7 h-7 rounded-lg bg-cyan-500/20 text-cyan-300 font-mono text-xs font-bold flex items-center justify-center border border-cyan-500/30">
                    #{idx + 1}
                  </span>
                  <span className="font-mono text-xs font-bold text-white bg-slate-900 px-2.5 py-1 rounded-md border border-white/10">
                    {res.case_id}
                  </span>
                  {res.intent && (
                    <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30 font-mono">
                      {res.intent}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-cyan-300 font-bold bg-slate-950 px-2.5 py-1 rounded-md border border-white/10">
                    Score: {res.score.toFixed(4)}
                  </span>
                  <ArrowRight className="w-4 h-4 text-slate-400 group-hover:text-white group-hover:translate-x-1 transition-all" />
                </div>
              </div>

              {/* Problem Snippet */}
              <p className="text-sm text-slate-200 font-sans line-clamp-2 leading-relaxed">
                "{res.customer_problem}"
              </p>
            </div>
          ))}
        </div>
      ) : hasSearched && !loading ? (
        <div className="glass-panel p-12 rounded-2xl text-center space-y-3 border border-white/10">
          <Database className="w-8 h-8 text-slate-500 mx-auto" />
          <h4 className="text-base font-bold text-white">No Historical Cases Found</h4>
          <p className="text-xs text-slate-400">Try broadening your search term or select one of the preset queries above.</p>
        </div>
      ) : null}

      {/* Case Detail Inspection Modal */}
      {selectedCase && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-md">
          <div className="glass-panel p-6 rounded-2xl border border-cyan-500/40 max-w-xl w-full space-y-5 shadow-2xl relative">
            <button
              onClick={() => setSelectedCase(null)}
              className="absolute top-4 right-4 p-1.5 rounded-lg bg-slate-900 border border-white/10 text-slate-400 hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="flex items-center gap-3 border-b border-white/10 pb-4">
              <div className="p-2.5 rounded-xl bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                <FileText className="w-6 h-6" />
              </div>
              <div>
                <span className="text-xs font-mono font-bold text-slate-400">HISTORICAL EVIDENCE INSPECTOR</span>
                <h3 className="text-lg font-bold text-white font-mono">{selectedCase.case_id}</h3>
              </div>
            </div>

            <div className="space-y-4 text-xs">
              <div className="space-y-1">
                <span className="text-slate-400 font-semibold block uppercase tracking-wider text-[10px]">Customer Problem Text:</span>
                <div className="p-4 rounded-xl bg-slate-950 border border-white/10 text-slate-100 text-sm leading-relaxed">
                  "{selectedCase.customer_problem}"
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-xl bg-slate-900 border border-white/5 space-y-1">
                  <span className="text-slate-400 text-[10px] uppercase font-semibold">Intent Taxonomy</span>
                  <p className="font-mono text-purple-300 font-bold truncate">
                    {selectedCase.intent || 'N/A'}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-slate-900 border border-white/5 space-y-1">
                  <span className="text-slate-400 text-[10px] uppercase font-semibold">FAISS Vector Similarity (Cosine Score)</span>
                  <p className="font-mono text-cyan-300 font-bold">
                    {selectedCase.score.toFixed(4)}
                  </p>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-purple-500/10 border border-purple-500/20 text-slate-300 space-y-1">
                <span className="font-bold text-white flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-purple-400" />
                  RAG Prompt Context Integration:
                </span>
                <p className="text-[11px] text-slate-400">
                  This historical customer turn is formatted into Gemini 1.5 Flash's system prompt to ground the generated response in proven AmazonHelp customer resolution patterns.
                </p>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedCase(null)}
                className="px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs shadow-glow-cyan"
              >
                Close Inspection
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
