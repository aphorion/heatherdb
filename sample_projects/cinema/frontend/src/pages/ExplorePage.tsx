import { useState } from 'react'
import { api, StabilityResult, ColdStartSnapshot, ScoredMovie } from '../api'
import SimilarityBar from '../components/SimilarityBar'

export default function ExplorePage() {
  const [stability, setStability] = useState<StabilityResult | null>(null)
  const [coldStart, setColdStart] = useState<ColdStartSnapshot[] | null>(null)
  const [loading, setLoading] = useState<string | null>(null)
  const [selectedSnapshot, setSelectedSnapshot] = useState(0)

  async function runStability() {
    setLoading('stability')
    try {
      const catalog = await api.getCatalog()
      const outlier = catalog.movies.find(m => m.genres?.includes('Romance') || m.genres?.includes('Comedy'))
      setStability(outlier ? await api.demoStability(outlier.imdb_id) : await api.demoStability())
    } catch { setStability(null) }
    setLoading(null)
  }

  async function runColdStart() {
    setLoading('coldstart')
    try {
      const res = await api.demoColdStart()
      setColdStart(res.snapshots)
      setSelectedSnapshot(0)
    } catch { setColdStart(null) }
    setLoading(null)
  }

  function RecList({ movies, label }: { movies: ScoredMovie[]; label: string }) {
    return (
      <div>
        <h3 className="label mb-3">{label}</h3>
        <div className="space-y-1">
          {movies.map((m, i) => (
            <div key={m.imdb_id} className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-cream/[0.02] transition-colors">
              <span className="font-mono text-[9px] text-cream-muted w-4 text-right">{i + 1}</span>
              <span className="font-serif text-[13px] text-cream/60 flex-1 truncate italic">{m.title}</span>
              <span className="font-mono text-[10px] text-gold/50 tabular-nums">{(m.similarity * 100).toFixed(1)}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="px-8 pt-8 pb-12 space-y-10 animate-fade-in">
      <div className="max-w-xl">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-6 h-[1px] bg-gold" />
          <span className="font-display text-[10px] font-bold tracking-[0.3em] text-gold uppercase">Laboratory</span>
        </div>
        <h1 className="font-display text-3xl font-extrabold text-cream tracking-tight mb-2">Explore SDM</h1>
        <p className="font-serif text-cream-muted italic text-sm leading-relaxed">
          Observe how Elastic Associative Memory forms taste fingerprints — stability under outliers and progressive learning.
        </p>
      </div>

      {/* Stability */}
      <div className="card-warm p-6 bg-warm-gradient">
        <div className="flex items-center justify-between mb-6 pb-4 border-b border-warm">
          <div>
            <h2 className="font-display text-base font-bold text-cream mb-1">Fingerprint Stability</h2>
            <p className="font-serif text-[12px] text-cream-muted italic">
              Watches 5 sci-fi films, then adds 1 romance — how much does the fingerprint shift?
            </p>
          </div>
          <button
            onClick={runStability}
            disabled={loading === 'stability'}
            className="font-display text-[11px] font-semibold px-5 py-2 rounded-xl border border-gold/20 text-gold hover:bg-gold/10 transition-all disabled:opacity-30"
          >
            {loading === 'stability' ? (
              <span className="flex items-center gap-2">
                <div className="w-3 h-3 border border-gold/30 border-t-gold rounded-full animate-spin" />
                Running...
              </span>
            ) : 'Run Demo'}
          </button>
        </div>
        {stability && (
          <div className="space-y-6 animate-fade-in">
            <div className="card-warm p-4">
              <div className="label mb-2">Similarity (before vs after)</div>
              <SimilarityBar value={stability.fingerprint_similarity} color="gold" />
              <p className="font-serif text-[11px] text-cream-muted italic mt-2">
                {stability.fingerprint_similarity > 0.95 ? 'Excellent — barely moved despite the outlier' :
                 stability.fingerprint_similarity > 0.85 ? 'Good — fairly stable fingerprint' :
                 'Some drift — more core films would help'}
              </p>
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <RecList movies={stability.before_recommendations} label="Before outlier" />
              <RecList movies={stability.after_recommendations} label="After outlier" />
            </div>
          </div>
        )}
      </div>

      {/* Cold Start */}
      <div className="card-warm p-6">
        <div className="flex items-center justify-between mb-6 pb-4 border-b border-warm">
          <div>
            <h2 className="font-display text-base font-bold text-cream mb-1">Cold Start Progression</h2>
            <p className="font-serif text-[12px] text-cream-muted italic">
              Watch how recommendations improve with each additional film
            </p>
          </div>
          <button
            onClick={runColdStart}
            disabled={loading === 'coldstart'}
            className="font-display text-[11px] font-semibold px-5 py-2 rounded-xl border border-gold/20 text-gold hover:bg-gold/10 transition-all disabled:opacity-30"
          >
            {loading === 'coldstart' ? (
              <span className="flex items-center gap-2">
                <div className="w-3 h-3 border border-gold/30 border-t-gold rounded-full animate-spin" />
                Running...
              </span>
            ) : 'Run Demo'}
          </button>
        </div>
        {coldStart && coldStart.length > 0 && (
          <div className="space-y-6 animate-fade-in">
            <div className="flex gap-1 overflow-x-auto no-scrollbar p-1 rounded-xl bg-film-dark border border-warm">
              {coldStart.map((snap, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedSnapshot(i)}
                  className={`font-display text-[10px] font-semibold px-4 py-1.5 rounded-lg transition-all whitespace-nowrap ${
                    i === selectedSnapshot ? 'bg-gold/15 text-gold' : 'text-cream-muted hover:text-cream-dim'
                  }`}
                >
                  {snap.movies_watched} film{snap.movies_watched !== 1 ? 's' : ''}
                </button>
              ))}
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="label mb-3">Genre affinities</h3>
                <div className="space-y-2">
                  {Object.entries(coldStart[selectedSnapshot].top_genres).map(([genre, score]) => (
                    <SimilarityBar key={genre} value={score} label={genre} />
                  ))}
                </div>
              </div>
              <RecList movies={coldStart[selectedSnapshot].recommendations} label="Recommendations" />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
