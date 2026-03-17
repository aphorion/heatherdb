import { useState, useEffect } from 'react'
import { api, ScoredMovie, FidelityData, ExplainResult } from '../api'
import MovieCard from '../components/MovieCard'
import SimilarityBar from '../components/SimilarityBar'

interface RecommendPageProps {
  currentUser: string
}

const LEVELS = [
  { label: 'Safe', threshold: 0.3 },
  { label: 'Balanced', threshold: 0.1 },
  { label: 'Wild', threshold: -0.5 },
]

const TIERS = [
  { key: 'love', min: 0.5, title: 'Perfect for you', sub: 'Highest affinity to your taste fingerprint', marker: 'I' },
  { key: 'great', min: 0.4, title: 'Strong matches', sub: 'Closely aligned with your viewing patterns', marker: 'II' },
  { key: 'explore', min: 0.3, title: 'Worth discovering', sub: 'Outside your usual orbit — might surprise you', marker: 'III' },
  { key: 'wild', min: -Infinity, title: 'Wildcards', sub: 'Far from your comfort zone', marker: 'IV' },
]

export default function RecommendPage({ currentUser }: RecommendPageProps) {
  const [recs, setRecs] = useState<ScoredMovie[]>([])
  const [fidelity, setFidelity] = useState<FidelityData | null>(null)
  const [loading, setLoading] = useState(false)
  const [level, setLevel] = useState(1)
  const [explain, setExplain] = useState<ExplainResult | null>(null)
  const [explainLoading, setExplainLoading] = useState(false)

  useEffect(() => { if (currentUser) load() }, [currentUser, level])

  async function load() {
    setLoading(true)
    try {
      const res = await api.getRecommendations(currentUser, 50, LEVELS[level].threshold)
      setRecs(res.recommendations)
      setFidelity(res.fidelity)
    } catch { setRecs([]); setFidelity(null) }
    setLoading(false)
  }

  if (!currentUser) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
        <div className="w-12 h-12 rounded-full border border-gold/10 flex items-center justify-center">
          <div className="w-2 h-2 rounded-full bg-gold/20" />
        </div>
        <p className="font-serif text-cream-muted italic text-sm">Select a profile to see recommendations</p>
      </div>
    )
  }

  function tierMovies(movies: ScoredMovie[]) {
    const unwatched = movies.filter(m => !m.watched)
    const watched = movies.filter(m => m.watched)
    const tiers = TIERS.map((t, i) => ({
      ...t,
      movies: unwatched.filter(m => m.similarity >= t.min && (i === 0 || m.similarity < TIERS[i - 1].min)),
    }))
    if (watched.length) tiers.push({ key: 'watched', min: 0, title: 'From your history', sub: 'Ranked by evolved taste alignment', marker: 'V', movies: watched })
    return tiers.filter(t => t.movies.length > 0)
  }

  async function loadExplain(imdb_id: string) {
    setExplainLoading(true)
    try {
      const result = await api.explainRecommendation(currentUser, imdb_id)
      setExplain(result)
    } catch { setExplain(null) }
    setExplainLoading(false)
  }

  const fLabel = (v: number) => v >= 0.9 ? 'Excellent' : v >= 0.75 ? 'Good' : v >= 0.5 ? 'Growing' : 'Cold start'

  return (
    <div className="px-8 pt-8 pb-12 space-y-10 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <div className="flex items-center gap-3 mb-3">
            <div className="w-6 h-[1px] bg-gold" />
            <span className="font-display text-[10px] font-bold tracking-[0.3em] text-gold uppercase">Recommendations</span>
          </div>
          <h1 className="font-display text-3xl font-extrabold text-cream tracking-tight">For You</h1>
          <p className="font-serif text-cream-muted italic text-sm mt-1">Curated by Elastic Associative Memory</p>
        </div>

        <div className="flex p-1 rounded-xl bg-film-card border border-warm">
          {LEVELS.map((l, i) => (
            <button
              key={l.label}
              onClick={() => setLevel(i)}
              className={`font-display text-[11px] font-semibold px-5 py-2 rounded-lg tracking-wide transition-all duration-200 ${
                i === level
                  ? 'bg-gold/15 text-gold'
                  : 'text-cream-muted hover:text-cream-dim'
              }`}
            >
              {l.label}
            </button>
          ))}
        </div>
      </div>

      {/* Fidelity */}
      {fidelity && (
        <div className="card-warm p-6 max-w-2xl bg-warm-gradient">
          <div className="flex items-center gap-3 mb-5">
            <span className="label">Model Confidence</span>
            <span className="font-mono text-[10px] text-gold bg-gold/10 px-2 py-0.5 rounded border border-gold/15">
              {fLabel(fidelity.fingerprint)}
            </span>
          </div>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <div className="label mb-2">Fingerprint precision</div>
              <SimilarityBar value={fidelity.fingerprint} color="gold" />
            </div>
            <div>
              <div className="label mb-2">Memory recall</div>
              <SimilarityBar value={fidelity.mean_movie} color="cream" />
            </div>
          </div>
        </div>
      )}

      {/* Explain Panel */}
      {(explain || explainLoading) && (
        <div className="card-warm p-6 bg-warm-gradient animate-fade-in relative">
          <button
            onClick={() => setExplain(null)}
            className="absolute top-4 right-4 w-7 h-7 rounded-lg border border-warm hover:border-gold/20 flex items-center justify-center text-cream-muted hover:text-cream transition-all"
          >
            <svg width="10" height="10" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></svg>
          </button>

          {explainLoading ? (
            <div className="flex items-center gap-3 py-4">
              <div className="w-4 h-4 border border-gold/30 border-t-gold rounded-full animate-spin" />
              <span className="font-serif text-cream-muted italic text-sm">Tracing neural pathways...</span>
            </div>
          ) : explain && (
            <div className="space-y-5">
              <div>
                <div className="flex items-center gap-3 mb-1">
                  <span className="label">Why this recommendation?</span>
                </div>
                <h3 className="font-display text-xl font-bold text-cream">{explain.movie.title}</h3>
                <div className="flex items-center gap-3 mt-1.5 font-mono text-[10px] text-cream-muted">
                  <span>{explain.activated_location_count} memory locations activated</span>
                  <span className="w-0.5 h-0.5 rounded-full bg-gold/30" />
                  <span>{explain.hopfield_iterations} Hopfield iteration{explain.hopfield_iterations !== 1 ? 's' : ''}</span>
                  {explain.converged && (
                    <>
                      <span className="w-0.5 h-0.5 rounded-full bg-gold/30" />
                      <span className="text-gold/60">converged</span>
                    </>
                  )}
                </div>
              </div>

              {explain.explanations.length > 0 ? (
                <div className="space-y-2.5">
                  <span className="label">Because you watched</span>
                  {explain.explanations.map((ex, i) => (
                    <div key={ex.imdb_id} className="flex items-center gap-3">
                      <span className="font-mono text-[9px] text-cream-muted w-4 text-right">{i + 1}</span>
                      <div className="flex-1">
                        <div className="flex items-center gap-3">
                          <span className="font-serif text-[13px] text-cream/80 italic flex-1">{ex.title}</span>
                          <span className="font-display text-[12px] font-bold text-gold tabular-nums">{ex.percentage}</span>
                        </div>
                        <div className="mt-1 h-[2px] bg-cream/[0.04] rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gold rounded-full transition-all duration-700"
                            style={{ width: `${Math.round(ex.contribution * 100)}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="font-serif text-cream-muted italic text-sm">
                  No clear single-movie contribution — this emerges from the overall memory pattern.
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Content */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-24 gap-4">
          <div className="w-8 h-8 border border-gold/20 rounded-full flex items-center justify-center">
            <div className="w-2 h-2 rounded-full bg-gold animate-flicker" />
          </div>
          <p className="font-serif text-cream-muted italic text-sm">Computing neural pathways...</p>
        </div>
      ) : recs.length === 0 ? (
        <div className="text-center py-24">
          <p className="font-serif text-cream-muted italic">Watch some films first to get recommendations.</p>
        </div>
      ) : (
        <div className="space-y-14">
          {tierMovies(recs).map(tier => (
            <div key={tier.key} className="animate-fade-in">
              <div className="flex items-center gap-4 mb-2">
                <span className="font-serif text-[13px] text-gold/40 italic">{tier.marker}</span>
                <h2 className="font-display text-lg font-bold text-cream/90">{tier.title}</h2>
                <span className="font-mono text-[9px] text-cream-muted bg-cream/[0.03] px-2 py-0.5 rounded-full">{tier.movies.length}</span>
              </div>
              <p className="font-serif text-[13px] text-cream-muted italic mb-6 ml-8">{tier.sub}</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-5">
                {tier.movies.map(m => (
                  <MovieCard
                    key={m.imdb_id}
                    title={m.title}
                    year={m.year}
                    poster={m.poster}
                    genres={m.genres}
                    imdb_rating={m.imdb_rating}
                    similarity={m.similarity}
                    watched={m.watched}
                    fidelity={m.fidelity}
                    showFidelity
                    onClick={() => !m.watched && loadExplain(m.imdb_id)}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
