import { useState, useEffect } from 'react'
import { api, MovieDetail, Fingerprint, EvolutionStep } from '../api'
import MovieCard from '../components/MovieCard'
import SimilarityBar from '../components/SimilarityBar'

interface ProfilePageProps {
  currentUser: string
}

export default function ProfilePage({ currentUser }: ProfilePageProps) {
  const [history, setHistory] = useState<MovieDetail[]>([])
  const [fingerprint, setFingerprint] = useState<Fingerprint | null>(null)
  const [evolution, setEvolution] = useState<EvolutionStep[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => { if (currentUser) load() }, [currentUser])
  async function load() {
    setLoading(true)
    try {
      const [h, f, e] = await Promise.all([api.getHistory(currentUser), api.getFingerprint(currentUser), api.getEvolution(currentUser)])
      setHistory(h.movies)
      setFingerprint(f)
      setEvolution(e.timeline)
    } catch { setHistory([]); setFingerprint(null) }
    setLoading(false)
  }

  if (!currentUser) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
        <div className="w-12 h-12 rounded-full border border-gold/10 flex items-center justify-center">
          <div className="w-2 h-2 rounded-full bg-gold/20" />
        </div>
        <p className="font-serif text-cream-muted italic text-sm">Select a profile to view their list</p>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="w-8 h-8 border border-gold/20 rounded-full flex items-center justify-center">
          <div className="w-2 h-2 rounded-full bg-gold animate-flicker" />
        </div>
      </div>
    )
  }

  const topAffinities = fingerprint?.genre_affinities ? Object.entries(fingerprint.genre_affinities).slice(0, 10) : []

  return (
    <div className="px-8 pt-8 pb-12 space-y-10 animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-6">
        <div className="w-16 h-16 rounded-2xl bg-gold/10 border border-gold/20 flex items-center justify-center font-display text-2xl font-bold text-gold">
          {currentUser.substring(0, 2).toUpperCase()}
        </div>
        <div>
          <h1 className="font-display text-3xl font-extrabold text-cream tracking-tight">{currentUser}</h1>
          <p className="font-serif text-cream-muted italic text-sm">{history.length} films watched</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Stats */}
        <div className="space-y-4">
          <div className="card-warm p-5 space-y-4 bg-warm-gradient">
            <h3 className="label">Activity</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="font-display text-3xl font-bold text-cream tabular-nums">{history.length}</div>
                <div className="font-serif text-[11px] text-cream-muted italic mt-0.5">Films</div>
              </div>
              <div>
                <div className="font-display text-3xl font-bold text-cream tabular-nums">{history.length}</div>
                <div className="font-serif text-[11px] text-cream-muted italic mt-0.5">SDM Writes</div>
              </div>
            </div>
          </div>
          {fingerprint?.fidelity && (
            <div className="card-warm p-5 space-y-4">
              <h3 className="label">Neural Fidelity</h3>
              <div className="space-y-3">
                <SimilarityBar value={fingerprint.fidelity.fingerprint} label="Fingerprint" color="gold" />
                <SimilarityBar value={fingerprint.fidelity.mean_movie} label="Recall" color="cream" />
              </div>
            </div>
          )}
        </div>

        {/* Taste DNA */}
        <div className="lg:col-span-2">
          {topAffinities.length > 0 && (
            <div className="card-warm p-6 h-full bg-warm-gradient">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-4 h-[1px] bg-gold" />
                <h3 className="label">Taste DNA</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-3">
                {topAffinities.map(([genre, score]) => (
                  <SimilarityBar key={genre} value={score} label={genre} />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Taste Evolution */}
      {evolution.length > 1 && (
        <div className="card-warm p-6 bg-warm-gradient">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-4 h-[1px] bg-gold" />
            <h3 className="label">Taste Evolution</h3>
            <span className="font-mono text-[9px] text-cream-muted bg-cream/[0.03] px-2 py-0.5 rounded-full">{evolution.length} steps</span>
          </div>

          {/* Stability chart — visual bars */}
          <div className="space-y-1.5 mb-6">
            {evolution.map((step, i) => (
              <div key={i} className="flex items-center gap-3 group/step hover:bg-cream/[0.02] rounded-lg px-2 py-1.5 transition-colors">
                <span className="font-mono text-[9px] text-cream-muted w-4 text-right shrink-0">{step.step}</span>
                <span className="font-serif text-[12px] text-cream/60 italic w-40 truncate shrink-0">{step.movie_title}</span>
                <div className="flex-1 h-[3px] bg-cream/[0.04] rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${
                      step.fingerprint_stability >= 0.95 ? 'bg-gold' :
                      step.fingerprint_stability >= 0.85 ? 'bg-gold/70' :
                      step.fingerprint_stability >= 0.5 ? 'bg-cream/40' : 'bg-crimson/50'
                    }`}
                    style={{ width: `${Math.max(3, step.fingerprint_stability * 100)}%` }}
                  />
                </div>
                <span className="font-mono text-[9px] text-cream-muted w-10 text-right tabular-nums shrink-0">
                  {(step.fingerprint_stability * 100).toFixed(0)}%
                </span>
                <span className="font-mono text-[8px] text-cream-muted/50 w-6 text-right tabular-nums shrink-0" title="Hopfield iterations">
                  {step.hopfield_iterations}it
                </span>
              </div>
            ))}
          </div>

          <p className="font-serif text-[11px] text-cream-muted italic leading-relaxed">
            {evolution[evolution.length - 1].fingerprint_stability >= 0.95
              ? 'Your taste attractor has stabilized — new watches cause minimal drift.'
              : evolution[evolution.length - 1].fingerprint_stability >= 0.8
              ? 'Your taste is settling — a few more watches should lock it in.'
              : 'Your taste is still forming — keep watching to build a stronger fingerprint.'}
          </p>
        </div>
      )}

      {/* History */}
      <div>
        <div className="flex items-center gap-3 mb-6">
          <div className="w-4 h-[1px] bg-gold/30" />
          <h2 className="font-display text-lg font-bold text-cream/80">Watch History</h2>
        </div>
        {history.length === 0 ? (
          <p className="font-serif text-cream-muted italic text-sm">No films watched yet.</p>
        ) : (
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-7 gap-3">
            {history.map(m => (
              <MovieCard key={m.imdb_id} title={m.title} year={m.year} poster={m.poster} genres={m.genres} compact watched />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
