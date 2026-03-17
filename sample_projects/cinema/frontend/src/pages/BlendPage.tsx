import { useState, useEffect } from 'react'
import { api, BlendResult } from '../api'
import MovieCard from '../components/MovieCard'
import SimilarityBar from '../components/SimilarityBar'

export default function BlendPage() {
  const [users, setUsers] = useState<string[]>([])
  const [user1, setUser1] = useState('')
  const [user2, setUser2] = useState('')
  const [result, setResult] = useState<BlendResult | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => { api.listUsers().then(r => setUsers(r.users)).catch(() => {}) }, [])

  async function runBlend() {
    if (!user1 || !user2 || user1 === user2) return
    setLoading(true)
    try { setResult(await api.blend(user1, user2)) } catch { setResult(null) }
    setLoading(false)
  }

  function AffinityList({ affinities, label }: { affinities: Record<string, number>; label: string }) {
    return (
      <div>
        <h3 className="label mb-3">{label}</h3>
        <div className="space-y-2">
          {Object.entries(affinities).slice(0, 8).map(([genre, score]) => (
            <SimilarityBar key={genre} value={score} label={genre} />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="px-8 pt-8 pb-12 space-y-10 animate-fade-in">
      <div>
        <div className="flex items-center gap-3 mb-3">
          <div className="w-6 h-[1px] bg-gold" />
          <span className="font-display text-[10px] font-bold tracking-[0.3em] text-gold uppercase">Blend</span>
        </div>
        <h1 className="font-display text-3xl font-extrabold text-cream tracking-tight mb-1">Blend Tastes</h1>
        <p className="font-serif text-cream-muted italic text-sm">Merge two fingerprints to find films you'll both enjoy.</p>
      </div>

      {/* Selector */}
      <div className="card-warm p-6 max-w-xl bg-warm-gradient">
        <div className="flex items-end gap-4 flex-wrap">
          <div className="flex-1 min-w-[120px]">
            <label className="label block mb-2">Profile 1</label>
            <select
              value={user1}
              onChange={e => setUser1(e.target.value)}
              className="w-full bg-film-dark border border-warm rounded-xl px-3 py-2.5 font-serif text-sm text-cream focus:outline-none focus:border-gold/20 transition-colors appearance-none cursor-pointer"
            >
              <option value="" className="bg-film-dark">Select...</option>
              {users.map(u => <option key={u} value={u} className="bg-film-dark">{u}</option>)}
            </select>
          </div>

          <div className="flex items-center justify-center w-10 h-10 rounded-full border border-gold/15 text-gold/30 font-serif text-lg italic mb-0.5">+</div>

          <div className="flex-1 min-w-[120px]">
            <label className="label block mb-2">Profile 2</label>
            <select
              value={user2}
              onChange={e => setUser2(e.target.value)}
              className="w-full bg-film-dark border border-warm rounded-xl px-3 py-2.5 font-serif text-sm text-cream focus:outline-none focus:border-gold/20 transition-colors appearance-none cursor-pointer"
            >
              <option value="" className="bg-film-dark">Select...</option>
              {users.filter(u => u !== user1).map(u => <option key={u} value={u} className="bg-film-dark">{u}</option>)}
            </select>
          </div>

          <button
            onClick={runBlend}
            disabled={!user1 || !user2 || user1 === user2 || loading}
            className="bg-gold hover:bg-gold-bright disabled:bg-film-card disabled:text-cream-muted text-film-black px-6 py-2.5 rounded-xl font-display text-sm font-bold tracking-wide transition-all hover:scale-[1.02] active:scale-[0.98] shadow-[0_4px_16px_rgba(232,179,75,0.2)] disabled:shadow-none"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <div className="w-3.5 h-3.5 border-2 border-film-black/30 border-t-film-black rounded-full animate-spin" />
                Blending...
              </span>
            ) : 'Blend'}
          </button>
        </div>
      </div>

      {result && (
        <div className="space-y-8 animate-fade-in">
          {/* Compatibility */}
          <div className="card-warm p-6 max-w-xl">
            <div className="flex items-center gap-5 mb-4">
              <div className="w-8 h-8 rounded-lg bg-gold/15 border border-gold/20 flex items-center justify-center font-display text-[10px] font-bold uppercase text-gold">{user1.substring(0, 2)}</div>
              <div className="flex-1">
                <div className="label mb-2">Taste compatibility</div>
                <SimilarityBar value={result.taste_similarity} color={result.taste_similarity > 0.5 ? 'gold' : 'cream'} />
              </div>
              <div className="w-8 h-8 rounded-lg bg-crimson/15 border border-crimson/20 flex items-center justify-center font-display text-[10px] font-bold uppercase text-crimson">{user2.substring(0, 2)}</div>
            </div>
            <p className="font-serif text-[12px] text-cream-muted italic">
              {result.taste_similarity > 0.7 ? 'Very compatible — rich taste overlap' :
               result.taste_similarity > 0.4 ? 'Some common ground with interesting divergences' :
               'Very different palates — the blend should produce surprises'}
            </p>
          </div>

          {/* Side by side */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="card-warm p-5 bg-warm-gradient">
              <AffinityList affinities={result.user1_affinities} label={`${user1}'s DNA`} />
            </div>
            <div className="card-warm p-5">
              <AffinityList affinities={result.user2_affinities} label={`${user2}'s DNA`} />
            </div>
          </div>

          {/* Recs */}
          <div>
            <div className="flex items-center gap-3 mb-6">
              <div className="w-4 h-[1px] bg-gold/30" />
              <h2 className="font-display text-lg font-bold text-cream/80">
                Films for {user1} + {user2}
              </h2>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-5">
              {result.recommendations.map(m => (
                <MovieCard
                  key={m.imdb_id}
                  title={m.title}
                  year={m.year}
                  poster={m.poster}
                  genres={m.genres}
                  similarity={m.similarity}
                  watchedBy={m.watched_by}
                  showFidelity
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
