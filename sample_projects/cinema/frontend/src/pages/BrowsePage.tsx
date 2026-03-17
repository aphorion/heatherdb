import { useState, useEffect } from 'react'
import { api, MovieSummary, MovieDetail } from '../api'
import MovieCard from '../components/MovieCard'
import Hero from '../components/Hero'
import ContentRow from '../components/ContentRow'

interface BrowsePageProps {
  currentUser: string
}

const GENRE_ORDER = ['Action', 'Sci-Fi', 'Thriller', 'Comedy', 'Drama', 'Horror', 'Romance', 'Adventure', 'Animation', 'Crime', 'Mystery', 'Fantasy']

export default function BrowsePage({ currentUser }: BrowsePageProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<MovieSummary[]>([])
  const [catalog, setCatalog] = useState<MovieDetail[]>([])
  const [catalogIds, setCatalogIds] = useState<Set<string>>(new Set())
  const [seeding, setSeeding] = useState(false)
  const [watchedIds, setWatchedIds] = useState<Set<string>>(new Set())
  const [heroMovie, setHeroMovie] = useState<MovieDetail | null>(null)
  const [searching, setSearching] = useState(false)

  useEffect(() => { loadCatalog() }, [])
  useEffect(() => {
    if (currentUser) {
      api.getHistory(currentUser).then(r => setWatchedIds(new Set(r.movies.map(m => m.imdb_id)))).catch(() => {})
    }
  }, [currentUser])

  async function loadCatalog() {
    const res = await api.getCatalog()
    setCatalog(res.movies)
    setCatalogIds(new Set(res.movies.map(m => m.imdb_id)))
    if (res.movies.length > 0) {
      const good = res.movies.filter(m => m.plot && m.plot !== 'N/A' && m.poster && m.poster !== 'N/A')
      const pool = good.length > 0 ? good : res.movies
      setHeroMovie(pool[Math.floor(Math.random() * pool.length)])
    }
  }

  useEffect(() => {
    const t = setTimeout(() => {
      if (query) doSearch()
      else setResults([])
    }, 400)
    return () => clearTimeout(t)
  }, [query])

  async function doSearch() {
    setSearching(true)
    try { setResults((await api.search(query)).movies) } catch { setResults([]) }
    setSearching(false)
  }

  async function addToCatalog(id: string) { await api.addToCatalog(id); await loadCatalog() }
  async function watchMovie(id: string) {
    if (!currentUser) return
    if (!catalogIds.has(id)) { await api.addToCatalog(id); await loadCatalog() }
    await api.watchMovie(currentUser, id)
    setWatchedIds(prev => new Set([...prev, id]))
  }

  const moviesByGenre: Record<string, MovieDetail[]> = {}
  GENRE_ORDER.forEach(g => moviesByGenre[g] = [])
  catalog.forEach(m => m.genres?.forEach(g => {
    if (!moviesByGenre[g]) moviesByGenre[g] = []
    moviesByGenre[g].push(m)
  }))
  const validGenres = Object.keys(moviesByGenre).filter(g => moviesByGenre[g].length > 0)
  validGenres.sort((a, b) => {
    const ia = GENRE_ORDER.indexOf(a), ib = GENRE_ORDER.indexOf(b)
    if (ia !== -1 && ib !== -1) return ia - ib
    if (ia !== -1) return -1
    if (ib !== -1) return 1
    return a.localeCompare(b)
  })

  return (
    <div className="pb-16">
      {/* Hero */}
      {!query && catalog.length > 0 && (
        <Hero
          movie={heroMovie}
          onWatch={() => heroMovie && watchMovie(heroMovie.imdb_id)}
          onAdd={() => heroMovie && addToCatalog(heroMovie.imdb_id)}
          added={heroMovie ? catalogIds.has(heroMovie.imdb_id) : false}
        />
      )}

      {/* Search */}
      <div className={`sticky top-14 z-40 px-6 transition-all duration-300 ${
        query ? 'pt-6' : catalog.length > 0 ? '-mt-8 mb-4' : 'pt-6'
      }`}>
        <div className="relative max-w-sm">
          <div className="absolute inset-y-0 left-0 flex items-center pl-4 pointer-events-none">
            <svg className="w-4 h-4 text-gold/30" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
          </div>
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search films..."
            className="w-full py-3 pl-11 pr-4 font-serif text-sm text-cream bg-film-dark/80 border border-warm rounded-xl focus:outline-none focus:border-gold/20 placeholder-cream-muted backdrop-blur-xl transition-all"
          />
        </div>
      </div>

      {/* Search results */}
      {query && (
        <div className="px-6 mt-6 animate-fade-in">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-4 h-[1px] bg-gold/30" />
            <h2 className="font-display text-lg font-bold text-cream/80">
              {searching ? 'Searching...' : `"${query}"`}
            </h2>
          </div>
          {results.length === 0 && !searching ? (
            <p className="text-cream-muted font-serif italic text-center py-20">No films found.</p>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {results.map(m => (
                <MovieCard
                  key={m.imdb_id}
                  title={m.title}
                  year={m.year}
                  poster={m.poster}
                  added={catalogIds.has(m.imdb_id)}
                  watched={watchedIds.has(m.imdb_id)}
                  onAdd={() => addToCatalog(m.imdb_id)}
                  onWatch={currentUser ? () => watchMovie(m.imdb_id) : undefined}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Genre rows */}
      {!query && (
        <div className="space-y-2">
          {validGenres.map(genre => (
            <ContentRow
              key={genre}
              title={genre}
              movies={moviesByGenre[genre]}
              catalogIds={catalogIds}
              watchedIds={watchedIds}
              onAdd={addToCatalog}
              onWatch={watchMovie}
              currentUser={currentUser}
            />
          ))}

          {/* Empty state */}
          {catalog.length === 0 && (
            <div className="flex flex-col items-center justify-center py-32 gap-8 animate-fade-in">
              {/* Film reel decoration */}
              <div className="relative">
                <div className="w-24 h-24 rounded-full border-2 border-gold/10 flex items-center justify-center">
                  <div className="w-16 h-16 rounded-full border border-gold/20 flex items-center justify-center">
                    <div className="w-3 h-3 rounded-full bg-gold/20" />
                  </div>
                </div>
                {/* Sprocket holes */}
                {[0, 60, 120, 180, 240, 300].map(deg => (
                  <div key={deg} className="absolute w-2 h-2 rounded-full bg-gold/10" style={{
                    top: `${50 - 46 * Math.cos(deg * Math.PI / 180)}%`,
                    left: `${50 + 46 * Math.sin(deg * Math.PI / 180)}%`,
                    transform: 'translate(-50%, -50%)',
                  }} />
                ))}
              </div>

              <div className="text-center space-y-3 max-w-xs">
                <p className="font-display text-lg font-bold text-cream/50">The reel is empty</p>
                <p className="font-serif text-sm text-cream-muted italic leading-relaxed">
                  Load our curated collection of iconic films, or search above to add titles one by one.
                </p>
              </div>

              <button
                onClick={async () => {
                  setSeeding(true)
                  try { await api.seed(); await loadCatalog() } catch {}
                  setSeeding(false)
                }}
                disabled={seeding}
                className="group bg-gold hover:bg-gold-bright disabled:bg-film-card disabled:text-cream-muted text-film-black px-8 py-3.5 rounded-xl font-display text-sm font-bold tracking-wide transition-all hover:scale-[1.02] active:scale-[0.98] shadow-[0_4px_24px_rgba(232,179,75,0.2)] disabled:shadow-none"
              >
                {seeding ? (
                  <span className="flex items-center gap-3">
                    <div className="w-4 h-4 border-2 border-film-black/30 border-t-film-black rounded-full animate-spin" />
                    Loading films...
                  </span>
                ) : 'Load Popular Films'}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
