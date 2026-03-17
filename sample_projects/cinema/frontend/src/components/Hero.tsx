import { MovieDetail } from '../api'

interface HeroProps {
  movie: MovieDetail | null
  onWatch: () => void
  onAdd: () => void
  added: boolean
}

export default function Hero({ movie, onWatch, onAdd, added }: HeroProps) {
  if (!movie) {
    return (
      <div className="h-[70vh] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border border-gold/30 rounded-full flex items-center justify-center">
            <div className="w-1.5 h-1.5 rounded-full bg-gold animate-flicker" />
          </div>
          <span className="font-serif text-cream-dim text-sm italic">Loading feature...</span>
        </div>
      </div>
    )
  }

  const bgImage = movie.poster !== 'N/A' ? movie.poster : ''

  return (
    <div className="relative h-[72vh] overflow-hidden">
      {/* Background image */}
      {bgImage && (
        <div className="absolute inset-0">
          <img src={bgImage} alt="" className="w-full h-full object-cover object-top opacity-30 animate-zoom-slow" />
          {/* Warm overlay */}
          <div className="absolute inset-0 bg-gradient-to-r from-film-black via-film-black/80 to-film-black/40" />
          <div className="absolute inset-0 bg-gradient-to-t from-film-black via-transparent to-film-black/50" />
          {/* Gold warm wash */}
          <div className="absolute inset-0 bg-gradient-to-br from-gold/[0.03] via-transparent to-crimson/[0.02]" />
        </div>
      )}

      {/* Content */}
      <div className="relative z-10 h-full flex items-end px-8 pb-16">
        <div className="max-w-xl space-y-6">
          {/* Tag line */}
          <div className="flex items-center gap-4 animate-slide-up">
            <div className="flex items-center gap-2">
              <div className="w-6 h-[1px] bg-gold" />
              <span className="font-display text-[10px] font-bold tracking-[0.3em] text-gold uppercase">Featured</span>
            </div>
            {movie.imdb_rating && movie.imdb_rating !== 'N/A' && (
              <span className="font-mono text-[11px] text-gold/60">{movie.imdb_rating}/10</span>
            )}
          </div>

          {/* Title */}
          <h1 className="font-display text-5xl md:text-6xl font-extrabold text-cream leading-[1.05] tracking-tight text-glow-gold animate-slide-up delay-75">
            {movie.title}
          </h1>

          {/* Metadata */}
          <div className="flex items-center gap-4 text-cream-dim animate-slide-up delay-150">
            <span className="font-serif text-lg">{movie.year}</span>
            <span className="w-1 h-1 rounded-full bg-gold/30" />
            <span className="font-serif text-lg">{movie.runtime !== 'N/A' ? movie.runtime : ''}</span>
            <span className="w-1 h-1 rounded-full bg-gold/30" />
            <span className="font-display text-[10px] font-semibold tracking-[0.15em] uppercase border border-cream/10 px-2 py-0.5 rounded">
              {movie.rated !== 'N/A' ? movie.rated : 'PG-13'}
            </span>
          </div>

          {/* Plot */}
          <p className="font-serif text-[17px] text-cream/50 leading-relaxed line-clamp-3 max-w-lg animate-slide-up delay-200">
            {movie.plot !== 'N/A' ? movie.plot : ''}
          </p>

          {/* Genres */}
          <div className="flex items-center gap-3 animate-slide-up delay-300">
            {movie.genres?.map(g => (
              <span key={g} className="font-mono text-[9px] tracking-[0.2em] text-cream-muted uppercase border border-warm px-2.5 py-1 rounded-full">
                {g}
              </span>
            ))}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3 pt-3 animate-slide-up delay-500">
            <button
              onClick={onWatch}
              className="group bg-gold hover:bg-gold-bright text-film-black px-7 py-3 rounded-xl font-display text-sm font-bold tracking-wide flex items-center gap-2.5 hover:scale-[1.02] active:scale-[0.98] transition-all shadow-[0_4px_20px_rgba(232,179,75,0.25)]"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
              Watch
            </button>
            <button
              onClick={onAdd}
              className="group bg-cream/[0.04] hover:bg-cream/[0.08] border border-cream/10 hover:border-cream/20 text-cream px-7 py-3 rounded-xl font-display text-sm font-medium tracking-wide flex items-center gap-2.5 hover:scale-[1.02] active:scale-[0.98] transition-all"
            >
              {added ? (
                <>
                  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>
                  Listed
                </>
              ) : (
                <>
                  <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
                  My List
                </>
              )}
            </button>
          </div>
        </div>

        {/* Decorative film frame markers */}
        <div className="absolute bottom-8 right-8 flex items-center gap-2 opacity-20">
          <span className="font-mono text-[9px] text-gold tracking-[0.3em]">REEL 01</span>
          <div className="w-8 h-[1px] bg-gold/50" />
        </div>
      </div>
    </div>
  )
}
