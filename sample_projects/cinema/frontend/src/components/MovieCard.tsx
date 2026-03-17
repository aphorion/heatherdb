interface MovieCardProps {
  title: string
  year: string
  poster: string
  genres?: string[]
  imdb_rating?: string
  similarity?: number
  fidelity?: number
  watched?: boolean
  watchedBy?: string[]
  onClick?: () => void
  onWatch?: () => void
  onAdd?: () => void
  added?: boolean
  compact?: boolean
  showFidelity?: boolean
  landscape?: boolean
}

const PLACEHOLDER = 'data:image/svg+xml,' + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450" fill="%231a1816"><rect width="300" height="450"/><text x="150" y="225" text-anchor="middle" fill="%23332f2a" font-size="16" font-family="serif">No Poster</text></svg>'
)

export default function MovieCard({
  title, year, poster, genres, imdb_rating, similarity, fidelity,
  watched, watchedBy, onClick, onWatch, onAdd, added, compact, showFidelity, landscape,
}: MovieCardProps) {
  const imgSrc = poster && poster !== 'N/A' ? poster : PLACEHOLDER

  const matchColor = (s: number) =>
    s >= 0.4 ? 'text-gold' : s >= 0.2 ? 'text-cream' : 'text-cream-dim'
  const matchBar = (s: number) =>
    s >= 0.4 ? 'bg-gold' : s >= 0.2 ? 'bg-cream/40' : 'bg-cream/15'

  return (
    <div
      onClick={onClick}
      className={`group relative flex-shrink-0 transition-all duration-300 ${onClick ? 'cursor-pointer' : ''} ${
      landscape ? 'w-[260px] md:w-[280px]' : compact ? 'w-[120px]' : 'w-[150px] md:w-[170px]'
    }`}>
      <div className={`relative ${landscape ? 'aspect-[16/10]' : 'aspect-[2/3]'} rounded-xl overflow-hidden bg-film-card poster-shadow group-hover:shadow-[0_12px_40px_rgba(0,0,0,0.5)] group-hover:scale-[1.04] transition-all duration-300 ease-out`}>
        <img
          src={imgSrc}
          alt={title}
          className={`w-full h-full object-cover transition-all duration-500 ${
            watched ? 'opacity-25 saturate-0' : 'group-hover:scale-105 group-hover:opacity-80'
          }`}
          loading="lazy"
        />

        {/* Warm vignette */}
        <div className="absolute inset-0 bg-gradient-to-t from-film-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        <div className="absolute inset-0 bg-gradient-to-br from-gold/[0.03] to-transparent pointer-events-none" />

        {/* Top corner badges */}
        <div className="absolute top-2 left-2 right-2 flex justify-between items-start z-10">
          {imdb_rating && imdb_rating !== 'N/A' && !landscape && (
            <span className="bg-film-black/70 backdrop-blur text-gold font-mono text-[9px] font-medium px-1.5 py-0.5 rounded-md border border-gold/10">
              {imdb_rating}
            </span>
          )}
          {similarity !== undefined && showFidelity && (
            <span className={`ml-auto bg-film-black/70 backdrop-blur font-mono text-[9px] font-medium px-2 py-0.5 rounded-md border border-warm ${matchColor(similarity)}`}>
              {Math.round(similarity * 100)}%
            </span>
          )}
        </div>

        {/* Watched */}
        {watched && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="bg-film-black/60 backdrop-blur-md px-3 py-1.5 rounded-lg border border-gold/10">
              <span className="font-display text-[10px] font-semibold tracking-[0.2em] text-gold/60 uppercase">Watched</span>
            </div>
          </div>
        )}

        {/* Hover info */}
        <div className="absolute bottom-0 left-0 right-0 p-3 translate-y-2 opacity-0 group-hover:translate-y-0 group-hover:opacity-100 transition-all duration-300 z-10">
          <h3 className="font-display text-[13px] font-semibold text-cream leading-tight line-clamp-2 mb-1">{title}</h3>
          <div className="flex items-center gap-2 text-[10px] text-cream-dim font-serif mb-2.5">
            <span>{year}</span>
            {genres && genres.length > 0 && (
              <>
                <span className="w-0.5 h-0.5 rounded-full bg-gold/30" />
                <span className="truncate italic">{genres.slice(0, 2).join(', ')}</span>
              </>
            )}
          </div>
          <div className="flex gap-1.5">
            {onWatch && !watched && (
              <button
                onClick={(e) => { e.stopPropagation(); onWatch() }}
                className="flex-1 bg-gold text-film-black font-display text-[10px] font-bold py-1.5 rounded-lg hover:bg-gold-bright transition-all flex items-center justify-center gap-1"
              >
                <svg width="9" height="9" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                Watch
              </button>
            )}
            {onAdd && !added && (
              <button
                onClick={(e) => { e.stopPropagation(); onAdd() }}
                className="w-8 h-7 bg-cream/[0.06] hover:bg-cream/[0.12] rounded-lg flex items-center justify-center border border-warm transition-all"
              >
                <svg width="11" height="11" fill="none" stroke="#ede8df" strokeWidth="2.5" viewBox="0 0 24 24"><path d="M12 5v14m-7-7h14"/></svg>
              </button>
            )}
            {onAdd && added && (
              <div className="w-8 h-7 bg-gold/10 rounded-lg flex items-center justify-center border border-gold/20">
                <svg width="11" height="11" fill="none" stroke="#e8b34b" strokeWidth="2.5" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"/></svg>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Below card — fidelity bar */}
      {showFidelity && similarity !== undefined && (
        <div className="mt-2.5 space-y-1.5 px-0.5">
          <p className="font-serif text-[11px] text-cream-dim truncate italic">{title}</p>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-[2px] bg-cream/[0.06] rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-700 ${matchBar(similarity)}`}
                style={{ width: `${Math.min(100, Math.max(3, similarity * 100))}%` }}
              />
            </div>
            <span className={`font-mono text-[9px] tabular-nums ${matchColor(similarity)}`}>
              {Math.round(similarity * 100)}%
            </span>
          </div>
        </div>
      )}

      {/* Landscape info */}
      {landscape && !showFidelity && (
        <div className="mt-2 px-0.5">
          <p className="font-display text-[12px] font-medium text-cream/60 truncate">{title}</p>
          <div className="flex items-center gap-2 font-serif text-[11px] text-cream-muted">
            <span>{year}</span>
            {imdb_rating && imdb_rating !== 'N/A' && (
              <>
                <span className="w-0.5 h-0.5 rounded-full bg-gold/20" />
                <span className="text-gold/50">{imdb_rating}</span>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
