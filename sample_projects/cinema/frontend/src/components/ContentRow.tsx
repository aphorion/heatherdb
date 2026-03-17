import { useRef, useState, useEffect } from 'react'
import MovieCard from './MovieCard'

interface ContentRowProps {
  title: string
  movies: any[]
  catalogIds: Set<string>
  watchedIds: Set<string>
  onAdd: (id: string) => void
  onWatch: (id: string) => void
  currentUser: string
}

export default function ContentRow({
  title, movies, catalogIds, watchedIds, onAdd, onWatch, currentUser,
}: ContentRowProps) {
  const rowRef = useRef<HTMLDivElement>(null)
  const [canScrollLeft, setCanScrollLeft] = useState(false)
  const [canScrollRight, setCanScrollRight] = useState(true)

  function updateScroll() {
    if (!rowRef.current) return
    const { scrollLeft, scrollWidth, clientWidth } = rowRef.current
    setCanScrollLeft(scrollLeft > 10)
    setCanScrollRight(scrollLeft < scrollWidth - clientWidth - 10)
  }

  useEffect(() => {
    updateScroll()
    const el = rowRef.current
    if (el) el.addEventListener('scroll', updateScroll)
    return () => { if (el) el.removeEventListener('scroll', updateScroll) }
  }, [movies])

  const scroll = (dir: 'left' | 'right') => {
    if (!rowRef.current) return
    rowRef.current.scrollBy({ left: dir === 'left' ? -rowRef.current.clientWidth * 0.75 : rowRef.current.clientWidth * 0.75, behavior: 'smooth' })
  }

  if (!movies || movies.length === 0) return null

  return (
    <div className="relative group/row px-6 py-3">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-4 h-[1px] bg-gold/30" />
        <h2 className="font-display text-[14px] font-bold tracking-wide text-cream/70 group-hover/row:text-cream transition-colors">
          {title}
        </h2>
        <span className="font-mono text-[9px] text-cream-muted">{movies.length}</span>
      </div>

      <div className="relative -mx-6">
        {canScrollLeft && (
          <button
            onClick={() => scroll('left')}
            className="absolute left-0 top-0 bottom-10 z-40 w-12 bg-gradient-to-r from-film-black/90 to-transparent flex items-center justify-center opacity-0 group-hover/row:opacity-100 transition-opacity"
          >
            <div className="w-8 h-8 rounded-full border border-gold/20 bg-film-dark/80 backdrop-blur flex items-center justify-center hover:bg-gold/10 transition-all">
              <svg width="14" height="14" fill="none" stroke="#e8b34b" strokeWidth="2" viewBox="0 0 24 24"><path d="m15 18-6-6 6-6"/></svg>
            </div>
          </button>
        )}

        <div ref={rowRef} className="flex gap-3 overflow-x-auto no-scrollbar scroll-smooth px-6 pb-4">
          {movies.map(m => (
            <MovieCard
              key={m.imdb_id}
              title={m.title}
              year={m.year}
              poster={m.poster}
              genres={m.genres}
              imdb_rating={m.imdb_rating}
              watched={watchedIds.has(m.imdb_id)}
              added={catalogIds.has(m.imdb_id)}
              onAdd={() => onAdd(m.imdb_id)}
              onWatch={currentUser ? () => onWatch(m.imdb_id) : undefined}
              landscape
            />
          ))}
        </div>

        {canScrollRight && (
          <button
            onClick={() => scroll('right')}
            className="absolute right-0 top-0 bottom-10 z-40 w-12 bg-gradient-to-l from-film-black/90 to-transparent flex items-center justify-center opacity-0 group-hover/row:opacity-100 transition-opacity"
          >
            <div className="w-8 h-8 rounded-full border border-gold/20 bg-film-dark/80 backdrop-blur flex items-center justify-center hover:bg-gold/10 transition-all">
              <svg width="14" height="14" fill="none" stroke="#e8b34b" strokeWidth="2" viewBox="0 0 24 24"><path d="m9 18 6-6-6-6"/></svg>
            </div>
          </button>
        )}
      </div>
    </div>
  )
}
