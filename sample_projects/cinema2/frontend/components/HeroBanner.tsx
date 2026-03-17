"use client";

import { useRouter } from "next/navigation";
import type { Movie } from "@/lib/types";
import { useProfile } from "@/hooks/useProfile";
import { watchMovie } from "@/lib/api";

export default function HeroBanner({ movie }: { movie: Movie }) {
  const router = useRouter();
  const { profile } = useProfile();

  async function handleWatch() {
    if (!profile) return;
    await watchMovie(profile, movie.imdb_id);
    router.refresh();
  }

  return (
    <div className="relative w-full h-[85vh] min-h-[600px] overflow-hidden group">
      {/* Backdrop Image */}
      <div
        className="absolute inset-0 bg-cover bg-center transition-transform duration-[10s] ease-out group-hover:scale-105"
        style={{ backgroundImage: `url(${movie.poster})` }}
      />

      {/* Gradient Overlays - Prime Video Style */}
      <div className="absolute inset-0 bg-gradient-to-t from-film-black via-film-black/40 to-transparent" />
      <div className="absolute inset-0 bg-gradient-to-r from-film-black via-film-black/60 to-transparent" />

      {/* Main Content */}
      <div className="relative z-10 h-full max-w-[1800px] mx-auto px-4 sm:px-6 lg:px-8 flex items-end pb-16 md:pb-24">
        <div className="max-w-2xl animate-fade-in-up">
          {/* Metadata Row */}
          <div className="flex items-center gap-4 text-sm md:text-base font-medium text-film-muted mb-4 uppercase tracking-wider">
            {movie.rated && <span className="px-2 py-0.5 border border-film-muted/50 rounded text-film-cream text-xs">{movie.rated}</span>}
            <span>{movie.year}</span>
            <span>{movie.runtime}</span>
            <span className="flex items-center gap-1 text-film-gold">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4">
                <path fillRule="evenodd" d="M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.007 5.404.433c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.433 2.082-5.006z" clipRule="evenodd" />
              </svg>
              {movie.imdb_rating}
            </span>
            <span className="hidden md:inline decoration-film-gold/30 underline decoration-2 underline-offset-4">
              {movie.genres[0]}
            </span>
          </div>

          {/* Title */}
          <h1 className="font-heading text-5xl md:text-7xl font-extrabold text-white mb-6 leading-tight drop-shadow-lg">
            {movie.title}
          </h1>

          {/* Plot */}
          <p className="text-film-cream/90 text-lg md:text-xl leading-relaxed mb-8 line-clamp-3 md:line-clamp-4 max-w-xl font-light">
            {movie.plot}
          </p>

          {/* Buttons */}
          <div className="flex flex-wrap gap-4">
            <button
              onClick={() => router.push(`/movie/${movie.imdb_id}`)}
              className="bg-film-gold text-white font-bold text-lg px-8 py-4 rounded-full hover:bg-film-gold-bright transition-all transform hover:scale-105 shadow-lg shadow-film-gold/20 flex items-center gap-2"
            >
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 text-white">
                <path fillRule="evenodd" d="M4.5 5.653c0-1.426 1.529-2.33 2.779-1.643l11.54 6.348c1.295.712 1.295 2.573 0 3.285L7.28 19.991c-1.25.687-2.779-.217-2.779-1.643V5.653z" clipRule="evenodd" />
              </svg>
              Watch Now
            </button>

            {profile && (
              <button
                onClick={handleWatch}
                className="bg-film-card/80 backdrop-blur-md border border-film-border text-white font-medium text-lg px-8 py-4 rounded-full hover:bg-film-card hover:border-white transition-all transform hover:scale-105 flex items-center gap-2"
              >
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-6 h-6">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                </svg>
                Watchlist
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
