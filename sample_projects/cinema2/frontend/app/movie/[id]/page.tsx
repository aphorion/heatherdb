"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Image from "next/image";
import { useProfile } from "@/hooks/useProfile";
import { getMovie, getSimilarMovies, getWhyRecommended, watchMovie } from "@/lib/api";
import type { Movie, ScoredMovie, WhyResult } from "@/lib/types";
import MovieRow from "@/components/MovieRow";
import WhyRecommended from "@/components/WhyRecommended";

export default function MovieDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { profile } = useProfile();
  const [movie, setMovie] = useState<Movie | null>(null);
  const [similar, setSimilar] = useState<ScoredMovie[]>([]);
  const [why, setWhy] = useState<WhyResult | null>(null);
  const [watched, setWatched] = useState(false);

  useEffect(() => {
    if (!id) return;
    getMovie(id).then(setMovie).catch(() => { });
    getSimilarMovies(id).then(setSimilar).catch(() => { });
  }, [id]);

  useEffect(() => {
    if (!id || !profile) return;
    getWhyRecommended(profile, id).then(setWhy).catch(() => { });
  }, [id, profile]);

  async function handleWatch() {
    if (!profile || !id) return;
    await watchMovie(profile, id);
    setWatched(true);
  }

  if (!movie) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <p className="text-film-muted">Loading...</p>
      </div>
    );
  }

  const hasPoster =
    movie.poster && movie.poster !== "N/A" && movie.poster.startsWith("http");

  return (
    <div>
      {/* Hero */}
      <div className="relative w-full h-[50vh] min-h-[350px] overflow-hidden">
        {hasPoster && (
          <div
            className="absolute inset-0 bg-cover bg-center blur-sm scale-110"
            style={{ backgroundImage: `url(${movie.poster})` }}
          />
        )}
        <div className="absolute inset-0 bg-film-black/70" />

        <div className="relative z-10 h-full max-w-[1800px] mx-auto px-4 md:px-8 flex items-end pb-8 gap-8">
          {hasPoster && (
            <div className="hidden md:block w-48 h-72 rounded-lg overflow-hidden shadow-2xl relative flex-shrink-0">
              <Image
                src={movie.poster}
                alt={movie.title}
                fill
                sizes="192px"
                className="object-cover"
              />
            </div>
          )}

          <div className="flex-1 min-w-0">
            <h1 className="font-heading text-3xl md:text-4xl font-bold text-white mb-2">
              {movie.title}
            </h1>
            <div className="flex flex-wrap items-center gap-3 text-sm text-film-muted mb-3">
              <span>{movie.year}</span>
              <span>{movie.runtime}</span>
              <span>{movie.rated}</span>
              <span className="text-film-gold">{movie.imdb_rating} IMDb</span>
              {why && (
                <span className="match-badge">{why.match}% match</span>
              )}
            </div>
            <div className="flex flex-wrap gap-2 mb-3">
              {movie.genres.map((g) => (
                <span
                  key={g}
                  className="text-xs px-2 py-0.5 border border-film-border rounded-full text-film-muted"
                >
                  {g}
                </span>
              ))}
            </div>
            <p className="text-film-cream/90 max-w-2xl leading-relaxed mb-4 line-clamp-4 md:line-clamp-6 font-light">
              {movie.plot}
            </p>
            <div className="flex flex-wrap gap-2 text-sm text-film-muted mb-4">
              <span>Directed by <span className="text-film-cream">{movie.director}</span></span>
              <span>· Starring <span className="text-film-cream">{movie.actors.join(", ")}</span></span>
            </div>
            {profile && !watched && (
              <button
                onClick={handleWatch}
                className="bg-film-gold text-film-black font-heading font-bold px-6 py-2 rounded hover:bg-film-gold-bright transition"
              >
                Mark as Watched
              </button>
            )}
            {watched && (
              <span className="text-green-400 text-sm font-heading">Watched!</span>
            )}
          </div>
        </div>
      </div>

      <div className="max-w-[1800px] mx-auto px-4 md:px-8 py-8 md:py-12 grid lg:grid-cols-4 gap-8 md:gap-12">
        {/* Why recommended */}
        <div className="lg:col-span-1">
          {why && <WhyRecommended influences={why.influences} />}
        </div>

        {/* Similar movies */}
        <div className="lg:col-span-3">
          <MovieRow title="Similar Movies" movies={similar} />
        </div>
      </div>
    </div>
  );
}
