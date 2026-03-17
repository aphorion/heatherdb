"use client";

import Link from "next/link";
import Image from "next/image";
import type { ScoredMovie } from "@/lib/types";

export default function MovieCard({ movie }: { movie: ScoredMovie }) {
  const hasPoster =
    movie.poster && movie.poster !== "N/A" && movie.poster.startsWith("http");

  return (
    <Link
      href={`/movie/${movie.imdb_id}`}
      className="group flex-shrink-0 w-[180px] md:w-[200px] relative transition-all duration-300 ease-out hover:z-30 hover:scale-105"
    >
      <div className="relative aspect-[2/3] rounded-md overflow-hidden bg-film-card shadow-lg transition-transform duration-300 ease-out border border-transparent group-hover:border-film-gold/50">
        {hasPoster ? (
          <Image
            src={movie.poster}
            alt={movie.title}
            fill
            sizes="(max-width: 768px) 180px, 200px"
            className="object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-film-muted text-xs p-2 text-center bg-film-card">
            {movie.title}
          </div>
        )}

        {/* Hover overlay with Play Button */}
        <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col items-center justify-center gap-2">
          <div className="w-12 h-12 rounded-full border-2 border-white flex items-center justify-center bg-black/30 backdrop-blur-sm">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6 text-white ml-0.5">
              <path fillRule="evenodd" d="M4.5 5.653c0-1.426 1.529-2.33 2.779-1.643l11.54 6.348c1.295.712 1.295 2.573 0 3.285L7.28 19.991c-1.25.687-2.779-.217-2.779-1.643V5.653z" clipRule="evenodd" />
            </svg>
          </div>
          <span className="text-white font-bold text-sm tracking-wide">PLAY</span>
        </div>

        {/* Match badge */}
        {movie.match !== undefined && (
          <div className="absolute top-2 left-2 bg-film-gold text-white text-[10px] font-bold px-2 py-0.5 rounded-sm shadow-sm uppercase tracking-wider">
            {movie.match}% Match
          </div>
        )}
      </div>

      {/* Title below card */}
      <div className="mt-2 opacity-80 group-hover:opacity-100 transition-opacity">
        <h3 className="text-film-cream text-sm font-bold truncate">{movie.title}</h3>
      </div>
    </Link>
  );
}
