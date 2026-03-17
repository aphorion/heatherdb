"use client";

import type { ScoredMovie } from "@/lib/types";
import MovieCard from "./MovieCard";
import LandscapeMovieCard from "./LandscapeMovieCard"; // Import new component
import Link from "next/link";

interface Props {
  title: string;
  subtitle?: string;
  movies: ScoredMovie[];
  variant?: "default" | "landscape"; // New prop
}

export default function MovieRow({ title, subtitle, movies, variant = "default" }: Props) {
  if (!movies.length) return null;

  return (
    <section className="mb-12">
      <div className="px-4 md:px-12 mb-4 flex items-end justify-between">
        <div>
          <h2 className="font-heading text-xl md:text-2xl font-bold text-film-cream flex items-center gap-2">
            {title}
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2.5} stroke="currentColor" className="w-4 h-4 text-film-gold opacity-100 transition-all duration-300">
              <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 4.5l7.5 7.5-7.5 7.5" />
            </svg>
          </h2>
          {subtitle && (
            <p className="text-film-muted text-sm font-medium mt-1 line-clamp-1">{subtitle}</p>
          )}
        </div>
        <Link href="#" className="hidden sm:block text-xs font-bold text-film-gold hover:text-white uppercase tracking-wider transition-opacity duration-300">
          See All
        </Link>
      </div>

      <div className="scroll-row px-4 md:px-12 pb-8">
        {movies.map((m, i) => (
          variant === "landscape" ? (
            // Mock progress for demo purposes
            <LandscapeMovieCard key={m.imdb_id} movie={m} progress={Math.floor(Math.random() * 80) + 10} />
          ) : (
            <MovieCard key={m.imdb_id} movie={m} />
          )
        ))}
      </div>
    </section>
  );
}
