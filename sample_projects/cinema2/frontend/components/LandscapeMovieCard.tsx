"use client";

import Link from "next/link";
import Image from "next/image";
import type { ScoredMovie } from "@/lib/types";

export default function LandscapeMovieCard({
    movie,
    progress,
}: {
    movie: ScoredMovie;
    progress?: number;
}) {
    // Use a fallback backdrop if needed, though API typically provides posters.
    // Ideally, we'd have a 'backdrop' property. For now, we'll center crop the poster or use a placeholder.
    // In a real app, you'd want actual 16:9 assets.
    const imageSrc = movie.poster && movie.poster !== "N/A" ? movie.poster : "/placeholder.png";

    return (
        <Link
            href={`/movie/${movie.imdb_id}`}
            className="group flex-shrink-0 w-[280px] sm:w-[320px] relative transition-all duration-300 ease-out hover:z-30 hover:scale-105"
        >
            <div className="relative aspect-video rounded-md overflow-hidden bg-film-card shadow-lg border border-transparent group-hover:border-film-gold/50">
                <Image
                    src={imageSrc}
                    alt={movie.title}
                    fill
                    className="object-cover"
                    sizes="(max-width: 768px) 280px, 320px"
                />

                {/* Play overlay */}
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <div className="w-10 h-10 rounded-full bg-film-gold text-white flex items-center justify-center shadow-lg transform scale-0 group-hover:scale-100 transition-transform duration-300 delay-75">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5 ml-0.5">
                            <path fillRule="evenodd" d="M4.5 5.653c0-1.426 1.529-2.33 2.779-1.643l11.54 6.348c1.295.712 1.295 2.573 0 3.285L7.28 19.991c-1.25.687-2.779-.217-2.779-1.643V5.653z" clipRule="evenodd" />
                        </svg>
                    </div>
                </div>

                {/* Progress Bar (Visual only for now) */}
                {progress !== undefined && (
                    <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-600/50">
                        <div
                            className="h-full bg-film-gold"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                )}
            </div>

            <div className="mt-2 px-1">
                <h3 className="text-film-cream font-bold text-sm truncate">{movie.title}</h3>
                <div className="flex justify-between items-center text-xs text-film-muted mt-0.5">
                    <span>{movie.year}</span>
                    {progress !== undefined && <span className="text-film-gold">{progress}m left</span>}
                </div>
                <p className="text-film-muted/70 text-xs mt-1 line-clamp-1">{movie.plot}</p>
            </div>
        </Link>
    );
}
