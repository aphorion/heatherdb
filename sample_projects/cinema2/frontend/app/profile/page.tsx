"use client";

import { useEffect, useState } from "react";
import { useProfile } from "@/hooks/useProfile";
import { getTasteProfile, getHistory } from "@/lib/api";
import type { TasteProfile, Movie } from "@/lib/types";
import GenreDNA from "@/components/GenreDNA";
import TasteJourney from "@/components/TasteJourney";
import MovieCard from "@/components/MovieCard";
import type { ScoredMovie } from "@/lib/types";

export default function ProfilePage() {
  const { profile } = useProfile();
  const [taste, setTaste] = useState<TasteProfile | null>(null);
  const [history, setHistory] = useState<Movie[]>([]);

  useEffect(() => {
    if (!profile) return;
    getTasteProfile(profile).then(setTaste).catch(() => {});
    getHistory(profile).then(setHistory).catch(() => {});
  }, [profile]);

  if (!profile) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <p className="text-film-muted">Select a profile to view your taste.</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="font-heading text-3xl font-bold text-film-cream mb-1">
          {profile}&apos;s Taste Profile
        </h1>
        {taste && (
          <p className="text-film-muted">{taste.status}</p>
        )}
      </div>

      {taste && (
        <div className="grid md:grid-cols-2 gap-8 mb-10">
          <div className="bg-film-card border border-film-border rounded-xl p-6">
            <GenreDNA dna={taste.genre_dna} />
            {taste.confidence > 0 && (
              <div className="mt-4 pt-4 border-t border-film-border">
                <p className="text-sm text-film-muted">
                  Match Confidence:{" "}
                  <span className="text-film-gold font-heading font-bold">
                    {taste.confidence}%
                  </span>
                </p>
              </div>
            )}
          </div>

          <div className="bg-film-card border border-film-border rounded-xl p-6">
            <TasteJourney journey={taste.journey} />
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div>
          <h2 className="font-heading text-lg font-bold text-film-cream mb-4">
            Watch History ({history.length} movies)
          </h2>
          <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-4">
            {history.map((m) => (
              <MovieCard key={m.imdb_id} movie={m as ScoredMovie} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
