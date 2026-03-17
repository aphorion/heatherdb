"use client";

import { Suspense, useEffect, useState, useCallback } from "react";
import { useSearchParams } from "next/navigation";
import { useProfile } from "@/hooks/useProfile";
import {
  getCatalog,
  seedLocal,
  seedCatalog,
  getBecauseYouWatched,
  getRecommendations,
  getWildcards,
  getContinueWatching,
  getGenres,
  getGenreMovies,
  searchMovies,
} from "@/lib/api";
import type { Movie, ScoredMovie, BecauseRow, TieredRecs } from "@/lib/types";
import MovieRow from "@/components/MovieRow";
import MovieCard from "@/components/MovieCard";
import HeroBanner from "@/components/HeroBanner";

function round(n: number) {
  return Math.round(n * 10) / 10;
}

export default function HomePage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center min-h-[60vh]">
          <p className="text-film-muted">Loading...</p>
        </div>
      }
    >
      <HomeContent />
    </Suspense>
  );
}

function HomeContent() {
  const { profile } = useProfile();
  const searchParams = useSearchParams();
  const query = searchParams.get("q");

  const [catalog, setCatalog] = useState<Movie[]>([]);
  const [searchResults, setSearchResults] = useState<Movie[] | null>(null);
  const [becauseRows, setBecauseRows] = useState<BecauseRow[]>([]);
  const [tiered, setTiered] = useState<TieredRecs | null>(null);
  const [wildcards, setWildcards] = useState<ScoredMovie[]>([]);
  const [continueWatching, setContinueWatching] = useState<ScoredMovie[]>([]);
  const [genreRows, setGenreRows] = useState<{ genre: string; movies: ScoredMovie[] }[]>([]);
  const [seeding, setSeeding] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let cancelled = false;

    // Reset personalized state immediately
    setBecauseRows([]);
    setTiered(null);
    setWildcards([]);
    setContinueWatching([]);

    async function load() {
      try {
        const movies = await getCatalog();
        if (cancelled) return;
        setCatalog(movies);
        if (!movies.length) return;
        setLoaded(true);

        // Load genre rows (pick top genres)
        const genres = await getGenres().catch(() => [] as string[]);
        if (cancelled) return;
        const topGenres = genres.slice(0, 6);
        const genreData = await Promise.all(
          topGenres.map(async (g) => ({
            genre: g,
            movies: await getGenreMovies(g, profile || undefined, 15).catch(() => []),
          }))
        );
        if (cancelled) return;
        setGenreRows(genreData.filter((r) => r.movies.length > 0));

        if (profile) {
          const [rows, recs, wild, cont] = await Promise.all([
            getBecauseYouWatched(profile).catch(() => []),
            getRecommendations(profile).catch(() => null),
            getWildcards(profile, 12).catch(() => []),
            getContinueWatching(profile, 12).catch(() => []),
          ]);
          if (cancelled) return;
          setBecauseRows(rows);
          setTiered(recs);
          setWildcards(wild);
          setContinueWatching(cont);
        }
      } catch {
        // backend not ready
      }
    }

    load();
    return () => { cancelled = true; };
  }, [profile, reloadKey]);

  useEffect(() => {
    if (query) {
      searchMovies(query)
        .then(setSearchResults)
        .catch(() => setSearchResults([]));
    } else {
      setSearchResults(null);
    }
  }, [query]);

  async function handleSeed() {
    setSeeding(true);
    try {
      await seedCatalog();
      setReloadKey((k) => k + 1);
    } finally {
      setSeeding(false);
    }
  }

  async function handleSeedLocal() {
    setSeeding(true);
    try {
      await seedLocal();
      setReloadKey((k) => k + 1);
    } finally {
      setSeeding(false);
    }
  }

  // Search results view
  if (searchResults !== null) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <h2 className="font-heading text-2xl font-bold text-film-cream mb-6">
          Results for &quot;{query}&quot;
        </h2>
        {searchResults.length === 0 ? (
          <p className="text-film-muted">No movies found.</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-4">
            {searchResults.map((m) => (
              <MovieCard key={m.imdb_id} movie={m as ScoredMovie} />
            ))}
          </div>
        )}
      </div>
    );
  }

  // Empty state
  if (!loaded) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-6">
        <h1 className="font-heading text-3xl font-bold text-film-gold">
          Welcome to Cinema
        </h1>
        <p className="text-film-muted text-center max-w-md">
          Discover movies you&apos;ll love. Start by seeding the catalog.
        </p>
        <div className="flex gap-3">
          <button
            onClick={handleSeed}
            disabled={seeding}
            className="bg-film-gold text-film-black font-heading font-bold px-8 py-3 rounded-lg hover:bg-film-gold-bright transition disabled:opacity-50"
          >
            {seeding ? "Loading..." : "Seed from OMDb (200+ movies)"}
          </button>
          <button
            onClick={handleSeedLocal}
            disabled={seeding}
            className="border border-film-border text-film-cream font-heading px-6 py-3 rounded-lg hover:bg-film-card transition disabled:opacity-50"
          >
            Seed Local (40 movies)
          </button>
        </div>
      </div>
    );
  }

  // Build the "We Recommend" row: top recs if profile exists, otherwise top-rated catalog
  const weRecommend: ScoredMovie[] = (() => {
    if (profile && tiered) {
      const all = [
        ...(tiered.perfect_for_you || []),
        ...(tiered.great_match || []),
        ...(tiered.worth_watching || []),
        ...(tiered.discover || []),
      ];
      if (all.length > 0) return all.slice(0, 15);
    }
    // No profile: show top-rated from catalog
    return catalog
      .map((m) => ({
        ...m,
        match: round(parseFloat(m.imdb_rating || "0") * 10),
      }))
      .sort((a, b) => b.match - a.match)
      .slice(0, 15) as ScoredMovie[];
  })();

  // Pick featured movie for hero: top rec or top-rated
  const heroMovie = weRecommend[0] || null;

  return (
    <div>
      {/* Hero Banner */}
      {heroMovie && <HeroBanner movie={heroMovie} />}

      <div className="max-w-[1800px] mx-auto pt-8 px-4 md:px-8">
        {/* We Recommend */}
        {weRecommend.length > 1 && (
          <MovieRow
            title={profile ? "We Recommend" : "Top Rated"}
            subtitle={profile ? "Picked for you based on your taste" : "The highest rated movies in our catalog"}
            movies={weRecommend.slice(1)}
          />
        )}

        {/* Continue Watching */}
        {continueWatching.length > 0 && (
          <MovieRow
            title="Continue Watching"
            movies={continueWatching}
            variant="landscape"
          />
        )}

        {/* Because you watched rows */}
        {becauseRows.map((row) => (
          <MovieRow
            key={row.because.imdb_id}
            title={`Because you watched ${row.because.title}`}
            movies={row.movies}
          />
        ))}

        {/* Perfect For You (separate from We Recommend for deeper matches) */}
        {tiered?.perfect_for_you && tiered.perfect_for_you.length > 5 ? (
          <MovieRow title="Perfect For You" movies={tiered.perfect_for_you.slice(5)} />
        ) : null}

        {/* Try Something New (wildcards) */}
        {wildcards.length > 0 && (
          <MovieRow
            title="Try Something New"
            subtitle="Movies outside your comfort zone"
            movies={wildcards}
          />
        )}

        {/* Genre rows */}
        {genreRows.map((row) => (
          <MovieRow
            key={row.genre}
            title={row.genre}
            movies={row.movies}
          />
        ))}

        {!profile && (
          <div className="text-center py-12">
            <p className="text-film-muted mb-2">
              Create a profile to get personalized recommendations
            </p>
            <p className="text-film-muted text-sm">
              Use the profile switcher in the top right
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
