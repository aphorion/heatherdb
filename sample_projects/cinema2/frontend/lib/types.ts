export interface Movie {
  imdb_id: string;
  title: string;
  year: string;
  rated?: string;
  runtime?: string;
  genres: string[];
  director: string;
  actors: string[];
  plot: string;
  poster: string;
  imdb_rating?: string;
  imdb_votes?: string;
}

export interface ScoredMovie extends Movie {
  match: number;
  watched_by?: string[];
}

export interface BecauseRow {
  because: {
    imdb_id: string;
    title: string;
    poster: string;
  };
  movies: ScoredMovie[];
}

export interface TieredRecs {
  perfect_for_you: ScoredMovie[];
  great_match: ScoredMovie[];
  worth_watching: ScoredMovie[];
  discover: ScoredMovie[];
}

export interface Influence {
  imdb_id: string;
  title: string;
  poster: string;
  contribution: number;
}

export interface WhyResult {
  movie: Movie;
  match: number;
  influences: Influence[];
}

export interface GenreDNA {
  [genre: string]: number;
}

export interface JourneyStep {
  step: number;
  movie: string;
  stability: string;
  top_genres: GenreDNA;
}

export interface TasteProfile {
  status: string;
  genre_dna: GenreDNA;
  confidence: number;
  journey: JourneyStep[];
  movies_watched: number;
}

export interface BlendResult {
  profile1_dna: GenreDNA;
  profile2_dna: GenreDNA;
  compatibility: number;
  recommendations: ScoredMovie[];
}
