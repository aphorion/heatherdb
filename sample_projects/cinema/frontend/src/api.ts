const BASE = '/api'

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export interface MovieSummary {
  imdb_id: string
  title: string
  year: string
  poster: string
}

export interface MovieDetail extends MovieSummary {
  rated: string
  runtime: string
  genres: string[]
  director: string
  actors: string[]
  plot: string
  imdb_rating: string
  imdb_votes: string
}

export interface ScoredMovie extends MovieDetail {
  similarity: number
  watched: boolean
  fidelity?: number
}

export interface FidelityData {
  fingerprint: number
  per_movie: Record<string, number>
  mean_movie: number
}

export interface BlendedMovie extends MovieDetail {
  similarity: number
  watched_by: string[]
}

export interface Fingerprint {
  vector: number[] | null
  genre_affinities: Record<string, number>
  movies_watched: number
  fidelity: FidelityData | null
}

export interface BlendResult {
  user1_affinities: Record<string, number>
  user2_affinities: Record<string, number>
  taste_similarity: number
  recommendations: BlendedMovie[]
}

export interface StabilityResult {
  fingerprint_similarity: number
  before_recommendations: ScoredMovie[]
  after_recommendations: ScoredMovie[]
  sci_fi_count: number
  outlier_added: boolean
}

export interface ColdStartSnapshot {
  movies_watched: number
  recommendations: ScoredMovie[]
  top_genres: Record<string, number>
}

export interface EvolutionStep {
  step: number
  movie_watched: string
  movie_title: string
  fingerprint_stability: number
  hopfield_iterations: number
  converged: boolean
  top_genres: Record<string, number>
}

export interface ExplainContribution {
  imdb_id: string
  title: string
  contribution: number
  percentage: string
}

export interface ExplainResult {
  movie: MovieDetail
  explanations: ExplainContribution[]
  hopfield_iterations: number
  converged: boolean
  activated_location_count: number
}

export const api = {
  health: () => request<{ status: string; heather_ok: boolean }>('/health'),

  search: (q: string) => request<{ movies: MovieSummary[] }>(`/search?q=${encodeURIComponent(q)}`),

  getMovie: (id: string) => request<MovieDetail>(`/movie/${id}`),

  addToCatalog: (imdb_id: string) =>
    request<{ status: string }>('/catalog/add', {
      method: 'POST',
      body: JSON.stringify({ imdb_id }),
    }),

  getCatalog: () => request<{ movies: MovieDetail[] }>('/catalog'),

  listUsers: () => request<{ users: string[] }>('/users'),

  watchMovie: (username: string, imdb_id: string) =>
    request<{ status: string }>(`/users/${username}/watch`, {
      method: 'POST',
      body: JSON.stringify({ imdb_id }),
    }),

  getHistory: (username: string) => request<{ movies: MovieDetail[] }>(`/users/${username}/history`),

  getFingerprint: (username: string) => request<Fingerprint>(`/users/${username}/fingerprint`),

  getRecommendations: (username: string, n = 10, threshold = 0.0) =>
    request<{ recommendations: ScoredMovie[]; fidelity: FidelityData | null }>(
      `/users/${username}/recommend?n=${n}&threshold=${threshold}`
    ),

  deleteUser: (username: string) =>
    request<{ status: string }>(`/users/${username}`, { method: 'DELETE' }),

  blend: (user1: string, user2: string) =>
    request<BlendResult>('/blend', {
      method: 'POST',
      body: JSON.stringify({ users: [user1, user2] }),
    }),

  demoStability: (outlier_id?: string) =>
    request<StabilityResult>('/demo/stability', {
      method: 'POST',
      body: JSON.stringify({ outlier_id: outlier_id ?? null }),
    }),

  demoColdStart: () =>
    request<{ snapshots: ColdStartSnapshot[] }>('/demo/coldstart', { method: 'POST' }),

  getEvolution: (username: string) =>
    request<{ username: string; total_watches: number; timeline: EvolutionStep[] }>(
      `/users/${username}/evolution`
    ),

  explainRecommendation: (username: string, imdb_id: string) =>
    request<ExplainResult>(`/users/${username}/recommend/${imdb_id}/explain`),

  seed: () =>
    request<{ added: number; skipped: number; failed: number }>('/seed', { method: 'POST' }),
}
