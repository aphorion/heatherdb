import type {
  Movie,
  ScoredMovie,
  BecauseRow,
  TieredRecs,
  WhyResult,
  TasteProfile,
  BlendResult,
} from "./types";

const API = "/api";

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${url}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

// --- Seed & Catalog ---

export async function seedLocal() {
  return fetchJSON<{ added: number; skipped: number }>("/seed/local", { method: "POST" });
}

export async function seedCatalog() {
  return fetchJSON<{ added: number; skipped: number }>("/seed", { method: "POST" });
}

export async function getCatalog() {
  return fetchJSON<{ movies: Movie[] }>("/catalog").then((r) => r.movies);
}

export async function searchMovies(q: string) {
  return fetchJSON<{ movies: Movie[] }>(`/search?q=${encodeURIComponent(q)}`).then((r) => r.movies);
}

export async function getMovie(id: string) {
  return fetchJSON<Movie>(`/movie/${id}`);
}

export async function getSimilarMovies(id: string, n = 10) {
  return fetchJSON<{ movies: ScoredMovie[] }>(`/movie/${id}/similar?n=${n}`).then((r) => r.movies);
}

// --- Profiles ---

export async function getProfiles() {
  return fetchJSON<{ profiles: string[] }>("/profiles").then((r) => r.profiles);
}

export async function createProfile(name: string) {
  return fetchJSON<{ status: string }>(`/profiles/${encodeURIComponent(name)}`, { method: "POST" });
}

export async function deleteProfile(name: string) {
  return fetchJSON<{ status: string }>(`/profiles/${encodeURIComponent(name)}`, { method: "DELETE" });
}

export async function watchMovie(profile: string, imdbId: string) {
  return fetchJSON<{ status: string }>(`/profiles/${encodeURIComponent(profile)}/watch`, {
    method: "POST",
    body: JSON.stringify({ imdb_id: imdbId }),
  });
}

export async function getHistory(profile: string) {
  return fetchJSON<{ movies: Movie[] }>(`/profiles/${encodeURIComponent(profile)}/history`).then(
    (r) => r.movies
  );
}

// --- Recommendations ---

export async function getRecommendations(profile: string) {
  return fetchJSON<TieredRecs>(`/profiles/${encodeURIComponent(profile)}/recommendations`);
}

export async function getBecauseYouWatched(profile: string) {
  return fetchJSON<{ rows: BecauseRow[] }>(
    `/profiles/${encodeURIComponent(profile)}/because-you-watched`
  ).then((r) => r.rows);
}

export async function getWhyRecommended(profile: string, imdbId: string) {
  return fetchJSON<WhyResult>(`/profiles/${encodeURIComponent(profile)}/movie/${imdbId}/why`);
}

// --- Wildcards & Continue Watching ---

export async function getWildcards(profile: string, n = 10) {
  return fetchJSON<{ movies: ScoredMovie[] }>(
    `/profiles/${encodeURIComponent(profile)}/wildcards?n=${n}`
  ).then((r) => r.movies);
}

export async function getContinueWatching(profile: string, n = 10) {
  return fetchJSON<{ movies: ScoredMovie[] }>(
    `/profiles/${encodeURIComponent(profile)}/continue-watching?n=${n}`
  ).then((r) => r.movies);
}

// --- Genres ---

export async function getGenres() {
  return fetchJSON<{ genres: string[] }>("/genres").then((r) => r.genres);
}

export async function getGenreMovies(genre: string, profile?: string, n = 20) {
  const params = new URLSearchParams({ n: String(n) });
  if (profile) params.set("profile", profile);
  return fetchJSON<{ genre: string; movies: ScoredMovie[] }>(
    `/genres/${encodeURIComponent(genre)}?${params}`
  ).then((r) => r.movies);
}

// --- Taste ---

export async function getTasteProfile(profile: string) {
  return fetchJSON<TasteProfile>(`/profiles/${encodeURIComponent(profile)}/taste`);
}

// --- Blend ---

export async function blendProfiles(profile1: string, profile2: string) {
  return fetchJSON<BlendResult>("/profiles/blend", {
    method: "POST",
    body: JSON.stringify({ profiles: [profile1, profile2] }),
  });
}
