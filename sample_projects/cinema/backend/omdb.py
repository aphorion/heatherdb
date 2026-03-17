"""OMDb API client."""

import httpx

OMDB_BASE = "https://www.omdbapi.com/"


class OMDbClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=15.0)

    async def search(self, query: str, page: int = 1) -> list[dict]:
        resp = await self._client.get(
            OMDB_BASE,
            params={"apikey": self.api_key, "s": query, "type": "movie", "page": page},
        )
        if resp.status_code == 401:
            raise ValueError("Invalid OMDb API key")
        resp.raise_for_status()
        data = resp.json()
        if data.get("Response") == "False":
            return []
        results = []
        for item in data.get("Search", []):
            results.append({
                "imdb_id": item["imdbID"],
                "title": item["Title"],
                "year": item["Year"],
                "poster": item.get("Poster", "N/A"),
            })
        return results

    async def get_movie(self, imdb_id: str) -> dict | None:
        resp = await self._client.get(
            OMDB_BASE,
            params={"apikey": self.api_key, "i": imdb_id, "plot": "full"},
        )
        if resp.status_code == 401:
            raise ValueError("Invalid OMDb API key")
        resp.raise_for_status()
        data = resp.json()
        if data.get("Response") == "False":
            return None
        return {
            "imdb_id": data["imdbID"],
            "title": data.get("Title", ""),
            "year": data.get("Year", ""),
            "rated": data.get("Rated", "N/A"),
            "runtime": data.get("Runtime", "N/A"),
            "genres": [g.strip() for g in data.get("Genre", "").split(",") if g.strip()],
            "director": data.get("Director", "N/A"),
            "actors": [a.strip() for a in data.get("Actors", "").split(",") if a.strip()],
            "plot": data.get("Plot", ""),
            "poster": data.get("Poster", "N/A"),
            "imdb_rating": data.get("imdbRating", "N/A"),
            "imdb_votes": data.get("imdbVotes", "N/A"),
        }

    async def close(self):
        await self._client.aclose()
