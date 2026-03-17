"""Thin HTTP client for HeatherDB."""

import httpx


class HeatherClient:
    def __init__(self, url: str = "http://localhost:6380"):
        self.url = url.rstrip("/")
        self._client = httpx.Client(base_url=self.url, timeout=30.0)

    def write(self, vectors: list[list[float]]) -> int:
        """Write vectors to HeatherDB. Returns count of vectors written."""
        resp = self._client.post("/write", json={"vectors": vectors})
        resp.raise_for_status()
        return resp.json()["count"]

    def read(self, query: list[float], strategy: str = "iterative") -> list[float]:
        """Read (reconstruct) a vector from HeatherDB using the query as a cue."""
        resp = self._client.post("/read", json={"query": query, "strategy": strategy})
        resp.raise_for_status()
        return resp.json()["result"]

    def stats(self) -> dict:
        """Get HeatherDB statistics."""
        resp = self._client.get("/stats")
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        """Check if HeatherDB is reachable."""
        try:
            resp = self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def close(self):
        self._client.close()
