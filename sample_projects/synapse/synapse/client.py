"""Thin HTTP client for HeatherDB (collection-aware)."""

import httpx


class HeatherClient:
    def __init__(self, url: str = "http://localhost:6380", collection: str = "mnist"):
        self.url = url.rstrip("/")
        self.collection = collection
        self._client = httpx.Client(base_url=self.url, timeout=30.0)

    def _col(self, path: str) -> str:
        return f"/collections/{self.collection}{path}"

    def ensure_collection(self):
        """Create the collection if it doesn't exist."""
        resp = self._client.post("/collections", json={"name": self.collection})
        resp.raise_for_status()

    def write(self, vectors: list[list[float]]) -> int:
        resp = self._client.post(self._col("/write"), json={"vectors": vectors})
        resp.raise_for_status()
        return resp.json()["count"]

    def read(self, query: list[float], strategy: str = "iterative") -> list[float]:
        resp = self._client.post(
            self._col("/read"), json={"query": query, "strategy": strategy}
        )
        resp.raise_for_status()
        return resp.json()["result"]

    def stats(self) -> dict:
        resp = self._client.get(self._col("/stats"))
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        try:
            resp = self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def close(self):
        self._client.close()
