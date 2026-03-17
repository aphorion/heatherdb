"""Thin HTTP client for HeatherDB."""

import httpx


class HeatherClient:
    def __init__(self, url: str = "http://localhost:6380"):
        self.url = url.rstrip("/")
        self._client = httpx.Client(base_url=self.url, timeout=30.0)

    def write(self, vectors: list[list[float]]) -> int:
        resp = self._client.post("/write", json={"vectors": vectors})
        resp.raise_for_status()
        return resp.json()["count"]

    def read(self, query: list[float], strategy: str = "iterative") -> list[float]:
        resp = self._client.post("/read", json={"query": query, "strategy": strategy})
        resp.raise_for_status()
        return resp.json()["result"]

    def stats(self) -> dict:
        resp = self._client.get("/stats")
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
