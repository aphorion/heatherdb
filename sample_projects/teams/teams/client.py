"""Thin HTTP client for HeatherDB with collection + compose support."""

import httpx


class HeatherClient:
    def __init__(self, url: str = "http://localhost:6380"):
        self.url = url.rstrip("/")
        self._client = httpx.Client(base_url=self.url, timeout=30.0)

    def health(self) -> bool:
        try:
            resp = self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def create_collection(self, name: str) -> dict:
        resp = self._client.post("/collections", json={"name": name})
        resp.raise_for_status()
        return resp.json()

    def list_collections(self) -> list[str]:
        resp = self._client.get("/collections")
        resp.raise_for_status()
        return resp.json()["collections"]

    def drop_collection(self, name: str) -> bool:
        resp = self._client.delete(f"/collections/{name}")
        resp.raise_for_status()
        return resp.json()["dropped"]

    def write(self, collection: str, vectors: list[list[float]], metadata: list[dict] | None = None) -> dict:
        body: dict = {"vectors": vectors}
        if metadata is not None:
            body["metadata"] = metadata
        resp = self._client.post(f"/collections/{collection}/write", json=body)
        resp.raise_for_status()
        return resp.json()

    def read(self, collection: str, query: list[float], strategy: str = "iterative") -> list[float]:
        resp = self._client.post(
            f"/collections/{collection}/read",
            json={"query": query, "strategy": strategy},
        )
        resp.raise_for_status()
        return resp.json()["result"]

    def compose_read(
        self,
        collections: list[str],
        query: list[float],
        routing_sharpness: float = 20.0,
    ) -> dict:
        resp = self._client.post(
            "/compose/read",
            json={
                "collections": collections,
                "query": query,
                "routing_sharpness": routing_sharpness,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def stats(self, collection: str) -> dict:
        resp = self._client.get(f"/collections/{collection}/stats")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()
