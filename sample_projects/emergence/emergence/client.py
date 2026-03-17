"""HeatherDB client with multi-collection and algebra support."""

import httpx


class HeatherClient:
    def __init__(self, url: str = "http://localhost:6380"):
        self.url = url.rstrip("/")
        self._client = httpx.Client(base_url=self.url, timeout=30.0)

    # --- Core ---

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

    # --- Read/Write ---

    def write(self, collection: str, vectors: list[list[float]],
              metadata: list[dict] | None = None) -> dict:
        body: dict = {"vectors": vectors}
        if metadata is not None:
            body["metadata"] = metadata
        resp = self._client.post(f"/collections/{collection}/write", json=body)
        resp.raise_for_status()
        return resp.json()

    def read(self, collection: str, query: list[float],
             strategy: str = "iterative") -> list[float]:
        resp = self._client.post(
            f"/collections/{collection}/read",
            json={"query": query, "strategy": strategy},
        )
        resp.raise_for_status()
        return resp.json()["result"]

    def query_documents(self, collection: str, query: list[float],
                        n: int = 5) -> list[dict]:
        resp = self._client.post(
            f"/collections/{collection}/documents/query",
            json={"query": query, "n": n},
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def stats(self, collection: str) -> dict:
        resp = self._client.get(f"/collections/{collection}/stats")
        resp.raise_for_status()
        return resp.json()

    # --- Algebra ---

    def algebra_add(self, source_a: str, source_b: str, target: str) -> dict:
        resp = self._client.post("/algebra/add", json={
            "source_a": source_a,
            "source_b": source_b,
            "target": target,
        })
        resp.raise_for_status()
        return resp.json()

    def algebra_sub(self, source_a: str, source_b: str, target: str) -> dict:
        resp = self._client.post("/algebra/sub", json={
            "source_a": source_a,
            "source_b": source_b,
            "target": target,
        })
        resp.raise_for_status()
        return resp.json()

    def algebra_scale(self, source: str, target: str, alpha: float) -> dict:
        resp = self._client.post("/algebra/scale", json={
            "source": source,
            "target": target,
            "alpha": alpha,
        })
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self._client.close()
