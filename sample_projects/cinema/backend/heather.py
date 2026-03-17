"""Async HTTP client for HeatherDB with collection support."""

import httpx


class HeatherClient:
    def __init__(self, url: str = "http://localhost:6380"):
        self.url = url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.url, timeout=30.0)

    async def write(self, collection: str, vectors: list[list[float]], metadata: list[dict] | None = None) -> dict:
        body: dict = {"vectors": vectors}
        if metadata is not None:
            body["metadata"] = metadata
        resp = await self._client.post(
            f"/collections/{collection}/write", json=body
        )
        resp.raise_for_status()
        return resp.json()

    async def read(self, collection: str, query: list[float], strategy: str = "iterative") -> list[float]:
        resp = await self._client.post(
            f"/collections/{collection}/read", json={"query": query, "strategy": strategy}
        )
        resp.raise_for_status()
        return resp.json()["result"]

    async def collections(self) -> list[str]:
        resp = await self._client.get("/collections")
        resp.raise_for_status()
        return resp.json().get("collections", [])

    async def fingerprint(self, collection: str) -> list[float] | None:
        resp = await self._client.get(f"/collections/{collection}/fingerprint")
        resp.raise_for_status()
        return resp.json().get("fingerprint")

    async def analyze(self, collection: str, query: list[float], strategy: str = "iterative") -> dict:
        resp = await self._client.post(
            f"/collections/{collection}/analyze", json={"query": query, "strategy": strategy}
        )
        resp.raise_for_status()
        return resp.json()

    async def batch_analyze(self, collection: str, queries: list[list[float]], strategy: str = "iterative") -> list[dict]:
        resp = await self._client.post(
            f"/collections/{collection}/batch_analyze", json={"queries": queries, "strategy": strategy}
        )
        resp.raise_for_status()
        return resp.json()["results"]

    async def locations(self, collection: str) -> list[dict]:
        resp = await self._client.get(f"/collections/{collection}/locations")
        resp.raise_for_status()
        return resp.json().get("locations", [])

    async def get_documents(self, collection: str) -> list[dict]:
        resp = await self._client.get(f"/collections/{collection}/documents")
        resp.raise_for_status()
        return resp.json().get("documents", [])

    async def get_document(self, collection: str, doc_id: int) -> dict:
        resp = await self._client.get(f"/collections/{collection}/documents/{doc_id}")
        resp.raise_for_status()
        return resp.json()

    async def query_documents(self, collection: str, query: list[float], n: int = 10) -> list[dict]:
        resp = await self._client.post(
            f"/collections/{collection}/documents/query", json={"query": query, "n": n}
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self):
        await self._client.aclose()
