import type {
  HealthResponse,
  WriteRequest,
  WriteResponse,
  ReadRequest,
  ReadResponse,
  StatsResponse,
  CreateCollectionRequest,
  CreateCollectionResponse,
  ListCollectionsResponse,
  DropCollectionResponse,
  ConfigResponse,
  LocationsResponse,
  AnalyzeRequest,
  AnalyzeResponse,
  DocumentsResponse,
  DocumentResponse,
  QueryDocumentsRequest,
  QueryDocumentsResponse,
} from './types'

const BASE = import.meta.env.DEV ? '' : ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(body.error || res.statusText)
  }
  return res.json()
}

export const api = {
  health: () => request<HealthResponse>('/health'),

  listCollections: () => request<ListCollectionsResponse>('/collections'),

  createCollection: (name: string) =>
    request<CreateCollectionResponse>('/collections', {
      method: 'POST',
      body: JSON.stringify({ name } satisfies CreateCollectionRequest),
    }),

  dropCollection: (name: string) =>
    request<DropCollectionResponse>(`/collections/${name}`, {
      method: 'DELETE',
    }),

  write: (collection: string, vectors: number[][], metadata?: Record<string, unknown>[]) =>
    request<WriteResponse>(`/collections/${collection}/write`, {
      method: 'POST',
      body: JSON.stringify({ vectors, metadata } satisfies WriteRequest),
    }),

  read: (collection: string, query: number[], strategy?: 'iterative' | 'fast') =>
    request<ReadResponse>(`/collections/${collection}/read`, {
      method: 'POST',
      body: JSON.stringify({ query, strategy } satisfies ReadRequest),
    }),

  stats: (collection: string) =>
    request<StatsResponse>(`/collections/${collection}/stats`),

  config: (collection: string) =>
    request<ConfigResponse>(`/collections/${collection}/config`),

  locations: (collection: string) =>
    request<LocationsResponse>(`/collections/${collection}/locations`),

  analyze: (collection: string, query: number[], strategy?: 'iterative' | 'fast') =>
    request<AnalyzeResponse>(`/collections/${collection}/analyze`, {
      method: 'POST',
      body: JSON.stringify({ query, strategy } satisfies AnalyzeRequest),
    }),

  getDocuments: (collection: string) =>
    request<DocumentsResponse>(`/collections/${collection}/documents`),

  getDocument: (collection: string, id: number) =>
    request<DocumentResponse>(`/collections/${collection}/documents/${id}`),

  queryDocuments: (collection: string, query: number[], n?: number) =>
    request<QueryDocumentsResponse>(`/collections/${collection}/documents/query`, {
      method: 'POST',
      body: JSON.stringify({ query, n } satisfies QueryDocumentsRequest),
    }),
}
