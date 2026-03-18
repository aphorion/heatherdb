// --- Request types ---

export interface WriteRequest {
  vectors: number[][];
  metadata?: Record<string, unknown>[];
}

export interface ReadRequest {
  query: number[];
  strategy?: 'iterative' | 'fast';
}

export interface CreateCollectionRequest {
  name: string;
}

export interface AnalyzeRequest {
  query: number[];
  strategy?: 'iterative' | 'fast';
}

// --- Response types ---

export interface HealthResponse {
  status: string;
}

export interface WriteResponse {
  count: number;
  ids?: number[];
}

export interface ReadResponse {
  result: number[];
}

export interface StatsResponse {
  num_locations: number;
  total_writes: number;
  current_eta: number;
  avg_write_count: number;
  max_write_count: number;
}

export interface CreateCollectionResponse {
  name: string;
  created: boolean;
}

export interface ListCollectionsResponse {
  collections: string[];
}

export interface DropCollectionResponse {
  dropped: boolean;
}

export interface EAMConfig {
  d: number;
  l_0: number;
  k: number;
  eta_0: number;
  lambda: number;
  eta_min: number;
  tau_split: number;
  tau_merge: number;
  gamma: number;
  tau_damp: number;
  tau_overload: number;
  beta: number;
  t_max: number;
  epsilon: number;
}

export interface ConfigResponse {
  config: EAMConfig;
}

export interface LocationSummaryItem {
  id: number;
  write_count: number;
  avg_counter_magnitude: number;
}

export interface LocationsResponse {
  locations: LocationSummaryItem[];
}

export interface ActivatedLocationItem {
  id: number;
  similarity: number;
  weight: number;
}

export interface AnalyzeResponse {
  iterations: number;
  converged: boolean;
  activated_locations: ActivatedLocationItem[];
  result: number[];
}

export interface QueryDocumentsRequest {
  query: number[];
  n?: number;
}

export interface DocumentItem {
  id: number;
  metadata: Record<string, unknown>;
}

export interface DocumentsResponse {
  documents: DocumentItem[];
}

export interface DocumentResponse {
  id: number;
  metadata: Record<string, unknown>;
}

export interface QueryDocumentResult {
  id: number;
  similarity: number;
  metadata: Record<string, unknown>;
}

export interface QueryDocumentsResponse {
  results: QueryDocumentResult[];
}

export interface ErrorResponse {
  error: string;
}
