use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// --- Requests ---

#[derive(Debug, Deserialize)]
pub struct WriteRequest {
    pub vectors: Vec<Vec<f64>>,
    pub metadata: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
pub struct ReadRequest {
    pub query: Vec<f64>,
    /// "iterative" (default) or "fast"
    pub strategy: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct AnalyzeRequest {
    pub query: Vec<f64>,
    pub strategy: Option<String>,
}

// --- Responses ---

#[derive(Debug, Serialize)]
pub struct WriteResponse {
    pub count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ids: Option<Vec<u64>>,
}

#[derive(Debug, Serialize)]
pub struct ReadResponse {
    pub result: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub num_locations: usize,
    pub total_writes: f64,
    pub current_eta: f64,
    pub avg_write_count: f64,
    pub max_write_count: f64,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Serialize)]
pub struct CreateCollectionResponse {
    pub name: String,
    pub created: bool,
}

#[derive(Debug, Serialize)]
pub struct ListCollectionsResponse {
    pub collections: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DropCollectionResponse {
    pub dropped: bool,
}

#[derive(Debug, Serialize)]
pub struct ConfigResponse {
    pub config: heather_db::EAMConfig,
}

#[derive(Debug, Serialize)]
pub struct LocationSummaryItem {
    pub id: usize,
    pub write_count: f64,
    pub avg_counter_magnitude: f64,
}

#[derive(Debug, Serialize)]
pub struct LocationsResponse {
    pub locations: Vec<LocationSummaryItem>,
}

#[derive(Debug, Serialize)]
pub struct ActivatedLocationItem {
    pub id: usize,
    pub similarity: f64,
    pub weight: f64,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeResponse {
    pub iterations: usize,
    pub converged: bool,
    pub activated_locations: Vec<ActivatedLocationItem>,
    pub result: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct BatchAnalyzeRequest {
    pub queries: Vec<Vec<f64>>,
    pub strategy: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BatchAnalyzeResponse {
    pub results: Vec<AnalyzeResponse>,
}

#[derive(Debug, Serialize)]
pub struct FingerprintResponse {
    pub fingerprint: Option<Vec<f64>>,
}

#[derive(Debug, Serialize)]
pub struct DocumentItem {
    pub id: u64,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct DocumentsResponse {
    pub documents: Vec<DocumentItem>,
}

#[derive(Debug, Serialize)]
pub struct DocumentResponse {
    pub id: u64,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct QueryDocumentsRequest {
    pub query: Vec<f64>,
    #[serde(default = "default_n")]
    pub n: usize,
}

fn default_n() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct QueryDocumentResult {
    pub id: u64,
    pub similarity: f64,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct QueryDocumentsResponse {
    pub results: Vec<QueryDocumentResult>,
}

// --- Algebra ---

#[derive(Debug, Deserialize)]
pub struct AlgebraAddRequest {
    pub source_a: String,
    pub source_b: String,
    pub target: String,
    #[serde(default)]
    pub max_cross_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct AlgebraSubRequest {
    pub source_a: String,
    pub source_b: String,
    pub target: String,
    #[serde(default)]
    pub max_cross_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct AlgebraScaleRequest {
    pub source: String,
    pub target: String,
    pub alpha: f64,
}

#[derive(Debug, Deserialize)]
pub struct AlgebraIntersectRequest {
    pub source_a: String,
    pub source_b: String,
    pub target: String,
    #[serde(default = "default_threshold")]
    pub threshold: f64,
}

fn default_threshold() -> f64 {
    0.95
}

#[derive(Debug, Serialize)]
pub struct AlgebraResponse {
    pub collection: String,
    pub num_locations: usize,
}

// --- Compose ---

#[derive(Debug, Deserialize)]
pub struct ComposeReadRequest {
    pub collections: Vec<String>,
    pub query: Vec<f64>,
    #[serde(default = "default_routing_sharpness")]
    pub routing_sharpness: f64,
}

fn default_routing_sharpness() -> f64 {
    20.0
}

#[derive(Debug, Serialize)]
pub struct ComposeReadResponse {
    pub result: Vec<f64>,
    pub weights: HashMap<String, f64>,
    pub confidences: HashMap<String, f64>,
}
