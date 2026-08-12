use std::collections::HashMap;

use heather_db::RoleCleanup;
use serde::{Deserialize, Serialize};

// --- Requests ---

/// Two raw vectors for a single-vector algebra op (`/vec/bind`, `/vec/unbind`).
#[derive(Debug, Deserialize)]
pub struct VecPairRequest {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    /// `/vec/bind` only: unit-normalise the result (default true).
    /// Set false for spiral-plane ops where spectrum magnitude is the payload.
    pub normalize: Option<bool>,
    /// `/vec/unbind` only: spectral division (true inverse for
    /// non-unit-spectrum keys) instead of correlation. Default false.
    pub exact: Option<bool>,
    /// Null-bin threshold for exact unbind. Default 1e-9.
    pub eps: Option<f64>,
}

/// One raw vector and a real exponent for `/vec/pow` (spectral power).
#[derive(Debug, Deserialize)]
pub struct VecPowRequest {
    pub a: Vec<f64>,
    pub t: f64,
}

/// Knobs for description-length-minimising compression (both optional).
#[derive(Debug, Default, Deserialize)]
pub struct CompressRequest {
    /// Model cost per location (bits). Default: the dimension.
    pub kappa: Option<f64>,
    /// Weight on the data-fit (variance) cost of a merge. Default 1.0.
    pub lambda: Option<f64>,
}

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

/// Raw dot-product attention read at a fixed inverse-temperature `scale`.
/// `exclude_id` drops one stored location from the activated set (leave-one-out).
#[derive(Debug, Deserialize)]
pub struct AttentionRequest {
    pub query: Vec<f64>,
    pub scale: f64,
    pub exclude_id: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct AttentionResponse {
    pub result: Vec<f64>,
    pub beta: f64,
}

/// Calibrate the attention temperature by leave-one-out value-reconstruction
/// description length. `betas` overrides the default log-spaced sweep.
#[derive(Debug, Deserialize)]
pub struct CalibrateRequest {
    pub betas: Option<Vec<f64>>,
}

#[derive(Debug, Serialize)]
pub struct CalibrateResponse {
    pub beta: f64,
    pub dl: f64,
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

// ─── Database (multi-tenancy) ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateDatabaseRequest {
    pub name: String,
    pub dimension: usize,
    /// Optional LMDB map size in MB (default: 4096).
    #[serde(default)]
    pub map_size_mb: Option<usize>,
    /// Opt into the MDL allocation gate: spawn/split by description length
    /// (surprise·recurrence > engram bits) instead of fixed tau_split/tau_overload.
    #[serde(default)]
    pub mdl_gate: bool,
    /// Optional EAM-config overrides. Any unset field keeps the engine default.
    #[serde(default)]
    pub eam: Option<EamConfigOverride>,
}

/// Per-database EAM knobs settable at creation. Unset fields keep the engine
/// default; supplied fields are validated before the database is created. Lets
/// callers (e.g. ingested-model DBs) request full-softmax `k`, single-step
/// `t_max`, or a disabled neighbor graph without hand-editing `db.toml`.
#[derive(Debug, Deserialize, Default)]
pub struct EamConfigOverride {
    pub k: Option<usize>,
    pub t_max: Option<usize>,
    pub beta: Option<f64>,
    pub neighbor_cap: Option<usize>,
    pub num_landmarks: Option<usize>,
    pub l_0: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct DatabaseInfo {
    pub name: String,
    pub created_at: u64,
    pub dimension: usize,
    pub map_size_mb: usize,
    pub collections: usize,
}

#[derive(Debug, Serialize)]
pub struct ListDatabasesResponse {
    pub databases: Vec<DatabaseInfo>,
}

#[derive(Debug, Serialize)]
pub struct DropDatabaseResponse {
    pub dropped: bool,
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

#[derive(Debug, Deserialize, Default)]
pub struct LocationsQuery {
    /// When true, the response includes each location's full
    /// `address` and `counter` vectors instead of just the summary.
    #[serde(default)]
    pub full: bool,
}

#[derive(Debug, Serialize)]
pub struct LocationFullItem {
    pub id: usize,
    pub write_count: f64,
    pub address: Vec<f64>,
    pub counter: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct LocationsFullResponse {
    pub locations: Vec<LocationFullItem>,
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

#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteDocumentResponse {
    pub deleted: bool,
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
    /// When present, `query` is read as the filler expected at this role: each
    /// candidate is unbound by `unbind_role` and the recovered filler compared
    /// to `query`, instead of the stored superposition being compared whole.
    /// Absent keeps full-bundle scoring, so existing clients are unaffected.
    /// See `Collection::query_documents_scoped` for why this exists.
    #[serde(default)]
    pub unbind_role: Option<Vec<f64>>,
    /// Multi-role scoring. When non-empty, `query` is used **only** for recall
    /// and ranking (plain full-bundle cosine) and each pair contributes one
    /// similarity per returned document, in this order. Mutually exclusive
    /// with `unbind_role`.
    ///
    /// Ranking ignores the pairs entirely, so a document that would rank well
    /// under the caller's weighting can fail to be recalled at all — ask for
    /// an `n` far larger than you intend to display. See
    /// `Collection::query_documents_multi_role`.
    #[serde(default)]
    pub role_pairs: Vec<RoleFillerPair>,
    /// Cleanup of the recovered filler; omitted means MDL-calibrated cleanup.
    #[serde(default)]
    pub cleanup: CleanupSpec,
}

/// One scoring criterion: the role to unbind by and the filler expected there.
///
/// Each criterion carries its own filler. `unbind_role` reuses the recall
/// `query` as the filler, which conflates two different vectors; this does
/// not.
#[derive(Debug, Deserialize)]
pub struct RoleFillerPair {
    pub role: Vec<f64>,
    pub filler: Vec<f64>,
}

/// Wire form of `heather_db::RoleCleanup`.
///
/// `"mdl"` (the default), `"off"`, or `{"beta": 100.0}`. Betas below the
/// engine's floor are clamped; the temperature actually used comes back as
/// `cleanup_beta`. See `heather_db::RoleCleanup` for the measured sharpness
/// curve — a soft beta yields a plausible-looking but wrong ordering.
#[derive(Debug, Deserialize, Default, Clone, Copy, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CleanupSpec {
    #[default]
    Mdl,
    Off,
    Beta(f64),
}

impl From<CleanupSpec> for RoleCleanup {
    fn from(spec: CleanupSpec) -> Self {
        match spec {
            CleanupSpec::Mdl => RoleCleanup::Mdl,
            CleanupSpec::Off => RoleCleanup::Off,
            CleanupSpec::Beta(b) => RoleCleanup::Beta(b),
        }
    }
}

fn default_n() -> usize {
    10
}

#[derive(Debug, Serialize)]
pub struct QueryDocumentResult {
    pub id: u64,
    pub similarity: f64,
    pub metadata: serde_json::Value,
    /// One similarity per requested `role_pairs` entry, in request order.
    /// Omitted when no pairs were requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role_scores: Option<Vec<f64>>,
}

#[derive(Debug, Serialize)]
pub struct QueryDocumentsResponse {
    pub results: Vec<QueryDocumentResult>,
    /// Cleanup temperature the engine actually ran at. Omitted when cleanup
    /// was off or no `role_pairs` were requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cleanup_beta: Option<f64>,
}

// --- Bulk load (deterministic location placement) ---

/// Replace a collection's hard locations atomically with a caller-
/// supplied set. Bypasses competitive learning — useful when the
/// caller wants the collection to hold exactly the vectors specified,
/// with no novelty splits or address migration. Required for snapshot
/// algebra (bind / unbind / add / sub) to operate on known operands.
#[derive(Debug, Deserialize)]
pub struct BulkLoadRequest {
    pub addresses: Vec<Vec<f64>>,
    pub counters: Vec<Vec<f64>>,
    #[serde(default)]
    pub write_counts: Option<Vec<f64>>,
}

#[derive(Debug, Serialize)]
pub struct BulkLoadResponse {
    pub n_loaded: usize,
    pub dim: usize,
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

/// Apply a permutation ρ^k (the non-commutative S_D bind) to every
/// location of `source`, writing the result to `target`.
///
/// The permutation is identified by exactly one of `seed` (a raw u64) or
/// `name` (hashed to a seed via SHA-256, the project's symbol convention);
/// supplying neither or both is an error. `power` (default 1) is the
/// integer exponent k — negative applies ρ⁻¹.
#[derive(Debug, Deserialize)]
pub struct AlgebraPermuteRequest {
    pub source: String,
    pub target: String,
    /// Seed for the permutation. Mutually exclusive with `name`.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Name hashed (SHA-256) into a seed. Mutually exclusive with `seed`.
    #[serde(default)]
    pub name: Option<String>,
    /// Exponent k: ρ^k. Negative applies the inverse. Defaults to 1.
    #[serde(default = "default_permute_power")]
    pub power: i64,
}

fn default_permute_power() -> i64 {
    1
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

#[derive(Debug, Deserialize)]
pub struct AlgebraBindRequest {
    pub source_a: String,
    pub source_b: String,
    pub target: String,
    #[serde(default)]
    pub max_cross_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct AlgebraUnbindRequest {
    /// The bundle to be unbound (multi-location collection).
    pub source: String,
    /// The binding key as a raw vector. Provided directly in the
    /// request because the production engine's adaptive memory means
    /// collections rarely have exactly one location — users hold their
    /// binding keys as plain vectors, not as separate collections.
    pub key_vector: Vec<f64>,
    pub target: String,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_db_request_without_eam_defaults_to_none() {
        let req: CreateDatabaseRequest =
            serde_json::from_str(r#"{"name":"d","dimension":128}"#).unwrap();
        assert!(req.eam.is_none());
        assert!(req.map_size_mb.is_none());
    }

    #[test]
    fn create_db_request_parses_partial_eam_overrides() {
        let req: CreateDatabaseRequest = serde_json::from_str(
            r#"{"name":"enc","dimension":1024,"eam":{"k":3840,"t_max":1,"neighbor_cap":0}}"#,
        )
        .unwrap();
        let eam = req.eam.expect("eam present");
        assert_eq!(eam.k, Some(3840));
        assert_eq!(eam.t_max, Some(1));
        assert_eq!(eam.neighbor_cap, Some(0));
        // unsupplied knobs stay None so the engine default is kept
        assert_eq!(eam.beta, None);
        assert_eq!(eam.num_landmarks, None);
    }

    /// The multi-role wire contract. Omitting `cleanup` must mean MDL cleanup,
    /// not "no cleanup" — a client that forgets the field gets the correct
    /// behaviour, and only an explicit `"off"` opts out.
    #[test]
    fn query_documents_request_defaults_to_mdl_cleanup_and_no_pairs() {
        let req: QueryDocumentsRequest = serde_json::from_str(r#"{"query":[1.0,0.0]}"#).unwrap();
        assert_eq!(req.cleanup, CleanupSpec::Mdl);
        assert!(req.role_pairs.is_empty());
        assert!(req.unbind_role.is_none());
        assert!(matches!(RoleCleanup::from(req.cleanup), RoleCleanup::Mdl));
    }

    #[test]
    fn cleanup_spec_parses_all_three_modes() {
        let parse = |s: &str| -> CleanupSpec { serde_json::from_str(s).unwrap() };
        assert_eq!(parse(r#""mdl""#), CleanupSpec::Mdl);
        assert_eq!(parse(r#""off""#), CleanupSpec::Off);
        assert_eq!(parse(r#"{"beta":100.0}"#), CleanupSpec::Beta(100.0));
        assert!(matches!(
            RoleCleanup::from(parse(r#"{"beta":100.0}"#)),
            RoleCleanup::Beta(b) if b == 100.0
        ));
    }

    /// Pair order is the caller's, and `role_scores` comes back in it — so the
    /// request must preserve it verbatim rather than, say, keying by role.
    #[test]
    fn role_pairs_deserialize_in_request_order() {
        let req: QueryDocumentsRequest = serde_json::from_str(
            r#"{"query":[1.0,0.0],"n":50,"role_pairs":[
                 {"role":[1.0,0.0],"filler":[0.0,1.0]},
                 {"role":[0.0,1.0],"filler":[1.0,0.0]}],
               "cleanup":"off"}"#,
        )
        .unwrap();
        assert_eq!(req.n, 50);
        assert_eq!(req.cleanup, CleanupSpec::Off);
        assert_eq!(req.role_pairs.len(), 2);
        assert_eq!(req.role_pairs[0].role, vec![1.0, 0.0]);
        assert_eq!(req.role_pairs[1].role, vec![0.0, 1.0]);
    }

    /// `role_scores` and `cleanup_beta` are additive: a response with neither
    /// must be byte-identical to what pre-multi-role clients already parse.
    #[test]
    fn single_role_responses_are_unchanged_on_the_wire() {
        let body = serde_json::to_string(&QueryDocumentsResponse {
            results: vec![QueryDocumentResult {
                id: 7,
                similarity: 0.5,
                metadata: serde_json::Value::Null,
                role_scores: None,
            }],
            cleanup_beta: None,
        })
        .unwrap();
        assert_eq!(
            body,
            r#"{"results":[{"id":7,"similarity":0.5,"metadata":null}]}"#
        );
    }
}
