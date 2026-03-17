use axum::extract::Path;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Redirect, Response};
use rust_embed::Embed;

#[derive(Embed)]
#[folder = "../heather_cortex/dist"]
struct UiAssets;

pub async fn redirect_to_ui() -> Redirect {
    Redirect::permanent("/ui/")
}

pub async fn serve_ui(path: Option<Path<String>>) -> Response {
    let path = path.map(|p| p.0).unwrap_or_default();
    let path = if path.is_empty() { "index.html" } else { &path };

    // Try to serve the exact file
    if let Some(file) = UiAssets::get(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, mime.as_ref().to_string())],
            file.data.to_vec(),
        )
            .into_response()
    } else {
        // SPA fallback: serve index.html for client-side routing
        match UiAssets::get("index.html") {
            Some(file) => (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/html".to_string())],
                file.data.to_vec(),
            )
                .into_response(),
            None => (StatusCode::NOT_FOUND, "UI not available").into_response(),
        }
    }
}
