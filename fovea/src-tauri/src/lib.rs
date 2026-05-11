/// Fovea — desktop shell for the HeatherDB inspector.
///
/// Almost everything happens in the WebView (React + the heather.ts client).
/// The Rust side is just a Tauri runtime that:
///   - mounts the HTTP plugin so the WebView can fetch http:// URLs without
///     CORS pain (HeatherDB ships plain HTTP)
///   - mounts the Store plugin for persisting connection metadata across
///     processes (we currently use localStorage but switching is one prop)
///
/// Add Tauri commands here when we want server-side helpers (cross-process
/// snapshot diffing, encoder bridges, etc.) — none yet for v0.1.

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_store::Builder::new().build())
        .setup(|_app| Ok(()))
        .run(tauri::generate_context!())
        .expect("error while running fovea");
}
