//! Simple video playback without inference
//!
//! Based on demo's playback mode

use gstpup::config::AppConfig;
use tracing::info;

pub fn run(
    input: String,
    _global_config: Option<AppConfig>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Play mode - input: {}", input);

    // TODO: Implement simple playback using demo's approach
    // For now, placeholder that shows the structure works

    Ok(())
}
