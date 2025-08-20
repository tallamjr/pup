//! Detection-only processing (no video display)
//!
//! Based on demo's detection mode with pup's error handling

use crate::subcommands::OutputFormat;
use gstpup::config::AppConfig;
use std::path::PathBuf;
use tracing::info;

pub fn run(
    input: String,
    output: PathBuf,
    model: PathBuf,
    confidence: f32,
    format: OutputFormat,
    _global_config: Option<AppConfig>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!(
        "Detection mode - processing {} to {}",
        input,
        output.display()
    );
    info!(
        "Model: {}, Confidence: {}, Format: {:?}",
        model.display(),
        confidence,
        format
    );

    // TODO: Implement detection-only processing using demo's approach
    // For now, placeholder that shows the structure works

    Ok(())
}
