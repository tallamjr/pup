//! Subcommand implementations combining the best features from pup and demo
//!
//! Each module implements a specific subcommand using the best implementation
//! from the existing codebase.

use clap::ValueEnum;

pub mod detect;
pub mod live;
pub mod play;
pub mod process;
pub mod record;

#[derive(ValueEnum, Debug, Clone)]
pub enum OutputFormat {
    /// Plain text format
    Text,
    /// JSON format
    Json,
    /// CSV format
    Csv,
}

#[derive(ValueEnum, Debug, Clone)]
pub enum VideoFormat {
    /// MP4 container
    Mp4,
    /// AVI container
    Avi,
    /// MOV container
    Mov,
}
