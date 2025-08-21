//! Mock infrastructure for testing GStreamer plugins
//!
//! This module provides comprehensive mocking functionality for GStreamer
//! elements, pipelines, and related components to enable thorough unit testing
//! of GStreamer plugins without requiring the full GStreamer runtime.

pub mod gstreamer_mocks;

pub use gstreamer_mocks::*;
