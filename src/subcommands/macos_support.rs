//! macOS-specific support for video display
//!
//! Provides platform-specific functionality for macOS video windows, including
//! NSApplication initialization and CFRunLoop management required for proper
//! video display on macOS systems.

#[cfg(target_os = "macos")]
pub mod macos {
    use anyhow::Result;
    use cocoa::appkit::{NSApplication, NSApplicationActivationPolicyRegular};
    use cocoa::base::nil;
    use core_foundation::runloop::{CFRunLoop, CFRunLoopRun};
    use std::thread;
    use tracing::{debug, info};

    /// Runs the provided function within a macOS CFRunLoop context
    ///
    /// This is required for proper video window display on macOS. The function
    /// initializes NSApplication, runs the provided function in a separate thread,
    /// and manages the CFRunLoop on the main thread.
    pub fn run_with_runloop<F: FnOnce() -> Result<()> + Send + 'static>(
        main_func: F,
    ) -> Result<()> {
        info!("Initializing macOS NSApplication and CFRunLoop for video display");

        // Initialize NSApplication on macOS to fix video window display
        unsafe {
            let app = NSApplication::sharedApplication(nil);
            app.setActivationPolicy_(NSApplicationActivationPolicyRegular);
            debug!("NSApplication initialized with regular activation policy");
        }

        // Get the main run loop
        let main_run_loop = CFRunLoop::get_main();

        // Run the main function in a separate thread
        let main_run_loop_clone = main_run_loop.clone();
        let handle = thread::spawn(move || {
            debug!("Starting main function in background thread");
            let result = main_func();

            // Stop the run loop when done
            debug!("Main function completed, stopping CFRunLoop");
            main_run_loop_clone.stop();
            result
        });

        // Run the CFRunLoop on the main thread
        debug!("Starting CFRunLoop on main thread");
        unsafe {
            CFRunLoopRun();
        }

        debug!("CFRunLoop stopped, joining background thread");
        handle.join().unwrap()
    }
}

#[cfg(not(target_os = "macos"))]
pub mod macos {
    use anyhow::Result;

    /// No-op implementation for non-macOS platforms
    pub fn run_with_runloop<F: FnOnce() -> Result<()>>(main_func: F) -> Result<()> {
        main_func()
    }
}

// Re-export for easier use
pub use macos::run_with_runloop;
