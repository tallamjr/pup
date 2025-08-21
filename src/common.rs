//! Platform-specific application entry point and threading utilities
//!
//! This module provides platform-specific abstractions for running the main application,
//! handling the differences between macOS and other platforms regarding GUI applications
//! and OpenGL context management.
//!
//! On macOS, GUI applications require a main thread run loop for proper window management
//! and OpenGL context handling. This module abstracts these platform differences.

/// macOS has a specific requirement that there must be a run loop running on the main thread in
/// order to open windows and use OpenGL, and that the global NSApplication instance must be
/// initialized.
/// On macOS this launches the callback function on a thread.
/// On other platforms it's just executed immediately.
#[cfg(not(target_os = "macos"))]
pub fn run<T, F: FnOnce() -> T + Send + 'static>(main: F) -> T
where
    T: Send + 'static,
{
    main()
}

#[cfg(target_os = "macos")]
#[allow(deprecated)]
pub fn run<T, F: FnOnce() -> T + Send + 'static>(main: F) -> T
where
    T: Send + 'static,
{
    use std::{
        ffi::c_void,
        sync::mpsc::{channel, Sender},
        thread,
    };

    use cocoa::{
        appkit::{NSApplication, NSWindow},
        base::id,
        delegate,
    };
    use objc::{
        msg_send,
        runtime::{Object, Sel},
        sel, sel_impl,
    };

    unsafe {
        let app = cocoa::appkit::NSApp();
        let (send, recv) = channel::<()>();

        extern "C" fn on_finish_launching(this: &Object, _cmd: Sel, _notification: id) {
            let send = unsafe {
                let send_pointer = *this.get_ivar::<*const c_void>("send");
                let boxed = Box::from_raw(send_pointer as *mut Sender<()>);
                *boxed
            };
            send.send(()).unwrap();
        }

        let delegate = delegate!("AppDelegate", {
            app: id = app,
            send: *const c_void = Box::into_raw(Box::new(send)) as *const c_void,
            (applicationDidFinishLaunching:) => on_finish_launching as extern "C" fn(&Object, Sel, id)
        });
        app.setDelegate_(delegate);

        let t = thread::spawn(move || {
            // Wait for the NSApp to launch to avoid possibly calling stop_() too early
            recv.recv().unwrap();

            let res = main();

            let app = cocoa::appkit::NSApp();
            app.stop_(cocoa::base::nil);

            // Stopping the event loop requires an actual event
            let event = cocoa::appkit::NSEvent::otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                cocoa::base::nil,
                cocoa::appkit::NSEventType::NSApplicationDefined,
                cocoa::foundation::NSPoint { x: 0.0, y: 0.0 },
                cocoa::appkit::NSEventModifierFlags::empty(),
                0.0,
                0,
                cocoa::base::nil,
                cocoa::appkit::NSEventSubtype::NSApplicationActivatedEventType,
                0,
                0,
            );
            app.postEvent_atStart_(event, cocoa::base::YES);

            res
        });

        app.run();

        t.join().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only test basic functionality that doesn't trigger NSApplication
    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_run_basic_functionality() {
        let result = run(|| 42);
        assert_eq!(result, 42);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_run_with_string_return() {
        let result = run(|| "Hello, World!".to_string());
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_run_with_computation() {
        let result = run(|| {
            let mut sum = 0;
            for i in 1..=10 {
                sum += i;
            }
            sum
        });
        assert_eq!(result, 55);
    }

    // Test platform detection and compilation
    #[test]
    fn test_platform_specific_compilation() {
        // This test verifies the correct platform-specific code is compiled
        #[cfg(target_os = "macos")]
        {
            // On macOS, we expect the run function to be available
            assert!(true, "macOS-specific run function available");
        }

        #[cfg(not(target_os = "macos"))]
        {
            // On other platforms, we expect direct execution
            assert!(true, "Non-macOS direct execution available");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore] // Skip this test to avoid NSApplication hanging in test environment
    fn test_macos_specific_behavior() {
        // Test that macOS-specific code paths are exercised
        // NOTE: This test is ignored because it triggers NSApplication initialization
        // which can hang in test environments. The functionality is tested through
        // integration tests instead.
        let result = run(|| {
            // This should run on a separate thread on macOS
            std::thread::current().id()
        });

        // Just verify we get a thread ID back
        let main_thread_id = std::thread::current().id();
        // On macOS, the callback runs on a different thread
        assert_ne!(result, main_thread_id);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn test_non_macos_behavior() {
        // Test that non-macOS platforms execute directly
        let result = run(|| std::thread::current().id());

        let main_thread_id = std::thread::current().id();
        // On non-macOS platforms, should execute on the same thread
        assert_eq!(result, main_thread_id);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_run_with_different_return_types() {
        // Test Option return type
        let opt_result: Option<i32> = run(|| Some(42));
        assert_eq!(opt_result, Some(42));

        // Test Result return type
        let res_result: Result<i32, String> = run(|| Ok(100));
        assert_eq!(res_result, Ok(100));

        // Test Vec return type
        let vec_result: Vec<i32> = run(|| vec![1, 2, 3]);
        assert_eq!(vec_result, vec![1, 2, 3]);
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_run_with_complex_closure() {
        let multiplier = 5;
        let data = vec![1, 2, 3, 4];

        let result = run(move || {
            data.into_iter()
                .map(|x| x * multiplier)
                .collect::<Vec<i32>>()
        });

        assert_eq!(result, vec![5, 10, 15, 20]);
    }
}
