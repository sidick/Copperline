// SPDX-License-Identifier: GPL-3.0-or-later

//! Android entry point for the full Copperline emulator.
//!
//! Stage 1 of the Android port (see the Android port plan): proves the
//! native library loads under GameActivity and that logcat carries its
//! output -- verified end-to-end (build via `cargo ndk`, package, install,
//! launch, confirm via logcat) using a throwaway NativeActivity stand-in,
//! since packaging a real GameActivity APK needs the `androidx.games:
//! games-activity` Java/Kotlin glue and a Gradle project, neither of which
//! exist in this repo yet. Nothing here drives `copperline`'s `App`/window
//! machinery yet -- that's WP4 (lifecycle/surface), once a Gradle-based
//! packaging step exists to actually test it running.

use android_activity::{AndroidApp, MainEvent, PollEvent};

#[unsafe(no_mangle)]
fn android_main(app: AndroidApp) {
    copperline_android_host::init_logging();
    log::info!(
        "copperline-android: native library loaded, copperline {}",
        env!("CARGO_PKG_VERSION")
    );

    let mut quit = false;
    while !quit {
        app.poll_events(None, |event| {
            if let PollEvent::Main(MainEvent::Destroy) = event {
                quit = true;
            }
        });
    }
}
