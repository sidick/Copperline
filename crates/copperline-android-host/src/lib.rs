// SPDX-License-Identifier: GPL-3.0-or-later

//! Reusable Android host-layer code, shared by the full emulator
//! (`copperline-android`) and, later, the publisher-kit player
//! (`copperline-player-android`).
//!
//! Stage 1 of the Android port (see the Android port plan): logcat
//! logging setup, proven to link and run under GameActivity. Activity
//! lifecycle, surface handling, storage, input and display policy land
//! here as their own work packages (WP4-WP8) once each is built and
//! verified on a device/AVD.

/// Route `log` output to logcat under the `copperline` tag. Call once, at
/// the start of `android_main`, before anything else logs.
pub fn init_logging() {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("copperline"),
    );
}
