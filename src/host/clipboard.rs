// SPDX-License-Identifier: GPL-3.0-or-later

//! Host clipboard paste, backed by arboard on desktop.
//!
//! Read-only (paste into a text field): nothing in Copperline copies to the
//! clipboard, so that's the only operation this needs.

pub trait ClipboardBackend {
    /// The clipboard's text contents, or the reason it couldn't be read.
    fn paste(&self) -> Result<String, String>;
}

pub fn clipboard() -> Box<dyn ClipboardBackend> {
    platform::clipboard()
}

#[cfg(not(target_os = "android"))]
mod platform {
    use super::ClipboardBackend;

    pub(super) fn clipboard() -> Box<dyn ClipboardBackend> {
        Box::new(Desktop)
    }

    struct Desktop;

    impl ClipboardBackend for Desktop {
        fn paste(&self) -> Result<String, String> {
            arboard::Clipboard::new()
                .and_then(|mut c| c.get_text())
                .map_err(|e| e.to_string())
        }
    }
}

/// No clipboard hook yet; every paste reports itself unavailable, the same
/// message a desktop host with no clipboard service would give.
#[cfg(target_os = "android")]
mod platform {
    use super::ClipboardBackend;

    pub(super) fn clipboard() -> Box<dyn ClipboardBackend> {
        Box::new(Android)
    }

    struct Android;

    impl ClipboardBackend for Android {
        fn paste(&self) -> Result<String, String> {
            Err("clipboard is not implemented on Android yet".to_string())
        }
    }
}
