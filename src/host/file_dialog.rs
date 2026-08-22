// SPDX-License-Identifier: GPL-3.0-or-later

//! Native file/folder picker, backed by rfd on desktop.
//!
//! The trait mirrors rfd's own builder methods and names exactly, so a call
//! site converts by swapping `rfd::FileDialog::new()` for
//! [`file_dialog()`] and nothing else changes underneath it.

use std::path::{Path, PathBuf};

pub trait FileDialogBackend {
    fn set_title(self: Box<Self>, title: &str) -> Box<dyn FileDialogBackend>;
    fn add_filter(self: Box<Self>, name: &str, extensions: &[&str]) -> Box<dyn FileDialogBackend>;
    fn set_directory(self: Box<Self>, dir: &Path) -> Box<dyn FileDialogBackend>;
    fn set_file_name(self: Box<Self>, name: &str) -> Box<dyn FileDialogBackend>;
    fn pick_file(self: Box<Self>) -> Option<PathBuf>;
    fn pick_files(self: Box<Self>) -> Option<Vec<PathBuf>>;
    fn pick_folder(self: Box<Self>) -> Option<PathBuf>;
    /// A combined file-or-folder picker; only macOS's native dialog actually
    /// offers this, per rfd. Callers already gate their use of it on
    /// `cfg(target_os = "macos")`.
    fn pick_file_or_folder(self: Box<Self>) -> Option<PathBuf>;
    fn save_file(self: Box<Self>) -> Option<PathBuf>;
}

/// Start building a native file dialog for the current platform.
pub fn file_dialog() -> Box<dyn FileDialogBackend> {
    platform::file_dialog()
}

#[cfg(not(target_os = "android"))]
mod platform {
    use super::{FileDialogBackend, Path, PathBuf};

    pub(super) fn file_dialog() -> Box<dyn FileDialogBackend> {
        Box::new(Desktop(rfd::FileDialog::new()))
    }

    struct Desktop(rfd::FileDialog);

    impl FileDialogBackend for Desktop {
        fn set_title(self: Box<Self>, title: &str) -> Box<dyn FileDialogBackend> {
            Box::new(Desktop(self.0.set_title(title)))
        }

        fn add_filter(
            self: Box<Self>,
            name: &str,
            extensions: &[&str],
        ) -> Box<dyn FileDialogBackend> {
            Box::new(Desktop(self.0.add_filter(name, extensions)))
        }

        fn set_directory(self: Box<Self>, dir: &Path) -> Box<dyn FileDialogBackend> {
            Box::new(Desktop(self.0.set_directory(dir)))
        }

        fn set_file_name(self: Box<Self>, name: &str) -> Box<dyn FileDialogBackend> {
            Box::new(Desktop(self.0.set_file_name(name)))
        }

        fn pick_file(self: Box<Self>) -> Option<PathBuf> {
            self.0.pick_file()
        }

        fn pick_files(self: Box<Self>) -> Option<Vec<PathBuf>> {
            self.0.pick_files()
        }

        fn pick_folder(self: Box<Self>) -> Option<PathBuf> {
            self.0.pick_folder()
        }

        fn pick_file_or_folder(self: Box<Self>) -> Option<PathBuf> {
            self.0.pick_file_or_folder()
        }

        fn save_file(self: Box<Self>) -> Option<PathBuf> {
            self.0.save_file()
        }
    }
}

/// No Storage Access Framework hook yet (lands with host-directory support
/// for Android); every terminal method reports "the user picked nothing",
/// same as a desktop dialog that was cancelled.
#[cfg(target_os = "android")]
mod platform {
    use super::{FileDialogBackend, Path, PathBuf};

    pub(super) fn file_dialog() -> Box<dyn FileDialogBackend> {
        Box::new(Android)
    }

    struct Android;

    impl FileDialogBackend for Android {
        fn set_title(self: Box<Self>, _title: &str) -> Box<dyn FileDialogBackend> {
            self
        }

        fn add_filter(
            self: Box<Self>,
            _name: &str,
            _extensions: &[&str],
        ) -> Box<dyn FileDialogBackend> {
            self
        }

        fn set_directory(self: Box<Self>, _dir: &Path) -> Box<dyn FileDialogBackend> {
            self
        }

        fn set_file_name(self: Box<Self>, _name: &str) -> Box<dyn FileDialogBackend> {
            self
        }

        fn pick_file(self: Box<Self>) -> Option<PathBuf> {
            None
        }

        fn pick_files(self: Box<Self>) -> Option<Vec<PathBuf>> {
            None
        }

        fn pick_folder(self: Box<Self>) -> Option<PathBuf> {
            None
        }

        fn pick_file_or_folder(self: Box<Self>) -> Option<PathBuf> {
            None
        }

        fn save_file(self: Box<Self>) -> Option<PathBuf> {
            None
        }
    }
}
