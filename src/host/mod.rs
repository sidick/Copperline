// SPDX-License-Identifier: GPL-3.0-or-later

//! Desktop-only host integration points, target-gated so the `frontend`
//! feature compiles on Android too.
//!
//! rfd (file dialogs) and arboard (clipboard) have nothing to offer on
//! Android -- there is no native file-picker or clipboard API behind them
//! there -- so each gets a small trait here instead, backed by the real
//! crate on desktop and by a stub everywhere else. The stub is honest about
//! doing nothing: the real Android hooks (Storage Access Framework for
//! files) are their own work package, not part of this gating.

pub mod clipboard;
pub mod file_dialog;
