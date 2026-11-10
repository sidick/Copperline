// SPDX-License-Identifier: GPL-3.0-or-later

//! `c3d.trace_*`: the control protocol's live C3D command-stream capture
//! methods. Split out of [`super::exec`] so the `[c3d]` surface costs that
//! shared file only its four `CoreOp` dispatch arms. Each method has a
//! `#[cfg(not(feature = "c3d"))]` twin reporting the board's absence, so
//! the dispatch arms need no feature gate of their own.
use super::proto::CtlError;
use crate::emulator::Emulator;
use serde_json::{json, Value};
use std::path::PathBuf;

/// Shared JSON shape for `c3d.trace_start`/`stop`/`status`/`snapshot`'s
/// reply -- `rolling`/`rolling_byte_budget` surface
/// `TraceCaptureStatus`'s own distinction between a bounded in-memory
/// capture and the original unbounded-to-disk one (`src/c3d/board.rs`'s
/// `CaptureMode` doc comment).
#[cfg(feature = "c3d")]
fn c3d_trace_status_json(status: &crate::c3d::board::TraceCaptureStatus) -> Value {
    json!({
        "active": true,
        "path": status.path.display().to_string(),
        "rings": status.rings,
        "blobs": status.blobs,
        "bytes": status.bytes,
        "rolling": status.rolling,
        "rolling_byte_budget": status.rolling_byte_budget,
    })
}

/// `c3d.trace_start`: arms a live command-stream capture on the C3D
/// board, if one is configured and this build has the `c3d` feature.
/// Mirrors `copperhf_attach`'s "board not configured" stance
/// (`CtlError::unsupported`, not a hard error) for both ways this can be
/// unavailable -- no `[c3d] enabled` board fitted, or a build without
/// `--features c3d` at all, the latter needing its own cfg-gated
/// function body since `Bus::c3d_board_mut` itself does not exist
/// otherwise (`src/video/window/app_debugger.rs`'s `#[cfg(feature =
/// "mhi")]` call-site guard is the same shape for the same reason).
/// `rolling_bytes` set arms a bounded capture instead of the original
/// unbounded one (`C3dBoard::trace_start`'s own doc comment);
/// `trigger_texture` set arms-but-defers instead of starting immediately
/// (`C3dBoard::trace_arm_on_texture_bind`) -- the two are mutually
/// exclusive at the protocol layer the same way `path` is "either an
/// explicit value or the fallback chain", not a combination.
#[cfg(feature = "c3d")]
pub(super) fn c3d_trace_start(
    emu: &mut Emulator,
    path: Option<PathBuf>,
    rolling_bytes: Option<u64>,
    trigger_texture: Option<u32>,
) -> Result<Value, CtlError> {
    let Some(board) = emu.bus_mut().c3d_board_mut() else {
        return Err(CtlError::unsupported("no [c3d] board configured"));
    };
    match trigger_texture {
        Some(id) => {
            board.trace_arm_on_texture_bind(id, path, rolling_bytes);
            Ok(json!({"active": false, "trigger_pending": id}))
        }
        None => {
            board
                .trace_start(path, rolling_bytes)
                .map_err(|e| CtlError::io(format!("starting c3d trace capture: {e}")))?;
            Ok(c3d_trace_status_value(emu))
        }
    }
}
#[cfg(not(feature = "c3d"))]
pub(super) fn c3d_trace_start(
    _emu: &mut Emulator,
    _path: Option<PathBuf>,
    _rolling_bytes: Option<u64>,
    _trigger_texture: Option<u32>,
) -> Result<Value, CtlError> {
    Err(CtlError::unsupported(
        "the c3d board needs a build with --features c3d",
    ))
}

/// `c3d.trace_snapshot`: replay a rolling capture's currently-buffered
/// records to disk without stopping the roll (`C3dBoard::trace_snapshot`'s
/// own doc comment covers the `Direct`-mode case too).
#[cfg(feature = "c3d")]
pub(super) fn c3d_trace_snapshot(
    emu: &mut Emulator,
    path: Option<PathBuf>,
) -> Result<Value, CtlError> {
    let Some(board) = emu.bus_mut().c3d_board_mut() else {
        return Err(CtlError::unsupported("no [c3d] board configured"));
    };
    match board
        .trace_snapshot(path)
        .map_err(|e| CtlError::io(format!("snapshotting c3d trace capture: {e}")))?
    {
        Some(status) => Ok(c3d_trace_status_json(&status)),
        None => Ok(json!({"active": false})),
    }
}
#[cfg(not(feature = "c3d"))]
pub(super) fn c3d_trace_snapshot(
    _emu: &mut Emulator,
    _path: Option<PathBuf>,
) -> Result<Value, CtlError> {
    Err(CtlError::unsupported(
        "the c3d board needs a build with --features c3d",
    ))
}

#[cfg(feature = "c3d")]
pub(super) fn c3d_trace_stop(emu: &mut Emulator) -> Value {
    match emu.bus_mut().c3d_board_mut().and_then(|b| b.trace_stop()) {
        Some(status) => {
            let mut v = c3d_trace_status_json(&status);
            v["active"] = json!(false);
            v
        }
        None => json!({"active": false}),
    }
}
#[cfg(not(feature = "c3d"))]
pub(super) fn c3d_trace_stop(_emu: &mut Emulator) -> Value {
    json!({"active": false})
}

#[cfg(feature = "c3d")]
pub(super) fn c3d_trace_status_value(emu: &mut Emulator) -> Value {
    let Some(board) = emu.bus_mut().c3d_board_mut() else {
        return json!({"active": false});
    };
    match board.trace_status() {
        Some(status) => c3d_trace_status_json(&status),
        None => match board.trace_trigger_pending() {
            Some(id) => json!({"active": false, "trigger_pending": id}),
            None => json!({"active": false}),
        },
    }
}
#[cfg(not(feature = "c3d"))]
pub(super) fn c3d_trace_status_value(_emu: &mut Emulator) -> Value {
    json!({"active": false})
}
