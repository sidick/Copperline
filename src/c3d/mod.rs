// SPDX-License-Identifier: GPL-3.0-or-later

//! C3D: a virtual fixed-function 3D accelerator board.
//!
//! `docs/internals/c3d.md` is the device specification and the contract
//! these modules implement; it is the source of truth, not this code.
//!
//! The modules here are the parts of the board that are pure logic: they
//! depend on nothing else in the emulator and on no GPU API, so they build
//! and test everywhere the core does (including wasm32) and are shared with
//! the native conformance-trace runner. The board itself (autoconfig,
//! register decode, `ZorroDevice` glue) and the wgpu renderer arrive with
//! the `c3d` cargo feature in a later milestone.
//!
//! - [`proto`] -- wire constants and decoded command types
//! - [`ring`] -- the command-ring reader, decoder and validator
//! - [`state`] -- the per-context OpenGL 1.x state machine
//! - [`dispatch`] -- connects `ring` to `state`: one context's registers,
//!   the doorbell, and the renderer-op stream it produces
//! - [`trace`] -- the conformance/capture trace container
//!
//! One module is not pure logic and is gated accordingly: [`render`]
//! (cargo feature `c3d`, off by default) is the wgpu backend that turns
//! [`dispatch`]'s renderer operations into pixels, and it is what the
//! `copperline-c3d-trace` conformance runner drives.

pub mod dispatch;
pub mod proto;
pub mod ring;
pub mod state;
pub mod trace;

/// The wgpu renderer. Gated because it is the one part of the board that
/// creates a GPU device; everything above it is pure logic and builds
/// everywhere the core does.
#[cfg(feature = "c3d")]
pub mod render;
