// SPDX-License-Identifier: GPL-3.0-or-later

//! The C3D board itself: autoconfig, register decode, and the doorbell
//! path that drives the pure-logic modules ([`super::ring`], [`super::state`],
//! [`super::dispatch`]) into [`super::render`]'s wgpu backend. The
//! protocol specification (linked from `docs/internals/c3d.md`) is the
//! contract; this module is Copperline's one implementation of it.
//!
//! ## Milestone scope (M2)
//!
//! This board fits the baseline rasteriser tier only: `CAPS0` reports
//! [`CAP_IRQ`](super::proto::CAP_IRQ) and
//! [`CAP_REF_SYNC`](super::proto::CAP_REF_SYNC) and nothing else, so
//! `CAP_TRANSFORM`/`CAP_MULTITEXTURE`/`CAP_GUESTMEM`/
//! `CAP_SURFACE_GUESTADDR` all read clear and the ring decoder rejects
//! their opcodes as `E_BAD_OPCODE` before they ever reach the renderer --
//! [`super::render`] does not yet implement the GL-space draw path those
//! bits would admit, so the register truthfully says so rather than
//! advertising a capability the board cannot deliver. Every surface is
//! aperture-backed; a `CAP_SURFACE_GUESTADDR` surface reaching into
//! another board's VRAM is scheduled for M3's `DeviceHost` hook (see
//! `C3D-SURVEY.md`'s Q5 finding). `MAX_CONTEXTS` is fixed at
//! [`CONTEXT_COUNT`] and never configurable from `[c3d]` yet.
//!
//! ## Window layout and the aperture buffer
//!
//! One Zorro III autoconfig window ([`crate::zorro::BoardSpec::c3d`]):
//! 64 KiB of global registers, [`CONTEXT_COUNT`] context pages of 256
//! bytes each starting at [`CONTEXT_PAGE_BASE`], and the data aperture
//! from [`proto::APERTURE_OFFSET_DEFAULT`] to the end of the configured
//! window. The aperture is a plain `Vec<u8>` this board owns directly --
//! not a second, RAM-backed chained autoconfig identity -- because the
//! spec's `APERTURE_OFFSET` register is windowrelative to the *same*
//! base the guest found with `FindConfigDev()`; a chained identity gets
//! its own, unrelated base, and nothing would keep the two contiguous
//! across an autoconfig placement the guest does not control. Owning the
//! buffer directly keeps the register's promise exactly.
//!
//! ## The doorbell and the determinism contract
//!
//! A `RING_TAIL` write is the doorbell. This board handles it entirely
//! inside the register write, synchronously: it copies the addressed
//! context's ring bytes out of the aperture into a local buffer (which is
//! also, incidentally, exactly what `CAP_REF_SYNC` promises -- referenced
//! data is captured before the write returns), decodes and applies every
//! command through [`dispatch::Context::submit`], executes the resulting
//! [`RenderOp`]s against [`Renderer`] with `pollster`'s blocking wait
//! already inside it, and advances `FENCE_COMPLETED` for every fence the
//! submission reached. The copy is what lets the aperture be mutably
//! borrowed again immediately afterward for a `SURFACE_READBACK`'s write,
//! without which the ring's own borrow (commands may reference bytes
//! inline) and the readback's write would alias the same `Vec`.
//!
//! Executing on the emulation thread inside the write, rather than on a
//! worker thread drained by `tick`, satisfies the specification's
//! "Determinism and timing" contract trivially: nothing is asynchronous,
//! so there is nothing for the guest to observe before it is done. It is
//! also the simplest correct thing to build first. A render worker
//! thread -- the shape `copperhf`'s worker already establishes, see
//! `docs/internals/copperhf.md`'s "The determinism model" -- is a
//! performance optimisation for later milestones once real workloads
//! argue for it, not a M2 requirement.

use super::dispatch::{Context, RenderOp};
use super::proto;
use super::render::{MemLoc, Memory, Renderer};
use super::ring::DeviceConfig;
use super::state;
use crate::zorro_device::{DeviceHost, ZorroDevice};

/// Context register pages start here, one [`CONTEXT_PAGE_SIZE`]-byte page
/// per context (`docs/internals/c3d.md`, "Context register pages").
pub const CONTEXT_PAGE_BASE: u32 = 0x0001_0000;
pub const CONTEXT_PAGE_SIZE: u32 = 0x0100;

/// Copperline's fixed context count (the spec allows any board-chosen
/// value up to 16; not yet a `[c3d]` config knob). Named distinctly from
/// the `MAX_CONTEXTS` *register* (`greg::MAX_CONTEXTS`, the offset) --
/// the two sharing a name once `use greg::*` is in scope let a match
/// arm's body silently read the wrong one; see the git history for the
/// bug clippy's `unnecessary_cast` lint caught.
pub const CONTEXT_COUNT: usize = 4;

/// Copperline's fixed `MAX_RING_SIZE` *register value*: large enough for
/// a real command stream, small enough that copying a whole ring out of
/// the aperture on every doorbell (see this module's doc comment) stays
/// cheap. Named distinctly from `greg::MAX_RING_SIZE` (the register
/// offset) for the same reason as [`CONTEXT_COUNT`]/`greg::MAX_CONTEXTS`.
pub const RING_SIZE_LIMIT: u32 = 0x0004_0000; // 256 KiB

/// Copperline's fitted `CAPS0` for this milestone -- see the module doc
/// comment's "Milestone scope" for why `CAP_TRANSFORM` and friends are
/// not here yet.
pub const CAPS0: u32 = proto::CAP_IRQ | proto::CAP_REF_SYNC;

// ---------------------------------------------------------------------
// Global register offsets (`docs/internals/c3d.md`, "Global registers")
// ---------------------------------------------------------------------

mod greg {
    pub const ID: u32 = 0x000;
    pub const VERSION: u32 = 0x004;
    pub const CAPS0: u32 = 0x008;
    pub const CAPS1: u32 = 0x00C;
    pub const STATUS: u32 = 0x010;
    pub const CONTROL: u32 = 0x014;
    pub const IRQ_STATUS: u32 = 0x018;
    pub const IRQ_ENABLE: u32 = 0x01C;
    pub const APERTURE_OFFSET: u32 = 0x020;
    pub const APERTURE_SIZE: u32 = 0x024;
    pub const MAX_CONTEXTS: u32 = 0x028;
    pub const MAX_RING_SIZE: u32 = 0x02C;
    pub const MAX_TEXTURE_SIZE: u32 = 0x040;
    pub const MAX_TEXTURE_UNITS: u32 = 0x044;
    pub const MAX_TEXTURES: u32 = 0x048;
    pub const MAX_SURFACES: u32 = 0x04C;
    pub const MAX_LIGHTS: u32 = 0x050;
    pub const MAX_CLIP_PLANES: u32 = 0x054;
    pub const MAX_MATRIX_DEPTH_MV: u32 = 0x058;
    pub const MAX_MATRIX_DEPTH_PROJ: u32 = 0x05C;
    pub const MAX_MATRIX_DEPTH_TEX: u32 = 0x060;
    pub const TEXFMT_SUPPORTED: u32 = 0x064;
    pub const SURFFMT_SUPPORTED_LO: u32 = 0x068;
    pub const SURFFMT_SUPPORTED_HI: u32 = 0x06C;
    pub const MAX_SURFACE_WIDTH: u32 = 0x070;
    pub const MAX_SURFACE_HEIGHT: u32 = 0x074;
}

// ---------------------------------------------------------------------
// Context register offsets, relative to a context's own page
// ---------------------------------------------------------------------

mod creg {
    pub const CONTROL: u32 = 0x00;
    pub const STATUS: u32 = 0x04;
    pub const RING_BASE: u32 = 0x08;
    pub const RING_SIZE: u32 = 0x0C;
    pub const RING_TAIL: u32 = 0x10;
    pub const RING_HEAD: u32 = 0x14;
    pub const FENCE_COMPLETED: u32 = 0x18;
    pub const ERROR_CODE: u32 = 0x1C;
    pub const ERROR_OFFSET: u32 = 0x20;
    pub const ERROR_ACK: u32 = 0x24;
    pub const GL_ERROR: u32 = 0x28;
    pub const GL_ERROR_ACK: u32 = 0x2C;
    pub const FENCE_IRQ_TARGET: u32 = 0x30;
}

const CTX_CONTROL_ALLOC: u32 = 1 << 0;
const CTX_CONTROL_ENABLE: u32 = 1 << 1;
const CTX_CONTROL_RESET: u32 = 1 << 2;

const CTX_STATUS_HALTED: u32 = 1 << 1;
const CTX_STATUS_IDLE_RING: u32 = 1 << 2;

const CONTROL_ENABLE: u32 = 1 << 0;
const CONTROL_RESET: u32 = 1 << 1;

const STATUS_READY: u32 = 1 << 0;

/// One allocated context's register-visible state layered over
/// [`dispatch::Context`]: `CTX_CONTROL`'s `ALLOC`/`ENABLE` bits and
/// `FENCE_IRQ_TARGET`, which belong to the board's register file rather
/// than to the protocol-level [`Context`] itself.
#[derive(serde::Serialize, serde::Deserialize)]
struct ContextSlot {
    ctx: Context,
    alloc: bool,
    enable: bool,
    fence_irq_target: u32,
}

impl Default for ContextSlot {
    fn default() -> Self {
        ContextSlot {
            ctx: Context::new(state::Limits::default()),
            alloc: false,
            enable: false,
            fence_irq_target: 0,
        }
    }
}

/// The C3D board. See the module doc comment for the doorbell path and
/// the milestone's capability scope.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct C3dBoard {
    window_bytes: u32,
    aperture_offset: u32,
    aperture: Vec<u8>,
    control: u32,
    irq_status: u32,
    irq_enable: u32,
    contexts: Vec<ContextSlot>,
    activity: bool,

    /// Lazily created on the first doorbell that needs it -- board
    /// construction must not require a GPU adapter, since `[c3d] enabled
    /// = true` alone (before any context is even allocated) must not
    /// fail a headless run that never touches the board. `None` after
    /// construction, after a full reset, and after a failed creation
    /// attempt (`renderer_failed` distinguishes "not tried yet" from
    /// "tried and there is truly no adapter", so a doorbell does not
    /// retry adapter enumeration on every single command stream once it
    /// is known to be hopeless).
    #[serde(skip)]
    renderer: Option<Renderer>,
    #[serde(skip)]
    renderer_failed: bool,
}

impl C3dBoard {
    /// `window_bytes` is the whole autoconfig window
    /// ([`crate::zorro::BoardSpec::c3d`] takes the same value); the data
    /// aperture is everything from `APERTURE_OFFSET_DEFAULT` to the end
    /// of it.
    pub fn new(window_bytes: u32) -> Self {
        let aperture_offset = proto::APERTURE_OFFSET_DEFAULT;
        let aperture_len = window_bytes.saturating_sub(aperture_offset) as usize;
        let mut contexts = Vec::with_capacity(CONTEXT_COUNT);
        contexts.resize_with(CONTEXT_COUNT, ContextSlot::default);
        C3dBoard {
            window_bytes,
            aperture_offset,
            aperture: vec![0u8; aperture_len],
            control: CONTROL_ENABLE,
            irq_status: 0,
            irq_enable: 0,
            contexts,
            activity: false,
            renderer: None,
            renderer_failed: false,
        }
    }

    fn aperture_size(&self) -> u32 {
        self.window_bytes - self.aperture_offset
    }

    /// The decoder configuration this milestone's `CAPS0` promises --
    /// kept as one function so the register value and the decoder's own
    /// idea of what is legal can never drift apart.
    fn device_config(&self) -> DeviceConfig {
        DeviceConfig {
            guestmem: CAPS0 & proto::CAP_GUESTMEM != 0,
            transform: CAPS0 & proto::CAP_TRANSFORM != 0,
            multitexture: CAPS0 & proto::CAP_MULTITEXTURE != 0,
            surface_guestaddr: CAPS0 & proto::CAP_SURFACE_GUESTADDR != 0,
            aperture_size: self.aperture_size(),
            max_texture_units: 1,
            ..DeviceConfig::default()
        }
    }

    /// A free function over just the two fields it needs (rather than a
    /// `&mut self` method) so the borrow checker sees it as disjoint from
    /// `self.contexts`/`self.aperture`/`self.aperture_offset`, all of
    /// which `doorbell` also needs live across the same call.
    fn ensure_renderer<'a>(
        renderer: &'a mut Option<Renderer>,
        renderer_failed: &mut bool,
    ) -> Option<&'a mut Renderer> {
        if renderer.is_none() && !*renderer_failed {
            match Renderer::new() {
                Ok(r) => {
                    log::info!("c3d: renderer ready ({})", r.describe());
                    *renderer = Some(r);
                }
                Err(e) => {
                    log::warn!("c3d: no renderer available ({e}); the board answers registers but every doorbell's GPU work is dropped");
                    *renderer_failed = true;
                }
            }
        }
        renderer.as_mut()
    }

    /// `STATUS.READY` is unconditional in this milestone (no reset is
    /// ever in progress by the time a guest can observe it, and there is
    /// no `FATAL` condition this board can reach yet).
    fn status(&self) -> u32 {
        STATUS_READY
    }

    fn read_global(&self, off: u32) -> u32 {
        use greg::*;
        match off {
            ID => proto::ID_MAGIC,
            VERSION => proto::PROTOCOL_VERSION,
            CAPS0 => self::CAPS0,
            CAPS1 => 0,
            STATUS => self.status(),
            CONTROL => self.control,
            IRQ_STATUS => self.irq_status,
            IRQ_ENABLE => self.irq_enable,
            APERTURE_OFFSET => self.aperture_offset,
            APERTURE_SIZE => self.aperture_size(),
            MAX_CONTEXTS => CONTEXT_COUNT as u32,
            MAX_RING_SIZE => RING_SIZE_LIMIT,
            MAX_TEXTURE_SIZE => 1024,
            MAX_TEXTURE_UNITS => 1,
            MAX_TEXTURES => 256,
            MAX_SURFACES => 16,
            // Transform-tier limits read zero while CAP_TRANSFORM is
            // clear, per the spec's own note on MAX_LIGHTS/MAX_CLIP_PLANES.
            MAX_LIGHTS
            | MAX_CLIP_PLANES
            | MAX_MATRIX_DEPTH_MV
            | MAX_MATRIX_DEPTH_PROJ
            | MAX_MATRIX_DEPTH_TEX => 0,
            TEXFMT_SUPPORTED => self.device_config().texfmt_supported,
            SURFFMT_SUPPORTED_LO => (self.device_config().surffmt_supported & 0xFFFF_FFFF) as u32,
            SURFFMT_SUPPORTED_HI => (self.device_config().surffmt_supported >> 32) as u32,
            MAX_SURFACE_WIDTH => self.device_config().max_surface_width,
            MAX_SURFACE_HEIGHT => self.device_config().max_surface_height,
            _ => 0,
        }
    }

    fn write_global(&mut self, off: u32, value: u32) {
        use greg::*;
        match off {
            CONTROL => {
                if value & CONTROL_RESET != 0 {
                    self.do_full_reset();
                } else {
                    self.control = value & CONTROL_ENABLE;
                }
            }
            IRQ_ENABLE => self.irq_enable = value,
            // Write-1-to-clear.
            IRQ_STATUS => self.irq_status &= !value,
            _ => {} // every other global offset is RO; writes discarded
        }
    }

    fn do_full_reset(&mut self) {
        for slot in &mut self.contexts {
            *slot = ContextSlot::default();
        }
        self.irq_status = 0;
        self.irq_enable = 0;
        self.control = CONTROL_ENABLE;
        self.renderer = None;
        self.renderer_failed = false;
    }

    fn read_context(&self, n: usize, off: u32) -> u32 {
        let Some(slot) = self.contexts.get(n) else {
            return 0;
        };
        use creg::*;
        match off {
            CONTROL => {
                (if slot.alloc { CTX_CONTROL_ALLOC } else { 0 })
                    | (if slot.enable { CTX_CONTROL_ENABLE } else { 0 })
            }
            STATUS => {
                let mut v = 0;
                if slot.ctx.halted {
                    v |= CTX_STATUS_HALTED;
                }
                if slot.ctx.ring_head == slot.ctx.ring_tail {
                    v |= CTX_STATUS_IDLE_RING;
                }
                v
            }
            RING_BASE => slot.ctx.ring_base,
            RING_SIZE => slot.ctx.ring_size,
            RING_TAIL => slot.ctx.ring_tail,
            RING_HEAD => slot.ctx.ring_head,
            FENCE_COMPLETED => slot.ctx.fence_completed,
            ERROR_CODE => slot.ctx.error_code,
            ERROR_OFFSET => slot.ctx.error_offset,
            GL_ERROR => slot.ctx.state.gl_error(),
            FENCE_IRQ_TARGET => slot.fence_irq_target,
            _ => 0, // ERROR_ACK/GL_ERROR_ACK are WO; reserved offsets are 0
        }
    }

    fn write_context(&mut self, n: usize, off: u32, value: u32) {
        if n >= self.contexts.len() {
            return;
        }
        use creg::*;
        match off {
            CONTROL => {
                let slot = &mut self.contexts[n];
                let was_alloc = slot.alloc;
                slot.alloc = value & CTX_CONTROL_ALLOC != 0;
                slot.enable = value & CTX_CONTROL_ENABLE != 0;
                if was_alloc && !slot.alloc {
                    // Freed: every object this context held is gone.
                    self.contexts[n] = ContextSlot {
                        alloc: false,
                        ..ContextSlot::default()
                    };
                } else if value & CTX_CONTROL_RESET != 0 {
                    self.contexts[n].ctx.reset_all();
                    self.contexts[n].ctx.error_code = 0;
                    self.contexts[n].ctx.error_offset = 0;
                    self.contexts[n].ctx.halted = false;
                }
            }
            RING_BASE => self.contexts[n].ctx.ring_base = value,
            RING_SIZE => {
                // "Written only while ENABLE is clear; writing it also
                // resets RING_HEAD and RING_TAIL to 0."
                if !self.contexts[n].enable {
                    self.contexts[n].ctx.ring_size = value;
                    self.contexts[n].ctx.ring_head = 0;
                    self.contexts[n].ctx.ring_tail = 0;
                }
            }
            RING_TAIL => self.doorbell(n, value),
            ERROR_ACK => {
                self.contexts[n].ctx.error_ack();
            }
            GL_ERROR_ACK => {
                self.contexts[n].ctx.state.ack_gl_error();
            }
            FENCE_IRQ_TARGET => self.contexts[n].fence_irq_target = value,
            _ => {} // RO offsets (STATUS, RING_HEAD, FENCE_COMPLETED, ERROR_CODE/OFFSET, GL_ERROR) discard writes
        }
        self.update_irq_status(n);
    }

    /// The doorbell: `RING_TAIL` has just been written `new_tail`. See
    /// the module doc comment for why this runs synchronously and why
    /// the ring is copied out first.
    fn doorbell(&mut self, n: usize, new_tail: u32) {
        if n >= self.contexts.len() || !self.contexts[n].alloc || !self.contexts[n].enable {
            return;
        }
        self.activity = true;

        let ring_base = self.contexts[n].ctx.ring_base;
        let ring_len = self.contexts[n].ctx.ring_size & 0x7FFF_FFFF;
        let ring_bytes = self.read_aperture_range(ring_base, ring_len);

        let config = self.device_config();
        let mut ops: Vec<RenderOp<'_>> = Vec::new();
        self.contexts[n]
            .ctx
            .submit(&ring_bytes, new_tail, &config, &mut ops);

        if !ops.is_empty() {
            if let Some(renderer) =
                Self::ensure_renderer(&mut self.renderer, &mut self.renderer_failed)
            {
                let state = &self.contexts[n].ctx.state;
                let mut mem = ApertureMemory {
                    aperture: &mut self.aperture,
                };
                let errors = renderer.execute(&ops, state, &mut mem);
                for e in &errors {
                    log::warn!("c3d: context {n}: render error: {e}");
                }
            }
            for op in &ops {
                if let RenderOp::Fence { id } = op {
                    self.contexts[n].ctx.complete_fence(*id);
                }
            }
        }
        self.update_irq_status(n);
    }

    /// Reads `len` bytes at aperture offset `addr` (already resolved to
    /// be relative to the aperture's own start, not the window's), zero
    /// filled past the aperture's end -- a `RING_SIZE` reaching past a
    /// misconfigured `RING_BASE` degrades to zero commands rather than
    /// panicking.
    fn read_aperture_range(&self, addr: u32, len: u32) -> Vec<u8> {
        let start = addr as usize;
        let end = start.saturating_add(len as usize).min(self.aperture.len());
        if start >= self.aperture.len() || start >= end {
            return Vec::new();
        }
        self.aperture[start..end].to_vec()
    }

    /// `IRQ_STATUS`'s per-context fence and error bits, and the level
    /// this board's `int2_line` reports -- recomputed after anything that
    /// could change either (a doorbell, an ack, an explicit `IRQ_STATUS`
    /// clear).
    fn update_irq_status(&mut self, n: usize) {
        if n >= self.contexts.len() {
            return;
        }
        let slot = &self.contexts[n];
        let fence_pending = slot.fence_irq_target != 0
            && slot.ctx.fence_completed.wrapping_sub(slot.fence_irq_target) as i32 >= 0;
        let error_pending = slot.ctx.error_code != 0;
        let bit = 1u32 << n;
        if fence_pending {
            self.irq_status |= bit;
        }
        if error_pending {
            self.irq_status |= bit << 16;
        }
    }
}

/// Adapts the board's aperture buffer to [`Memory`]. A [`MemLoc::Aperture`]
/// address is already **aperture-relative** -- `0` is the aperture's own
/// first byte, not the window's -- exactly as the spec's "an aperture
/// offset" phrasing means it for `RING_BASE` and for a space-`0`
/// reference, so it is used directly as an index into `aperture` with no
/// translation. (There is nothing to translate *from*: this struct does
/// not even carry the window's `APERTURE_OFFSET`, on purpose, so that
/// reintroducing a subtraction here is a type error, not a silent
/// regression back to the bug this comment used to describe.)
/// [`MemLoc::Guest`] is always `None`/`false`: this milestone's `CAPS0`
/// has `CAP_GUESTMEM` clear, so the ring decoder never produces a guest
/// reference in the first place, and a surface can only be aperture
/// backed (`CAP_SURFACE_GUESTADDR` is likewise clear).
struct ApertureMemory<'a> {
    aperture: &'a mut Vec<u8>,
}

impl ApertureMemory<'_> {
    fn range(&self, aperture_addr: u32, len: usize) -> Option<std::ops::Range<usize>> {
        let start = aperture_addr as usize;
        let end = start.checked_add(len)?;
        (end <= self.aperture.len()).then_some(start..end)
    }
}

impl Memory for ApertureMemory<'_> {
    fn read(&self, loc: MemLoc, len: usize) -> Option<&[u8]> {
        match loc {
            MemLoc::Aperture(addr) => {
                let r = self.range(addr, len)?;
                Some(&self.aperture[r])
            }
            MemLoc::Guest(_) => None,
        }
    }

    fn write(&mut self, loc: MemLoc, data: &[u8]) -> bool {
        match loc {
            MemLoc::Aperture(addr) => match self.range(addr, data.len()) {
                Some(r) => {
                    self.aperture[r].copy_from_slice(data);
                    true
                }
                None => false,
            },
            MemLoc::Guest(_) => false,
        }
    }
}

impl ZorroDevice for C3dBoard {
    fn read(&mut self, off: u32, size: usize, _host: &mut DeviceHost) -> u32 {
        let value = if off < CONTEXT_PAGE_BASE {
            self.read_global(off & !0x3)
        } else if off < self.aperture_offset {
            let rel = off - CONTEXT_PAGE_BASE;
            let n = (rel / CONTEXT_PAGE_SIZE) as usize;
            let page_off = rel % CONTEXT_PAGE_SIZE;
            self.read_context(n, page_off & !0x3)
        } else {
            // Plain aperture read: ordinary board memory, window-relative
            // `off` translated to an offset from the aperture's own start.
            let mut buf = [0u8; 4];
            let reg_off = off & !0x3;
            if let Some(rel) = reg_off.checked_sub(self.aperture_offset) {
                let rel = rel as usize;
                let n = self.aperture.len().saturating_sub(rel).min(4);
                if n > 0 {
                    buf[..n].copy_from_slice(&self.aperture[rel..rel + n]);
                }
            }
            u32::from_be_bytes(buf)
        };
        extract_be(value, off & 0x3, size)
    }

    fn write(&mut self, off: u32, size: usize, value: u32, _host: &mut DeviceHost) {
        let reg_off = off & !0x3;
        let sub = off & 0x3;
        if off < CONTEXT_PAGE_BASE {
            let full = if size == 4 {
                value
            } else {
                patch_be(self.read_global(reg_off), sub, size, value)
            };
            self.write_global(reg_off, full);
        } else if off < self.aperture_offset {
            let rel = off - CONTEXT_PAGE_BASE;
            let n = (rel / CONTEXT_PAGE_SIZE) as usize;
            let page_off = rel % CONTEXT_PAGE_SIZE;
            let reg_off = page_off & !0x3;
            let full = if size == 4 {
                value
            } else {
                patch_be(self.read_context(n, reg_off), sub, size, value)
            };
            self.write_context(n, reg_off, full);
        } else {
            // Plain aperture write -- ordinary board memory, no side
            // effects, matching the spec's "the guest reads and writes it
            // like RAM". `off` is window-relative; translate to an offset
            // from the aperture's own start before indexing.
            if let Some(start) = (off as usize).checked_sub(self.aperture_offset as usize) {
                if start + size <= self.aperture.len() {
                    let bytes = value.to_be_bytes();
                    self.aperture[start..start + size].copy_from_slice(&bytes[4 - size..]);
                }
            }
        }
    }

    fn tick(&mut self, _cck: u32, _host: &mut DeviceHost) {
        // The doorbell executes synchronously (see the module doc
        // comment), so there is nothing outstanding for a tick to drain
        // in this milestone.
    }

    fn int2_line(&self) -> bool {
        self.irq_status & self.irq_enable != 0
    }

    fn take_activity(&mut self) -> bool {
        std::mem::take(&mut self.activity)
    }

    fn reset(&mut self) {
        self.do_full_reset();
    }

    fn kind(&self) -> &'static str {
        "c3d"
    }
}

/// Extracts `size` bytes (1, 2 or 4) starting at big-endian byte offset
/// `sub` (0..=3) of `value`.
fn extract_be(value: u32, sub: u32, size: usize) -> u32 {
    let shift = (4 - sub as usize - size) * 8;
    let mask: u32 = if size >= 4 {
        u32::MAX
    } else {
        (1u32 << (size * 8)) - 1
    };
    (value >> shift) & mask
}

/// Read-modify-write: replaces `size` bytes of `full` at big-endian byte
/// offset `sub` with the low `size` bytes of `patch`, keeping the rest of
/// `full` unchanged.
fn patch_be(full: u32, sub: u32, size: usize, patch: u32) -> u32 {
    let shift = (4 - sub as usize - size) * 8;
    let mask: u32 = if size >= 4 {
        u32::MAX
    } else {
        (1u32 << (size * 8)) - 1
    };
    (full & !(mask << shift)) | ((patch & mask) << shift)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c3d::proto::ErrorCode;

    fn board() -> C3dBoard {
        C3dBoard::new(0x0200_0000) // 32 MiB window
    }

    fn host<'a>(mem: &'a mut crate::memory::Memory) -> DeviceHost<'a> {
        DeviceHost::new(mem)
    }

    fn dummy_mem() -> crate::memory::Memory {
        crate::memory::Memory {
            chip_ram: vec![0u8; 0x1000],
            slow_ram: Vec::new(),
            mb_ram: Vec::new(),
            accel_ram: Vec::new(),
            rom: Vec::new(),
            overlay: false,
            zorro: crate::zorro::ZorroChain::default(),
            extended_rom: Vec::new(),
            extended_rom_base: 0,
            wcs: Vec::new(),
            wcs_write_protected: false,
        }
    }

    #[test]
    fn id_and_version_registers_read_the_spec_magic() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        assert_eq!(b.read(greg::ID, 4, &mut h), proto::ID_MAGIC);
        assert_eq!(b.read(greg::VERSION, 4, &mut h), proto::PROTOCOL_VERSION);
        assert_eq!(b.read(greg::CAPS0, 4, &mut h), CAPS0);
    }

    #[test]
    fn transform_tier_limits_read_zero_while_the_capability_is_clear() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        assert_eq!(b.read(greg::MAX_LIGHTS, 4, &mut h), 0);
        assert_eq!(b.read(greg::MAX_CLIP_PLANES, 4, &mut h), 0);
    }

    #[test]
    fn byte_and_word_reads_extract_the_correct_slice_of_a_longword_register() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        // ID = 0x4333_4420 = "C3D ".
        assert_eq!(b.read(greg::ID, 1, &mut h), 0x43);
        assert_eq!(b.read(greg::ID + 1, 1, &mut h), 0x33);
        assert_eq!(b.read(greg::ID, 2, &mut h), 0x4333);
        assert_eq!(b.read(greg::ID + 2, 2, &mut h), 0x4420);
    }

    #[test]
    fn a_plain_aperture_write_and_read_round_trips_like_ram() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        let addr = b.aperture_offset + 0x100;
        b.write(addr, 4, 0xDEAD_BEEF, &mut h);
        assert_eq!(b.read(addr, 4, &mut h), 0xDEAD_BEEF);
        assert_eq!(b.read(addr, 1, &mut h), 0xDE);
    }

    #[test]
    fn allocating_and_freeing_a_context_flips_the_alloc_bit() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        let ctl = CONTEXT_PAGE_BASE + creg::CONTROL;
        assert_eq!(b.read(ctl, 4, &mut h) & CTX_CONTROL_ALLOC, 0);
        b.write(ctl, 4, CTX_CONTROL_ALLOC | CTX_CONTROL_ENABLE, &mut h);
        assert_eq!(
            b.read(ctl, 4, &mut h),
            CTX_CONTROL_ALLOC | CTX_CONTROL_ENABLE
        );
        b.write(ctl, 4, 0, &mut h);
        assert_eq!(b.read(ctl, 4, &mut h), 0);
    }

    #[test]
    fn a_doorbell_on_a_disabled_context_is_ignored() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        // Not allocated: writing RING_TAIL must not panic or decode.
        b.write(CONTEXT_PAGE_BASE + creg::RING_TAIL, 4, 64, &mut h);
        assert_eq!(b.read(CONTEXT_PAGE_BASE + creg::RING_HEAD, 4, &mut h), 0);
    }

    /// End-to-end: allocate a context, point its ring at the aperture,
    /// write a `SURFACE_DEFINE` + `SET_DRAW_SURFACE` + `CLEAR_COLOR` +
    /// `CLEAR` + `SURFACE_READBACK` + `FENCE` command stream into the
    /// aperture by hand, ring the doorbell, and check the readback
    /// landed in the aperture with the right pixels -- without a GPU,
    /// `renderer_failed` just means the render step is skipped, so this
    /// exercises every step except the actual pixels.
    #[test]
    fn a_full_command_stream_is_decoded_and_a_readback_produces_the_cleared_pixels() {
        // Every address in the command stream below is a ref/RING_BASE
        // "aperture offset" per the spec: 0-based from the *aperture's*
        // own start, not the window's -- deliberately small numbers, to
        // catch exactly the window-vs-aperture confusion this test used
        // to have (see git history: ApertureMemory briefly subtracted
        // APERTURE_OFFSET from an address that was already
        // aperture-relative, silently misplacing every readback).
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);

        const RING_BASE_APERTURE_REL: u32 = 0;
        const RING_SIZE: u32 = 0x1000;
        const SURFACE_APERTURE_REL: u32 = RING_SIZE; // right after the ring
        const SURFACE_W: u32 = 4;
        const SURFACE_H: u32 = 4;
        const SURFACE_STRIDE: u32 = SURFACE_W * 4; // A8R8G8B8: 4 bytes/pixel

        let mut cmds = Vec::<u32>::new();
        let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;
        // SURFACE_DEFINE id, width, height, stride_bytes, format(5=A8R8G8B8),
        // flags(0=aperture-backed), address.
        cmds.push(opcode_len(proto::OP_SURFACE_DEFINE, 8));
        cmds.extend([
            1,
            SURFACE_W,
            SURFACE_H,
            SURFACE_STRIDE,
            5,
            0,
            SURFACE_APERTURE_REL,
        ]);
        cmds.push(opcode_len(proto::OP_SET_DRAW_SURFACE, 2));
        cmds.push(1);
        // CLEAR_COLOR r=1 g=0 b=0 a=1 (opaque red).
        cmds.push(opcode_len(proto::OP_CLEAR_COLOR, 5));
        cmds.extend([1.0f32.to_bits(), 0, 0, 1.0f32.to_bits()]);
        cmds.push(opcode_len(proto::OP_CLEAR, 2));
        cmds.push(proto::CLEAR_MASK_COLOR);
        // SURFACE_READBACK the whole 4x4 rect.
        cmds.push(opcode_len(proto::OP_SURFACE_READBACK, 5));
        cmds.extend([0, 0, SURFACE_W, SURFACE_H]);
        cmds.push(opcode_len(proto::OP_FENCE, 2));
        cmds.push(1);

        let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        let ring_window_off = b.aperture_offset + RING_BASE_APERTURE_REL;
        for (i, b_) in bytes.iter().enumerate() {
            b.write(ring_window_off + i as u32, 1, *b_ as u32, &mut h);
        }

        b.write(
            CONTEXT_PAGE_BASE + creg::RING_BASE,
            4,
            RING_BASE_APERTURE_REL,
            &mut h,
        );
        b.write(CONTEXT_PAGE_BASE + creg::RING_SIZE, 4, RING_SIZE, &mut h);
        b.write(
            CONTEXT_PAGE_BASE + creg::CONTROL,
            4,
            CTX_CONTROL_ALLOC | CTX_CONTROL_ENABLE,
            &mut h,
        );
        b.write(
            CONTEXT_PAGE_BASE + creg::RING_TAIL,
            4,
            bytes.len() as u32,
            &mut h,
        );

        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::ERROR_CODE, 4, &mut h),
            0,
            "no protocol error decoding a well-formed stream"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::RING_HEAD, 4, &mut h),
            bytes.len() as u32,
            "every command consumed"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::FENCE_COMPLETED, 4, &mut h),
            1,
            "the fence completed (synchronously, whether or not a GPU rendered anything)"
        );

        if b.renderer.is_none() {
            eprintln!(
                "skipping pixel check: no wgpu adapter available (renderer_failed={})",
                b.renderer_failed
            );
            return;
        }

        // A8R8G8B8 is "A R G B" bytes per the spec's surface-format table:
        // opaque red is 0xFF 0xFF 0x00 0x00, every one of the 16 pixels.
        let start = (b.aperture_offset + SURFACE_APERTURE_REL) as usize;
        for row in 0..SURFACE_H {
            for col in 0..SURFACE_W {
                let px = start + (row * SURFACE_STRIDE) as usize + (col * 4) as usize
                    - b.aperture_offset as usize;
                assert_eq!(
                    &b.aperture[px..px + 4],
                    &[0xFFu8, 0xFF, 0x00, 0x00],
                    "pixel ({col}, {row}) is not opaque red after CLEAR + SURFACE_READBACK"
                );
            }
        }
    }

    #[test]
    fn an_unknown_opcode_latches_a_protocol_error_and_advances_error_offset() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);

        let ring_off = b.aperture_offset;
        // Opcode 0xFFFF is not in the map: E_BAD_OPCODE, skip.
        let word: u32 = (0xFFFFu32 << 16) | 1;
        for (i, byte) in word.to_be_bytes().iter().enumerate() {
            b.write(ring_off + i as u32, 1, *byte as u32, &mut h);
        }
        b.write(
            CONTEXT_PAGE_BASE + creg::RING_BASE,
            4,
            ring_off - b.aperture_offset,
            &mut h,
        );
        b.write(CONTEXT_PAGE_BASE + creg::RING_SIZE, 4, 0x1000, &mut h);
        b.write(
            CONTEXT_PAGE_BASE + creg::CONTROL,
            4,
            CTX_CONTROL_ALLOC | CTX_CONTROL_ENABLE,
            &mut h,
        );
        b.write(CONTEXT_PAGE_BASE + creg::RING_TAIL, 4, 4, &mut h);

        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::ERROR_CODE, 4, &mut h),
            ErrorCode::BadOpcode as u32
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::STATUS, 4, &mut h) & CTX_STATUS_HALTED,
            0
        );

        // ERROR_ACK clears it.
        b.write(CONTEXT_PAGE_BASE + creg::ERROR_ACK, 4, 1, &mut h);
        assert_eq!(b.read(CONTEXT_PAGE_BASE + creg::ERROR_CODE, 4, &mut h), 0);
    }
}
