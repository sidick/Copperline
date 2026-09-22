// SPDX-License-Identifier: GPL-3.0-or-later

//! The C3D board itself: autoconfig, register decode, and the doorbell
//! path that drives the pure-logic modules ([`super::ring`], [`super::state`],
//! [`super::dispatch`]) into [`super::render`]'s wgpu backend. The
//! protocol specification (linked from `docs/internals/c3d.md`) is the
//! contract; this module is Copperline's one implementation of it.
//!
//! ## Milestone scope
//!
//! `CAPS0` reports [`CAP_IRQ`](super::proto::CAP_IRQ),
//! [`CAP_REF_SYNC`](super::proto::CAP_REF_SYNC),
//! [`CAP_SURFACE_GUESTADDR`](super::proto::CAP_SURFACE_GUESTADDR)
//! (surfaces backed by another board's VRAM via the cross-board DMA
//! hook), [`CAP_GUESTMEM`](super::proto::CAP_GUESTMEM) (space-1 refs for
//! rings, bulk data and query results, described below), and -- since
//! the transform-tier MVP landed in [`super::render`] --
//! [`CAP_TRANSFORM`](super::proto::CAP_TRANSFORM) and
//! [`CAP_MULTITEXTURE`](super::proto::CAP_MULTITEXTURE): GL-space draws
//! in all three shapes (`DRAW_INLINE`/`DRAW_ARRAYS`/`DRAW_ELEMENTS`),
//! viewport/depth-range, two-unit multitexture and texgen are all real.
//! The transform-tier *state* the renderer does not yet act on
//! (lighting, `CLIP_PLANE` -- unused by the first client) is still
//! decoded and tracked per the spec; see `docs/internals/c3d.md`'s
//! implementation notes for the exact remainder. `MAX_CONTEXTS` is
//! fixed at [`CONTEXT_COUNT`] and never configurable from `[c3d]` yet.
//!
//! ## `CAP_GUESTMEM`
//!
//! A space-1 [`super::render::MemLoc::Guest`] address names ordinary
//! guest memory -- chip/slow/motherboard/accelerator RAM or a RAM-backed
//! Zorro board -- reached through [`DeviceHost::dma_read`]/
//! [`DeviceHost::dma_write`]'s existing 24/32-bit decode, the same one
//! the A2091/CDTV bus masters use. **Reads** happen synchronously inside
//! the doorbell: `dma_read` takes `&self`, so no borrow of `self`
//! conflicts with it, and this is also exactly what `CAP_REF_SYNC`
//! promises ("captured before the `RING_TAIL` write... returns") -- a
//! trivial truth here since the whole doorbell is synchronous, but
//! worth stating plainly since `CAP_REF_SYNC` is "meaningful only with
//! `CAP_GUESTMEM`" per the spec, and now it is meaningful. **Writes**
//! stay on the existing one-tick-deferred `pending_dma` path (see
//! `ApertureMemory::write` below and `take_pending_dma_writes`), not a
//! new synchronous `dma_write` call, even though the host is available
//! during the doorbell: `dma_write`/`dma_write_byte` only resolve
//! ordinary RAM and *RAM-backed* Zorro windows
//! (`ZorroChain::region_at`), never a *device-backed* board's window
//! (`device_region_at`/`BoardBacking::Device`) -- so a synchronous write
//! could never reach another board's aperture the way
//! `CAP_SURFACE_GUESTADDR`'s readback path needs to. Routing every
//! `Guest` write through `pending_dma` uniformly, regardless of
//! destination kind, means the bus's `drain_cross_board_dma` pass -- the
//! only code that already resolves both RAM and device-window
//! destinations correctly -- decides where the bytes actually land, and
//! a `QUERY` result written to ordinary guest RAM gets the exact same
//! one-tick deferral its own fence-completion accounting already
//! expects for a guest-address readback. A guest program simply cannot
//! observe an incomplete write anyway: it learns a write landed only via
//! `FENCE_COMPLETED` (or `RING_HEAD`), and both are deferred right along
//! with it (see `ContextSlot::deferred_fence`).
//!
//! **Ring in guest memory.** The spec's `RING_BASE` register: "an
//! aperture offset, or a guest address if bit 31 of `RING_SIZE` is set
//! (`CAP_GUESTMEM`)" -- so unlike a ref, the ring itself may live in
//! guest memory too, and the doorbell honours that: when the bit is set
//! (and `CAP_GUESTMEM` is not masked off), the ring bytes are fetched
//! with `DeviceHost::dma_read` instead of `read_aperture_range`, into
//! exactly the same `Vec<u8>` shape `Context::submit` already expects --
//! nothing downstream of the fetch needs to know which space the bytes
//! came from. A guest that sets the bit while `CAP_GUESTMEM` is masked
//! off gets an empty command stream (nothing decoded) rather than a
//! register-level protocol error: `RING_BASE`/`RING_SIZE` are ordinary
//! registers with no error-latch mechanism of their own (only ring
//! *commands* raise `ERROR_CODE`), and a conformant guest never sets the
//! bit when `CAPS0` does not advertise the capability in the first
//! place.
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

/// Copperline's fitted `CAPS0` -- see the module doc comment's
/// "Milestone scope" and "`CAP_GUESTMEM`". `CAP_TRANSFORM` and
/// `CAP_MULTITEXTURE` are advertised now that the renderer implements
/// the transform-tier MVP (GL-space draws in all three shapes,
/// viewport/depth-range, two-unit multitexture, texgen); `CAP_GUESTMEM`
/// is advertised now that `ApertureMemory` resolves space-1 refs (and
/// a guest-memory ring) through `DeviceHost`, completing the spec's
/// stated Copperline capability set (bits 0, 1, 2, 3, 4, 6).
pub const CAPS0: u32 = proto::CAP_IRQ
    | proto::CAP_REF_SYNC
    | proto::CAP_SURFACE_GUESTADDR
    | proto::CAP_TRANSFORM
    | proto::CAP_MULTITEXTURE
    | proto::CAP_GUESTMEM;

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

    /// A fence this context reached whose completion is held back because
    /// a `CAP_SURFACE_GUESTADDR` readback in the same doorbell queued
    /// bytes for the bus's cross-board DMA pass (see
    /// `C3dBoard::take_pending_dma_writes`) rather than landing
    /// immediately. Two-stage, not simply "pending": `deferred_fence` is
    /// this doorbell's own queued write, not yet even handed to the bus;
    /// `deferred_fence_ready` is last call's, now safe to complete
    /// because the bus has had one whole tick to apply it (draining
    /// happens synchronously, right after `take_pending_dma_writes`
    /// returns, in the same tick invocation). Neither is meaningful
    /// outside one doorbell/tick cycle, so both are transient.
    #[serde(skip)]
    deferred_fence: Option<u32>,
    #[serde(skip)]
    deferred_fence_ready: Option<u32>,
}

impl Default for ContextSlot {
    fn default() -> Self {
        ContextSlot {
            ctx: Context::new(state::Limits::default()),
            alloc: false,
            enable: false,
            fence_irq_target: 0,
            deferred_fence: None,
            deferred_fence_ready: None,
        }
    }
}

/// The C3D board. See the module doc comment for the doorbell path and
/// the milestone's capability scope.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct C3dBoard {
    window_bytes: u32,
    aperture_offset: u32,
    /// The effective `CAPS0` this board instance reports: the full fitted
    /// set ([`CAPS0`]) minus any bits masked off by `[c3d] mask_caps` --
    /// the spec's conformance switch ("a configuration switch that masks
    /// any subset off"). Config-derived like `window_bytes`, so save
    /// states never carry it. Everything capability-dependent
    /// (`device_config`, the limit registers) derives from this field,
    /// never from the constant, so a masked bit behaves exactly like a
    /// device built without it.
    caps0: u32,
    aperture: Vec<u8>,
    control: u32,
    irq_status: u32,
    irq_enable: u32,
    contexts: Vec<ContextSlot>,
    activity: bool,

    /// Bytes queued this tick for the bus's cross-board DMA pass; see
    /// `take_pending_dma_writes`. Transient -- drained every tick, so a
    /// save state never needs to carry it (the worst a save/restore mid
    /// doorbell can do is lose one tick's not-yet-applied readback,
    /// which happens before the doorbell's fence would ever complete
    /// anyway, so nothing guest-visible depends on it surviving).
    #[serde(skip)]
    pending_dma: Vec<crate::zorro_device::PendingDmaWrite>,

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
        Self::with_masked_caps(window_bytes, 0)
    }

    /// [`Self::new`] with `mask_caps` bits (`[c3d] mask_caps`, resolved
    /// to a `CAPS0` bitmask by config validation) forced clear.
    pub fn with_masked_caps(window_bytes: u32, mask_caps: u32) -> Self {
        let aperture_offset = proto::APERTURE_OFFSET_DEFAULT;
        let aperture_len = window_bytes.saturating_sub(aperture_offset) as usize;
        let mut contexts = Vec::with_capacity(CONTEXT_COUNT);
        contexts.resize_with(CONTEXT_COUNT, ContextSlot::default);
        C3dBoard {
            window_bytes,
            aperture_offset,
            caps0: CAPS0 & !mask_caps,
            aperture: vec![0u8; aperture_len],
            control: CONTROL_ENABLE,
            irq_status: 0,
            irq_enable: 0,
            contexts,
            activity: false,
            pending_dma: Vec::new(),
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
            guestmem: self.caps0 & proto::CAP_GUESTMEM != 0,
            transform: self.caps0 & proto::CAP_TRANSFORM != 0,
            multitexture: self.caps0 & proto::CAP_MULTITEXTURE != 0,
            surface_guestaddr: self.caps0 & proto::CAP_SURFACE_GUESTADDR != 0,
            aperture_size: self.aperture_size(),
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
            CAPS0 => self.caps0,
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
            // `MAX_TEXTURE_UNITS > 1` is documented to imply
            // `CAP_MULTITEXTURE`, so a masked bit caps the report at 1.
            MAX_TEXTURE_UNITS => {
                if self.caps0 & proto::CAP_MULTITEXTURE != 0 {
                    self.device_config().max_texture_units
                } else {
                    1
                }
            }
            MAX_TEXTURES => 256,
            MAX_SURFACES => 16,
            // Transform-tier limits report the same values every
            // per-context `state::Limits::default()` actually enforces
            // (the spec's stated minimums) -- and read zero while
            // `CAP_TRANSFORM` is clear (whether never fitted or masked
            // off by `[c3d] mask_caps`), per the spec's own note.
            MAX_LIGHTS
            | MAX_CLIP_PLANES
            | MAX_MATRIX_DEPTH_MV
            | MAX_MATRIX_DEPTH_PROJ
            | MAX_MATRIX_DEPTH_TEX
                if self.caps0 & proto::CAP_TRANSFORM == 0 =>
            {
                0
            }
            MAX_LIGHTS => state::Limits::default().max_lights as u32,
            MAX_CLIP_PLANES => state::Limits::default().max_clip_planes as u32,
            MAX_MATRIX_DEPTH_MV => state::Limits::default().modelview_depth as u32,
            MAX_MATRIX_DEPTH_PROJ => state::Limits::default().projection_depth as u32,
            MAX_MATRIX_DEPTH_TEX => state::Limits::default().texture_depth as u32,
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
        self.pending_dma.clear();
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

    fn write_context(&mut self, n: usize, off: u32, value: u32, host: &DeviceHost) {
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
            RING_TAIL => self.doorbell(n, value, host),
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
    fn doorbell(&mut self, n: usize, new_tail: u32, host: &DeviceHost) {
        if n >= self.contexts.len() || !self.contexts[n].alloc || !self.contexts[n].enable {
            return;
        }
        self.activity = true;

        let ring_base = self.contexts[n].ctx.ring_base;
        let ring_size_reg = self.contexts[n].ctx.ring_size;
        let ring_len = ring_size_reg & 0x7FFF_FFFF;
        // Bit 31 of RING_SIZE: RING_BASE names a guest address rather
        // than an aperture offset (spec, `RING_BASE`/`RING_SIZE`; see the
        // module doc comment's "Ring in guest memory"). Gated on the
        // *effective* CAPS0, not the constant, so a masked-off
        // CAP_GUESTMEM behaves as if the ring can only ever be
        // aperture-backed.
        let ring_bytes = if ring_size_reg & 0x8000_0000 != 0 && self.device_config().guestmem {
            let mut buf = vec![0u8; ring_len as usize];
            host.dma_read(ring_base, &mut buf);
            buf
        } else {
            self.read_aperture_range(ring_base, ring_len)
        };

        let config = self.device_config();
        let mut ops: Vec<RenderOp<'_>> = Vec::new();
        self.contexts[n]
            .ctx
            .submit(&ring_bytes, new_tail, &config, &mut ops);

        if !ops.is_empty() {
            let queued_before = self.pending_dma.len();
            if let Some(renderer) =
                Self::ensure_renderer(&mut self.renderer, &mut self.renderer_failed)
            {
                let state = &self.contexts[n].ctx.state;
                let mut mem = ApertureMemory {
                    aperture: &mut self.aperture,
                    pending_dma: &mut self.pending_dma,
                    host,
                    guest_scratch: Vec::new(),
                };
                let errors = renderer.execute(&ops, state, &mut mem);
                for e in &errors {
                    log::warn!("c3d: context {n}: render error: {e}");
                }
            }
            // If this submission queued a guest-address write (a
            // CAP_SURFACE_GUESTADDR readback into another board's
            // window), the bus has not applied it yet, so no fence this
            // submission reaches has actually taken effect -- defer the
            // last one to `deferred_fence` instead of completing any of
            // them now. Only the last one matters: fence IDs are
            // monotonic and everything is synchronous within one
            // doorbell, so the guest can never observe an intermediate
            // value before the final one supersedes it. Otherwise (the
            // common case: aperture-backed surfaces only) complete every
            // fence immediately, exactly as before this milestone.
            let deferred = self.pending_dma.len() > queued_before;
            let last_fence = ops.iter().rev().find_map(|op| match op {
                RenderOp::Fence { id } => Some(*id),
                _ => None,
            });
            if deferred {
                if let Some(id) = last_fence {
                    self.contexts[n].deferred_fence = Some(id);
                }
            } else {
                for op in &ops {
                    if let RenderOp::Fence { id } = op {
                        self.contexts[n].ctx.complete_fence(*id);
                    }
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

/// Adapts the board's aperture buffer (plus, since `CAP_GUESTMEM`, the
/// guest address space reached through [`DeviceHost`]) to [`Memory`]. A
/// [`MemLoc::Aperture`] address is already **aperture-relative** -- `0`
/// is the aperture's own first byte, not the window's -- exactly as the
/// spec's "an aperture offset" phrasing means it for `RING_BASE` and for
/// a space-`0` reference, so it is used directly as an index into
/// `aperture` with no translation. (There is nothing to translate
/// *from*: this struct does not even carry the window's
/// `APERTURE_OFFSET`, on purpose, so that reintroducing a subtraction
/// here is a type error, not a silent regression back to the bug this
/// comment used to describe.)
///
/// [`MemLoc::Guest`] reads and writes are handled asymmetrically -- see
/// the module doc comment's "`CAP_GUESTMEM`" for why: reads go straight
/// through `host.dma_read` (synchronous, `&self`, no conflict with any
/// other borrow live during the doorbell), writes are queued into
/// `pending_dma` for the bus to apply one tick later (the only path that
/// correctly reaches a *device-backed* board window, which a plain
/// synchronous `dma_write` cannot). This struct is constructed fresh
/// every doorbell that has ops to execute, never held across doorbells.
struct ApertureMemory<'a, 'h> {
    aperture: &'a mut Vec<u8>,
    /// Where a [`MemLoc::Guest`] write is queued -- see
    /// `C3dBoard::take_pending_dma_writes`'s doc comment for why this
    /// board cannot apply such a write itself.
    pending_dma: &'a mut Vec<crate::zorro_device::PendingDmaWrite>,
    /// Host-services view used for [`MemLoc::Guest`] *reads*
    /// (`dma_read`, `&self`, so borrowing it alongside `aperture`/
    /// `pending_dma` needs no special care). A separate lifetime `'h`
    /// from `'a`: this reference's own borrow is scoped to the doorbell
    /// like everything else here, but the `DeviceHost` it points at was
    /// itself already borrowed (from `ZorroDevice::write`'s caller) for
    /// however long that call lives, an unrelated and typically longer
    /// span.
    host: &'a DeviceHost<'h>,
    /// Owned backing for the slice a [`MemLoc::Guest`] read lends out --
    /// `Memory::read` takes `&mut self` for exactly this reason. Reused
    /// (not reallocated) across reads within one doorbell; its previous
    /// contents are irrelevant since every read resizes and refills it
    /// before returning a slice into it.
    guest_scratch: Vec<u8>,
}

impl ApertureMemory<'_, '_> {
    fn range(&self, aperture_addr: u32, len: usize) -> Option<std::ops::Range<usize>> {
        let start = aperture_addr as usize;
        let end = start.checked_add(len)?;
        (end <= self.aperture.len()).then_some(start..end)
    }
}

impl Memory for ApertureMemory<'_, '_> {
    fn read(&mut self, loc: MemLoc, len: usize) -> Option<&[u8]> {
        match loc {
            MemLoc::Aperture(addr) => {
                let r = self.range(addr, len)?;
                Some(&self.aperture[r])
            }
            // CAP_GUESTMEM: pull the bytes straight from guest memory.
            // `dma_read` never fails -- an address the bus cannot resolve
            // reads back as 0xFF, the same "unmapped" convention every
            // other DMA bus master in this codebase uses -- matching
            // ring.rs's own note that reachability of a space-1 address
            // is a higher layer's concern it cannot itself validate.
            MemLoc::Guest(addr) => {
                self.guest_scratch.resize(len, 0);
                self.host.dma_read(addr, &mut self.guest_scratch);
                Some(&self.guest_scratch[..len])
            }
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
            // Queued for the bus to apply after this tick -- see
            // `ZorroDevice::take_pending_dma_writes`'s doc comment and
            // the module doc comment's "CAP_GUESTMEM" (writes stay
            // deferred even for ordinary guest RAM, not just a
            // CAP_SURFACE_GUESTADDR cross-board target, so one code path
            // handles both). Accepted unconditionally (never `false`):
            // whether `addr` is actually reachable is the bus's question
            // to answer when it resolves the address, not this board's.
            MemLoc::Guest(addr) => {
                self.pending_dma.push(crate::zorro_device::PendingDmaWrite {
                    addr,
                    bytes: data.to_vec(),
                });
                true
            }
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

    fn write(&mut self, off: u32, size: usize, value: u32, host: &mut DeviceHost) {
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
            self.write_context(n, reg_off, full, host);
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

    fn take_pending_dma_writes(&mut self) -> Vec<crate::zorro_device::PendingDmaWrite> {
        // Two-stage drain -- see ContextSlot's own doc comment on
        // `deferred_fence`/`deferred_fence_ready`. This call happens once
        // per bus tick, immediately followed (same tick, same call to
        // `Bus`'s cross-board DMA pass) by the bus actually applying the
        // batch this returns, so "ready" entries left over from the
        // *previous* call are now safe to complete, and this call's own
        // newly-queued writes become "ready" for the *next* call.
        for slot in &mut self.contexts {
            if let Some(id) = slot.deferred_fence_ready.take() {
                slot.ctx.complete_fence(id);
            }
        }
        for slot in &mut self.contexts {
            slot.deferred_fence_ready = slot.deferred_fence.take();
        }
        std::mem::take(&mut self.pending_dma)
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
    fn transform_tier_limits_report_the_enforced_spec_minimums() {
        // With CAP_TRANSFORM/CAP_MULTITEXTURE advertised, these registers
        // must report exactly what each context's `state::Limits::default()`
        // actually enforces -- the spec's stated minimums -- never zero
        // (that was the pre-transform-tier behaviour) and never a value
        // the state layer would then reject.
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        assert_ne!(CAPS0 & proto::CAP_TRANSFORM, 0);
        assert_ne!(CAPS0 & proto::CAP_MULTITEXTURE, 0);
        let limits = state::Limits::default();
        assert_eq!(
            b.read(greg::MAX_LIGHTS, 4, &mut h),
            limits.max_lights as u32
        );
        assert_eq!(
            b.read(greg::MAX_CLIP_PLANES, 4, &mut h),
            limits.max_clip_planes as u32
        );
        assert_eq!(
            b.read(greg::MAX_MATRIX_DEPTH_MV, 4, &mut h),
            limits.modelview_depth as u32
        );
        assert_eq!(
            b.read(greg::MAX_MATRIX_DEPTH_PROJ, 4, &mut h),
            limits.projection_depth as u32
        );
        assert_eq!(
            b.read(greg::MAX_MATRIX_DEPTH_TEX, 4, &mut h),
            limits.texture_depth as u32
        );
        assert_eq!(
            b.read(greg::MAX_TEXTURE_UNITS, 4, &mut h),
            limits.texture_units as u32
        );
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

    /// `[c3d] mask_caps`'s contract: a masked bit reads clear in `CAPS0`,
    /// its dependent limit registers report as on a device without it,
    /// and its opcodes are rejected -- the spec's conformance switch
    /// ("baseline, each bit cleared singly, and the full set"). This is
    /// the each-bit-cleared-singly case for `CAP_TRANSFORM`, plus the
    /// `CAP_MULTITEXTURE`-caps-`MAX_TEXTURE_UNITS` rule.
    #[test]
    fn mask_caps_clears_the_bit_its_limits_and_its_opcodes() {
        let mut b = C3dBoard::with_masked_caps(0x0200_0000, proto::CAP_TRANSFORM);
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        assert_eq!(
            b.read(greg::CAPS0, 4, &mut h),
            CAPS0 & !proto::CAP_TRANSFORM
        );
        assert_eq!(b.read(greg::MAX_LIGHTS, 4, &mut h), 0);
        assert_eq!(b.read(greg::MAX_MATRIX_DEPTH_TEX, 4, &mut h), 0);

        // A transform-tier opcode through the real doorbell must be
        // E_BAD_OPCODE again, exactly as before the CAPS0 flip.
        let words = [((proto::OP_LOAD_IDENTITY as u32) << 16) | 1];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_be_bytes()).collect();
        for (i, byte) in bytes.iter().enumerate() {
            b.write(b.aperture_offset + i as u32, 1, *byte as u32, &mut h);
        }
        b.write(CONTEXT_PAGE_BASE + creg::RING_BASE, 4, 0, &mut h);
        b.write(CONTEXT_PAGE_BASE + creg::RING_SIZE, 4, 0x1000, &mut h);
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
        assert_ne!(
            b.read(CONTEXT_PAGE_BASE + creg::ERROR_CODE, 4, &mut h),
            0,
            "a masked-off CAP_TRANSFORM must reject its opcodes"
        );

        let mut b = C3dBoard::with_masked_caps(0x0200_0000, proto::CAP_MULTITEXTURE);
        let mut h = host(&mut mem);
        assert_eq!(b.read(greg::MAX_TEXTURE_UNITS, 4, &mut h), 1);
    }

    /// The CAPS0 flip's own end-to-end guard: a transform-tier opcode
    /// (`MATRIX_MODE`+`LOAD_IDENTITY`) and a multitexture opcode
    /// (`ACTIVE_UNIT` with unit 1) submitted through the real register
    /// path must decode cleanly -- before the flip, the very first of
    /// them was `E_BAD_OPCODE` at the decoder and the stream would stop
    /// there. Mirrors the full-command-stream test's harness.
    #[test]
    fn transform_and_multitexture_opcodes_pass_the_doorbell_with_caps0_advertised() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);

        let mut cmds = Vec::<u32>::new();
        let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;
        cmds.push(opcode_len(proto::OP_MATRIX_MODE, 2));
        cmds.push(0x1700); // GL_MODELVIEW
        cmds.push(opcode_len(proto::OP_LOAD_IDENTITY, 1));
        cmds.push(opcode_len(proto::OP_ACTIVE_UNIT, 2));
        cmds.push(1); // unit 1: E_BAD_OPCODE-adjacent E_LIMIT without CAP_MULTITEXTURE
        cmds.push(opcode_len(proto::OP_ACTIVE_UNIT, 2));
        cmds.push(0); // back to unit 0, leaving clean state

        let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        let ring_window_off = b.aperture_offset;
        for (i, b_) in bytes.iter().enumerate() {
            b.write(ring_window_off + i as u32, 1, *b_ as u32, &mut h);
        }

        b.write(CONTEXT_PAGE_BASE + creg::RING_BASE, 4, 0, &mut h);
        b.write(CONTEXT_PAGE_BASE + creg::RING_SIZE, 4, 0x1000, &mut h);
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
            "transform/multitexture opcodes must decode cleanly with CAPS0 advertising them"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::RING_HEAD, 4, &mut h),
            bytes.len() as u32,
            "every command consumed, none rejected as E_BAD_OPCODE"
        );
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

    // -----------------------------------------------------------------
    // CAP_GUESTMEM
    // -----------------------------------------------------------------

    /// A guest-backed surface (`SURFACE_DEFINE`'s `GUEST_ADDR` flag,
    /// `CAP_SURFACE_GUESTADDR`) round-tripped through `SURFACE_UPLOAD`
    /// (reads the surface's own backing memory into the GPU texture) and
    /// `SURFACE_READBACK` (reads the GPU texture back out to the same
    /// backing memory): this is a strong, pixel-level proof that
    /// `Memory::read`'s `MemLoc::Guest` arm actually reaches guest
    /// memory, not just that decoding tolerates a guest-space address.
    /// If the upload's read were broken, `ensure_surface` still creates
    /// a zero-initialized GPU texture before the read fails, so the
    /// *readback* half still runs (errors don't stop later ops -- see
    /// `Renderer::execute`) and writes zero pixels back over the guest
    /// bytes this test pre-filled with a distinct colour -- so a broken
    /// read is not silently invisible here, it flips every pixel to
    /// black. `TEX_IMAGE`'s `data` ref exercises the identical
    /// `Memory::read` arm (see the module doc comment's `CAP_GUESTMEM`);
    /// `SURFACE_UPLOAD` was chosen because its counterpart
    /// `SURFACE_READBACK` gives an observable pass/fail without probing
    /// the renderer's private texture storage from another module.
    #[test]
    fn surface_upload_then_readback_round_trips_guest_backed_pixels_through_the_gpu_texture() {
        let mut b = board();
        let mut mem = dummy_mem();

        const GUEST_SURFACE_ADDR: u32 = 0x0000_0400;
        const W: u32 = 4;
        const H: u32 = 4;
        const STRIDE: u32 = W * 4; // A8R8G8B8
                                   // Opaque green (A R G B byte order, matching the M2 test).
        let pixel = [0xFFu8, 0x00, 0xFF, 0x00];
        for row in 0..H {
            for col in 0..W {
                let off = (GUEST_SURFACE_ADDR + row * STRIDE + col * 4) as usize;
                mem.chip_ram[off..off + 4].copy_from_slice(&pixel);
            }
        }

        let mut h = host(&mut mem);

        let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;
        let mut cmds = Vec::<u32>::new();
        // SURFACE_DEFINE id=1, guest-backed A8R8G8B8.
        cmds.push(opcode_len(proto::OP_SURFACE_DEFINE, 8));
        cmds.extend([
            1,
            W,
            H,
            STRIDE,
            5, // A8R8G8B8
            proto::SURFACE_DEFINE_FLAG_GUEST_ADDR,
            GUEST_SURFACE_ADDR,
        ]);
        cmds.push(opcode_len(proto::OP_SET_DRAW_SURFACE, 2));
        cmds.push(1);
        cmds.push(opcode_len(proto::OP_SURFACE_UPLOAD, 5));
        cmds.extend([0, 0, W, H]);
        cmds.push(opcode_len(proto::OP_SURFACE_READBACK, 5));
        cmds.extend([0, 0, W, H]);
        cmds.push(opcode_len(proto::OP_FENCE, 2));
        cmds.push(1);

        let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        for (i, byte) in bytes.iter().enumerate() {
            b.write(b.aperture_offset + i as u32, 1, *byte as u32, &mut h);
        }
        b.write(CONTEXT_PAGE_BASE + creg::RING_BASE, 4, 0, &mut h);
        b.write(CONTEXT_PAGE_BASE + creg::RING_SIZE, 4, 0x1000, &mut h);
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
            "no protocol error decoding a guest-backed surface stream"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::RING_HEAD, 4, &mut h),
            bytes.len() as u32,
            "every command consumed"
        );

        if b.renderer.is_none() {
            eprintln!(
                "skipping guest round-trip check: no wgpu adapter available (renderer_failed={})",
                b.renderer_failed
            );
            return;
        }

        // SURFACE_READBACK addresses row by row, so expect one queued
        // write per row (see render.rs's op_surface_readback).
        let writes = b.take_pending_dma_writes();
        assert_eq!(
            writes.len(),
            H as usize,
            "SURFACE_READBACK's guest-space writes must be queued, not applied synchronously"
        );
        for w in &writes {
            for (i, byte) in w.bytes.iter().enumerate() {
                crate::zorro_device::dma_write_byte(&mut mem, w.addr + i as u32, *byte);
            }
        }

        for row in 0..H {
            for col in 0..W {
                let off = (GUEST_SURFACE_ADDR + row * STRIDE + col * 4) as usize;
                assert_eq!(
                    &mem.chip_ram[off..off + 4],
                    &pixel,
                    "pixel ({col}, {row}) did not round-trip guest memory -> GPU texture -> guest memory"
                );
            }
        }
    }

    /// `QUERY`'s `dest` in guest space: the result bytes must land in
    /// actual guest memory. Since a `MemLoc::Guest` write always queues
    /// into `pending_dma` (see the module doc comment's `CAP_GUESTMEM`
    /// section) rather than landing synchronously, this drains it by
    /// hand and applies it exactly the way `Bus::drain_cross_board_dma`
    /// does for a plain-RAM target (`dma_write_byte`), the same
    /// machinery the `CAP_SURFACE_GUESTADDR` bus-level test already
    /// trusts. Needs a real renderer (`RenderOp::Query` is executed by
    /// `Renderer::execute`, unlike `FENCE`) so this skips gracefully
    /// without a wgpu adapter, mirroring the M2 pixel-check test.
    #[test]
    fn query_writes_its_result_into_guest_memory_via_the_deferred_dma_path() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);

        const GUEST_DEST_ADDR: u32 = 0x0000_0200;
        const GL_CURRENT_COLOR: u32 = 0x0B00;

        let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;
        let mut cmds = Vec::<u32>::new();
        cmds.push(opcode_len(proto::OP_QUERY, 4));
        cmds.extend([
            GL_CURRENT_COLOR,
            GUEST_DEST_ADDR,
            0x8000_0000 | 16, // space 1 (guest), length 16 (vec4 of f32)
        ]);

        let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        for (i, byte) in bytes.iter().enumerate() {
            b.write(b.aperture_offset + i as u32, 1, *byte as u32, &mut h);
        }
        b.write(CONTEXT_PAGE_BASE + creg::RING_BASE, 4, 0, &mut h);
        b.write(CONTEXT_PAGE_BASE + creg::RING_SIZE, 4, 0x1000, &mut h);
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
            "no protocol error decoding a well-formed guest-dest QUERY"
        );

        if b.renderer.is_none() {
            eprintln!(
                "skipping guest-write check: no wgpu adapter available (renderer_failed={})",
                b.renderer_failed
            );
            return;
        }

        let writes = b.take_pending_dma_writes();
        assert_eq!(
            writes.len(),
            1,
            "QUERY's guest-space write must be queued, not applied synchronously"
        );
        assert_eq!(writes[0].addr, GUEST_DEST_ADDR);
        // GL_CURRENT_COLOR defaults to opaque white (state.rs's
        // CurrentVertex::default): four big-endian 1.0f32 values.
        let expected: Vec<u8> = [1.0f32; 4].iter().flat_map(|f| f.to_be_bytes()).collect();
        assert_eq!(writes[0].bytes, expected);

        // Apply it exactly the way Bus::drain_cross_board_dma would for a
        // plain-RAM guest target.
        for (i, byte) in writes[0].bytes.iter().enumerate() {
            crate::zorro_device::dma_write_byte(&mut mem, GUEST_DEST_ADDR + i as u32, *byte);
        }
        assert_eq!(
            &mem.chip_ram[GUEST_DEST_ADDR as usize..GUEST_DEST_ADDR as usize + 16],
            expected.as_slice(),
            "the query result must land in guest RAM once the bus applies the deferred write"
        );
    }

    /// The masking side of the conformance switch for this bit: with
    /// `CAP_GUESTMEM` cleared, a space-1 ref is `E_BAD_REF` (ring.rs's
    /// `validate_ref`, exercised here through the real doorbell rather
    /// than the decoder's own unit test).
    #[test]
    fn mask_caps_rejects_a_guest_space_ref_with_e_bad_ref() {
        let mut b = C3dBoard::with_masked_caps(0x0200_0000, proto::CAP_GUESTMEM);
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);

        assert_eq!(b.read(greg::CAPS0, 4, &mut h), CAPS0 & !proto::CAP_GUESTMEM);

        let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;
        let mut cmds = Vec::<u32>::new();
        cmds.push(opcode_len(proto::OP_QUERY, 4));
        cmds.extend([0x0B00u32, 0x0000_0200, 0x8000_0000 | 16]);

        let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        for (i, byte) in bytes.iter().enumerate() {
            b.write(b.aperture_offset + i as u32, 1, *byte as u32, &mut h);
        }
        b.write(CONTEXT_PAGE_BASE + creg::RING_BASE, 4, 0, &mut h);
        b.write(CONTEXT_PAGE_BASE + creg::RING_SIZE, 4, 0x1000, &mut h);
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
            ErrorCode::BadRef as u32,
            "a masked-off CAP_GUESTMEM must reject a space-1 ref"
        );
    }

    /// `RING_BASE`/`RING_SIZE`: bit 31 of `RING_SIZE` names a guest
    /// address for the ring itself (spec, "an aperture offset, or a
    /// guest address if bit 31 of RING_SIZE is set"), not just for refs
    /// inside the stream. The command bytes here live only in
    /// `dummy_mem`'s `chip_ram`, never written to the aperture at all,
    /// so this only passes if the doorbell's ring fetch itself reaches
    /// guest memory.
    #[test]
    fn a_ring_living_in_guest_memory_is_fetched_through_the_host() {
        let mut b = board();
        let mut mem = dummy_mem();

        const GUEST_RING_ADDR: u32 = 0x0000_0300;
        let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;
        // FENCE id=7: needs no ref, and its completion doesn't depend on
        // a renderer existing (see the module doc comment's doorbell
        // section), so this proves the ring fetch alone, nothing else.
        let cmds = [opcode_len(proto::OP_FENCE, 2), 7u32];
        let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        mem.chip_ram[GUEST_RING_ADDR as usize..GUEST_RING_ADDR as usize + bytes.len()]
            .copy_from_slice(&bytes);

        let mut h = host(&mut mem);
        b.write(
            CONTEXT_PAGE_BASE + creg::RING_BASE,
            4,
            GUEST_RING_ADDR,
            &mut h,
        );
        // Bit 31 set: RING_BASE is a guest address.
        b.write(
            CONTEXT_PAGE_BASE + creg::RING_SIZE,
            4,
            0x8000_0000 | 0x1000,
            &mut h,
        );
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
            "a well-formed guest-memory ring must decode cleanly"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::RING_HEAD, 4, &mut h),
            bytes.len() as u32,
            "the FENCE command was consumed from the guest-memory ring"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::FENCE_COMPLETED, 4, &mut h),
            7,
            "the fence decoded from the guest-memory ring completed"
        );
    }

    /// A guest that sets `RING_SIZE`'s guest-address bit while
    /// `CAP_GUESTMEM` is masked off gets nothing decoded -- no register
    /// exists to latch a protocol error for a misconfigured `RING_BASE`/
    /// `RING_SIZE` (only ring *commands* raise `ERROR_CODE`), so "ignore
    /// the whole ring" is this board's answer, documented in the module
    /// doc comment's "Ring in guest memory".
    #[test]
    fn a_masked_cap_guestmem_ignores_a_guest_memory_ring_entirely() {
        let mut b = C3dBoard::with_masked_caps(0x0200_0000, proto::CAP_GUESTMEM);
        let mut mem = dummy_mem();

        const GUEST_RING_ADDR: u32 = 0x0000_0300;
        let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;
        let cmds = [opcode_len(proto::OP_FENCE, 2), 7u32];
        let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        mem.chip_ram[GUEST_RING_ADDR as usize..GUEST_RING_ADDR as usize + bytes.len()]
            .copy_from_slice(&bytes);

        let mut h = host(&mut mem);
        b.write(
            CONTEXT_PAGE_BASE + creg::RING_BASE,
            4,
            GUEST_RING_ADDR,
            &mut h,
        );
        b.write(
            CONTEXT_PAGE_BASE + creg::RING_SIZE,
            4,
            0x8000_0000 | 0x1000,
            &mut h,
        );
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
            b.read(CONTEXT_PAGE_BASE + creg::RING_HEAD, 4, &mut h),
            0,
            "nothing was decoded from a guest-memory ring CAP_GUESTMEM cannot reach"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::FENCE_COMPLETED, 4, &mut h),
            0
        );
    }
}
