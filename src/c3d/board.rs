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
//! expects for a guest-address readback. **`RING_HEAD`/`CTX_STATUS.
//! IDLE_RING` are not part of that deferral and must not be read as if
//! they were**: `Context::submit` advances `RING_HEAD` synchronously,
//! every doorbell, regardless of whether anything it decoded is still
//! pending a deferred write -- and the spec says exactly that:
//! `RING_HEAD` is "commands before it have been *consumed* (decoded
//! and, if by-reference and `CAP_REF_SYNC`, captured), not necessarily
//! executed". A guest that infers "the write has landed" from
//! `RING_HEAD`/`IDLE_RING` rather than `FENCE_COMPLETED` is reading a
//! signal the spec never promised that meaning for; `FENCE_COMPLETED`
//! is the only architected completion signal, and it is the one this
//! board actually defers alongside the write (see
//! `ContextSlot::deferred_fence`).
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
//! off gets nothing decoded (`doorbell` returns before ever building a
//! ring slice) rather than a register-level protocol error:
//! `RING_BASE`/`RING_SIZE` are ordinary registers with no error-latch
//! mechanism of their own (only ring *commands* raise `ERROR_CODE`, and
//! this case never reaches a command decoder), and a conformant guest
//! never sets the bit when `CAPS0` does not advertise the capability in
//! the first place. This is a dedicated early return, not a third arm
//! folded into the aperture-vs-guest `if` below: `ring_base` is a guest
//! address here, and letting it fall through to
//! `read_aperture_range(ring_base, ...)` would decode whatever
//! unrelated aperture bytes happen to sit at that offset as commands --
//! wrong, not merely a missed capability.
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
use super::render::{MemLoc, Memory, RenderError, Renderer};
use super::ring::DeviceConfig;
use super::state;
use super::trace::TraceWriter;
use crate::zorro_device::{DeviceHost, ZorroDevice};
use std::fs::File;
use std::io::{self, BufWriter};
use std::path::PathBuf;

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
/// a guest-memory ring) through `DeviceHost`; `CAP_PROGRAMMABLE` is
/// advertised now that the full object model `proto::CAP_PROGRAMMABLE`'s
/// doc comment lists -- shaders/programs, buffers and vertex attributes,
/// framebuffer/renderbuffer objects, programmable draw, and the stencil
/// test -- is implemented, per the spec's porting guidance ("set
/// `CAP_PROGRAMMABLE` only if it implements the full ... object model").
/// Also advertises `CAP_SHARED_CONTEXTS` (bit 8) now that `Renderer`'s
/// GPU-resource maps are properly partitioned by [`SharePartition`]
/// (textures, shaders, programs, buffers, renderbuffers, palettes) --
/// investigating this feature surfaced that they previously were not
/// partitioned by context *at all* (every context's objects landed in
/// one accidentally-global id-keyed map), a latent cross-context
/// aliasing bug independent of this capability; see `SharePartition`'s
/// doc comment.
///
/// This completes the spec's stated Copperline capability set (bits 0,
/// 1, 2, 3, 4, 6, 7, 8).
pub const CAPS0: u32 = proto::CAP_IRQ
    | proto::CAP_REF_SYNC
    | proto::CAP_SURFACE_GUESTADDR
    | proto::CAP_TRANSFORM
    | proto::CAP_MULTITEXTURE
    | proto::CAP_GUESTMEM
    | proto::CAP_PROGRAMMABLE
    | proto::CAP_SHARED_CONTEXTS;

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
    pub const MAX_SHADERS: u32 = 0x078;
    pub const MAX_PROGRAMS: u32 = 0x07C;
    pub const MAX_BUFFERS: u32 = 0x080;
    pub const MAX_FRAMEBUFFERS: u32 = 0x084;
    pub const MAX_RENDERBUFFERS: u32 = 0x088;
    pub const MAX_VERTEX_ATTRIBS: u32 = 0x08C;
    pub const MAX_VERTEX_UNIFORM_VECTORS: u32 = 0x090;
    pub const MAX_FRAGMENT_UNIFORM_VECTORS: u32 = 0x094;
    pub const MAX_VARYING_VECTORS: u32 = 0x098;
    pub const MAX_COLOR_ATTACHMENTS: u32 = 0x09C;
    pub const MAX_RENDERBUFFER_SIZE: u32 = 0x0A0;
    pub const MAX_SAMPLES: u32 = 0x0A4;
    pub const PROGRAM_BINARY_FORMAT: u32 = 0x0A8;
    pub const ALIASED_POINT_SIZE_MIN: u32 = 0x0AC;
    pub const ALIASED_POINT_SIZE_MAX: u32 = 0x0B0;
    pub const MAX_SYNC_OBJECTS: u32 = 0x0B4;
    pub const SHADER_LANGS_SUPPORTED: u32 = 0x0B8;
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
    pub const SHARE_GROUP: u32 = 0x34;
}

const CTX_CONTROL_ALLOC: u32 = 1 << 0;
const CTX_CONTROL_ENABLE: u32 = 1 << 1;
const CTX_CONTROL_RESET: u32 = 1 << 2;

const CTX_STATUS_HALTED: u32 = 1 << 1;
const CTX_STATUS_IDLE_RING: u32 = 1 << 2;
const CTX_STATUS_AWAITING_SYNC: u32 = 1 << 3;

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

/// One share group's live storage, as `board.rs` sees it: `Weak` handles
/// into the same six `Rc<RefCell<_>>`s a group's member contexts'
/// `State`s hold strong references to -- see `C3dBoard::share_groups`'s
/// doc comment for why `Weak`, not `Rc`, is essential here. `Default`
/// gives every field a `Weak::new()` (upgrades to `None`), used as the
/// "no live member yet" starting point in [`Self::resolve`].
#[derive(Default)]
struct ShareGroupStores {
    textures:
        std::rc::Weak<std::cell::RefCell<std::collections::BTreeMap<u32, state::TextureObject>>>,
    shaders:
        std::rc::Weak<std::cell::RefCell<std::collections::BTreeMap<u32, state::ShaderObject>>>,
    programs:
        std::rc::Weak<std::cell::RefCell<std::collections::BTreeMap<u32, state::ProgramObject>>>,
    buffers: std::rc::Weak<std::cell::RefCell<std::collections::BTreeSet<u32>>>,
    renderbuffers:
        std::rc::Weak<std::cell::RefCell<std::collections::BTreeMap<u32, state::Renderbuffer>>>,
    sync_objects:
        std::rc::Weak<std::cell::RefCell<std::collections::BTreeMap<u32, state::SyncObject>>>,
}

type SharedMap<T> = std::rc::Rc<std::cell::RefCell<T>>;

impl ShareGroupStores {
    /// Returns this group's six live `Rc`s, upgrading the stored `Weak`s
    /// if another member is still alive, or creating (and remembering)
    /// a fresh set if this is the group's first member (or its previous
    /// members have all since dropped out -- `Weak::upgrade` correctly
    /// returns `None` either way, and either way the right thing is "make
    /// a new one").
    #[allow(clippy::type_complexity)]
    fn resolve(
        &mut self,
    ) -> (
        SharedMap<std::collections::BTreeMap<u32, state::TextureObject>>,
        SharedMap<std::collections::BTreeMap<u32, state::ShaderObject>>,
        SharedMap<std::collections::BTreeMap<u32, state::ProgramObject>>,
        SharedMap<std::collections::BTreeSet<u32>>,
        SharedMap<std::collections::BTreeMap<u32, state::Renderbuffer>>,
        SharedMap<std::collections::BTreeMap<u32, state::SyncObject>>,
    ) {
        let textures = self.textures.upgrade().unwrap_or_else(|| {
            let fresh =
                std::rc::Rc::new(std::cell::RefCell::new(std::collections::BTreeMap::new()));
            self.textures = std::rc::Rc::downgrade(&fresh);
            fresh
        });
        let shaders = self.shaders.upgrade().unwrap_or_else(|| {
            let fresh =
                std::rc::Rc::new(std::cell::RefCell::new(std::collections::BTreeMap::new()));
            self.shaders = std::rc::Rc::downgrade(&fresh);
            fresh
        });
        let programs = self.programs.upgrade().unwrap_or_else(|| {
            let fresh =
                std::rc::Rc::new(std::cell::RefCell::new(std::collections::BTreeMap::new()));
            self.programs = std::rc::Rc::downgrade(&fresh);
            fresh
        });
        let buffers = self.buffers.upgrade().unwrap_or_else(|| {
            let fresh =
                std::rc::Rc::new(std::cell::RefCell::new(std::collections::BTreeSet::new()));
            self.buffers = std::rc::Rc::downgrade(&fresh);
            fresh
        });
        let renderbuffers = self.renderbuffers.upgrade().unwrap_or_else(|| {
            let fresh =
                std::rc::Rc::new(std::cell::RefCell::new(std::collections::BTreeMap::new()));
            self.renderbuffers = std::rc::Rc::downgrade(&fresh);
            fresh
        });
        let sync_objects = self.sync_objects.upgrade().unwrap_or_else(|| {
            let fresh =
                std::rc::Rc::new(std::cell::RefCell::new(std::collections::BTreeMap::new()));
            self.sync_objects = std::rc::Rc::downgrade(&fresh);
            fresh
        });
        (
            textures,
            shaders,
            programs,
            buffers,
            renderbuffers,
            sync_objects,
        )
    }
}

impl ContextSlot {
    /// `index` is this slot's fixed position in `C3dBoard::contexts`,
    /// threaded down into [`state::State::set_context_index`] so
    /// [`state::Namespace::Private`] identity is stable and distinct
    /// per slot -- see that method's doc comment for why it cannot be
    /// derived any other way (`State`/`Context` have no board-level
    /// view of "which slot is this").
    /// `programmable` is this board's fixed `CAP_PROGRAMMABLE` bit
    /// (board-fixed CAPS0 never changes after construction, so it's
    /// safe to bake into `Limits` once here rather than re-deriving it
    /// per access). A `CAP_PROGRAMMABLE` device's `MAX_TEXTURE_UNITS`
    /// must be `>= 8` (spec commit 1ec7e4a), which is meaningless
    /// unless the per-context state actually has 8 texture units' worth
    /// of storage (`texture`/`texgen`/`enables`/`current.texcoord`
    /// vectors, all sized from `Limits::texture_units` in
    /// `State::new`) -- a register claiming 8 units while the state
    /// backing only has 2 would make unit 2..7 silently out of bounds.
    fn new(index: u32, programmable: bool) -> Self {
        let limits = if programmable {
            state::Limits {
                texture_units: 8,
                ..state::Limits::default()
            }
        } else {
            state::Limits::default()
        };
        let mut ctx = Context::new(limits);
        ctx.state.set_context_index(index);
        ContextSlot {
            ctx,
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

    /// `CAP_SHARED_CONTEXTS`'s share groups: `CTX_SHARE_GROUP`'s write
    /// path (`write_context`) looks a group up here to find the `Rc`s an
    /// already-live member holds, or creates a fresh set and registers
    /// them if this is the group's first member. Deliberately `Weak`,
    /// not `Rc`: a strong reference here would keep a group's storage
    /// alive forever, even after every member context has gone, which
    /// would silently break the spec's "a shared object outlives this
    /// one context's release as long as another member exists" -- the
    /// *last* member's own drop (see `write_context`'s `CONTROL` arm)
    /// must be what frees it. `#[serde(skip)]`: a save state round-trips
    /// each context's `State` independently via `serde`'s `rc` feature,
    /// which does not preserve `Rc` identity across separately
    /// (de)serialized values -- restoring from a save state therefore
    /// does not currently preserve a share group's aliasing across its
    /// members, a known, narrow limitation (ordinary single-context
    /// save/restore is completely unaffected). This registry is rebuilt
    /// from empty on restore and repopulated the next time any context
    /// writes a nonzero `CTX_SHARE_GROUP` -- which, since that write is
    /// only honoured before a context's first command, cannot happen to
    /// an already-populated restored context anyway.
    #[serde(skip)]
    share_groups: std::collections::HashMap<u32, ShareGroupStores>,

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

    /// A live `c3d.trace_start`/`trace_stop` capture, when armed -- see
    /// [`TraceCapture`]'s own doc comment for the file this produces and
    /// why it cannot survive a save state the same way `renderer` cannot:
    /// an open `File` has no serializable representation, so a captured
    /// session silently stops (rather than writing a truncated,
    /// unreadable trace tail) across a save/restore, the same tradeoff
    /// `renderer`/`pending_dma` already make for the same reason.
    #[serde(skip)]
    capture: Option<CaptureMode>,
    /// `[c3d] trace`'s configured default path -- what `trace_start`
    /// falls back to when called with `None` (a `c3d.trace_start` with
    /// no explicit `path` of its own, or `[c3d] trace_enabled`'s
    /// boot-time auto-start). An ordinary `PathBuf`, unlike `capture`
    /// itself, so it serializes with everything else here.
    configured_trace_path: Option<PathBuf>,
    /// `[c3d] trace_trigger_texture`/`c3d.trace_start`'s `trigger_texture`
    /// param, armed-but-not-yet-fired: see [`PendingTrigger`]'s own doc
    /// comment. Not serialized for the same reason `capture` is not --
    /// nothing live to resume waiting for across a save state.
    #[serde(skip)]
    pending_trigger: Option<PendingTrigger>,
}

/// Either shape a `c3d.trace_start`-armed capture can take: streaming
/// straight to disk (the original, unbounded M3 shape) or bounded in
/// memory until a snapshot (added once real sessions showed the former
/// grows without limit -- see [`RollingCapture`]'s own doc comment).
enum CaptureMode {
    Direct(TraceCapture),
    Rolling(RollingCapture),
}

/// A live `c3d.trace_start`-armed capture: every doorbell this board
/// handles while armed is written to `writer` as `BLOB*` sections (the
/// refs it resolved) followed by one `RING` section
/// ([`C3dBoard::record_doorbell`]), producing a file
/// `copperline-c3d-trace info`/`run`/`compare` can load directly -- the
/// same container the M1 trace runner builds, just filled from a real
/// session instead of hand-written in code. This is the M3 "trace
/// capture" deliverable's actual point: debugging a real driver (e.g. a
/// guest running real 3D software) without a live interactive session
/// every time.
struct TraceCapture {
    writer: TraceWriter<BufWriter<File>>,
    path: PathBuf,
    /// One monotonic logical tail per context -- see
    /// `C3dBoard::try_record_doorbell`'s doc comment for why this is not
    /// simply the live board's own circular `RING_TAIL`.
    logical_tail: std::collections::HashMap<usize, u32>,
    rings: u64,
    blobs: u64,
    /// Total payload bytes written across every `RING`/`BLOB` section
    /// (not counting each section's own tag/length/address prefix) --
    /// a rough progress figure for `c3d.trace_status`, not an exact file
    /// size.
    bytes: u64,
}

impl Drop for TraceCapture {
    /// Best-effort finalize so quitting a session with `[c3d]
    /// trace_enabled` still leaves a loadable trace, mirroring
    /// `VideoRecorder`'s own same-shaped `Drop` (`src/recorder.rs`) --
    /// the primary use case this exists for is exactly "boot with
    /// `[c3d] trace` set, play for a while, quit normally", with no
    /// explicit `c3d.trace_stop` call ever in the picture.
    /// `c3d.trace_stop` itself also flushes before dropping its own
    /// `TraceCapture`, so this runs twice on that path -- harmless, a
    /// flush on an already-flushed writer is a no-op.
    fn drop(&mut self) {
        if let Err(e) = self.writer.flush() {
            log::warn!(
                "c3d: trace capture {} failed to flush on drop: {e}",
                self.path.display()
            );
        }
    }
}

/// A snapshot of a live capture's progress -- the reply
/// `c3d.trace_start`/`trace_stop`/`trace_status` all share.
pub struct TraceCaptureStatus {
    pub path: PathBuf,
    pub rings: u64,
    pub blobs: u64,
    pub bytes: u64,
    /// Whether this status describes a [`RollingCapture`] rather than a
    /// [`TraceCapture`] -- when true, `rings`/`blobs`/`bytes` describe
    /// what is currently *buffered* (not yet written to `path`) rather
    /// than what has actually been written there; a
    /// `c3d.trace_snapshot`/`trace_stop` call is what makes `path` real.
    pub rolling: bool,
    /// Rolling only: the byte budget `rolling`'s buffer is bounded to.
    pub rolling_byte_budget: Option<usize>,
}

impl TraceCaptureStatus {
    fn direct(cap: &TraceCapture) -> Self {
        TraceCaptureStatus {
            path: cap.path.clone(),
            rings: cap.rings,
            blobs: cap.blobs,
            bytes: cap.bytes,
            rolling: false,
            rolling_byte_budget: None,
        }
    }
}

/// One doorbell's captured payload, held in memory for
/// [`RollingCapture`] -- the same content [`C3dBoard::try_record_doorbell`]
/// writes straight to a [`TraceWriter`] for `Direct` mode, buffered here
/// instead so older doorbells can be evicted before ever reaching disk.
struct DoorbellRecord {
    context: u32,
    ring_tail: u32,
    ring_bytes: Vec<u8>,
    blobs: Vec<(u32, u32, Vec<u8>)>,
    /// `ring_bytes.len()` plus every blob's payload length, cached at
    /// push time so eviction accounting (`RollingCapture::push`) is O(1)
    /// per record instead of re-summing the blob list on every push.
    bytes: usize,
}

/// A `c3d.trace_start`-with-`rolling_bytes`-armed capture: doorbells
/// accumulate in `records`, oldest evicted first once `bytes` exceeds
/// `byte_budget`, and nothing touches disk until
/// [`C3dBoard::trace_snapshot`] or [`C3dBoard::trace_stop`] replays the
/// currently-buffered records through a fresh [`TraceWriter`]
/// ([`Self::write_to`]). The tradeoff is the opposite of `Direct`
/// (`TraceCapture`): bounded memory instead of an ever-growing file, at
/// the cost of only ever holding "the last `byte_budget` or so" rather
/// than the whole session -- `ClipRing`'s own "rolling last N seconds"
/// tradeoff (`src/gifclip.rs`), just bounded by bytes here rather than
/// emulated time, since no emulated-time reference reaches the doorbell
/// path this reads from (`C3dBoard::doorbell` takes a raw `DeviceHost`,
/// not a clock).
struct RollingCapture {
    caps0: u32,
    path: PathBuf,
    records: std::collections::VecDeque<DoorbellRecord>,
    bytes: usize,
    byte_budget: usize,
    /// Per-context monotonic logical tail, the same bookkeeping
    /// `TraceCapture::logical_tail` does and for the same reason (see
    /// `C3dBoard::try_record_doorbell`'s doc comment) -- kept separately
    /// per mode rather than shared, since a mode switch
    /// (`c3d.trace_start` replacing a running capture with the other
    /// kind) starts both the buffer and the logical numbering fresh.
    logical_tail: std::collections::HashMap<usize, u32>,
}

impl RollingCapture {
    fn new(caps0: u32, path: PathBuf, byte_budget: usize) -> Self {
        RollingCapture {
            caps0,
            path,
            records: std::collections::VecDeque::new(),
            bytes: 0,
            byte_budget,
            logical_tail: std::collections::HashMap::new(),
        }
    }

    fn push(
        &mut self,
        context: u32,
        ring_tail: u32,
        ring_bytes: Vec<u8>,
        blobs: Vec<(u32, u32, Vec<u8>)>,
    ) {
        let bytes = ring_bytes.len() + blobs.iter().map(|(_, _, d)| d.len()).sum::<usize>();
        self.bytes += bytes;
        self.records.push_back(DoorbellRecord {
            context,
            ring_tail,
            ring_bytes,
            blobs,
            bytes,
        });
        // A single record larger than the whole budget still gets kept
        // (evicting it would leave the ring permanently empty for every
        // doorbell after); only records *beyond* the newest are dropped
        // to make room.
        while self.bytes > self.byte_budget && self.records.len() > 1 {
            let Some(oldest) = self.records.pop_front() else {
                break;
            };
            self.bytes -= oldest.bytes;
        }
    }

    /// A status snapshot of what is currently buffered (not what was
    /// last written to disk -- `records`/`bytes` describe the live
    /// ring). O(records), not cached: bounded by `byte_budget` so this
    /// never iterates more than a capture's own configured ceiling,
    /// cheap enough for an on-demand `c3d.trace_status` reply.
    fn status(&self) -> TraceCaptureStatus {
        let blobs = self.records.iter().map(|r| r.blobs.len() as u64).sum();
        TraceCaptureStatus {
            path: self.path.clone(),
            rings: self.records.len() as u64,
            blobs,
            bytes: self.bytes as u64,
            rolling: true,
            rolling_byte_budget: Some(self.byte_budget),
        }
    }

    /// Replay every currently-buffered record through a fresh
    /// `TraceWriter` at `path`, in original order -- the file a
    /// `c3d.trace_snapshot`/`trace_stop` call produces. Does not touch
    /// `self.records`: a snapshot does not stop the roll.
    fn write_to(&self, path: &std::path::Path) -> io::Result<TraceCaptureStatus> {
        crate::paths::ensure_parent(path)?;
        let file = File::create(path)?;
        let mut writer = TraceWriter::new(BufWriter::new(file));
        writer.header(proto::PROTOCOL_VERSION, self.caps0)?;
        let (mut rings, mut blobs, mut bytes) = (0u64, 0u64, 0u64);
        for record in &self.records {
            for (space, address, data) in &record.blobs {
                writer.blob(*space, *address, data)?;
                blobs += 1;
                bytes += data.len() as u64;
            }
            writer.ring(record.context, record.ring_tail, &record.ring_bytes)?;
            rings += 1;
            bytes += record.ring_bytes.len() as u64;
        }
        writer.flush()?;
        Ok(TraceCaptureStatus {
            path: path.to_path_buf(),
            rings,
            blobs,
            bytes,
            rolling: true,
            rolling_byte_budget: Some(self.byte_budget),
        })
    }
}

/// A `c3d.trace_start`-with-`trigger_texture`-armed-but-not-yet-running
/// capture: nothing is buffered or written until texture `texture_id` is
/// bound (`TEX_BIND`) on any context, at which point
/// [`C3dBoard::check_trace_trigger`] resolves this into an ordinary
/// `trace_start` call with the `path`/`rolling_bytes` this was armed
/// with. `TEX_BIND` is pure state-machine bookkeeping with no
/// [`RenderOp`] of its own, so there is no render-op stream to hook --
/// the cheapest signal is asking each context's own `texture_units`
/// whether any of them now names the target id, checked once per
/// doorbell after `submit_with` has applied it.
struct PendingTrigger {
    texture_id: u32,
    path: Option<PathBuf>,
    rolling_bytes: Option<u64>,
}

/// The `[old_head, new_tail)` span of `ring` the guest appended this
/// doorbell, unwrapped into one flat byte vector -- wrapping once if the
/// span crosses the circular buffer's physical end, matching
/// `RingCursor`'s own `occupied` computation (`ring.rs`) for the same
/// head/tail pair. `% len` defends against a guest-supplied
/// `old_head`/`new_tail` that strayed outside `[0, len)` before decode
/// had a chance to reject it -- a diagnostic capture must never panic on
/// input the board itself merely logs and moves on from.
fn appended_ring_bytes(ring: &[u8], old_head: u32, new_tail: u32) -> Vec<u8> {
    let len = ring.len();
    if len == 0 {
        return Vec::new();
    }
    let head = old_head as usize % len;
    let tail = new_tail as usize % len;
    if tail >= head {
        ring[head..tail].to_vec()
    } else {
        let mut v = ring[head..].to_vec();
        v.extend_from_slice(&ring[..tail]);
        v
    }
}

/// Fits a C3D board on the Zorro chain and applies `[c3d]`'s boot-time
/// trace-capture settings, for `crate::emulator`'s machine construction.
/// Kept here, rather than inline at that call site, so the board owns its
/// own wiring and `build_machine_inner` carries one call.
///
/// The caller has already decided the board is wanted ([`C3dConfig::enabled`]);
/// this does not re-check it.
///
/// Returns the Zorro slot the board took, which the caller logs: the
/// "board on the Zorro chain" line belongs to the `copperline::emulator`
/// log target like every other board's, and `RUST_LOG` filters built
/// around that target (`tests/c3d.rs`'s own included) would not see it
/// from here.
pub(crate) fn attach(
    zorro: &mut crate::zorro::ZorroChain,
    devices: &mut Vec<crate::zorro_device::BoardDevice>,
    cfg: &crate::config::C3dConfig,
) -> anyhow::Result<usize> {
    // The whole Zorro III window; not yet a `[c3d]` config knob (see this
    // module's doc comment). A legal Zorro III size -- BoardSpec::validate
    // checks this the same as any other board.
    const WINDOW_BYTES: u32 = 0x0200_0000; // 32 MiB
    let slot = devices.len();
    zorro.add_board(crate::zorro::BoardSpec::c3d(slot, WINDOW_BYTES as usize))?;
    let mut board = C3dBoard::with_masked_caps(WINDOW_BYTES, cfg.mask_caps);
    // `[c3d] trace`/`trace_enabled`/`trace_rolling_bytes`/
    // `trace_trigger_texture`: see `docs/internals/c3d.md`'s "Trace
    // capture" and `C3dBoard::trace_start`'s doc comment for the full
    // fallback chain `trace` alone (no `trace_enabled`) participates in
    // without auto-starting anything itself. `trace_trigger_texture` set
    // means "wait", so it takes precedence over starting immediately --
    // the two are the same "now vs. deferred" choice `c3d.trace_start`'s
    // own `trigger_texture` parameter makes at the control layer.
    if let Some(path) = &cfg.trace {
        board.set_configured_trace_path(path.clone());
    }
    if cfg.trace_enabled {
        if let Some(texture_id) = cfg.trace_trigger_texture {
            board.trace_arm_on_texture_bind(texture_id, None, cfg.trace_rolling_bytes);
        } else if let Err(e) = board.trace_start(None, cfg.trace_rolling_bytes) {
            log::warn!("c3d: [c3d] trace_enabled capture failed to start: {e}");
        }
    }
    devices.push(crate::zorro_device::BoardDevice::C3d(Box::new(board)));
    Ok(slot)
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
        let caps0 = CAPS0 & !mask_caps;
        let programmable = caps0 & proto::CAP_PROGRAMMABLE != 0;
        let contexts: Vec<ContextSlot> = (0..CONTEXT_COUNT as u32)
            .map(|i| ContextSlot::new(i, programmable))
            .collect();
        C3dBoard {
            window_bytes,
            aperture_offset,
            caps0,
            aperture: vec![0u8; aperture_len],
            control: CONTROL_ENABLE,
            irq_status: 0,
            irq_enable: 0,
            contexts,
            activity: false,
            share_groups: std::collections::HashMap::new(),
            pending_dma: Vec::new(),
            renderer: None,
            renderer_failed: false,
            capture: None,
            configured_trace_path: None,
            pending_trigger: None,
        }
    }

    fn aperture_size(&self) -> u32 {
        self.window_bytes - self.aperture_offset
    }

    /// The decoder configuration this milestone's `CAPS0` promises --
    /// kept as one function so the register value and the decoder's own
    /// idea of what is legal can never drift apart.
    fn device_config(&self) -> DeviceConfig {
        let programmable = self.caps0 & proto::CAP_PROGRAMMABLE != 0;
        DeviceConfig {
            guestmem: self.caps0 & proto::CAP_GUESTMEM != 0,
            transform: self.caps0 & proto::CAP_TRANSFORM != 0,
            multitexture: self.caps0 & proto::CAP_MULTITEXTURE != 0,
            surface_guestaddr: self.caps0 & proto::CAP_SURFACE_GUESTADDR != 0,
            programmable,
            shared_contexts: self.caps0 & proto::CAP_SHARED_CONTEXTS != 0,
            aperture_size: self.aperture_size(),
            // `CAP_PROGRAMMABLE` requires `MAX_TEXTURE_UNITS >= 8` (spec
            // commit 1ec7e4a, matching GLES2's own `MAX_TEXTURE_IMAGE_
            // UNITS`/`MAX_COMBINED_TEXTURE_IMAGE_UNITS` minimum); this is
            // the ring decoder's own bounds check (`unit >= self.config.
            // max_texture_units`), kept consistent with the register
            // read and with `ContextSlot::new`'s `Limits::texture_units`
            // so all three never drift apart.
            max_texture_units: if programmable {
                8
            } else {
                DeviceConfig::default().max_texture_units
            },
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
            // `device_config().max_texture_units` is itself `>= 8`
            // whenever `CAP_PROGRAMMABLE` is set (spec commit 1ec7e4a) --
            // Copperline's real fitted CAPS0 always sets both bits
            // together, so that value is reached here unconditionally in
            // practice; requesting `CAP_PROGRAMMABLE` with `CAP_MULTITEXTURE`
            // masked off singly is a spec-illegal device configuration
            // this decoder makes no attempt to handle specially (see the
            // board.rs test suite: that exact combination is no longer
            // exercised singly, by design, not by oversight).
            MAX_TEXTURE_UNITS => {
                if self.caps0 & proto::CAP_MULTITEXTURE != 0 {
                    self.device_config().max_texture_units
                } else {
                    1
                }
            }
            MAX_TEXTURES => self.device_config().max_textures,
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
            // `0` without `CAP_PROGRAMMABLE`, per the spec's tier table --
            // same guarded-arm shape as the transform-tier limits above.
            MAX_SHADERS
            | MAX_PROGRAMS
            | MAX_BUFFERS
            | MAX_FRAMEBUFFERS
            | MAX_RENDERBUFFERS
            | MAX_VERTEX_ATTRIBS
            | MAX_VERTEX_UNIFORM_VECTORS
            | MAX_FRAGMENT_UNIFORM_VECTORS
            | MAX_VARYING_VECTORS
            | MAX_COLOR_ATTACHMENTS
            | MAX_RENDERBUFFER_SIZE
            | MAX_SAMPLES
            | PROGRAM_BINARY_FORMAT
            | SHADER_LANGS_SUPPORTED
                if self.caps0 & proto::CAP_PROGRAMMABLE == 0 =>
            {
                0
            }
            MAX_SHADERS => self.device_config().max_shaders,
            MAX_PROGRAMS => self.device_config().max_programs,
            MAX_BUFFERS => self.device_config().max_buffers,
            MAX_FRAMEBUFFERS => self.device_config().max_framebuffers,
            MAX_RENDERBUFFERS => self.device_config().max_renderbuffers,
            // Like MAX_LIGHTS etc. above, reports the same value every
            // per-context `state::Limits::default()` enforces.
            MAX_VERTEX_ATTRIBS => state::Limits::default().max_vertex_attribs as u32,
            // Fixed values comfortably above the spec's stated minimums
            // (128/16/8) -- not measured against `wgpu`/`naga`'s actual
            // per-adapter limits, which are always far higher for any
            // backend this board runs on; see `proto::GLOBAL_MAX_SHADERS`'s
            // doc comment for why there is nothing to measure against.
            MAX_VERTEX_UNIFORM_VECTORS => 256,
            MAX_FRAGMENT_UNIFORM_VECTORS => 64,
            MAX_VARYING_VECTORS => 16,
            // Like MAX_VERTEX_ATTRIBS above, reports the same value every
            // per-context `state::Limits::default()` enforces -- real
            // multiple-render-target support (`DRAW_BUFFERS`/`READ_BUFFER`),
            // not just the attachment-list shape.
            MAX_COLOR_ATTACHMENTS => state::Limits::default().max_color_attachments as u32,
            MAX_RENDERBUFFER_SIZE => self.device_config().max_renderbuffer_size,
            // Like MAX_COLOR_ATTACHMENTS above, reports the same value
            // every per-context `state::Limits::default()` enforces --
            // real multisample storage (`RENDERBUFFER_STORAGE_COLOR_
            // MULTISAMPLE`/`_DEPTH_STENCIL_MULTISAMPLE`, `BLIT_FRAMEBUFFER`
            // resolve), not just an advertised number. `2` is the ceiling
            // (see `state::Limits::max_samples`'s doc comment for why no
            // real backend can honestly advertise more under this spec's
            // "every integer up to the limit" rule).
            MAX_SAMPLES => state::Limits::default().max_samples as u32,
            // Like MAX_SAMPLES above, reports the same value every
            // per-context `state::Limits::default()` enforces -- real
            // PROGRAM_BINARY/PROGRAM_GET_BINARY support, not just an
            // advertised id. See `state::Limits::program_binary_format`'s
            // doc comment for why `0x4350_3031` is independently
            // hardcoded here rather than referencing `render.rs`.
            PROGRAM_BINARY_FORMAT => state::Limits::default().program_binary_format,
            // Bit 0 (GLSL ES 1.00): always set once CAP_PROGRAMMABLE is on
            // -- the only dialect this device's GLSL frontend (glsl_lang +
            // naga::front::glsl) accepts today. Bit 1 (GLSL ES 3.00): not
            // set -- no `#version 300 es` support exists in the compile
            // pipeline yet (the point of this register's existence: a
            // device may legitimately not have it).
            SHADER_LANGS_SUPPORTED => 0b01,
            // `0` without `CAP_SHARED_CONTEXTS` -- its own independent
            // guard, not folded into the `CAP_PROGRAMMABLE` block above:
            // cross-context sharing is independent of the programmable
            // tier, per the spec.
            MAX_SYNC_OBJECTS if self.caps0 & proto::CAP_SHARED_CONTEXTS == 0 => 0,
            MAX_SYNC_OBJECTS => self.device_config().max_sync_objects,
            // Deliberately NOT gated by `CAP_PROGRAMMABLE`, unlike every
            // other register in this guarded block above: this device
            // draws every point at one pixel regardless of `POINT_SIZE`
            // (baseline/`CAP_TRANSFORM`) or `gl_PointSize`
            // (`CAP_PROGRAMMABLE`)'s requested value -- a spec-legal
            // `[1.0, 1.0]` `ALIASED_POINT_SIZE_RANGE` that governs both
            // tiers alike, so it must read the same whether or not
            // `CAP_PROGRAMMABLE` is set.
            ALIASED_POINT_SIZE_MIN => 1.0f32.to_bits(),
            ALIASED_POINT_SIZE_MAX => 1.0f32.to_bits(),
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
        let programmable = self.caps0 & proto::CAP_PROGRAMMABLE != 0;
        for (i, slot) in self.contexts.iter_mut().enumerate() {
            *slot = ContextSlot::new(i as u32, programmable);
        }
        self.share_groups.clear();
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
                if slot.ctx.awaiting_sync {
                    v |= CTX_STATUS_AWAITING_SYNC;
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
            SHARE_GROUP => slot.ctx.state.share_group(),
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
                    // Freed: every PRIVATE object this context held is
                    // gone (wholesale-replacing its `State` drops its
                    // `Rc` handles to the five shareable kinds and to
                    // sync objects). A SHARED object survives exactly as
                    // long as `Rc::strong_count` stays above zero across
                    // this drop, i.e. as long as another live member of
                    // the same group still holds its own clone of the
                    // same `Rc` -- correct for free, no explicit "am I
                    // the last member" bookkeeping needed. See
                    // `state::State::set_share_group`'s doc comment.
                    self.contexts[n] =
                        ContextSlot::new(n as u32, self.caps0 & proto::CAP_PROGRAMMABLE != 0);
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
            RING_TAIL => {
                self.doorbell(n, value, host);
                self.retry_awaiting_sync_contexts(host);
            }
            ERROR_ACK => {
                self.contexts[n].ctx.error_ack();
            }
            GL_ERROR_ACK => {
                self.contexts[n].ctx.state.ack_gl_error();
            }
            FENCE_IRQ_TARGET => self.contexts[n].fence_irq_target = value,
            // `CAP_SHARED_CONTEXTS`-gated (ignored, stays private, without
            // it -- per the spec's CAPS0 bit-8 row) and "written only
            // while this context has decoded no command since its last
            // ALLOC transition" -- the same immutability shape `RING_SIZE`
            // already enforces above, just against `ring_head == 0`
            // (nothing consumed yet) rather than `!enable`. Falling through
            // to the `_` arm's no-op when the guard fails is exactly a
            // discarded write, the same as every other gated case here.
            SHARE_GROUP
                if self.caps0 & proto::CAP_SHARED_CONTEXTS != 0
                    && self.contexts[n].ctx.ring_head == 0 =>
            {
                if value == 0 {
                    self.contexts[n].ctx.state.set_private();
                } else {
                    let stores = self.share_groups.entry(value).or_default();
                    let (textures, shaders, programs, buffers, renderbuffers, sync_objects) =
                        stores.resolve();
                    self.contexts[n].ctx.state.set_share_group(
                        value,
                        textures,
                        shaders,
                        programs,
                        buffers,
                        renderbuffers,
                        sync_objects,
                    );
                }
            }
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
        // `ring_head` as it stands *before* `submit_with` below moves it:
        // the start of whatever the guest appended since its last
        // doorbell, for `record_doorbell`'s capture of exactly that span
        // (`TraceCapture`'s doc comment). Irrelevant, and not read again,
        // when no capture is armed.
        let old_head = self.contexts[n].ctx.ring_head;

        let ring_base = self.contexts[n].ctx.ring_base;
        let ring_size_reg = self.contexts[n].ctx.ring_size;
        let ring_len = ring_size_reg & 0x7FFF_FFFF;
        // Bit 31 of RING_SIZE: RING_BASE names a guest address rather
        // than an aperture offset (spec, `RING_BASE`/`RING_SIZE`; see the
        // module doc comment's "Ring in guest memory"). Gated on the
        // *effective* CAPS0, not the constant, so a masked-off
        // CAP_GUESTMEM behaves as if the ring can only ever be
        // aperture-backed -- and, critically, the aperture-read branch
        // below must never run for a bit-31 ring: `ring_base` is a guest
        // address, not an aperture offset, and reading the aperture at
        // that number would decode whatever unrelated bytes happen to
        // live there as commands, not "nothing". A guest-address ring
        // with `CAP_GUESTMEM` masked off is therefore handled by its own
        // early return below, before this two-way `if` can be reached at
        // all, rather than folded into it as a third arm of the same
        // condition.
        let ring_is_guest = ring_size_reg & 0x8000_0000 != 0;
        if ring_is_guest && !self.device_config().guestmem {
            // No register-level error exists for this: RING_BASE/
            // RING_SIZE are ordinary registers with no error latch of
            // their own (only ring *commands*, decoded from ring bytes
            // this device never reaches here, raise ERROR_CODE -- see
            // the spec's error table), and a guest-address ring is not a
            // ref either, so E_BAD_REF does not apply. RING_TAIL still
            // echoes the write, matching every other RW register's
            // ordinary behaviour, but nothing is decoded: RING_HEAD does
            // not advance, no fence completes, no error latches.
            self.contexts[n].ctx.ring_tail = new_tail;
            self.update_irq_status(n);
            return;
        }
        let ring_bytes = if ring_is_guest {
            let mut buf = vec![0u8; ring_len as usize];
            host.dma_read(ring_base, &mut buf);
            buf
        } else {
            self.read_aperture_range(ring_base, ring_len)
        };

        let config = self.device_config();
        let mut ops: Vec<RenderOp<'_>> = Vec::new();
        // `CALL`'s buffer-fetch closure: the same two primitives this
        // function already used above to fetch the ring itself when it
        // lives in guest memory, one level down. `ring.rs`'s `validate_ref`
        // has already checked `data`'s alignment, aperture bound and
        // `CAP_GUESTMEM` gating as part of ordinary decoding, so this
        // closure only needs to resolve bytes, not re-validate. Borrows
        // only `self.aperture` (disjoint from `self.contexts`, which
        // `submit_with` below needs `&mut` access to) and `host`, which is
        // already a separate parameter.
        let aperture = &self.aperture;
        // Tee for a live trace capture: every ref this doorbell actually
        // resolves, `CALL` buffers here and ordinary by-reference op
        // payloads (textures, vertex/index arrays, palettes...) via
        // `ApertureMemory` below, collected locally and only touching
        // `self.capture` once both are done -- `self.contexts` needs
        // `&mut` for `submit_with` right below, so this closure (like the
        // one it replaces) must not borrow `self` itself.
        let capturing = self.capture.is_some();
        let mut captured_blobs: Vec<(u32, u32, Vec<u8>)> = Vec::new();
        let mut fetch_call_buffer = |r: proto::Ref| -> Option<Vec<u8>> {
            let bytes = match r.space {
                proto::RefSpace::Aperture => {
                    let start = r.address as usize;
                    let end = start.saturating_add(r.length as usize).min(aperture.len());
                    if start >= aperture.len() || start >= end {
                        Vec::new()
                    } else {
                        aperture[start..end].to_vec()
                    }
                }
                proto::RefSpace::Guest => {
                    let mut buf = vec![0u8; r.length as usize];
                    host.dma_read(r.address, &mut buf);
                    buf
                }
            };
            if capturing && !bytes.is_empty() {
                let space = match r.space {
                    proto::RefSpace::Aperture => 0,
                    proto::RefSpace::Guest => 1,
                };
                captured_blobs.push((space, r.address, bytes.clone()));
            }
            Some(bytes)
        };
        let mut call_storage: Vec<Vec<u8>> = Vec::new();
        self.contexts[n].ctx.submit_with(
            &ring_bytes,
            new_tail,
            &config,
            &mut ops,
            &mut call_storage,
            &mut fetch_call_buffer,
        );
        self.check_trace_trigger(n);

        if !ops.is_empty() {
            let queued_before = self.pending_dma.len();
            if let Some(renderer) =
                Self::ensure_renderer(&mut self.renderer, &mut self.renderer_failed)
            {
                let mut mem = ApertureMemory {
                    aperture: &mut self.aperture,
                    pending_dma: &mut self.pending_dma,
                    host,
                    guest_scratch: Vec::new(),
                    capture: capturing.then_some(&mut captured_blobs),
                };
                let errors = renderer.execute(&ops, &mut mem);
                for e in &errors {
                    if let RenderError::GlError(gl_err) = e {
                        // Not a render failure -- a GL-semantic error the
                        // op only had the GPU-side state to detect (see
                        // `RenderError::GlError`'s doc comment). Applied
                        // to the submitting context's own GL_ERROR, same
                        // first-error-wins semantics as every other
                        // `raise_gl_error` call site.
                        self.contexts[n].ctx.state.raise_gl_error(*gl_err);
                    } else {
                        log::warn!("c3d: context {n}: render error: {e}");
                    }
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
        if capturing {
            self.record_doorbell(n, old_head, new_tail, &ring_bytes, captured_blobs);
        }
        self.update_irq_status(n);
    }

    /// `WAIT_SYNC`'s device-side wait (`CAP_SHARED_CONTEXTS`): a context
    /// stalled there needs no guest CPU polling or interrupt round-trip
    /// to resume -- the device itself retries it, here, after every
    /// top-level doorbell (called from `write_context`'s `RING_TAIL`
    /// arm, *not* from [`Self::doorbell`] itself, so a retry's own calls
    /// back into `doorbell` cannot recursively re-trigger another retry
    /// pass). Re-attempts every other currently-`alloc`+`enable`+
    /// `awaiting_sync` context by replaying its last-known `RING_TAIL`
    /// against its own (unchanged) `ring_head` -- exactly the ordinary
    /// doorbell path, which naturally either advances past the `WAIT_
    /// SYNC` now (if its dependency has since been signalled) or stalls
    /// again at the identical position (if not). Looped until a pass
    /// makes no further progress, bounded by `contexts.len() + 1`
    /// passes: a `WAIT_SYNC` cycle (context A waiting on a sync only B
    /// can signal, B waiting on one only A can signal) can never
    /// resolve regardless of how many passes run, so a bound is exactly
    /// as correct as an unbounded loop for every satisfiable case and
    /// strictly safer for an unsatisfiable one.
    fn retry_awaiting_sync_contexts(&mut self, host: &DeviceHost) {
        for _ in 0..=self.contexts.len() {
            let mut made_progress = false;
            for m in 0..self.contexts.len() {
                if self.contexts[m].alloc
                    && self.contexts[m].enable
                    && self.contexts[m].ctx.awaiting_sync
                {
                    let retry_tail = self.contexts[m].ctx.ring_tail;
                    self.doorbell(m, retry_tail, host);
                    if !self.contexts[m].ctx.awaiting_sync {
                        made_progress = true;
                    }
                }
            }
            if !made_progress {
                break;
            }
        }
    }

    /// `[c3d] trace`'s resolved default path, if the config set one --
    /// what `trace_start(None, None)` falls back to.
    pub fn set_configured_trace_path(&mut self, path: PathBuf) {
        self.configured_trace_path = Some(path);
    }

    /// Arm a live trace capture, replacing any capture already running
    /// (matching `ui_wave_start`'s own "replace, don't refuse" stance for
    /// a file-backed diagnostic) and disarming any pending trigger (a
    /// fresh explicit start wins over a trigger armed earlier). `path`
    /// wins outright when given; `None` falls back to `[c3d] trace`'s
    /// configured default (`self.configured_trace_path`, set once at
    /// boot by `set_configured_trace_path`) and only then to a
    /// timestamped file in the `[paths]` traces directory
    /// ([`crate::paths::c3d_trace_file`]) -- the same fallback chain
    /// `[c3d] trace_enabled`'s boot-time auto-start
    /// (`emulator::build_machine`) and a bare `c3d.trace_start` with no
    /// `path` argument both resolve through.
    ///
    /// `rolling_bytes`: `None` streams straight to `path` immediately,
    /// unbounded (`Direct`, the original M3 shape -- `path` is created
    /// and the `C3DT` header written right here). `Some(budget)` instead
    /// arms a bounded in-memory [`RollingCapture`] ([`CaptureMode`]'s own
    /// doc comment) -- `path` is resolved and remembered but nothing is
    /// created on disk until [`Self::trace_snapshot`]/[`Self::trace_stop`]
    /// replays the buffered records through it. Both modes write the
    /// `C3DT` header against this board's *actual* capability mask
    /// (`self.caps0`, not the [`CAPS0`] constant -- a capture taken under
    /// `[c3d] mask_caps` must replay against that reduced set, not
    /// whatever the board could do unmasked) and the protocol version it
    /// implements, so a later `copperline-c3d-trace run`/`compare` knows
    /// up front whether it can even attempt replay.
    pub fn trace_start(
        &mut self,
        path: Option<PathBuf>,
        rolling_bytes: Option<u64>,
    ) -> io::Result<()> {
        self.pending_trigger = None;
        let path = path
            .or_else(|| self.configured_trace_path.clone())
            .unwrap_or_else(crate::paths::c3d_trace_file);
        self.capture = Some(match rolling_bytes {
            None => {
                crate::paths::ensure_parent(&path)?;
                let file = File::create(&path)?;
                let mut writer = TraceWriter::new(BufWriter::new(file));
                writer.header(proto::PROTOCOL_VERSION, self.caps0)?;
                CaptureMode::Direct(TraceCapture {
                    writer,
                    path,
                    logical_tail: std::collections::HashMap::new(),
                    rings: 0,
                    blobs: 0,
                    bytes: 0,
                })
            }
            Some(budget) => {
                CaptureMode::Rolling(RollingCapture::new(self.caps0, path, budget as usize))
            }
        });
        Ok(())
    }

    /// Stop and finalise the capture, if one is running, returning its
    /// final status. For `Direct`, `cap` dropping at the end of this
    /// function flushes the underlying file via `TraceCapture`'s own
    /// `Drop` impl (reading the status fields first rather than
    /// partially moving them out of `cap`, which `Drop` forbids), so the
    /// result is immediately safe to hand to `copperline-c3d-trace`. For
    /// `Rolling`, this is the same replay [`Self::trace_snapshot`] does
    /// -- "stop" always leaves a file on disk, matching `Direct`'s own
    /// guarantee, rather than silently discarding whatever was still
    /// buffered.
    pub fn trace_stop(&mut self) -> Option<TraceCaptureStatus> {
        match self.capture.take()? {
            CaptureMode::Direct(cap) => Some(TraceCaptureStatus::direct(&cap)),
            CaptureMode::Rolling(roll) => match roll.write_to(&roll.path.clone()) {
                Ok(status) => Some(status),
                Err(e) => {
                    log::warn!(
                        "c3d: rolling trace capture {} failed to finalise on stop: {e}",
                        roll.path.display()
                    );
                    None
                }
            },
        }
    }

    /// Replay a [`RollingCapture`]'s currently-buffered records to disk
    /// without stopping the roll -- the "grab what's in the ring right
    /// now" operation `c3d.trace_snapshot` exists for, e.g. right after
    /// noticing a bug on screen. `path` defaults to the capture's own
    /// resolved path the same way `trace_start`'s own fallback chain
    /// does (so repeated snapshots with no `path` argument overwrite the
    /// same file unless the caller wants otherwise). For a `Direct`
    /// capture this just flushes the already-open file and reports its
    /// current progress -- there is nothing bounded to snapshot, but the
    /// call is still meaningful rather than an error. `None` means no
    /// capture is armed at all.
    pub fn trace_snapshot(
        &mut self,
        path: Option<PathBuf>,
    ) -> io::Result<Option<TraceCaptureStatus>> {
        let Some(mode) = self.capture.as_mut() else {
            return Ok(None);
        };
        match mode {
            CaptureMode::Direct(cap) => {
                cap.writer.flush()?;
                Ok(Some(TraceCaptureStatus::direct(cap)))
            }
            CaptureMode::Rolling(roll) => {
                let target = path.unwrap_or_else(|| roll.path.clone());
                Ok(Some(roll.write_to(&target)?))
            }
        }
    }

    /// The running capture's progress, if one is armed -- for `Rolling`,
    /// what is currently *buffered* ([`RollingCapture::status`]), not
    /// what was last written to disk (nothing may have been snapshotted
    /// yet at all).
    pub fn trace_status(&self) -> Option<TraceCaptureStatus> {
        self.capture.as_ref().map(|mode| match mode {
            CaptureMode::Direct(cap) => TraceCaptureStatus::direct(cap),
            CaptureMode::Rolling(roll) => roll.status(),
        })
    }

    /// Arm a trigger: no capture starts (buffered or on disk) until
    /// texture `texture_id` is bound on any context, at which point this
    /// resolves into an ordinary [`Self::trace_start`] call with `path`/
    /// `rolling_bytes`. Replaces any trigger armed earlier; does not
    /// touch a capture already running (matching `trace_start`'s own
    /// "an explicit start always wins" stance -- arming a trigger while
    /// already capturing just means "and also do this if it stops and
    /// you want to re-trigger", not "stop now").
    pub fn trace_arm_on_texture_bind(
        &mut self,
        texture_id: u32,
        path: Option<PathBuf>,
        rolling_bytes: Option<u64>,
    ) {
        self.pending_trigger = Some(PendingTrigger {
            texture_id,
            path,
            rolling_bytes,
        });
    }

    /// The texture id a `c3d.trace_start`/`[c3d] trace_trigger_texture`
    /// trigger is still waiting on, if one is armed and hasn't fired.
    pub fn trace_trigger_pending(&self) -> Option<u32> {
        self.pending_trigger.as_ref().map(|t| t.texture_id)
    }

    /// Checked once per doorbell on context `n`, after `submit_with` has
    /// applied its commands to `state` -- see [`PendingTrigger`]'s doc
    /// comment for why this is a state scan rather than a hook on the
    /// render-op stream. A no-op whenever no trigger is armed (the
    /// common case), so this costs nothing on an ordinary doorbell.
    fn check_trace_trigger(&mut self, n: usize) {
        let Some(trigger) = &self.pending_trigger else {
            return;
        };
        let Some(ctx) = self.contexts.get(n) else {
            return;
        };
        let fired = ctx
            .ctx
            .state
            .texture_units
            .iter()
            .any(|u| u.bound_texture == trigger.texture_id);
        if !fired {
            return;
        }
        // `.unwrap()` is safe: `trigger` above borrowed `Some(..)` from
        // the same `self.pending_trigger` this `.take()` reads.
        let trigger = self.pending_trigger.take().unwrap();
        if let Err(e) = self.trace_start(trigger.path, trigger.rolling_bytes) {
            log::warn!(
                "c3d: trace trigger fired (texture {}) but failed to start capture: {e}",
                trigger.texture_id
            );
        }
    }

    /// Flush one doorbell's capture: `BLOB` sections for every ref
    /// `blobs` names (ahead of the `RING` section they belong to, per
    /// `trace.rs`'s "materialised... before the `RING` section that
    /// references them" contract), then one `RING` section for the
    /// `[old_head, new_tail)` span of `ring_bytes` the guest just
    /// appended -- or, for a `Rolling` capture, the same content pushed
    /// into its in-memory ring instead of written straight to a file. A
    /// write failure (disk full, say -- `Rolling` can only fail this way
    /// at an eventual snapshot/stop, never here, since nothing touches
    /// disk per doorbell) disarms the capture rather than panicking or
    /// silently losing the rest of the session -- matching
    /// `ensure_renderer`'s "no GPU adapter" stance: a diagnostic
    /// capability degrading gracefully, not a guest-visible failure.
    fn record_doorbell(
        &mut self,
        n: usize,
        old_head: u32,
        new_tail: u32,
        ring_bytes: &[u8],
        blobs: Vec<(u32, u32, Vec<u8>)>,
    ) {
        if let Err(e) = self.try_record_doorbell(n, old_head, new_tail, ring_bytes, blobs) {
            log::warn!("c3d: trace capture write failed, stopping capture: {e}");
            self.capture = None;
        }
    }

    fn try_record_doorbell(
        &mut self,
        n: usize,
        old_head: u32,
        new_tail: u32,
        ring_bytes: &[u8],
        blobs: Vec<(u32, u32, Vec<u8>)>,
    ) -> io::Result<()> {
        let Some(mode) = self.capture.as_mut() else {
            return Ok(());
        };
        // `ring_tail` recorded below is **not** the live board's own
        // circular offset: `copperline-c3d-trace`'s replay model treats
        // a context's ring as an ever-growing linear buffer that never
        // wraps (`TraceCapture`'s doc comment), so this tracks one
        // monotonic logical tail per context and translates the real
        // (circular) span into that flat numbering -- the bytes
        // themselves are identical either way, only the offsets they are
        // addressed by differ.
        let appended = appended_ring_bytes(ring_bytes, old_head, new_tail);
        match mode {
            CaptureMode::Direct(cap) => {
                for (space, address, data) in &blobs {
                    cap.writer.blob(*space, *address, data)?;
                    cap.blobs += 1;
                    cap.bytes += data.len() as u64;
                }
                let old_logical = *cap.logical_tail.get(&n).unwrap_or(&0);
                let new_logical = old_logical + appended.len() as u32;
                cap.writer.ring(n as u32, new_logical, &appended)?;
                cap.logical_tail.insert(n, new_logical);
                cap.rings += 1;
                cap.bytes += appended.len() as u64;
            }
            CaptureMode::Rolling(roll) => {
                let old_logical = *roll.logical_tail.get(&n).unwrap_or(&0);
                let new_logical = old_logical + appended.len() as u32;
                roll.push(n as u32, new_logical, appended, blobs);
                roll.logical_tail.insert(n, new_logical);
            }
        }
        Ok(())
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
    /// Tee for a live trace capture: every successful read, appended as
    /// `(space, address, bytes)` -- `0`/`1` matching `TraceWriter::blob`'s
    /// own space flag -- so [`C3dBoard::record_doorbell`] can write them
    /// as `BLOB` sections ahead of this doorbell's own `RING` section.
    /// `None` when no capture is armed, the common case, so an
    /// uncaptured session pays only the one `is_some()` check per read.
    capture: Option<&'a mut Vec<(u32, u32, Vec<u8>)>>,
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
                if let Some(cap) = self.capture.as_deref_mut() {
                    cap.push((0, addr, self.aperture[r.clone()].to_vec()));
                }
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
                if let Some(cap) = self.capture.as_deref_mut() {
                    cap.push((1, addr, self.guest_scratch[..len].to_vec()));
                }
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
        // Unlike the other limits above, `MAX_TEXTURE_UNITS` is no
        // longer simply `state::Limits::default().texture_units`: with
        // `CAP_PROGRAMMABLE` set (as `board()`'s full CAPS0 always is)
        // it must be `>= 8` regardless (spec commit 1ec7e4a) -- see
        // `cap_programmable_requires_at_least_eight_texture_units_and_
        // they_actually_work` for the dedicated coverage of that rule.
        assert!(b.read(greg::MAX_TEXTURE_UNITS, 4, &mut h) >= 8);
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

    /// Spec draft 0.15's `CTX_CONTROL` rule, now also covering
    /// `CAP_SHARED_CONTEXTS`'s `CTX_SHARE_GROUP`: freeing a context
    /// (`ALLOC` cleared) and reallocating it must not carry
    /// `FENCE_COMPLETED`/`FENCE_IRQ_TARGET`/`CTX_SHARE_GROUP` forward --
    /// a stale high fence value from the previous owner would satisfy
    /// the new owner's every fence wait instantly (unsigned
    /// reached-or-passed compare) before any of its work had run, and a
    /// stale share group would silently join the new owner into a
    /// stranger's shared-object namespace it never asked to be part of.
    /// `write_context`'s freeing branch already resets the whole
    /// `ContextSlot` to its default on `ALLOC` 1->0 (every object this
    /// context held is destroyed, per `CTX_CONTROL`'s own doc), which
    /// already zeroes all three well before any later realloc --
    /// behaviourally equivalent to zeroing on the 0->1 transition itself,
    /// since a context can only reach `ALLOC=1` a second time by first
    /// having passed back through `ALLOC=0`. This test pins that down as
    /// an explicit, spec-cited invariant rather than an incidental
    /// side effect of the freeing branch's existing full wipe.
    #[test]
    fn fence_state_and_share_group_do_not_survive_a_free_and_realloc() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        let ctl = CONTEXT_PAGE_BASE + creg::CONTROL;
        let ring_off = 0u32;
        let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;
        let cmds = [opcode_len(proto::OP_FENCE, 2), 5];
        let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        let ring_window_off = b.aperture_offset + ring_off;
        for (i, b_) in bytes.iter().enumerate() {
            b.write(ring_window_off + i as u32, 1, *b_ as u32, &mut h);
        }
        b.write(CONTEXT_PAGE_BASE + creg::RING_BASE, 4, ring_off, &mut h);
        b.write(CONTEXT_PAGE_BASE + creg::RING_SIZE, 4, 0x1000, &mut h);
        b.write(ctl, 4, CTX_CONTROL_ALLOC | CTX_CONTROL_ENABLE, &mut h);
        b.write(CONTEXT_PAGE_BASE + creg::FENCE_IRQ_TARGET, 4, 42, &mut h);
        // Written before the ring's first command decodes below, so
        // this join is honoured -- same immutability window `CTX_SHARE_
        // GROUP` itself enforces.
        b.write(CONTEXT_PAGE_BASE + creg::SHARE_GROUP, 4, 7, &mut h);
        b.write(
            CONTEXT_PAGE_BASE + creg::RING_TAIL,
            4,
            bytes.len() as u32,
            &mut h,
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::FENCE_COMPLETED, 4, &mut h),
            5,
            "sanity: the fence actually completed before freeing"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::SHARE_GROUP, 4, &mut h),
            7,
            "sanity: the join actually took before freeing"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::FENCE_IRQ_TARGET, 4, &mut h),
            42
        );

        // Free, then reallocate the same page.
        b.write(ctl, 4, 0, &mut h);
        b.write(ctl, 4, CTX_CONTROL_ALLOC | CTX_CONTROL_ENABLE, &mut h);

        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::FENCE_COMPLETED, 4, &mut h),
            0,
            "FENCE_COMPLETED must not carry the previous owner's progress"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::FENCE_IRQ_TARGET, 4, &mut h),
            0,
            "FENCE_IRQ_TARGET must be disarmed for the new owner"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::SHARE_GROUP, 4, &mut h),
            0,
            "a reallocated page must not silently inherit the previous owner's share group"
        );
    }

    /// The other half of spec draft 0.15's `CTX_CONTROL` rule:
    /// `CTX_CONTROL.RESET` (bit 2) is explicitly excluded -- a live
    /// owner's mid-life reset must leave `FENCE_COMPLETED`/
    /// `FENCE_IRQ_TARGET` untouched, since zeroing them would strand a
    /// wait on an already-emitted fence ID until the next fence happened
    /// to land.
    #[test]
    fn ctx_control_reset_leaves_fence_state_untouched() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        let ctl = CONTEXT_PAGE_BASE + creg::CONTROL;
        let ring_off = 0u32;
        let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;
        let cmds = [opcode_len(proto::OP_FENCE, 2), 9];
        let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        let ring_window_off = b.aperture_offset + ring_off;
        for (i, b_) in bytes.iter().enumerate() {
            b.write(ring_window_off + i as u32, 1, *b_ as u32, &mut h);
        }
        b.write(CONTEXT_PAGE_BASE + creg::RING_BASE, 4, ring_off, &mut h);
        b.write(CONTEXT_PAGE_BASE + creg::RING_SIZE, 4, 0x1000, &mut h);
        b.write(ctl, 4, CTX_CONTROL_ALLOC | CTX_CONTROL_ENABLE, &mut h);
        b.write(CONTEXT_PAGE_BASE + creg::FENCE_IRQ_TARGET, 4, 7, &mut h);
        b.write(
            CONTEXT_PAGE_BASE + creg::RING_TAIL,
            4,
            bytes.len() as u32,
            &mut h,
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::FENCE_COMPLETED, 4, &mut h),
            9
        );

        b.write(
            ctl,
            4,
            CTX_CONTROL_ALLOC | CTX_CONTROL_ENABLE | CTX_CONTROL_RESET,
            &mut h,
        );

        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::FENCE_COMPLETED, 4, &mut h),
            9,
            "RESET must not strand a wait on an already-emitted fence ID"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::FENCE_IRQ_TARGET, 4, &mut h),
            7,
            "RESET must not disarm a live IRQ target"
        );
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

        // `CAP_PROGRAMMABLE` now requires `MAX_TEXTURE_UNITS >= 8` (spec
        // commit 1ec7e4a), which on its own forces `CAP_MULTITEXTURE`
        // too (that register's pre-existing `> 1` rule) -- a device
        // with `CAP_MULTITEXTURE` masked off but `CAP_PROGRAMMABLE`
        // still set is now spec-illegal, not a configuration a real
        // board could report. Masking both together is the only
        // spec-legal way to simulate "a device without multitexture".
        let mut b = C3dBoard::with_masked_caps(
            0x0200_0000,
            proto::CAP_MULTITEXTURE | proto::CAP_PROGRAMMABLE,
        );
        let mut h = host(&mut mem);
        assert_eq!(b.read(greg::MAX_TEXTURE_UNITS, 4, &mut h), 1);
    }

    /// `MAX_VERTEX_UNIFORM_VECTORS`/`MAX_FRAGMENT_UNIFORM_VECTORS`/
    /// `MAX_VARYING_VECTORS` report fixed values at or above the spec's
    /// stated minimums (128/16/8) when `CAP_PROGRAMMABLE` is set, and
    /// `0` when it isn't -- the same guarded-register shape every other
    /// `CAP_PROGRAMMABLE` limit already follows.
    #[test]
    fn shader_resource_limit_registers_report_fixed_values_above_the_spec_minimums() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        assert!(b.read(greg::MAX_VERTEX_UNIFORM_VECTORS, 4, &mut h) >= 128);
        assert!(b.read(greg::MAX_FRAGMENT_UNIFORM_VECTORS, 4, &mut h) >= 16);
        assert!(b.read(greg::MAX_VARYING_VECTORS, 4, &mut h) >= 8);

        let mut b = C3dBoard::with_masked_caps(0x0200_0000, proto::CAP_PROGRAMMABLE);
        let mut h = host(&mut mem);
        assert_eq!(b.read(greg::MAX_VERTEX_UNIFORM_VECTORS, 4, &mut h), 0);
        assert_eq!(b.read(greg::MAX_FRAGMENT_UNIFORM_VECTORS, 4, &mut h), 0);
        assert_eq!(b.read(greg::MAX_VARYING_VECTORS, 4, &mut h), 0);
    }

    /// `MAX_TEXTURE_UNITS` must be `>= 8` with `CAP_PROGRAMMABLE` set
    /// (spec commit 1ec7e4a), which on its own forces `CAP_MULTITEXTURE`
    /// too -- and the bound must be real, not just the register's own
    /// story: `ACTIVE_UNIT`/`TEX_BIND` at unit 7 must actually decode
    /// (the ring decoder's own `max_texture_units` bound,
    /// `device_config()`) and the per-context state must actually have
    /// 8 units' worth of storage (`ContextSlot::new`'s `Limits`) rather
    /// than panicking or silently going out of bounds past unit 1.
    #[test]
    fn cap_programmable_requires_at_least_eight_texture_units_and_they_actually_work() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        assert!(b.read(greg::MAX_TEXTURE_UNITS, 4, &mut h) >= 8);
        assert_ne!(b.read(greg::CAPS0, 4, &mut h) & proto::CAP_MULTITEXTURE, 0);

        let words = [
            ((proto::OP_ACTIVE_UNIT as u32) << 16) | 2,
            7,
            ((proto::OP_TEX_BIND as u32) << 16) | 3,
            7,
            0,
        ];
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
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::ERROR_CODE, 4, &mut h),
            0,
            "ACTIVE_UNIT/TEX_BIND at unit 7 must decode cleanly on a CAP_PROGRAMMABLE device"
        );
    }

    /// `ALIASED_POINT_SIZE_MIN`/`_MAX` must read `1.0`/`1.0` always --
    /// unlike every other `CAP_PROGRAMMABLE`-era register above, these
    /// are NOT gated by `CAP_PROGRAMMABLE`: they also bound the
    /// baseline/`CAP_TRANSFORM` tier's pre-existing `POINT_SIZE` state,
    /// so they must report the same whether or not `CAP_PROGRAMMABLE`
    /// is set, never `0`.
    #[test]
    fn aliased_point_size_range_reports_one_to_one_regardless_of_cap_programmable() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        assert_eq!(
            b.read(greg::ALIASED_POINT_SIZE_MIN, 4, &mut h),
            1.0f32.to_bits()
        );
        assert_eq!(
            b.read(greg::ALIASED_POINT_SIZE_MAX, 4, &mut h),
            1.0f32.to_bits()
        );

        let mut b = C3dBoard::with_masked_caps(0x0200_0000, proto::CAP_PROGRAMMABLE);
        let mut h = host(&mut mem);
        assert_eq!(
            b.read(greg::ALIASED_POINT_SIZE_MIN, 4, &mut h),
            1.0f32.to_bits(),
            "must not read 0 just because every CAP_PROGRAMMABLE-gated register does"
        );
        assert_eq!(
            b.read(greg::ALIASED_POINT_SIZE_MAX, 4, &mut h),
            1.0f32.to_bits()
        );
    }

    /// `SHADER_LANGS_SUPPORTED` bit 0 (GLSL ES 1.00) must be set whenever
    /// `CAP_PROGRAMMABLE` is -- every device that has the programmable
    /// tier at all accepts the one mandatory dialect -- and the register
    /// must read `0` entirely without the capability, like every other
    /// `CAP_PROGRAMMABLE`-gated register.
    #[test]
    fn shader_langs_supported_reports_glsl_es_1_00_with_cap_programmable() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        assert_eq!(b.read(greg::SHADER_LANGS_SUPPORTED, 4, &mut h), 0b01);

        let mut b = C3dBoard::with_masked_caps(0x0200_0000, proto::CAP_PROGRAMMABLE);
        let mut h = host(&mut mem);
        assert_eq!(
            b.read(greg::SHADER_LANGS_SUPPORTED, 4, &mut h),
            0,
            "must read 0 once CAP_PROGRAMMABLE is masked off"
        );
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

    /// End-to-end `CALL`: the outer ring holds just a single `CALL`
    /// pointing at a buffer placed elsewhere in the aperture; the buffer
    /// holds the same `SURFACE_DEFINE`+`SET_DRAW_SURFACE`+`CLEAR_COLOR`+
    /// `CLEAR`+`SURFACE_READBACK`+`FENCE` stream the plain full-command-
    /// stream test above uses. Asserts the buffer's commands actually took
    /// effect (fence completed, no error, cleared pixels) and that
    /// `RING_HEAD` advanced past the `CALL` in the *outer* ring, not into
    /// the buffer.
    #[test]
    fn call_replays_a_buffer_and_its_commands_take_effect() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);

        const RING_BASE_APERTURE_REL: u32 = 0;
        const RING_SIZE: u32 = 0x1000;
        const BUFFER_APERTURE_REL: u32 = RING_SIZE;
        const SURFACE_APERTURE_REL: u32 = RING_SIZE + 0x1000;
        const SURFACE_W: u32 = 4;
        const SURFACE_H: u32 = 4;
        const SURFACE_STRIDE: u32 = SURFACE_W * 4;

        let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;

        let mut buf_cmds = Vec::<u32>::new();
        buf_cmds.push(opcode_len(proto::OP_SURFACE_DEFINE, 8));
        buf_cmds.extend([
            1,
            SURFACE_W,
            SURFACE_H,
            SURFACE_STRIDE,
            5,
            0,
            SURFACE_APERTURE_REL,
        ]);
        buf_cmds.push(opcode_len(proto::OP_SET_DRAW_SURFACE, 2));
        buf_cmds.push(1);
        buf_cmds.push(opcode_len(proto::OP_CLEAR_COLOR, 5));
        buf_cmds.extend([1.0f32.to_bits(), 0, 0, 1.0f32.to_bits()]);
        buf_cmds.push(opcode_len(proto::OP_CLEAR, 2));
        buf_cmds.push(proto::CLEAR_MASK_COLOR);
        buf_cmds.push(opcode_len(proto::OP_SURFACE_READBACK, 5));
        buf_cmds.extend([0, 0, SURFACE_W, SURFACE_H]);
        buf_cmds.push(opcode_len(proto::OP_FENCE, 2));
        buf_cmds.push(1);
        let buf_bytes: Vec<u8> = buf_cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        let buf_off = b.aperture_offset + BUFFER_APERTURE_REL;
        for (i, b_) in buf_bytes.iter().enumerate() {
            b.write(buf_off + i as u32, 1, *b_ as u32, &mut h);
        }

        // Outer ring: one CALL to the buffer above.
        let ring_cmds = [
            opcode_len(proto::OP_CALL, 3),
            BUFFER_APERTURE_REL,    // ref address (aperture space)
            buf_bytes.len() as u32, // ref length, space bit 31 clear
        ];
        let ring_bytes: Vec<u8> = ring_cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        let ring_off = b.aperture_offset + RING_BASE_APERTURE_REL;
        for (i, b_) in ring_bytes.iter().enumerate() {
            b.write(ring_off + i as u32, 1, *b_ as u32, &mut h);
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
            ring_bytes.len() as u32,
            &mut h,
        );

        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::ERROR_CODE, 4, &mut h),
            0,
            "no protocol error replaying a well-formed called buffer"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::RING_HEAD, 4, &mut h),
            ring_bytes.len() as u32,
            "RING_HEAD advances past the CALL in the OUTER ring, not into the buffer"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::FENCE_COMPLETED, 4, &mut h),
            1,
            "the FENCE inside the called buffer completed"
        );

        if b.renderer.is_none() {
            eprintln!(
                "skipping pixel check: no wgpu adapter available (renderer_failed={})",
                b.renderer_failed
            );
            return;
        }
        let start = (b.aperture_offset + SURFACE_APERTURE_REL) as usize;
        for row in 0..SURFACE_H {
            for col in 0..SURFACE_W {
                let px = start + (row * SURFACE_STRIDE) as usize + (col * 4) as usize
                    - b.aperture_offset as usize;
                assert_eq!(
                    &b.aperture[px..px + 4],
                    &[0xFFu8, 0xFF, 0x00, 0x00],
                    "pixel ({col}, {row}) is not opaque red after a CALLed CLEAR + SURFACE_READBACK"
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
    ///
    /// Deliberately prefills the *aperture* at the numeric offset
    /// `RING_BASE` names (a real command, not zero bytes) before the
    /// doorbell: an implementation that mishandles this case by falling
    /// through to `read_aperture_range(ring_base, ...)` -- treating the
    /// guest address as an aperture offset -- would decode that command
    /// instead of doing nothing, which a check against an untouched
    /// (all-zero) aperture cannot distinguish from correctly-ignored
    /// (zero bytes decode as `BadLength` and halt at offset zero too,
    /// landing on the same `RING_HEAD == 0`). Asserting `ERROR_CODE == 0`
    /// as well as `RING_HEAD == 0` is what actually tells "genuinely
    /// ignored" apart from "halted after misreading the aperture".
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

        // Garbage (a real, well-formed FENCE id=99) at the aperture
        // offset numerically equal to GUEST_RING_ADDR -- the offset the
        // old buggy fallback would have misread `RING_BASE` as.
        let garbage = [opcode_len(proto::OP_FENCE, 2), 99u32];
        let garbage_bytes: Vec<u8> = garbage.iter().flat_map(|w| w.to_be_bytes()).collect();
        for (i, byte) in garbage_bytes.iter().enumerate() {
            b.write(
                b.aperture_offset + GUEST_RING_ADDR + i as u32,
                1,
                *byte as u32,
                &mut h,
            );
        }

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
            b.read(CONTEXT_PAGE_BASE + creg::ERROR_CODE, 4, &mut h),
            0,
            "genuinely ignored, not halted on a misread of the aperture's garbage bytes"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::RING_HEAD, 4, &mut h),
            0,
            "nothing was decoded from a guest-memory ring CAP_GUESTMEM cannot reach"
        );
        assert_eq!(
            b.read(CONTEXT_PAGE_BASE + creg::FENCE_COMPLETED, 4, &mut h),
            0,
            "the aperture's garbage FENCE id=99 must not have been decoded and completed"
        );
    }

    #[test]
    fn appended_ring_bytes_unwraps_a_span_that_crosses_the_physical_end() {
        let ring: Vec<u8> = (0u8..16).collect();
        // No wrap: [4, 10).
        assert_eq!(appended_ring_bytes(&ring, 4, 10), &ring[4..10]);
        // Wraps once: [12, 16) then [0, 3).
        let wrapped = appended_ring_bytes(&ring, 12, 3);
        assert_eq!(wrapped, [12, 13, 14, 15, 0, 1, 2]);
        // Guest-controlled offsets past the buffer's own length must not
        // panic (`% len`, not a direct index).
        assert_eq!(
            appended_ring_bytes(&ring, 1000, 1003),
            appended_ring_bytes(&ring, 8, 11)
        );
        assert!(appended_ring_bytes(&[], 0, 5).is_empty());
    }

    #[test]
    fn trace_start_stop_status_track_an_armed_captures_lifecycle() {
        let mut b = board();
        assert!(b.trace_status().is_none(), "nothing armed yet");
        assert!(
            b.trace_stop().is_none(),
            "stopping with nothing armed is a no-op, not an error"
        );

        let path = std::env::temp_dir().join(format!(
            "copperline-c3d-trace-lifecycle-test-{}.c3dtrace",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        b.trace_start(Some(path.clone()), None)
            .expect("arm a capture");
        let status = b.trace_status().expect("armed");
        assert_eq!(status.path, path);
        assert_eq!((status.rings, status.blobs, status.bytes), (0, 0, 0));

        // Starting again replaces the first capture outright (matching
        // `ui_wave_start`'s own stance), rather than erroring.
        let path2 = std::env::temp_dir().join(format!(
            "copperline-c3d-trace-lifecycle-test-2-{}.c3dtrace",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path2);
        b.trace_start(Some(path2.clone()), None)
            .expect("replace the capture");
        assert_eq!(b.trace_status().unwrap().path, path2);

        let stopped = b.trace_stop().expect("was armed");
        assert_eq!(stopped.path, path2);
        assert!(b.trace_status().is_none(), "disarmed");
        assert!(path2.exists(), "the file was actually created");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&path2);
    }

    #[test]
    fn trace_start_with_no_path_falls_back_to_the_configured_default_then_a_timestamped_file() {
        let mut b = board();
        // No `set_configured_trace_path` call: falls all the way back to
        // `crate::paths::c3d_trace_file`'s timestamped default.
        b.trace_start(None, None)
            .expect("arm with the timestamped default");
        let default_path = b.trace_stop().expect("was armed").path;
        assert!(
            default_path
                .to_string_lossy()
                .contains("copperline-c3d-trace-"),
            "should use the timestamped default name: {default_path:?}"
        );
        let _ = std::fs::remove_file(&default_path);

        let configured = std::env::temp_dir().join(format!(
            "copperline-c3d-trace-configured-default-test-{}.c3dtrace",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&configured);
        b.set_configured_trace_path(configured.clone());
        b.trace_start(None, None)
            .expect("arm with the configured default");
        assert_eq!(b.trace_status().unwrap().path, configured);

        // An explicit path still wins over the configured default.
        let explicit = std::env::temp_dir().join(format!(
            "copperline-c3d-trace-explicit-test-{}.c3dtrace",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&explicit);
        b.trace_start(Some(explicit.clone()), None)
            .expect("arm with an explicit path");
        assert_eq!(b.trace_status().unwrap().path, explicit);
        b.trace_stop();

        let _ = std::fs::remove_file(&configured);
        let _ = std::fs::remove_file(&explicit);
    }

    /// The driver-facing proof this capability exists for: a captured
    /// session's `RING` section decodes back into the *exact* `Command`
    /// sequence the guest sent, in order -- so a state-setting opcode
    /// immediately preceding a draw (`ACTIVE_UNIT`, `TEX_BIND`,
    /// `ENABLE`/`DISABLE`, `TEX_ENV`...) is directly inspectable from the
    /// trace file alone, with no live session, via
    /// `crate::c3d::ring::decode_all` over a `TraceReader`-read `RING`
    /// section's bytes -- exactly what `copperline-c3d-trace`'s own
    /// `replay` already does internally, just surfaced here as a decode
    /// rather than a render.
    #[test]
    fn trace_capture_records_the_exact_command_order_a_doorbell_decoded() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);

        let path = std::env::temp_dir().join(format!(
            "copperline-c3d-trace-order-test-{}.c3dtrace",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        b.trace_start(Some(path.clone()), None)
            .expect("arm capture");

        let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;
        let cmds: Vec<u32> = vec![
            opcode_len(proto::OP_ACTIVE_UNIT, 2),
            1,
            opcode_len(proto::OP_MATRIX_MODE, 2),
            1, // an arbitrary legal MatrixMode value
            opcode_len(proto::OP_LOAD_IDENTITY, 1),
            opcode_len(proto::OP_FENCE, 2),
            7,
        ];
        let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();

        let ring_window_off = b.aperture_offset;
        for (i, byte) in bytes.iter().enumerate() {
            b.write(ring_window_off + i as u32, 1, *byte as u32, &mut h);
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
            "well-formed stream"
        );

        let status = b.trace_stop().expect("capture was armed");
        assert_eq!(status.rings, 1);

        let file_bytes = std::fs::read(&path).expect("read capture file");
        let _ = std::fs::remove_file(&path);

        let mut captured: Option<(u32, u32, Vec<u8>)> = None;
        for section in crate::c3d::trace::TraceReader::new(&file_bytes) {
            if let crate::c3d::trace::Section::Ring(r) = section.expect("well-formed trace") {
                captured = Some((r.context, r.ring_tail, r.bytes.to_vec()));
            }
        }
        let (context, ring_tail, ring_bytes) = captured.expect("exactly one RING section");
        assert_eq!(context, 0, "the first context slot");
        assert_eq!(ring_tail, bytes.len() as u32);
        assert_eq!(
            ring_bytes, bytes,
            "captured bytes match exactly what the guest appended"
        );

        // `RingCursor` indexes `[0, ring_tail)` into the *real, full-size*
        // ring (head/tail are absolute offsets into it, same as
        // `copperline-c3d-trace`'s own `replay()` places a captured span
        // at its growable replay buffer's previous tail rather than
        // decoding it as a standalone buffer) -- place the captured span
        // at the start of a buffer as large as the live ring was, not
        // one sized to the span alone.
        let mut replay_ring = vec![0u8; 0x1000];
        replay_ring[..ring_bytes.len()].copy_from_slice(&ring_bytes);
        let (steps, _) = crate::c3d::ring::decode_all(
            &replay_ring,
            0,
            ring_bytes.len() as u32,
            b.device_config(),
        );
        assert_eq!(
            steps.len(),
            4,
            "ACTIVE_UNIT, MATRIX_MODE, LOAD_IDENTITY, FENCE"
        );
        assert!(matches!(
            steps[0],
            crate::c3d::ring::Step::Command(proto::Command::ActiveUnit { unit: 1 })
        ));
        assert!(matches!(
            steps[1],
            crate::c3d::ring::Step::Command(proto::Command::MatrixMode { .. })
        ));
        assert!(matches!(
            steps[2],
            crate::c3d::ring::Step::Command(proto::Command::LoadIdentity)
        ));
        assert!(matches!(
            steps[3],
            crate::c3d::ring::Step::Command(proto::Command::Fence { id: 7 })
        ));
    }

    fn setup_ring_context(b: &mut C3dBoard, h: &mut DeviceHost) {
        b.write(CONTEXT_PAGE_BASE + creg::RING_BASE, 4, 0, h);
        b.write(CONTEXT_PAGE_BASE + creg::RING_SIZE, 4, 0x1000, h);
        b.write(
            CONTEXT_PAGE_BASE + creg::CONTROL,
            4,
            CTX_CONTROL_ALLOC | CTX_CONTROL_ENABLE,
            h,
        );
    }

    /// Appends `cmds` at the ring's current `*tail` and rings the
    /// doorbell -- one more doorbell each call, unlike
    /// `trace_capture_records_the_exact_command_order_a_doorbell_decoded`'s
    /// single big one, which is what the rolling-eviction and trigger
    /// tests below need.
    fn submit_doorbell(b: &mut C3dBoard, h: &mut DeviceHost, tail: &mut u32, cmds: &[u32]) {
        let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        let ring_window_off = b.aperture_offset;
        for (i, byte) in bytes.iter().enumerate() {
            b.write(ring_window_off + *tail + i as u32, 1, *byte as u32, h);
        }
        *tail += bytes.len() as u32;
        b.write(CONTEXT_PAGE_BASE + creg::RING_TAIL, 4, *tail, h);
    }

    fn fence_cmd(id: u32) -> Vec<u32> {
        vec![((proto::OP_FENCE as u32) << 16) | 2, id]
    }

    /// Decodes a captured `RING` section's bytes directly as a
    /// standalone command stream -- valid regardless of the section's
    /// own (possibly non-zero, possibly eviction-discontiguous) logical
    /// `ring_tail`, since a `RingCursor` only ever cares about the slice
    /// it is actually given. `RingCursor::new_linear` (the same
    /// constructor a `CALL`ed buffer decodes with), not `new`/
    /// `decode_all`: `new`'s circular "one word short of a full ring"
    /// framing exists only for a *live* ring whose head/tail chase each
    /// other, which does not apply to an already-extracted, immutable
    /// byte span -- using it here would spuriously reject a span that
    /// happens to fill its own slice exactly (every `bytes.len()` this
    /// function is ever called with, in fact: a captured span is never
    /// padded).
    fn decode_ring_commands(bytes: &[u8], config: DeviceConfig) -> Vec<proto::Command<'_>> {
        let mut cursor = crate::c3d::ring::RingCursor::new_linear(bytes, config);
        let mut out = Vec::new();
        while let Some(step) = cursor.step() {
            if let crate::c3d::ring::Step::Command(c) = step {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn rolling_capture_evicts_the_oldest_doorbell_once_the_byte_budget_is_exceeded() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        setup_ring_context(&mut b, &mut h);

        // Each doorbell below is one FENCE command: 8 bytes of ring
        // content, no refs, so the budget bounds purely by ring size --
        // easy to reason about exactly. Budget 20: doorbells 1+2 fit
        // (16 <= 20); doorbell 3 pushes it to 24 > 20, evicting doorbell
        // 1 (fence id 1) and leaving 16 bytes (fence ids 2, 3).
        b.trace_start(None, Some(20))
            .expect("arm a rolling capture");
        let mut tail = 0u32;
        for id in 1u32..=3 {
            submit_doorbell(&mut b, &mut h, &mut tail, &fence_cmd(id));
        }

        let status = b.trace_status().expect("still rolling");
        assert!(status.rolling);
        assert_eq!(status.rolling_byte_budget, Some(20));
        assert_eq!(
            status.rings, 2,
            "the first doorbell (unit 1) should have been evicted"
        );
        assert_eq!(status.bytes, 16);

        let path = std::env::temp_dir().join(format!(
            "copperline-c3d-trace-rolling-eviction-test-{}.c3dtrace",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        b.trace_snapshot(Some(path.clone()))
            .expect("snapshot")
            .expect("was armed");
        let file_bytes = std::fs::read(&path).expect("read capture file");
        let _ = std::fs::remove_file(&path);

        let config = b.device_config();
        let mut ids_seen = Vec::new();
        for section in crate::c3d::trace::TraceReader::new(&file_bytes) {
            if let crate::c3d::trace::Section::Ring(r) = section.expect("well-formed trace") {
                for cmd in decode_ring_commands(r.bytes, config) {
                    if let proto::Command::Fence { id } = cmd {
                        ids_seen.push(id);
                    }
                }
            }
        }
        assert_eq!(
            ids_seen,
            vec![2, 3],
            "only the two most recent doorbells should have survived eviction"
        );
    }

    #[test]
    fn a_single_doorbell_larger_than_the_byte_budget_is_kept_anyway() {
        // Evicting it would leave the ring permanently empty for every
        // doorbell after -- `RollingCapture::push`'s own
        // `records.len() > 1` guard.
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        setup_ring_context(&mut b, &mut h);

        b.trace_start(None, Some(1)).expect("arm a rolling capture");
        let mut tail = 0u32;
        submit_doorbell(&mut b, &mut h, &mut tail, &fence_cmd(1));

        let status = b.trace_status().expect("still rolling");
        assert_eq!(
            status.rings, 1,
            "the one oversized record is kept, not dropped"
        );
        assert_eq!(status.bytes, 8);
    }

    #[test]
    fn trace_snapshot_replays_the_buffer_without_stopping_the_roll() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        setup_ring_context(&mut b, &mut h);

        let path = std::env::temp_dir().join(format!(
            "copperline-c3d-trace-rolling-snapshot-test-{}.c3dtrace",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        b.trace_start(Some(path.clone()), Some(1024))
            .expect("arm a rolling capture");
        assert!(
            !path.exists(),
            "rolling mode must not touch disk until a snapshot"
        );

        let mut tail = 0u32;
        submit_doorbell(&mut b, &mut h, &mut tail, &fence_cmd(1));

        let status = b
            .trace_snapshot(None)
            .expect("snapshot call succeeds")
            .expect("was armed");
        assert_eq!(status.rings, 1);
        assert!(
            path.exists(),
            "snapshot with no path falls back to the configured one"
        );

        submit_doorbell(&mut b, &mut h, &mut tail, &fence_cmd(2));
        assert_eq!(
            b.trace_status().expect("still rolling").rings,
            2,
            "the roll continues after a snapshot"
        );

        let status2 = b
            .trace_snapshot(None)
            .expect("second snapshot")
            .expect("still armed");
        assert_eq!(
            status2.rings, 2,
            "the second snapshot includes both doorbells"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// The driver-facing proof this capability exists for: a capture
    /// does not need a precisely-timed manual `c3d.trace_start` -- it
    /// can be armed ahead of time and left to fire itself exactly when
    /// the texture of interest first appears.
    #[test]
    fn trace_arm_on_texture_bind_defers_the_capture_until_that_texture_binds() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        setup_ring_context(&mut b, &mut h);

        let path = std::env::temp_dir().join(format!(
            "copperline-c3d-trace-trigger-test-{}.c3dtrace",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        b.trace_arm_on_texture_bind(42, Some(path.clone()), None);
        assert_eq!(b.trace_trigger_pending(), Some(42));
        assert!(b.trace_status().is_none(), "armed but not yet capturing");

        let tex_create = |id: u32| vec![((proto::OP_TEX_CREATE as u32) << 16) | 2, id];
        let tex_bind = |unit: u32, id: u32| vec![((proto::OP_TEX_BIND as u32) << 16) | 3, unit, id];

        let mut tail = 0u32;
        // Doorbell 1: bind a different texture -- must not fire.
        let mut cmds = tex_create(7);
        cmds.extend(tex_bind(0, 7));
        submit_doorbell(&mut b, &mut h, &mut tail, &cmds);
        assert!(
            b.trace_status().is_none(),
            "an unrelated texture bind must not fire the trigger"
        );
        assert_eq!(b.trace_trigger_pending(), Some(42));

        // Doorbell 2: create and bind texture 42 -- fires the trigger.
        // The triggering doorbell itself is not captured; the capture
        // starts from the *next* one (`check_trace_trigger`'s doc
        // comment).
        let mut cmds = tex_create(42);
        cmds.extend(tex_bind(0, 42));
        submit_doorbell(&mut b, &mut h, &mut tail, &cmds);
        assert_eq!(b.trace_trigger_pending(), None, "trigger consumed");
        let status = b.trace_status().expect("capture started");
        assert_eq!(
            status.rings, 0,
            "the triggering doorbell itself is not captured"
        );

        // Doorbell 3: now captured.
        submit_doorbell(&mut b, &mut h, &mut tail, &fence_cmd(1));
        let status = b.trace_stop().expect("was capturing");
        assert_eq!(
            status.rings, 1,
            "only the post-trigger doorbell was captured"
        );
        assert_eq!(status.path, path);

        let _ = std::fs::remove_file(&path);
    }

    // -- CAP_SHARED_CONTEXTS: cross-context sharing ----------------------

    const SHARE_GROUP_REG: u32 = CONTEXT_PAGE_BASE + creg::SHARE_GROUP;

    fn context_page(n: u32) -> u32 {
        CONTEXT_PAGE_BASE + n * CONTEXT_PAGE_SIZE
    }

    /// As `setup_ring_context`, parameterised over which context slot and
    /// which ring-base offset within the aperture to use -- the plain
    /// single-context helper above always uses context `0` and ring base
    /// `0`, which two *simultaneously live* contexts in these tests can't
    /// both do without their rings overlapping in the shared aperture.
    fn setup_ring_context_n(b: &mut C3dBoard, h: &mut DeviceHost, n: u32, ring_base: u32) {
        b.write(context_page(n) + creg::RING_BASE, 4, ring_base, h);
        b.write(context_page(n) + creg::RING_SIZE, 4, 0x1000, h);
        b.write(
            context_page(n) + creg::CONTROL,
            4,
            CTX_CONTROL_ALLOC | CTX_CONTROL_ENABLE,
            h,
        );
    }

    /// As `submit_doorbell`, for context `n` at aperture offset
    /// `ring_base` (matching whatever `setup_ring_context_n` used).
    fn submit_doorbell_n(
        b: &mut C3dBoard,
        h: &mut DeviceHost,
        n: u32,
        ring_base: u32,
        tail: &mut u32,
        cmds: &[u32],
    ) {
        let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        let window_off = b.aperture_offset + ring_base;
        for (i, byte) in bytes.iter().enumerate() {
            b.write(window_off + *tail + i as u32, 1, *byte as u32, h);
        }
        *tail += bytes.len() as u32;
        b.write(context_page(n) + creg::RING_TAIL, 4, *tail, h);
    }

    fn fence_sync_create_cmd(id: u32) -> Vec<u32> {
        vec![((proto::OP_FENCE_SYNC_CREATE as u32) << 16) | 2, id]
    }

    fn fence_sync_destroy_cmd(id: u32) -> Vec<u32> {
        vec![((proto::OP_FENCE_SYNC_DESTROY as u32) << 16) | 2, id]
    }

    fn wait_sync_cmd(id: u32) -> Vec<u32> {
        vec![((proto::OP_WAIT_SYNC as u32) << 16) | 2, id]
    }

    fn error_code(b: &mut C3dBoard, h: &mut DeviceHost, n: u32) -> u32 {
        b.read(context_page(n) + creg::ERROR_CODE, 4, h)
    }

    fn ctx_status(b: &mut C3dBoard, h: &mut DeviceHost, n: u32) -> u32 {
        b.read(context_page(n) + creg::STATUS, 4, h)
    }

    fn ring_head(b: &mut C3dBoard, h: &mut DeviceHost, n: u32) -> u32 {
        b.read(context_page(n) + creg::RING_HEAD, 4, h)
    }

    fn ring_tail_reg(b: &mut C3dBoard, h: &mut DeviceHost, n: u32) -> u32 {
        b.read(context_page(n) + creg::RING_TAIL, 4, h)
    }

    /// Without `CAP_SHARED_CONTEXTS`, every `0x0Exx` opcode is
    /// `E_BAD_OPCODE` and `CTX_SHARE_GROUP` always reads/stays `0` --
    /// the spec's `CAPS0` bit-8 row, the "each bit cleared singly" case.
    #[test]
    fn cap_shared_contexts_gating_rejects_the_sync_opcodes_and_share_group_write() {
        let mut b = C3dBoard::with_masked_caps(0x0200_0000, proto::CAP_SHARED_CONTEXTS);
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        setup_ring_context_n(&mut b, &mut h, 0, 0);
        let mut tail = 0u32;
        submit_doorbell_n(&mut b, &mut h, 0, 0, &mut tail, &fence_sync_create_cmd(1));
        assert_eq!(
            error_code(&mut b, &mut h, 0),
            proto::ErrorCode::BadOpcode as u32
        );

        b.write(SHARE_GROUP_REG, 4, 5, &mut h);
        assert_eq!(
            b.read(SHARE_GROUP_REG, 4, &mut h),
            0,
            "CTX_SHARE_GROUP must stay 0 without the capability"
        );
    }

    /// The core feature: two contexts that write the same nonzero
    /// `CTX_SHARE_GROUP` *before their first command* share one sync-
    /// object namespace -- `FENCE_SYNC_CREATE` in context 0 is visible
    /// to `WAIT_SYNC` in context 1, immediately (synchronous signalling;
    /// see `SyncObject`'s doc comment), with no `E_BAD_ID`.
    #[test]
    fn two_contexts_in_the_same_share_group_see_each_others_sync_objects() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        b.write(context_page(0) + creg::SHARE_GROUP, 4, 5, &mut h);
        b.write(context_page(1) + creg::SHARE_GROUP, 4, 5, &mut h);
        setup_ring_context_n(&mut b, &mut h, 0, 0);
        setup_ring_context_n(&mut b, &mut h, 1, 0x1000);
        let mut tail0 = 0u32;
        let mut tail1 = 0u32;

        submit_doorbell_n(&mut b, &mut h, 0, 0, &mut tail0, &fence_sync_create_cmd(9));
        assert_eq!(error_code(&mut b, &mut h, 0), 0);

        submit_doorbell_n(&mut b, &mut h, 1, 0x1000, &mut tail1, &wait_sync_cmd(9));
        assert_eq!(
            error_code(&mut b, &mut h, 1),
            0,
            "context 1 must see context 0's sync object -- same share group"
        );
        assert_eq!(
            ring_head(&mut b, &mut h, 1),
            ring_tail_reg(&mut b, &mut h, 1),
            "already-signalled WAIT_SYNC must not stall"
        );
    }

    /// Two contexts in *different* share groups (or one left private)
    /// must not see each other's sync objects at all -- the id simply
    /// doesn't exist in the waiting context's own namespace, `E_BAD_ID`,
    /// indistinguishable from an id that was never created.
    #[test]
    fn contexts_in_different_share_groups_cannot_see_each_others_sync_objects() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        b.write(context_page(0) + creg::SHARE_GROUP, 4, 5, &mut h);
        b.write(context_page(1) + creg::SHARE_GROUP, 4, 6, &mut h);
        setup_ring_context_n(&mut b, &mut h, 0, 0);
        setup_ring_context_n(&mut b, &mut h, 1, 0x1000);
        let mut tail0 = 0u32;
        let mut tail1 = 0u32;

        submit_doorbell_n(&mut b, &mut h, 0, 0, &mut tail0, &fence_sync_create_cmd(9));
        assert_eq!(error_code(&mut b, &mut h, 0), 0);

        submit_doorbell_n(&mut b, &mut h, 1, 0x1000, &mut tail1, &wait_sync_cmd(9));
        assert_eq!(
            error_code(&mut b, &mut h, 1),
            proto::ErrorCode::BadId as u32,
            "different share groups must not alias each other's sync objects"
        );
    }

    /// `CTX_SHARE_GROUP` is "written only while this context has decoded
    /// no command since its last `ALLOC` transition" -- a write *after*
    /// the first command must be silently ignored, leaving the context
    /// exactly as private (or as shared) as it already was.
    #[test]
    fn ctx_share_group_written_after_the_first_command_has_no_effect() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        setup_ring_context_n(&mut b, &mut h, 0, 0);
        setup_ring_context_n(&mut b, &mut h, 1, 0x1000);
        let mut tail0 = 0u32;
        let mut tail1 = 0u32;

        // Context 0 decodes one command (still private) before trying to
        // join a group -- the write below must be ignored.
        submit_doorbell_n(&mut b, &mut h, 0, 0, &mut tail0, &fence_sync_create_cmd(1));
        b.write(context_page(0) + creg::SHARE_GROUP, 4, 5, &mut h);
        assert_eq!(
            b.read(context_page(0) + creg::SHARE_GROUP, 4, &mut h),
            0,
            "late write must not take effect"
        );

        // Context 1 genuinely joins group 5 before its first command.
        b.write(context_page(1) + creg::SHARE_GROUP, 4, 5, &mut h);

        submit_doorbell_n(&mut b, &mut h, 0, 0, &mut tail0, &fence_sync_create_cmd(9));
        submit_doorbell_n(&mut b, &mut h, 1, 0x1000, &mut tail1, &wait_sync_cmd(9));
        assert_eq!(
            error_code(&mut b, &mut h, 1),
            proto::ErrorCode::BadId as u32,
            "context 0 stayed private (its late join was ignored), so context \
             1's real group-5 membership still cannot see context 0's sync id"
        );
    }

    /// `WAIT_SYNC`'s device-side wait, the real substance of this
    /// capability, in the one shape this board's fully-synchronous-per-
    /// doorbell execution model (dispatch *and* `Renderer::execute` both
    /// complete before a doorbell call returns) can ever actually
    /// reach: a genuinely cross-context stall -- context B waiting on a
    /// sync object context A has not created *yet* -- is `E_BAD_ID`
    /// immediately, per the spec's "never created" rule, not a wait;
    /// `FENCE_SYNC_CREATE` signals synchronously at dispatch time (see
    /// `state::SyncObject`'s doc comment), so by the time any *other*
    /// context's separate doorbell could ever observe a sync object
    /// that exists at all, it is already signalled. The `Some(false)`
    /// branch in `Context::submit_with`'s `WAIT_SYNC` interception is
    /// therefore real, spec-required infrastructure for a device that
    /// *can* have a created-but-unsignalled window (a pipelined one, or
    /// a future revision of this one) -- but is unreachable through
    /// Copperline's own wire sequence today. This test exercises it
    /// directly instead, at the `dispatch`/`state` level this module
    /// cannot reach: two `State`s sharing one `sync_objects` `Rc`
    /// (mirroring what `board.rs`'s `ShareGroupStores::resolve` does for
    /// two real contexts), with `id` `9` manually inserted *unsignalled*
    /// -- a case no wire opcode can actually produce, exactly why this
    /// needs a lower-level test to reach at all.
    #[test]
    fn wait_sync_interception_actually_stalls_on_a_genuinely_unsignalled_sync_object() {
        use crate::c3d::dispatch::Context;
        use crate::c3d::state::{self as c3d_state, SyncObject};
        use std::cell::RefCell;
        use std::rc::Rc;

        let sync_objects = Rc::new(RefCell::new(std::collections::BTreeMap::from([(
            9,
            SyncObject { signalled: false },
        )])));
        let mut waiter = Context::new(c3d_state::Limits::default());
        waiter.state.set_context_index(1);
        waiter.state.set_share_group(
            5,
            Rc::new(RefCell::new(Default::default())),
            Rc::new(RefCell::new(Default::default())),
            Rc::new(RefCell::new(Default::default())),
            Rc::new(RefCell::new(Default::default())),
            Rc::new(RefCell::new(Default::default())),
            sync_objects.clone(),
        );

        let cmds = {
            let mut v = wait_sync_cmd(9);
            v.extend(fence_sync_create_cmd(10)); // the "follow-up" command
            v
        };
        let tail = (cmds.len() * 4) as u32;
        let mut bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        bytes.resize(bytes.len() + 4, 0); // RingCursor reserves one spare word
        let config = crate::c3d::ring::DeviceConfig::default();
        let mut out = Vec::new();
        waiter.submit_with(
            &bytes,
            tail,
            &config,
            &mut out,
            &mut Vec::new(),
            &mut |_| None,
        );

        assert_eq!(waiter.error_code, 0, "not an error, a stall");
        assert!(
            waiter.awaiting_sync,
            "AWAITING_SYNC must be set while stalled"
        );
        assert_eq!(
            waiter.ring_head, 0,
            "ring_head must sit at WAIT_SYNC itself, not past it"
        );
        assert!(
            waiter.state.sync_signalled(10).is_none(),
            "the follow-up FENCE_SYNC_CREATE must not have run yet"
        );

        // Something else (a real implementation's equivalent of another
        // context's GPU work actually completing) signals sync 9.
        sync_objects.borrow_mut().get_mut(&9).unwrap().signalled = true;

        // Retrying is exactly `board.rs`'s `retry_awaiting_sync_contexts`:
        // call `submit_with` again from the same `ring_tail`.
        out.clear();
        waiter.submit_with(
            &bytes,
            tail,
            &config,
            &mut out,
            &mut Vec::new(),
            &mut |_| None,
        );

        assert!(
            !waiter.awaiting_sync,
            "AWAITING_SYNC must clear once signalled"
        );
        assert_eq!(
            waiter.ring_head, tail,
            "the follow-up command must now have been decoded too"
        );
        assert_eq!(
            waiter.state.sync_signalled(10),
            Some(true),
            "the follow-up FENCE_SYNC_CREATE actually ran, not just ring_head moving"
        );
    }

    /// `WAIT_SYNC` on a sync object that is already signalled -- real
    /// GL's own "wait on a fence already passed" case -- is a no-op, not
    /// a stall: covered incidentally by the same-share-group test above,
    /// this isolates it with an explicit same-context round trip too.
    #[test]
    fn wait_sync_on_an_already_signalled_sync_object_does_not_stall() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        setup_ring_context_n(&mut b, &mut h, 0, 0);
        let mut tail = 0u32;
        let mut cmds = fence_sync_create_cmd(1);
        cmds.extend(wait_sync_cmd(1));
        submit_doorbell_n(&mut b, &mut h, 0, 0, &mut tail, &cmds);
        assert_eq!(error_code(&mut b, &mut h, 0), 0);
        assert_eq!(
            ring_head(&mut b, &mut h, 0),
            ring_tail_reg(&mut b, &mut h, 0)
        );
        assert_eq!(ctx_status(&mut b, &mut h, 0) & CTX_STATUS_AWAITING_SYNC, 0);
    }

    /// `FENCE_SYNC_DESTROY` frees `id` for reuse in the group's
    /// namespace -- a later `WAIT_SYNC` against the now-destroyed id is
    /// `E_BAD_ID`, indistinguishable from one that was never created.
    #[test]
    fn wait_sync_after_fence_sync_destroy_is_bad_id() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        setup_ring_context_n(&mut b, &mut h, 0, 0);
        let mut tail = 0u32;
        let mut cmds = fence_sync_create_cmd(1);
        cmds.extend(fence_sync_destroy_cmd(1));
        submit_doorbell_n(&mut b, &mut h, 0, 0, &mut tail, &cmds);
        assert_eq!(error_code(&mut b, &mut h, 0), 0);

        submit_doorbell_n(&mut b, &mut h, 0, 0, &mut tail, &wait_sync_cmd(1));
        assert_eq!(
            error_code(&mut b, &mut h, 0),
            proto::ErrorCode::BadId as u32
        );
    }

    /// `WAIT_SYNC` decoded from inside a `CALL`'d buffer is `E_BAD_OPCODE`
    /// -- a real device-side stall there has nowhere to persist a
    /// resumable position across the doorbells it can span (the outer
    /// ring's `ring_head` names the `CALL` command itself, not anywhere
    /// inside the buffer it names), so this must be rejected outright
    /// rather than silently falling through to a non-stalling no-op --
    /// mirroring `CALL`-inside-`CALL`'s own identical nesting rejection.
    #[test]
    fn wait_sync_inside_a_call_buffer_is_bad_opcode() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);

        const RING_BASE_REL: u32 = 0;
        const BUFFER_REL: u32 = 0x1000;

        // A sync object that genuinely exists (and is signalled) in this
        // context's own namespace -- so if the `CALL`-nesting rejection
        // were missing, this would wrongly succeed as a no-op rather
        // than correctly failing closed.
        let buf_cmds = wait_sync_cmd(1);
        let buf_bytes: Vec<u8> = buf_cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        let buf_off = b.aperture_offset + BUFFER_REL;
        for (i, byte) in buf_bytes.iter().enumerate() {
            b.write(buf_off + i as u32, 1, *byte as u32, &mut h);
        }

        let mut ring_cmds = fence_sync_create_cmd(1);
        ring_cmds.extend([
            ((proto::OP_CALL as u32) << 16) | 3,
            BUFFER_REL,
            buf_bytes.len() as u32,
        ]);
        let ring_bytes: Vec<u8> = ring_cmds.iter().flat_map(|w| w.to_be_bytes()).collect();
        let ring_off = b.aperture_offset + RING_BASE_REL;
        for (i, byte) in ring_bytes.iter().enumerate() {
            b.write(ring_off + i as u32, 1, *byte as u32, &mut h);
        }

        b.write(context_page(0) + creg::RING_BASE, 4, RING_BASE_REL, &mut h);
        b.write(context_page(0) + creg::RING_SIZE, 4, 0x4000, &mut h);
        b.write(
            context_page(0) + creg::CONTROL,
            4,
            CTX_CONTROL_ALLOC | CTX_CONTROL_ENABLE,
            &mut h,
        );
        b.write(
            context_page(0) + creg::RING_TAIL,
            4,
            ring_bytes.len() as u32,
            &mut h,
        );

        assert_eq!(
            error_code(&mut b, &mut h, 0),
            proto::ErrorCode::BadOpcode as u32,
            "WAIT_SYNC inside a CALL must fail closed, even against a real, \
             signalled, same-namespace sync object"
        );
    }

    /// The headline claim, proven end to end through a real GPU round
    /// trip, not a `state.rs`-level assertion that an `Rc` is shared:
    /// context 0 creates and uploads a texture; context 1, in the same
    /// `CTX_SHARE_GROUP`, binds the *same id* with zero re-upload and
    /// actually renders with it -- a window-space textured triangle
    /// sampling it, read back from the aperture-backed draw surface --
    /// producing exactly the colour context 0 uploaded. If the sharing
    /// plumbing (the `Rc<RefCell<_>>` namespace machinery and the
    /// `Renderer`-side `PartKey` partitioning fix it depends on) were
    /// wrong in either direction -- not actually shared, or aliased
    /// across the wrong namespace -- this would either fail to bind at
    /// all (`E_BAD_ID`) or render the wrong/undefined colour.
    #[test]
    fn a_texture_uploaded_in_one_context_is_sampled_by_a_real_draw_in_another_sharing_it() {
        let mut b = board();
        let mut mem = dummy_mem();
        let mut h = host(&mut mem);
        b.write(context_page(0) + creg::SHARE_GROUP, 4, 5, &mut h);
        b.write(context_page(1) + creg::SHARE_GROUP, 4, 5, &mut h);

        const RING0_REL: u32 = 0;
        const RING1_REL: u32 = 0x2000;
        const TEX_SRC_REL: u32 = 0x4000;
        const SURFACE_REL: u32 = 0x5000;
        const SURFACE_W: u32 = 4;
        const SURFACE_H: u32 = 4;
        const SURFACE_STRIDE: u32 = SURFACE_W * 4;
        const TEXTURE_ID: u32 = 7;

        let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;

        // Opaque red, RGBA8 byte order (TEX_IMAGE's wire format 0).
        let tex_src = [0xFFu8, 0x00, 0x00, 0xFF];
        let tex_src_off = b.aperture_offset + TEX_SRC_REL;
        for (i, byte) in tex_src.iter().enumerate() {
            b.write(tex_src_off + i as u32, 1, *byte as u32, &mut h);
        }

        setup_ring_context_n(&mut b, &mut h, 0, RING0_REL);
        setup_ring_context_n(&mut b, &mut h, 1, RING1_REL);
        let mut tail0 = 0u32;
        let mut tail1 = 0u32;

        // Context 0: creates the texture object and uploads its only
        // content. Context 1 never does either of these.
        let ctx0_cmds = [
            opcode_len(proto::OP_TEX_CREATE, 2),
            TEXTURE_ID,
            opcode_len(proto::OP_TEX_IMAGE, 9),
            TEXTURE_ID,
            0, // level
            0, // format: RGBA8
            1, // width
            1, // height
            4, // row_bytes
            TEX_SRC_REL,
            4, // ref length
        ];
        submit_doorbell_n(&mut b, &mut h, 0, RING0_REL, &mut tail0, &ctx0_cmds);
        assert_eq!(error_code(&mut b, &mut h, 0), 0, "context 0's own upload");

        // Context 1: binds the SAME id with no TEX_CREATE/TEX_IMAGE of
        // its own, defines a draw surface, enables texturing, and draws
        // one window-space triangle (position + texcoord0 only --
        // GLES2's default white CURRENT_COLOR and MODULATE TEX_ENV mean
        // the fragment colour is exactly the sampled texel) covering the
        // whole surface, then reads it back into the aperture.
        let tri_vertex = |x: f32, y: f32| -> [u32; 6] {
            [
                x.to_bits(),
                y.to_bits(),
                0.0f32.to_bits(),
                1.0f32.to_bits(),
                0,
                0,
            ]
        };
        let mut ctx1_cmds = vec![
            opcode_len(proto::OP_TEX_BIND, 3),
            0, // unit
            TEXTURE_ID,
            opcode_len(proto::OP_SURFACE_DEFINE, 8),
            1,
            SURFACE_W,
            SURFACE_H,
            SURFACE_STRIDE,
            5,
            0,
            SURFACE_REL,
            opcode_len(proto::OP_SET_DRAW_SURFACE, 2),
            1,
            opcode_len(proto::OP_ENABLE, 2),
            0x0DE1, // GL_TEXTURE_2D
            opcode_len(proto::OP_DRAW_INLINE_WIN, 1 + 3 + 3 * 6),
            4, // prim: TRIANGLES
            proto::VertexFormat::TEXCOORD0,
            3, // count
        ];
        ctx1_cmds.extend(tri_vertex(-2.0, -2.0));
        ctx1_cmds.extend(tri_vertex(16.0, -2.0));
        ctx1_cmds.extend(tri_vertex(-2.0, 16.0));
        ctx1_cmds.push(opcode_len(proto::OP_SURFACE_READBACK, 5));
        ctx1_cmds.extend([0, 0, SURFACE_W, SURFACE_H]);
        submit_doorbell_n(&mut b, &mut h, 1, RING1_REL, &mut tail1, &ctx1_cmds);
        assert_eq!(
            error_code(&mut b, &mut h, 1),
            0,
            "context 1 binding and drawing with context 0's texture"
        );

        if b.renderer.is_none() {
            eprintln!(
                "skipping pixel check: no wgpu adapter available (renderer_failed={})",
                b.renderer_failed
            );
            return;
        }
        let start = (b.aperture_offset + SURFACE_REL) as usize;
        for row in 0..SURFACE_H {
            for col in 0..SURFACE_W {
                let px = start + (row * SURFACE_STRIDE) as usize + (col * 4) as usize
                    - b.aperture_offset as usize;
                assert_eq!(
                    &b.aperture[px..px + 4],
                    &[0xFFu8, 0xFF, 0x00, 0x00],
                    "pixel ({col}, {row}) must be context 0's uploaded red, \
                     sampled by context 1's draw with zero re-upload"
                );
            }
        }
    }
}
