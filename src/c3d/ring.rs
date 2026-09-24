// SPDX-License-Identifier: GPL-3.0-or-later

//! The C3D command-ring reader: pure functions over a byte slice that turn
//! `RING_HEAD..RING_TAIL` into a sequence of [`proto::Command`]s or
//! [`ProtoError`]s, per `docs/internals/c3d.md`'s "Command stream" and
//! "Errors" sections.
//!
//! This module has no emulator or GPU dependency, does no I/O, and
//! allocates nothing in the decode path: a [`RingCursor`] borrows the ring
//! bytes it was built from and every draw command's bulk payload is handed
//! back as a sub-slice of them. It knows nothing about live board or
//! context state (bound surfaces, texture objects, the GL state machine),
//! but it does know the device's static configuration -- [`DeviceConfig`]
//! -- and raises every protocol error the spec's error table assigns to
//! something that is a pure function of the bytes plus that fixed
//! configuration:
//!
//! | Code | Cause |
//! |---|---|
//! | `E_BAD_LENGTH`, `E_RING_OVERRUN` | framing |
//! | `E_BAD_OPCODE` | unknown opcode, a known opcode of a tier `DeviceConfig` lacks, `TEX_PALETTE` without `TEXFMT_SUPPORTED` bit 9 (`I8_INDEXED`), or `CALL` nested inside an already-called buffer (raised by [`super::dispatch`], not this module -- see below) |
//! | `E_BAD_ARG` | length/count/format mismatch, a non-zero reserved field, a capability-less feature flag, `NORMAL` in a window-space draw, `stride_bytes` short of the row |
//! | `E_BAD_REF` | a ref outside the aperture/reachable space, misaligned, in space 1 without `CAP_GUESTMEM`, or an aperture-backed `SURFACE_DEFINE` extent outside the aperture |
//! | `E_BAD_ID` | a texture/surface ID of 0 (where 0 isn't the documented "none") or above the configured limit |
//! | `E_LIMIT` | a unit/light/plane/dimension beyond a configured maximum |
//! | `E_UNSUPPORTED_FORMAT` | a format bit clear in `TEXFMT_SUPPORTED`/`SURFFMT_SUPPORTED_*` |
//!
//! The two errors this module does *not* raise -- `E_NO_SURFACE` and
//! `E_BAD_RECT` -- depend on the *live* bound surface and its current
//! geometry, not on anything `DeviceConfig` or the bytes alone can answer;
//! those are [`super::state`]'s job.
//!
//! ## Framing
//!
//! A command's word 0 is `opcode[31:16] | length[15:0]`, `length` counting
//! word 0 itself. `docs/internals/c3d.md` states the ring never wraps a
//! command across its physical end -- the guest pads the remainder with a
//! `NOP` and continues at offset `0` -- but doesn't name a protocol error
//! for a malformed guest that violates this (it only defines `E_BAD_LENGTH`
//! for `length == 0` and for a command that "extends past `RING_TAIL`").
//! This decoder treats "extends past the physical ring end" the same as
//! "extends past `RING_TAIL`": both are the guest handing the decoder a
//! command it cannot honour without wrapping, which is exactly what
//! `E_BAD_LENGTH` is for, and a conforming guest never produces either.
//!
//! `E_RING_OVERRUN` is the doorbell-time check: the spec says the ring is
//! full one word short of empty, and a guest can write any `RING_TAIL` it
//! likes, so [`RingCursor::new`] -- constructed fresh from the current
//! register values on every poll, which is what "the doorbell" amounts to
//! in this pure module -- validates `head`/`tail` up front and raises it
//! (once, on the first [`RingCursor::step`] call) if the guest let `tail`
//! overtake `head`.
//!
//! ## Opcode handling
//!
//! An opcode outside the spec's map entirely, and `LOGIC_OP` (`0x0216`,
//! which the spec calls out as unconditionally `E_BAD_OPCODE` in version 1
//! regardless of capability -- see `proto::OP_LOGIC_OP`'s doc comment) are
//! rejected without reading `DeviceConfig` at all. Every other opcode is
//! additionally checked against `DeviceConfig.transform`: the spec's
//! `CAP_TRANSFORM` row states "without it those opcodes are unknown", so a
//! `CAP_TRANSFORM`-only opcode on a baseline device is `E_BAD_OPCODE`
//! exactly like an opcode outside the map -- both are skipped by trusting
//! the `length` word already validated by the framing check, per the
//! spec's "Unknown opcodes ... `length` is trusted" line.
//!
//! `TEX_PALETTE` gets the same "opcode doesn't exist" treatment, but keyed
//! off `TEXFMT_SUPPORTED` bit 9 (`I8_INDEXED`) instead of `transform`: the
//! spec makes the opcode itself optional, separately from any tier.
//!
//! ## GL enumerant legality stays out of this module
//!
//! Raw GL enumerants (blend factors, depth funcs, texenv modes, primitive
//! types, matrix/light/material `pname`s, and so on) are carried through
//! [`Command`] payloads as plain `u32`s, never converted to typed Rust
//! enums or checked for legality here: deciding whether an enumerant is
//! legal is a *GL error* (`GL_INVALID_ENUM`), which belongs to the
//! per-context state machine ([`super::state`]), not to this wire layer.
//! The one narrow exception is `LIGHT`'s `light` and `CLIP_PLANE`'s
//! `plane` fields: the Khronos registry defines `GL_LIGHTn`/
//! `GL_CLIP_PLANEn` as a contiguous run starting at a fixed, non-board-
//! configurable value, so recovering `n` to compare against
//! `max_lights`/`max_clip_planes` for `E_LIMIT` is arithmetic on a fixed
//! constant, not enumerant-legality decoding -- the raw word is still
//! passed through to `Command` unchanged, and `state.rs` performs the real
//! `GL_INVALID_ENUM` check independently.

use super::proto::{self, Command, ErrorCode, Ref, RefSpace, VertexFormat};

/// `GL_LIGHT0`. Fixed by the Khronos OpenGL registry, not board-configured;
/// see this module's doc comment on why `ring.rs` knows it.
const GL_LIGHT0: u32 = 0x4000;

/// `GL_CLIP_PLANE0`. Fixed by the Khronos OpenGL registry; see
/// `GL_LIGHT0`'s comment.
const GL_CLIP_PLANE0: u32 = 0x3000;

/// The device's static configuration: the capability bits and the
/// board-fixed limits registers from `docs/internals/c3d.md`'s "Global
/// registers" table that this decoder needs to raise `E_BAD_OPCODE` (tier
/// gating), `E_BAD_REF`, `E_BAD_ID`, `E_LIMIT` and `E_UNSUPPORTED_FORMAT`.
/// It carries no *live* state -- no bound surface, no texture objects --
/// which is exactly the line the module doc comment draws between this
/// module's errors and `state.rs`'s.
///
/// `surffmt_supported` folds `SURFFMT_SUPPORTED_LO`/`_HI` into one 64-bit
/// mask (bit `n` = format `n` supported) since nothing here needs them
/// kept as separate 32-bit registers; a caller reading the real registers
/// combines them as `lo as u64 | (hi as u64) << 32`.
#[derive(Debug, Clone, Copy)]
pub struct DeviceConfig {
    // Capability bits (`CAPS0`, `c3d-tiers`).
    /// `CAP_GUESTMEM`: space `1` refs are `E_BAD_REF` without it.
    pub guestmem: bool,
    /// `CAP_TRANSFORM`: gates the transform-tier opcodes (see this
    /// module's doc comment).
    pub transform: bool,
    /// `CAP_MULTITEXTURE`. `MAX_TEXTURE_UNITS > 1` is documented to imply
    /// this bit; `check_unit` enforces both independently rather than
    /// trusting a caller to keep them consistent.
    pub multitexture: bool,
    /// `CAP_SURFACE_GUESTADDR`: required for `SURFACE_DEFINE`'s
    /// guest-address `flags` bit.
    pub surface_guestaddr: bool,

    // Limits (`c3d-global-registers`).
    pub aperture_size: u32,
    pub max_texture_units: u32,
    pub max_textures: u32,
    pub max_surfaces: u32,
    pub max_lights: u32,
    pub max_clip_planes: u32,
    pub max_texture_size: u32,
    pub max_surface_width: u32,
    pub max_surface_height: u32,
    /// `TEXFMT_SUPPORTED`: bit `n` set means [`proto::TextureFormat`]
    /// value `n` is accepted.
    pub texfmt_supported: u32,
    /// `SURFFMT_SUPPORTED_LO`/`_HI` combined; bit `n` set means
    /// [`proto::SurfaceFormat`] value `n` is accepted. Values `>= 64`
    /// (notably `DEPTH` = `255`) are never representable here and so
    /// never "supported" by this bitmap -- `READ_PIXELS` special-cases
    /// `DEPTH` separately, exactly as the spec does.
    pub surffmt_supported: u64,
}

impl Default for DeviceConfig {
    /// Copperline's own configuration (`docs/internals/c3d.md`'s
    /// "Copperline implementation notes": bits 0, 1, 2, 3, 4 and 6 of
    /// `CAPS0`, i.e. every capability this decoder reads). Limits are
    /// Copperline's chosen values, at or above every spec-stated minimum;
    /// `max_texture_units` matches `state::Limits::default()`'s two units
    /// (the first client's two-unit multitexture target). `texfmt_supported`
    /// covers the spec's required minimum set (`RGBA8`..`A8`, bits 0-7);
    /// `surffmt_supported` covers the required minimum (`R5G6B5`,
    /// `A8R8G8B8`) plus the other chunky formats Copperline's surfaces
    /// also accept.
    fn default() -> Self {
        DeviceConfig {
            guestmem: true,
            transform: true,
            multitexture: true,
            surface_guestaddr: true,
            aperture_size: proto::APERTURE_SIZE_DEFAULT,
            max_texture_units: 2,
            max_textures: 256,
            max_surfaces: 16,
            max_lights: 8,
            max_clip_planes: 6,
            max_texture_size: 1024,
            max_surface_width: 1024,
            max_surface_height: 768,
            texfmt_supported: 0xFF, // RGBA8, RGB8, RGB565, RGBA4444, RGBA5551, L8, LA8, A8
            surffmt_supported: (1 << 1) // R5G6B5
                | (1 << 3) // R5G5B5
                | (1 << 5) // A8R8G8B8
                | (1 << 6) // B8G8R8A8 -- RTG_COLOR_FORMAT_BGRA (z3660), glQuake's screen format
                | (1 << 7) // R8G8B8A8
                | (1 << 8) // R8G8B8
                | (1 << 9), // B8G8R8
        }
    }
}

/// A latched protocol error: the code, the ring offset of the command that
/// raised it (`ERROR_OFFSET`), and whether it halts the context or is
/// skipped (`ErrorCode::halts`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProtoError {
    pub code: ErrorCode,
    pub offset: u32,
    pub halt: bool,
}

/// One step of decoding: either a command was decoded, or a protocol error
/// was raised (and, per `ErrorCode::halts`, decoding may or may not
/// continue from the next command).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Step<'a> {
    Command(Command<'a>),
    Error(ProtoError),
}

/// Word count of a ref (`word A` + `word B`).
const REF_WORDS: u32 = 2;

/// Decodes commands from `ring[head..]` up to `tail`, advancing `head` as
/// it goes. `ring` must be exactly `RING_SIZE` bytes (a context's whole
/// ring, whichever address space it lives in -- the caller resolves
/// `RING_BASE`/`RING_SIZE` before handing this module the bytes) and
/// `head`/`tail` are byte offsets into it, both `< ring.len()`, both
/// word-aligned, per `docs/internals/c3d.md`'s register descriptions.
///
/// A `RingCursor` is stateless beyond `head` and a latched `halted` flag:
/// it is built fresh from the current register values on every poll, so it
/// never needs to remember an `ERROR_ACK` across calls -- that lives in the
/// context registers, which [`super::state`] owns.
pub struct RingCursor<'a> {
    ring: &'a [u8],
    tail: u32,
    head: u32,
    config: DeviceConfig,
    halted: bool,
    /// Set by `new` when `head`/`tail` are already an `E_RING_OVERRUN`;
    /// consumed (and cleared) by the first `step` call, which is what
    /// actually reports it. See this module's doc comment.
    pending_overrun: bool,
    /// `true` for a [`Self::new_linear`] cursor over a `CALL`ed buffer:
    /// disables `step`'s "a command reaching exactly the physical end wraps
    /// `head` to `0`" behaviour, which is correct only for an actual
    /// circular ring (where `tail` is a separate, independently-wrapping
    /// offset). A called buffer's `tail` always equals `ring.len()`, so
    /// without this flag a command landing exactly on the buffer's end
    /// would wrap `head` back to `0` instead of leaving it equal to `tail`
    /// (buffer exhausted) -- silently re-decoding the buffer from the start
    /// instead of stopping.
    linear: bool,
}

impl<'a> RingCursor<'a> {
    /// Builds a cursor over `ring[head..tail]` (circularly). If `tail` has
    /// already overtaken `head` by more than one ring's worth of bytes --
    /// the guest violated "must not advance `RING_TAIL` past
    /// `RING_HEAD - 4`" -- the cursor starts halted and its first `step`
    /// reports `E_RING_OVERRUN`.
    pub fn new(ring: &'a [u8], head: u32, tail: u32, config: DeviceConfig) -> Self {
        let ring_size = ring.len() as u32;
        let mut cursor = RingCursor {
            ring,
            tail,
            head,
            config,
            halted: false,
            pending_overrun: false,
            linear: false,
        };
        if ring_size >= 4 {
            let occupied = if tail >= head {
                tail - head
            } else {
                ring_size - head + tail
            };
            // The ring is full when one word short of a full circle; more
            // than that means the guest let RING_TAIL overtake RING_HEAD.
            if occupied > ring_size - 4 {
                cursor.halted = true;
                cursor.pending_overrun = true;
            }
        }
        cursor
    }

    /// Builds a cursor over a `CALL`ed buffer: `ring[0..ring.len()]`, linear,
    /// no wraparound and no `RING_HEAD`/`RING_TAIL` register of its own (see
    /// `c3d-cmd-control`'s `CALL` prose). Deliberately does **not** run
    /// [`Self::new`]'s `E_RING_OVERRUN` check: that check exists only
    /// because a real ring reserves one word to tell "full" from "empty"
    /// when `head`/`tail` chase each other circularly: a called buffer has
    /// no such ambiguity (its whole extent is always fully "available", and
    /// nothing external can move its `tail` mid-decode), so applying the
    /// same "one word short of full" arithmetic here would spuriously halt
    /// on every buffer whose commands consume it exactly to the end. A
    /// command that would run past `ring.len()` is still caught, by the
    /// ordinary framing check in [`Self::step`] (`available` computed
    /// against `tail == ring.len()`), which is the spec's actual
    /// requirement: "a command in the buffer that would extend past the end
    /// of the referenced range is a framing error ... exactly as a ring
    /// overrun does (`E_BAD_LENGTH`)".
    pub(crate) fn new_linear(ring: &'a [u8], config: DeviceConfig) -> Self {
        let tail = ring.len() as u32;
        RingCursor {
            ring,
            tail,
            head: 0,
            config,
            halted: false,
            pending_overrun: false,
            linear: true,
        }
    }

    /// The current `RING_HEAD` value: bytes before it have been consumed.
    pub fn head(&self) -> u32 {
        self.head
    }

    /// Whether the cursor has halted (a halting error was just raised, or
    /// `RING_TAIL` overran `RING_HEAD` at construction). No further steps
    /// are produced once this is true; `super::state` clears it by
    /// resuming decode at `RING_HEAD` after `ERROR_ACK`, which in this pure
    /// module just means building a fresh `RingCursor`.
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Decodes and consumes the next command, advancing `head`. Returns
    /// `None` once the ring is empty (`head == tail`) or the cursor is
    /// halted (after yielding a pending `E_RING_OVERRUN`, if any, exactly
    /// once).
    pub fn step(&mut self) -> Option<Step<'a>> {
        if self.pending_overrun {
            self.pending_overrun = false;
            return Some(Step::Error(ProtoError {
                code: ErrorCode::RingOverrun,
                offset: self.head,
                halt: true,
            }));
        }
        if self.halted || self.head == self.tail {
            return None;
        }
        let ring_size = self.ring.len() as u32;
        let start = self.head;

        let word0 = read_u32(self.ring, start);
        let opcode = (word0 >> 16) as u16;
        let length_words = word0 & 0xFFFF;

        if length_words == 0 {
            return Some(self.raise(ErrorCode::BadLength, start));
        }
        let total_bytes = length_words * 4;

        // Space available before either RING_TAIL (if not wrapped past
        // `start`) or the physical ring end (if it has), per the framing
        // doc comment above.
        let available = if self.tail >= start {
            self.tail - start
        } else {
            ring_size - start
        };
        if total_bytes > available {
            return Some(self.raise(ErrorCode::BadLength, start));
        }

        let new_head = if !self.linear && start + total_bytes == ring_size {
            0
        } else {
            start + total_bytes
        };

        let step = self.decode_body(opcode, start, length_words);
        self.head = new_head;
        if let Step::Error(err) = &step {
            if err.halt {
                self.halted = true;
                // A halting error leaves RING_HEAD at the offending
                // command until ERROR_ACK, not past it.
                self.head = start;
            }
        }
        Some(step)
    }

    fn raise(&mut self, code: ErrorCode, offset: u32) -> Step<'a> {
        let halt = code.halts();
        if halt {
            self.halted = true;
            // `head` is restored to `offset` by the caller for the halting
            // path (see `step`); for a fresh cursor whose very first word
            // is malformed, `self.head` already equals `offset`.
        }
        Step::Error(ProtoError { code, offset, halt })
    }

    /// Whether `opcode` needs `CAP_TRANSFORM`: every `0x03xx`/`0x04xx`
    /// opcode, the GL-space draw commands (not their `_WIN` counterparts),
    /// and the two raster-state opcodes the spec calls out as transform
    /// tier (`DEPTH_RANGE`, `VIEWPORT`). See this module's doc comment.
    fn opcode_needs_transform(opcode: u16) -> bool {
        use proto::*;
        matches!(
            opcode,
            OP_MATRIX_MODE
                | OP_LOAD_MATRIX
                | OP_LOAD_IDENTITY
                | OP_MULT_MATRIX
                | OP_PUSH_MATRIX
                | OP_POP_MATRIX
                | OP_TRANSLATE
                | OP_ROTATE
                | OP_SCALE
                | OP_FRUSTUM
                | OP_ORTHO
                | OP_LIGHT
                | OP_LIGHT_MODEL
                | OP_MATERIAL
                | OP_COLOR_MATERIAL
                | OP_CLIP_PLANE
                | OP_TEXGEN
                | OP_TEXGEN_PLANE
                | OP_DEPTH_RANGE
                | OP_VIEWPORT
                | OP_DRAW_INLINE
                | OP_DRAW_ARRAYS
                | OP_DRAW_ELEMENTS
        )
    }

    /// Decodes the payload of a structurally-framed command (length
    /// already validated against the ring/tail bounds). `body` is the
    /// command's payload words, i.e. `ring[start+4 .. start+length*4]`.
    fn decode_body(&mut self, opcode: u16, start: u32, length_words: u32) -> Step<'a> {
        if !self.config.transform && Self::opcode_needs_transform(opcode) {
            return self.raise(ErrorCode::BadOpcode, start);
        }
        // TEX_PALETTE is spec-optional, gated by its own TEXFMT_SUPPORTED
        // bit rather than by tier: "a device reports it with
        // TEXFMT_SUPPORTED bit 9 (I8_INDEXED); without it TEX_PALETTE is
        // E_BAD_OPCODE" -- not E_UNSUPPORTED_FORMAT, which is for a format
        // value the opcode's own payload names (TEX_PALETTE's never does;
        // it only ever binds a palette, it doesn't select an image
        // format).
        if opcode == proto::OP_TEX_PALETTE
            && (self.config.texfmt_supported >> proto::TextureFormat::I8Indexed as u32) & 1 == 0
        {
            return self.raise(ErrorCode::BadOpcode, start);
        }

        // Copy the `&'a [u8]` out of `self` first so the resulting slice's
        // lifetime is `'a`, not tied to this method's `&mut self` borrow --
        // needed because several branches below call back into `self`
        // (`self.raise`, `self.validate_ref`, the `check_*` helpers) while
        // still holding `body`.
        let ring: &'a [u8] = self.ring;
        let body = &ring[(start + 4) as usize..(start + length_words * 4) as usize];

        macro_rules! fixed {
            ($payload_words:expr) => {{
                let expected = 1 + $payload_words;
                if length_words != expected {
                    return self.raise(ErrorCode::BadArg, start);
                }
            }};
        }
        macro_rules! check {
            ($result:expr) => {
                if let Err(e) = $result {
                    return e;
                }
            };
        }

        use proto::*;
        Step::Command(match opcode {
            OP_NOP => Command::Nop,
            OP_FENCE => {
                fixed!(1);
                Command::Fence { id: w32(body, 0) }
            }
            OP_FLUSH => {
                fixed!(0);
                Command::Flush
            }
            OP_FINISH => {
                fixed!(0);
                Command::Finish
            }
            OP_CTX_RESET_STATE => {
                fixed!(0);
                Command::CtxResetState
            }
            OP_CALL => {
                fixed!(2);
                let data = Ref::decode(w32(body, 0), w32(body, 1));
                check!(self.validate_ref(data, start));
                Command::Call { data }
            }

            OP_SURFACE_DEFINE => {
                fixed!(7);
                let id = w32(body, 0);
                let width = w32(body, 1);
                let height = w32(body, 2);
                let stride_bytes = w32(body, 3);
                let format = w32(body, 4);
                let flags = w32(body, 5);
                let address = w32(body, 6);
                if flags & !SURFACE_DEFINE_FLAG_GUEST_ADDR != 0 {
                    return self.raise(ErrorCode::BadArg, start);
                }
                check!(self.check_id(id, self.config.max_surfaces, start));
                let guest_addr = flags & SURFACE_DEFINE_FLAG_GUEST_ADDR != 0;
                if guest_addr && !self.config.surface_guestaddr {
                    return self.raise(ErrorCode::BadArg, start);
                }
                check!(self.check_surface_size(width, height, start));
                let surface_format = match self.surface_format(format, start) {
                    Ok(f) => f,
                    Err(e) => return e,
                };
                let row_bytes = match width.checked_mul(surface_format.bytes_per_pixel()) {
                    Some(b) => b,
                    None => return self.raise(ErrorCode::BadArg, start),
                };
                if stride_bytes < row_bytes {
                    return self.raise(ErrorCode::BadArg, start);
                }
                // A guest-address surface is unchecked here for the same
                // reason a space-1 Ref is: reachability of guest memory
                // depends on installed RAM and other boards' apertures,
                // which this module has no way to know.
                if !guest_addr {
                    check!(self.check_aperture_surface_extent(
                        address,
                        height,
                        stride_bytes,
                        row_bytes,
                        start
                    ));
                }
                Command::SurfaceDefine {
                    id,
                    width,
                    height,
                    stride_bytes,
                    format,
                    flags,
                    address,
                }
            }
            OP_SURFACE_DESTROY => {
                fixed!(1);
                let id = w32(body, 0);
                check!(self.check_id(id, self.config.max_surfaces, start));
                Command::SurfaceDestroy { id }
            }
            OP_SET_DRAW_SURFACE => {
                fixed!(1);
                let id = w32(body, 0);
                check!(self.check_id_or_none(id, self.config.max_surfaces, start));
                Command::SetDrawSurface { id }
            }
            OP_SURFACE_UPLOAD => {
                fixed!(4);
                Command::SurfaceUpload {
                    x: w32(body, 0),
                    y: w32(body, 1),
                    w: w32(body, 2),
                    h: w32(body, 3),
                }
            }
            OP_SURFACE_READBACK => {
                fixed!(4);
                Command::SurfaceReadback {
                    x: w32(body, 0),
                    y: w32(body, 1),
                    w: w32(body, 2),
                    h: w32(body, 3),
                }
            }
            OP_CLEAR => {
                fixed!(1);
                let mask = w32(body, 0);
                if mask & !(CLEAR_MASK_COLOR | CLEAR_MASK_DEPTH) != 0 {
                    return self.raise(ErrorCode::BadArg, start);
                }
                Command::Clear { mask }
            }

            OP_ENABLE => {
                fixed!(1);
                Command::Enable { cap: w32(body, 0) }
            }
            OP_DISABLE => {
                fixed!(1);
                Command::Disable { cap: w32(body, 0) }
            }
            OP_BLEND_FUNC => {
                fixed!(2);
                Command::BlendFunc {
                    sfactor: w32(body, 0),
                    dfactor: w32(body, 1),
                }
            }
            OP_DEPTH_FUNC => {
                fixed!(1);
                Command::DepthFunc { func: w32(body, 0) }
            }
            OP_DEPTH_MASK => {
                fixed!(1);
                Command::DepthMask { flag: w32(body, 0) }
            }
            OP_DEPTH_RANGE => {
                fixed!(2);
                Command::DepthRange {
                    near: f32b(body, 0),
                    far: f32b(body, 1),
                }
            }
            OP_ALPHA_FUNC => {
                fixed!(2);
                Command::AlphaFunc {
                    func: w32(body, 0),
                    reference: f32b(body, 1),
                }
            }
            OP_CULL_FACE => {
                fixed!(1);
                Command::CullFace { mode: w32(body, 0) }
            }
            OP_FRONT_FACE => {
                fixed!(1);
                Command::FrontFace { mode: w32(body, 0) }
            }
            OP_SHADE_MODEL => {
                fixed!(1);
                Command::ShadeModel { mode: w32(body, 0) }
            }
            OP_COLOR_MASK => {
                fixed!(4);
                Command::ColorMask {
                    r: w32(body, 0),
                    g: w32(body, 1),
                    b: w32(body, 2),
                    a: w32(body, 3),
                }
            }
            OP_SCISSOR => {
                fixed!(4);
                Command::Scissor {
                    x: w32(body, 0),
                    y: w32(body, 1),
                    w: w32(body, 2),
                    h: w32(body, 3),
                }
            }
            OP_VIEWPORT => {
                fixed!(4);
                Command::Viewport {
                    x: w32(body, 0),
                    y: w32(body, 1),
                    w: w32(body, 2),
                    h: w32(body, 3),
                }
            }
            OP_POLYGON_OFFSET => {
                fixed!(2);
                Command::PolygonOffset {
                    factor: f32b(body, 0),
                    units: f32b(body, 1),
                }
            }
            OP_CLEAR_COLOR => {
                fixed!(4);
                Command::ClearColor {
                    r: f32b(body, 0),
                    g: f32b(body, 1),
                    b: f32b(body, 2),
                    a: f32b(body, 3),
                }
            }
            OP_CLEAR_DEPTH => {
                fixed!(1);
                Command::ClearDepth {
                    depth: f32b(body, 0),
                }
            }
            OP_FOG_MODE => {
                fixed!(1);
                Command::FogMode { mode: w32(body, 0) }
            }
            OP_FOG_PARAMS => {
                fixed!(3);
                Command::FogParams {
                    density: f32b(body, 0),
                    start: f32b(body, 1),
                    end: f32b(body, 2),
                }
            }
            OP_FOG_COLOR => {
                fixed!(4);
                Command::FogColor {
                    r: f32b(body, 0),
                    g: f32b(body, 1),
                    b: f32b(body, 2),
                    a: f32b(body, 3),
                }
            }
            OP_HINT => {
                fixed!(2);
                Command::Hint {
                    target: w32(body, 0),
                    mode: w32(body, 1),
                }
            }
            OP_LINE_WIDTH => {
                fixed!(1);
                Command::LineWidth {
                    width: f32b(body, 0),
                }
            }
            OP_POINT_SIZE => {
                fixed!(1);
                Command::PointSize {
                    size: f32b(body, 0),
                }
            }
            OP_POLYGON_MODE => {
                fixed!(2);
                Command::PolygonMode {
                    face: w32(body, 0),
                    mode: w32(body, 1),
                }
            }
            OP_BLEND_EQUATION => {
                fixed!(1);
                Command::BlendEquation { mode: w32(body, 0) }
            }
            OP_BLEND_FUNC_SEPARATE => {
                fixed!(4);
                Command::BlendFuncSeparate {
                    src_rgb: w32(body, 0),
                    dst_rgb: w32(body, 1),
                    src_a: w32(body, 2),
                    dst_a: w32(body, 3),
                }
            }

            OP_MATRIX_MODE => {
                fixed!(1);
                Command::MatrixMode { mode: w32(body, 0) }
            }
            OP_LOAD_MATRIX => {
                fixed!(16);
                Command::LoadMatrix { m: matrix(body) }
            }
            OP_LOAD_IDENTITY => {
                fixed!(0);
                Command::LoadIdentity
            }
            OP_MULT_MATRIX => {
                fixed!(16);
                Command::MultMatrix { m: matrix(body) }
            }
            OP_PUSH_MATRIX => {
                fixed!(0);
                Command::PushMatrix
            }
            OP_POP_MATRIX => {
                fixed!(0);
                Command::PopMatrix
            }
            OP_TRANSLATE => {
                fixed!(3);
                Command::Translate {
                    x: f32b(body, 0),
                    y: f32b(body, 1),
                    z: f32b(body, 2),
                }
            }
            OP_ROTATE => {
                fixed!(4);
                Command::Rotate {
                    angle_deg: f32b(body, 0),
                    x: f32b(body, 1),
                    y: f32b(body, 2),
                    z: f32b(body, 3),
                }
            }
            OP_SCALE => {
                fixed!(3);
                Command::Scale {
                    x: f32b(body, 0),
                    y: f32b(body, 1),
                    z: f32b(body, 2),
                }
            }
            OP_FRUSTUM => {
                fixed!(6);
                Command::Frustum {
                    l: f32b(body, 0),
                    r: f32b(body, 1),
                    b: f32b(body, 2),
                    t: f32b(body, 3),
                    n: f32b(body, 4),
                    f: f32b(body, 5),
                }
            }
            OP_ORTHO => {
                fixed!(6);
                Command::Ortho {
                    l: f32b(body, 0),
                    r: f32b(body, 1),
                    b: f32b(body, 2),
                    t: f32b(body, 3),
                    n: f32b(body, 4),
                    f: f32b(body, 5),
                }
            }

            OP_LIGHT => {
                fixed!(6);
                let light = w32(body, 0);
                check!(self.check_light(light, start));
                Command::Light {
                    light,
                    pname: w32(body, 1),
                    v: vec4(body, 2),
                }
            }
            OP_LIGHT_MODEL => {
                fixed!(5);
                Command::LightModel {
                    pname: w32(body, 0),
                    v: vec4(body, 1),
                }
            }
            OP_MATERIAL => {
                fixed!(6);
                Command::Material {
                    face: w32(body, 0),
                    pname: w32(body, 1),
                    v: vec4(body, 2),
                }
            }
            OP_COLOR_MATERIAL => {
                fixed!(2);
                Command::ColorMaterial {
                    face: w32(body, 0),
                    mode: w32(body, 1),
                }
            }
            OP_CLIP_PLANE => {
                fixed!(5);
                let plane = w32(body, 0);
                check!(self.check_clip_plane(plane, start));
                Command::ClipPlane {
                    plane,
                    eq: vec4(body, 1),
                }
            }
            OP_TEXGEN => {
                fixed!(3);
                let unit = w32(body, 0);
                check!(self.check_unit(unit, start));
                Command::TexGen {
                    unit,
                    coord: w32(body, 1),
                    mode: w32(body, 2),
                }
            }
            OP_TEXGEN_PLANE => {
                fixed!(6);
                let unit = w32(body, 0);
                check!(self.check_unit(unit, start));
                Command::TexGenPlane {
                    unit,
                    coord: w32(body, 1),
                    plane: w32(body, 2),
                    eq: vec4(body, 3),
                }
            }

            OP_TEX_CREATE => {
                fixed!(1);
                let id = w32(body, 0);
                check!(self.check_id(id, self.config.max_textures, start));
                Command::TexCreate { id }
            }
            OP_TEX_DESTROY => {
                fixed!(1);
                let id = w32(body, 0);
                check!(self.check_id(id, self.config.max_textures, start));
                Command::TexDestroy { id }
            }
            OP_TEX_BIND => {
                fixed!(2);
                let unit = w32(body, 0);
                let id = w32(body, 1);
                check!(self.check_unit(unit, start));
                check!(self.check_id_or_none(id, self.config.max_textures, start));
                Command::TexBind { unit, id }
            }
            OP_TEX_IMAGE => {
                fixed!(8);
                let id = w32(body, 0);
                let format = w32(body, 2);
                let width = w32(body, 3);
                let height = w32(body, 4);
                check!(self.check_id(id, self.config.max_textures, start));
                check!(self.check_pow2_dims(width, height, start));
                check!(self.check_texture_size(width, height, start));
                check!(self.check_texfmt(format, start));
                let data = Ref::decode(w32(body, 6), w32(body, 7));
                check!(self.validate_ref(data, start));
                Command::TexImage {
                    id,
                    level: w32(body, 1),
                    format,
                    width,
                    height,
                    row_bytes: w32(body, 5),
                    data,
                }
            }
            OP_TEX_SUBIMAGE => {
                fixed!(10);
                let id = w32(body, 0);
                let width = w32(body, 4);
                let height = w32(body, 5);
                let format = w32(body, 6);
                check!(self.check_id(id, self.config.max_textures, start));
                check!(self.check_texture_size(width, height, start));
                check!(self.check_texfmt(format, start));
                let data = Ref::decode(w32(body, 8), w32(body, 9));
                check!(self.validate_ref(data, start));
                Command::TexSubImage {
                    id,
                    level: w32(body, 1),
                    x: w32(body, 2),
                    y: w32(body, 3),
                    width,
                    height,
                    format,
                    row_bytes: w32(body, 7),
                    data,
                }
            }
            OP_TEX_PARAM => {
                fixed!(3);
                let id = w32(body, 0);
                check!(self.check_id(id, self.config.max_textures, start));
                Command::TexParam {
                    id,
                    pname: w32(body, 1),
                    value: w32(body, 2),
                }
            }
            OP_TEX_ENV => {
                fixed!(3);
                let unit = w32(body, 0);
                check!(self.check_unit(unit, start));
                Command::TexEnv {
                    unit,
                    pname: w32(body, 1),
                    value: w32(body, 2),
                }
            }
            OP_TEX_ENV_COLOR => {
                fixed!(5);
                let unit = w32(body, 0);
                check!(self.check_unit(unit, start));
                Command::TexEnvColor {
                    unit,
                    r: f32b(body, 1),
                    g: f32b(body, 2),
                    b: f32b(body, 3),
                    a: f32b(body, 4),
                }
            }
            OP_ACTIVE_UNIT => {
                fixed!(1);
                let unit = w32(body, 0);
                check!(self.check_unit(unit, start));
                Command::ActiveUnit { unit }
            }
            OP_TEX_PALETTE => {
                fixed!(4);
                let id = w32(body, 0);
                let entries = w32(body, 1);
                check!(self.check_id(id, self.config.max_textures, start));
                if entries > 256 {
                    return self.raise(ErrorCode::BadArg, start);
                }
                let data = Ref::decode(w32(body, 2), w32(body, 3));
                check!(self.validate_ref(data, start));
                Command::TexPalette { id, entries, data }
            }
            OP_TEX_COPY_IMAGE => {
                fixed!(7);
                let id = w32(body, 0);
                let format = w32(body, 2);
                let width = w32(body, 5);
                let height = w32(body, 6);
                check!(self.check_id(id, self.config.max_textures, start));
                check!(self.check_texture_size(width, height, start));
                check!(self.check_texfmt(format, start));
                Command::TexCopyImage {
                    id,
                    level: w32(body, 1),
                    format,
                    x: w32(body, 3),
                    y: w32(body, 4),
                    width,
                    height,
                }
            }
            OP_TEX_COPY_SUBIMAGE => {
                fixed!(8);
                let id = w32(body, 0);
                let width = w32(body, 6);
                let height = w32(body, 7);
                check!(self.check_id(id, self.config.max_textures, start));
                check!(self.check_texture_size(width, height, start));
                Command::TexCopySubImage {
                    id,
                    level: w32(body, 1),
                    xoff: w32(body, 2),
                    yoff: w32(body, 3),
                    x: w32(body, 4),
                    y: w32(body, 5),
                    width,
                    height,
                }
            }

            OP_CURRENT_COLOR => {
                fixed!(4);
                Command::CurrentColor {
                    r: f32b(body, 0),
                    g: f32b(body, 1),
                    b: f32b(body, 2),
                    a: f32b(body, 3),
                }
            }
            OP_CURRENT_NORMAL => {
                fixed!(3);
                Command::CurrentNormal {
                    x: f32b(body, 0),
                    y: f32b(body, 1),
                    z: f32b(body, 2),
                }
            }
            OP_CURRENT_TEXCOORD => {
                fixed!(3);
                let unit = w32(body, 0);
                check!(self.check_unit(unit, start));
                Command::CurrentTexCoord {
                    unit,
                    s: f32b(body, 1),
                    t: f32b(body, 2),
                }
            }
            OP_CURRENT_FOGCOORD => {
                fixed!(1);
                Command::CurrentFogCoord { f: f32b(body, 0) }
            }

            OP_DRAW_INLINE => match self.decode_draw_inline(body, length_words, start, false) {
                Ok((prim, format, count, vertices)) => Command::DrawInline {
                    prim,
                    format,
                    count,
                    vertices,
                },
                Err(err) => return err,
            },
            OP_DRAW_INLINE_WIN => match self.decode_draw_inline(body, length_words, start, true) {
                Ok((prim, format, count, vertices)) => Command::DrawInlineWin {
                    prim,
                    format,
                    count,
                    vertices,
                },
                Err(err) => return err,
            },
            OP_DRAW_ARRAYS => match self.decode_draw_arrays(body, length_words, start, false) {
                Ok((prim, format, count, descriptors)) => Command::DrawArrays {
                    prim,
                    format,
                    count,
                    descriptors,
                },
                Err(err) => return err,
            },
            OP_DRAW_ARRAYS_WIN => match self.decode_draw_arrays(body, length_words, start, true) {
                Ok((prim, format, count, descriptors)) => Command::DrawArraysWin {
                    prim,
                    format,
                    count,
                    descriptors,
                },
                Err(err) => return err,
            },
            OP_DRAW_ELEMENTS => match self.decode_draw_elements(body, length_words, start, false) {
                Ok((
                    prim,
                    format,
                    count,
                    index_type,
                    min_index,
                    max_index,
                    index_ref,
                    descriptors,
                )) => Command::DrawElements {
                    prim,
                    format,
                    count,
                    index_type,
                    min_index,
                    max_index,
                    index_ref,
                    descriptors,
                },
                Err(err) => return err,
            },
            OP_DRAW_ELEMENTS_WIN => {
                match self.decode_draw_elements(body, length_words, start, true) {
                    Ok((
                        prim,
                        format,
                        count,
                        index_type,
                        min_index,
                        max_index,
                        index_ref,
                        descriptors,
                    )) => Command::DrawElementsWin {
                        prim,
                        format,
                        count,
                        index_type,
                        min_index,
                        max_index,
                        index_ref,
                        descriptors,
                    },
                    Err(err) => return err,
                }
            }

            OP_QUERY => {
                fixed!(3);
                let dest = Ref::decode(w32(body, 1), w32(body, 2));
                check!(self.validate_ref(dest, start));
                Command::Query {
                    what: w32(body, 0),
                    dest,
                }
            }
            OP_READ_PIXELS => {
                fixed!(9);
                let format = w32(body, 4);
                let flags = w32(body, 6);
                if flags & !READ_PIXELS_FLAG_ROWS_BOTTOM_UP != 0 {
                    return self.raise(ErrorCode::BadArg, start);
                }
                // DEPTH (255) is always allowed for READ_PIXELS and isn't
                // tracked by SURFFMT_SUPPORTED at all -- see
                // `check_surffmt`'s doc comment.
                if format != 255 {
                    check!(self.check_surffmt(format, start));
                }
                let dest = Ref::decode(w32(body, 7), w32(body, 8));
                check!(self.validate_ref(dest, start));
                Command::ReadPixels {
                    x: w32(body, 0),
                    y: w32(body, 1),
                    w: w32(body, 2),
                    h: w32(body, 3),
                    format,
                    row_bytes: w32(body, 5),
                    flags,
                    dest,
                }
            }

            // Not in the opcode map at all, or LOGIC_OP (see this module's
            // doc comment): E_BAD_OPCODE, skipped by trusting `length`.
            _ => return self.raise(ErrorCode::BadOpcode, start),
        })
    }

    /// `prim, format, count, vertices...` shared by `DRAW_INLINE`/
    /// `DRAW_INLINE_WIN`. `window_space` rejects a `NORMAL` format bit: a
    /// window-space vertex has already been transformed and lit, so a
    /// normal cannot mean anything there, on any device -- see
    /// `check_window_space_format`.
    fn decode_draw_inline(
        &mut self,
        body: &'a [u8],
        length_words: u32,
        start: u32,
        window_space: bool,
    ) -> Result<(u32, VertexFormat, u32, &'a [u8]), Step<'a>> {
        let prim = w32(body, 0);
        let format = VertexFormat(w32(body, 1));
        let count = w32(body, 2);
        self.check_window_space_format(format, window_space, start)?;
        let vertex_words = match format.vertex_words() {
            Some(w) => w,
            None => return Err(self.raise(ErrorCode::BadArg, start)),
        };
        let expected = 1 + 3 + u64::from(count) * u64::from(vertex_words);
        if u64::from(length_words) != expected {
            return Err(self.raise(ErrorCode::BadArg, start));
        }
        let vertices = &body[12..];
        Ok((prim, format, count, vertices))
    }

    /// `prim, format, count, arrays...` shared by `DRAW_ARRAYS`/
    /// `DRAW_ARRAYS_WIN`. See `decode_draw_inline`'s doc comment on
    /// `window_space`.
    fn decode_draw_arrays(
        &mut self,
        body: &'a [u8],
        length_words: u32,
        start: u32,
        window_space: bool,
    ) -> Result<(u32, VertexFormat, u32, &'a [u8]), Step<'a>> {
        let prim = w32(body, 0);
        let format = VertexFormat(w32(body, 1));
        let count = w32(body, 2);
        self.check_window_space_format(format, window_space, start)?;
        let descriptor_count = match format.descriptor_count() {
            Some(n) => n,
            None => return Err(self.raise(ErrorCode::BadArg, start)),
        };
        let expected = 1 + 3 + u64::from(descriptor_count) * u64::from(proto::DESCRIPTOR_WORDS);
        if u64::from(length_words) != expected {
            return Err(self.raise(ErrorCode::BadArg, start));
        }
        let descriptors = &body[12..];
        Ok((prim, format, count, descriptors))
    }

    /// `prim, format, count, index_type, min_index, max_index, index_ref,
    /// arrays...` shared by `DRAW_ELEMENTS`/`DRAW_ELEMENTS_WIN`. See
    /// `decode_draw_inline`'s doc comment on `window_space`.
    #[allow(clippy::type_complexity)]
    fn decode_draw_elements(
        &mut self,
        body: &'a [u8],
        length_words: u32,
        start: u32,
        window_space: bool,
    ) -> Result<(u32, VertexFormat, u32, u32, u32, u32, Ref, &'a [u8]), Step<'a>> {
        let prim = w32(body, 0);
        let format = VertexFormat(w32(body, 1));
        let count = w32(body, 2);
        self.check_window_space_format(format, window_space, start)?;
        let index_type = w32(body, 3);
        let min_index = w32(body, 4);
        let max_index = w32(body, 5);
        let index_ref = Ref::decode(w32(body, 6), w32(body, 7));
        self.validate_ref(index_ref, start)?;
        let descriptor_count = match format.descriptor_count() {
            Some(n) => n,
            None => return Err(self.raise(ErrorCode::BadArg, start)),
        };
        let fixed_words = 1 + 3 + 1 + 1 + 1 + REF_WORDS; // word0 + prim/format/count + index_type/min/max + index_ref
        let expected = u64::from(fixed_words)
            + u64::from(descriptor_count) * u64::from(proto::DESCRIPTOR_WORDS);
        if u64::from(length_words) != expected {
            return Err(self.raise(ErrorCode::BadArg, start));
        }
        let descriptors = &body[32..];
        Ok((
            prim,
            format,
            count,
            index_type,
            min_index,
            max_index,
            index_ref,
            descriptors,
        ))
    }

    /// Validates a ref against the structural rules ring.rs owns: longword
    /// alignment, `CAP_GUESTMEM` for space `1`, and the aperture bound for
    /// space `0`. Reachability of a space `1` guest address is left to a
    /// higher layer (it depends on installed RAM and other boards'
    /// apertures, which this module has no way to know).
    fn validate_ref(&mut self, r: Ref, start: u32) -> Result<(), Step<'a>> {
        if !r.address.is_multiple_of(4) {
            return Err(self.raise(ErrorCode::BadRef, start));
        }
        match r.space {
            RefSpace::Guest => {
                if !self.config.guestmem {
                    return Err(self.raise(ErrorCode::BadRef, start));
                }
            }
            RefSpace::Aperture => match r.address.checked_add(r.length) {
                Some(end) if end <= self.config.aperture_size => {}
                _ => return Err(self.raise(ErrorCode::BadRef, start)),
            },
        }
        Ok(())
    }

    /// `id` must be `1..=max`: `0` is the documented "none" sentinel and is
    /// never a legal object to name with a command that creates, destroys
    /// or otherwise addresses a specific object (`SURFACE_DEFINE`,
    /// `TEX_CREATE`, `TEX_IMAGE`, and so on).
    fn check_id(&mut self, id: u32, max: u32, start: u32) -> Result<(), Step<'a>> {
        if id == 0 || id > max {
            return Err(self.raise(ErrorCode::BadId, start));
        }
        Ok(())
    }

    /// `id` must be `<= max`; `0` is always legal here because the spec
    /// documents it as a real value for this opcode (`SET_DRAW_SURFACE`'s
    /// "no surface", `TEX_BIND`'s "null texture").
    fn check_id_or_none(&mut self, id: u32, max: u32, start: u32) -> Result<(), Step<'a>> {
        if id > max {
            return Err(self.raise(ErrorCode::BadId, start));
        }
        Ok(())
    }

    /// A texture unit must be below `max_texture_units`, and unit `> 0`
    /// additionally needs `CAP_MULTITEXTURE` -- checked independently of
    /// `max_texture_units` rather than trusting a caller to keep the two
    /// consistent, even though the spec documents `MAX_TEXTURE_UNITS > 1`
    /// as implying the capability bit.
    fn check_unit(&mut self, unit: u32, start: u32) -> Result<(), Step<'a>> {
        if unit >= self.config.max_texture_units || (unit > 0 && !self.config.multitexture) {
            return Err(self.raise(ErrorCode::Limit, start));
        }
        Ok(())
    }

    /// `light` is a wire `GL_LIGHTn` value; `n = light - GL_LIGHT0` must be
    /// below `max_lights`. A `light` word that isn't even `GL_LIGHT0`-shaped
    /// (below `GL_LIGHT0`, wrapping the subtraction to a huge value) falls
    /// out of range the same way, which is correct: `max_lights` is always
    /// `>= 8` under `CAP_TRANSFORM` per the spec, so nothing between
    /// `GL_LIGHT{max_lights}` and a huge wrapped value is ever a real GL
    /// enumerant either.
    fn check_light(&mut self, light: u32, start: u32) -> Result<(), Step<'a>> {
        let index = light.wrapping_sub(GL_LIGHT0);
        if index >= self.config.max_lights {
            return Err(self.raise(ErrorCode::Limit, start));
        }
        Ok(())
    }

    /// `plane` is a wire `GL_CLIP_PLANEn` value; see `check_light`'s
    /// reasoning, which applies identically here against `max_clip_planes`.
    fn check_clip_plane(&mut self, plane: u32, start: u32) -> Result<(), Step<'a>> {
        let index = plane.wrapping_sub(GL_CLIP_PLANE0);
        if index >= self.config.max_clip_planes {
            return Err(self.raise(ErrorCode::Limit, start));
        }
        Ok(())
    }

    fn check_texture_size(&mut self, width: u32, height: u32, start: u32) -> Result<(), Step<'a>> {
        if width > self.config.max_texture_size || height > self.config.max_texture_size {
            return Err(self.raise(ErrorCode::Limit, start));
        }
        Ok(())
    }

    fn check_surface_size(&mut self, width: u32, height: u32, start: u32) -> Result<(), Step<'a>> {
        if width > self.config.max_surface_width || height > self.config.max_surface_height {
            return Err(self.raise(ErrorCode::Limit, start));
        }
        Ok(())
    }

    /// `TEX_IMAGE`'s width/height must be powers of two (the spec's
    /// "Textures" section); `0` is not a power of two either, which is
    /// correct here since a `0`x`0` level is meaningless.
    fn check_pow2_dims(&mut self, width: u32, height: u32, start: u32) -> Result<(), Step<'a>> {
        if !width.is_power_of_two() || !height.is_power_of_two() {
            return Err(self.raise(ErrorCode::BadArg, start));
        }
        Ok(())
    }

    /// A window-space vertex has already been transformed *and lit*, so a
    /// `NORMAL` component cannot mean anything there -- rejected outright
    /// as `E_BAD_ARG`, on any device (not tier-gated: a `CAP_TRANSFORM`
    /// device is just as unable to make sense of a lit vertex's normal as
    /// a baseline one). Every other format bit stays legal in a
    /// window-space draw.
    fn check_window_space_format(
        &mut self,
        format: VertexFormat,
        window_space: bool,
        start: u32,
    ) -> Result<(), Step<'a>> {
        if window_space && format.has(VertexFormat::NORMAL) {
            return Err(self.raise(ErrorCode::BadArg, start));
        }
        Ok(())
    }

    fn check_texfmt(&mut self, format: u32, start: u32) -> Result<(), Step<'a>> {
        if format >= 32 || (self.config.texfmt_supported >> format) & 1 == 0 {
            return Err(self.raise(ErrorCode::UnsupportedFormat, start));
        }
        Ok(())
    }

    /// See `DeviceConfig::surffmt_supported`'s doc comment for why a
    /// format `>= 64` (in particular `DEPTH` = `255`) always fails this and
    /// why `READ_PIXELS` special-cases `DEPTH` before calling it.
    fn check_surffmt(&mut self, format: u32, start: u32) -> Result<(), Step<'a>> {
        if format >= 64 || (self.config.surffmt_supported >> format) & 1 == 0 {
            return Err(self.raise(ErrorCode::UnsupportedFormat, start));
        }
        Ok(())
    }

    /// `check_surffmt` plus decoding the value into a
    /// [`proto::SurfaceFormat`], for `SURFACE_DEFINE`'s `bytes_per_pixel`
    /// need (stride/extent validation). A format whose bit is set in
    /// `surffmt_supported` but that isn't a real `SurfaceFormat` variant
    /// (a misconfigured `DeviceConfig` advertising an undefined format) is
    /// also `E_UNSUPPORTED_FORMAT`: the device does not, in fact,
    /// meaningfully support it either way.
    fn surface_format(
        &mut self,
        format: u32,
        start: u32,
    ) -> Result<proto::SurfaceFormat, Step<'a>> {
        self.check_surffmt(format, start)?;
        proto::SurfaceFormat::try_from(format)
            .map_err(|_| self.raise(ErrorCode::UnsupportedFormat, start))
    }

    /// An aperture-backed surface (`SURFACE_DEFINE_FLAG_GUEST_ADDR` clear)
    /// must fit entirely inside the aperture: the last row begins at
    /// `address + (height - 1) * stride_bytes` and runs for `row_bytes`,
    /// so the extent's end is `address + (height - 1) * stride_bytes +
    /// row_bytes`, which must be `<= aperture_size`. `E_BAD_REF` is the
    /// spec's code for "outside the aperture", the same cause a `Ref`
    /// fails for; a surface's backing address is the same kind of thing
    /// even though it isn't the two-word `Ref` encoding. All arithmetic is
    /// checked: `height`, `stride_bytes` and `address` are guest-supplied
    /// and the product overflows a `u32` trivially.
    fn check_aperture_surface_extent(
        &mut self,
        address: u32,
        height: u32,
        stride_bytes: u32,
        row_bytes: u32,
        start: u32,
    ) -> Result<(), Step<'a>> {
        let rows_before_last = height.saturating_sub(1);
        let end = rows_before_last
            .checked_mul(stride_bytes)
            .and_then(|offset| offset.checked_add(address))
            .and_then(|last_row_start| last_row_start.checked_add(row_bytes));
        match end {
            Some(e) if e <= self.config.aperture_size => Ok(()),
            _ => Err(self.raise(ErrorCode::BadRef, start)),
        }
    }
}

/// Iterates a `RingCursor` to completion, decoding every command in one
/// submission in order. A convenience for callers (and tests) that don't
/// need to interleave decoding with anything else; `super::state` is free
/// to drive `RingCursor::step` itself instead when it does.
pub fn decode_all<'a>(
    ring: &'a [u8],
    head: u32,
    tail: u32,
    config: DeviceConfig,
) -> (Vec<Step<'a>>, u32) {
    let mut cursor = RingCursor::new(ring, head, tail, config);
    let mut steps = Vec::new();
    while let Some(step) = cursor.step() {
        steps.push(step);
    }
    (steps, cursor.head())
}

/// Reads big-endian word `index` (0-based, 4 bytes each) from `body`.
fn w32(body: &[u8], index: usize) -> u32 {
    let off = index * 4;
    u32::from_be_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]])
}

/// Reads big-endian word `index` from `body` as `f32`, per `docs/internals/
/// c3d.md`'s "IEEE-754 single precision, big-endian" convention.
fn f32b(body: &[u8], index: usize) -> f32 {
    f32::from_bits(w32(body, index))
}

fn vec4(body: &[u8], index: usize) -> [f32; 4] {
    [
        f32b(body, index),
        f32b(body, index + 1),
        f32b(body, index + 2),
        f32b(body, index + 3),
    ]
}

fn matrix(body: &[u8]) -> [f32; 16] {
    let mut m = [0f32; 16];
    for (i, slot) in m.iter_mut().enumerate() {
        *slot = f32b(body, i);
    }
    m
}

fn read_u32(ring: &[u8], offset: u32) -> u32 {
    w32(ring, (offset / 4) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c3d::proto::*;

    /// A generous default so tests that don't care about a particular
    /// limit don't have to think about it.
    fn config(guestmem: bool) -> DeviceConfig {
        DeviceConfig {
            guestmem,
            ..DeviceConfig::default()
        }
    }

    /// Builds a ring of `size` bytes and writes `words` starting at byte
    /// offset `at`, returning the ring.
    fn ring_with(size: usize, at: usize, words: &[u32]) -> Vec<u8> {
        let mut ring = vec![0u8; size];
        let mut off = at;
        for w in words {
            ring[off..off + 4].copy_from_slice(&w.to_be_bytes());
            off += 4;
        }
        ring
    }

    fn header(opcode: u16, length_words: u16) -> u32 {
        (opcode as u32) << 16 | length_words as u32
    }

    fn decode_one(ring: &[u8], head: u32, tail: u32) -> Step<'_> {
        let (mut steps, _) = decode_all(ring, head, tail, config(true));
        assert_eq!(steps.len(), 1, "expected exactly one decoded step");
        steps.remove(0)
    }

    fn decode_one_with(ring: &[u8], head: u32, tail: u32, c: DeviceConfig) -> Step<'_> {
        let (mut steps, _) = decode_all(ring, head, tail, c);
        assert_eq!(steps.len(), 1, "expected exactly one decoded step");
        steps.remove(0)
    }

    fn err(code: ErrorCode, offset: u32, halt: bool) -> Step<'static> {
        Step::Error(ProtoError { code, offset, halt })
    }

    // -- Framing --------------------------------------------------------

    #[test]
    fn empty_ring_yields_no_commands() {
        let ring = vec![0u8; 4096];
        let (steps, head) = decode_all(&ring, 0, 0, config(true));
        assert!(steps.is_empty());
        assert_eq!(head, 0);
    }

    #[test]
    fn a_single_fence_command_decodes_and_advances_head_by_its_length() {
        let ring = ring_with(4096, 0, &[header(OP_FENCE, 2), 0x0000_002A]);
        let (steps, head) = decode_all(&ring, 0, 8, config(true));
        assert_eq!(steps, vec![Step::Command(Command::Fence { id: 42 })]);
        assert_eq!(head, 8);
    }

    #[test]
    fn zero_length_is_e_bad_length_and_halts_at_the_command_offset() {
        let ring = ring_with(4096, 0, &[header(OP_FENCE, 0)]);
        let step = decode_one(&ring, 0, 4);
        assert_eq!(step, err(ErrorCode::BadLength, 0, true));
        let mut cursor = RingCursor::new(&ring, 0, 4, config(true));
        cursor.step();
        assert!(cursor.is_halted());
        assert_eq!(cursor.head(), 0);
    }

    #[test]
    fn a_command_extending_past_ring_tail_is_e_bad_length_and_halts() {
        // FENCE declares length 2 (8 bytes) but the tail only exposes 4.
        let ring = ring_with(4096, 0, &[header(OP_FENCE, 2), 0]);
        let step = decode_one(&ring, 0, 4);
        assert_eq!(step, err(ErrorCode::BadLength, 0, true));
    }

    #[test]
    fn a_command_that_would_wrap_the_physical_ring_end_is_e_bad_length() {
        // Ring is 16 bytes; word 0 of a FENCE (declared length 2, needing
        // 8 bytes) sits at byte 12, where only 4 bytes remain before the
        // physical end -- it would need to wrap to be readable, which
        // framing forbids. Only word 0 itself fits in the ring, which is
        // enough: the decoder must reject on the declared length alone,
        // never reading past it.
        let ring = ring_with(16, 12, &[header(OP_FENCE, 2)]);
        // tail < head models an already-wrapped tail (elsewhere in the
        // ring); available-before-end is then ring_size - head = 4 bytes,
        // less than the 8 the command declares.
        let (steps, _) = decode_all(&ring, 12, 4, config(true));
        assert_eq!(steps, vec![err(ErrorCode::BadLength, 12, true)]);
    }

    #[test]
    fn a_tail_exactly_one_word_short_of_a_full_circle_is_legal_not_overrun() {
        let ring = vec![0u8; 4096];
        let head = 100u32;
        let legal_tail = (head + (4096 - 4)) % 4096;
        let cursor = RingCursor::new(&ring, head, legal_tail, config(true));
        assert!(!cursor.is_halted());
    }

    #[test]
    fn ring_tail_overtaking_ring_head_raises_e_ring_overrun_and_halts() {
        // RING_SIZE 4096, head=100. The legal boundary (one word short of
        // a full circle) is `occupied == ring_size - 4`; nothing but the
        // register width stops a guest from writing a RING_TAIL one byte
        // further round than that (RING_TAIL's "longword aligned" is a
        // guest-library contract this decoder does not get to assume --
        // see this module's doc comment on E_RING_OVERRUN), which is
        // exactly the boundary the spec's "occupied > ring_size - 4"
        // description names. A tail exactly a further *word* on from the
        // boundary would land back on `head`, which this representation
        // cannot distinguish from an empty ring (an inherent property of
        // reducing the occupancy mod RING_SIZE, not a gap in this
        // decoder); a byte short of that is unambiguous and halts.
        let ring = vec![0u8; 4096];
        let head = 100u32;
        let legal_tail = (head + (4096 - 4)) % 4096;
        let overrun_tail = (legal_tail + 1) % 4096;
        let mut cursor = RingCursor::new(&ring, head, overrun_tail, config(true));
        assert!(cursor.is_halted());
        assert_eq!(cursor.step(), Some(err(ErrorCode::RingOverrun, head, true)));
        // Reported exactly once; further steps are silent, per a halted
        // cursor's contract.
        assert!(cursor.step().is_none());
        assert_eq!(cursor.head(), head, "head does not advance on overrun");
    }

    #[test]
    fn nop_padding_to_the_ring_end_is_consumed_and_head_wraps_to_zero() {
        // Ring of 16 bytes; head at byte 8; a NOP with length covering the
        // remaining 8 bytes (2 words) pads to the end.
        let mut ring = vec![0u8; 16];
        ring[8..12].copy_from_slice(&header(OP_NOP, 2).to_be_bytes());
        // A real command afterwards, once head has wrapped to 0.
        ring[0..4].copy_from_slice(&header(OP_FLUSH, 1).to_be_bytes());
        // tail wraps past the physical end back to offset 4 (past FLUSH).
        let (steps, head) = decode_all(&ring, 8, 4, config(true));
        assert_eq!(
            steps,
            vec![Step::Command(Command::Nop), Step::Command(Command::Flush),]
        );
        assert_eq!(head, 4);
    }

    #[test]
    fn nop_length_may_be_any_value_at_least_one() {
        let ring = ring_with(4096, 0, &[header(OP_NOP, 1)]);
        let (steps, head) = decode_all(&ring, 0, 4, config(true));
        assert_eq!(steps, vec![Step::Command(Command::Nop)]);
        assert_eq!(head, 4);

        let ring = ring_with(4096, 0, &[header(OP_NOP, 5), 0, 0, 0, 0]);
        let (steps, head) = decode_all(&ring, 0, 20, config(true));
        assert_eq!(steps, vec![Step::Command(Command::Nop)]);
        assert_eq!(head, 20);
    }

    #[test]
    fn exact_fit_at_the_ring_end_does_not_spuriously_wrap() {
        // A NOP covering exactly the ring's last 8 bytes: head advances to
        // ring_size, which the cursor reports as 0 (wrapped), matching
        // "continues at offset 0".
        let ring = ring_with(16, 8, &[header(OP_NOP, 2), 0]);
        let (steps, head) = decode_all(&ring, 8, 0, config(true));
        assert_eq!(steps, vec![Step::Command(Command::Nop)]);
        assert_eq!(head, 0);
    }

    #[test]
    fn a_full_ring_one_word_short_of_wrapping_decodes_completely() {
        // RING_SIZE 4096, head=0, tail = 4092 (one word short of full
        // circle): a single NOP covering exactly that much is legal.
        let ring = ring_with(4096, 0, &[header(OP_NOP, 1023)]);
        let (steps, head) = decode_all(&ring, 0, 4092, config(true));
        assert_eq!(steps, vec![Step::Command(Command::Nop)]);
        assert_eq!(head, 4092);
    }

    // -- Opcode errors ----------------------------------------------------

    #[test]
    fn an_opcode_outside_the_map_is_e_bad_opcode_and_is_skipped_not_halted() {
        let ring = ring_with(4096, 0, &[header(0x09FF, 3), 0, 0]);
        let step = decode_one(&ring, 0, 12);
        assert_eq!(step, err(ErrorCode::BadOpcode, 0, false));
    }

    #[test]
    fn logic_op_is_always_e_bad_opcode_in_version_one() {
        let ring = ring_with(4096, 0, &[header(OP_LOGIC_OP, 2), 0]);
        let step = decode_one(&ring, 0, 8);
        assert_eq!(step, err(ErrorCode::BadOpcode, 0, false));
    }

    #[test]
    fn decoding_continues_past_a_skipped_bad_opcode() {
        let ring = ring_with(4096, 0, &[header(0x09FF, 1), header(OP_FLUSH, 1)]);
        let (steps, head) = decode_all(&ring, 0, 8, config(true));
        assert_eq!(
            steps,
            vec![
                err(ErrorCode::BadOpcode, 0, false),
                Step::Command(Command::Flush),
            ]
        );
        assert_eq!(head, 8);
    }

    #[test]
    fn a_transform_tier_opcode_decodes_when_transform_is_set() {
        let ring = ring_with(4096, 0, &[header(OP_MATRIX_MODE, 2), 0x1700]);
        let step = decode_one(&ring, 0, 8);
        assert_eq!(step, Step::Command(Command::MatrixMode { mode: 0x1700 }));
    }

    #[test]
    fn every_transform_tier_opcode_is_e_bad_opcode_without_transform() {
        let mut c = config(true);
        c.transform = false;
        // One representative opcode from each transform-gated group: the
        // matrix group, the lighting group, the two raster-state opcodes,
        // and the GL-space (not _WIN) draw opcodes.
        let cases: [(u16, u16); 9] = [
            (OP_MATRIX_MODE, 2),
            (OP_LOAD_IDENTITY, 1),
            (OP_LIGHT, 7),
            (OP_CLIP_PLANE, 6),
            (OP_TEXGEN, 4),
            (OP_DEPTH_RANGE, 3),
            (OP_VIEWPORT, 5),
            (OP_DRAW_ARRAYS, 4), // format=0 -> 1 descriptor -> length 1+3+4=8, use a short bogus one instead
            (OP_DRAW_ELEMENTS, 9),
        ];
        for (opcode, length) in cases {
            let words = vec![header(opcode, length); 1]
                .into_iter()
                .chain(std::iter::repeat(0).take(length as usize - 1))
                .collect::<Vec<_>>();
            let ring = ring_with(4096, 0, &words);
            let step = decode_one_with(&ring, 0, length as u32 * 4, c);
            assert_eq!(
                step,
                err(ErrorCode::BadOpcode, 0, false),
                "opcode {opcode:#06x} should be E_BAD_OPCODE without transform"
            );
        }
    }

    #[test]
    fn window_space_draw_opcodes_are_unaffected_by_transform_being_clear() {
        let mut c = config(true);
        c.transform = false;
        let format = simple_format();
        let words = [header(OP_DRAW_INLINE_WIN, 7), 4, format.0, 1, 0, 0, 0];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 28, c);
        assert!(matches!(step, Step::Command(Command::DrawInlineWin { .. })));
    }

    // -- NORMAL is rejected outright in window-space draws -----------------

    fn format_with_normal() -> VertexFormat {
        VertexFormat((1 << VertexFormat::POS_COUNT_SHIFT) | VertexFormat::NORMAL)
    }

    #[test]
    fn normal_in_a_window_space_inline_draw_is_e_bad_arg() {
        let format = format_with_normal();
        let vertex_words = format.vertex_words().unwrap();
        let correct_length = 1 + 3 + vertex_words;
        let mut words = vec![
            header(OP_DRAW_INLINE_WIN, correct_length as u16),
            4,
            format.0,
            1,
        ];
        for _ in 0..vertex_words {
            words.push(0);
        }
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, correct_length * 4);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    #[test]
    fn normal_in_a_window_space_arrays_draw_is_e_bad_arg() {
        let format = format_with_normal();
        let descriptor_count = format.descriptor_count().unwrap();
        let correct_length = 1 + 3 + descriptor_count * DESCRIPTOR_WORDS;
        let mut words = vec![
            header(OP_DRAW_ARRAYS_WIN, correct_length as u16),
            4,
            format.0,
            1,
        ];
        for _ in 0..descriptor_count {
            words.extend_from_slice(&[0, 0, 0x0010_0000, 0]);
        }
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, correct_length * 4);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    #[test]
    fn normal_in_a_window_space_elements_draw_is_e_bad_arg() {
        let format = format_with_normal();
        let descriptor_count = format.descriptor_count().unwrap();
        let correct_length = 9 + descriptor_count * DESCRIPTOR_WORDS;
        let mut words = vec![
            header(OP_DRAW_ELEMENTS_WIN, correct_length as u16),
            4,
            format.0,
            1,
            0,
            0,
            0,
            0x0010_0000,
            0,
        ];
        for _ in 0..descriptor_count {
            words.extend_from_slice(&[0, 0, 0x0010_0000, 0]);
        }
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, correct_length * 4);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    #[test]
    fn normal_is_legal_in_gl_space_arrays_and_elements_draws() {
        let format = format_with_normal();
        let descriptor_count = format.descriptor_count().unwrap();
        let correct_length = 1 + 3 + descriptor_count * DESCRIPTOR_WORDS;
        let mut words = vec![
            header(OP_DRAW_ARRAYS, correct_length as u16),
            4,
            format.0,
            1,
        ];
        for _ in 0..descriptor_count {
            words.extend_from_slice(&[0, 0, 0x0010_0000, 0]);
        }
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, correct_length * 4);
        assert!(matches!(step, Step::Command(Command::DrawArrays { .. })));

        let correct_length = 9 + descriptor_count * DESCRIPTOR_WORDS;
        let mut words = vec![
            header(OP_DRAW_ELEMENTS, correct_length as u16),
            4,
            format.0,
            1,
            0,
            0,
            0,
            0x0010_0000,
            0,
        ];
        for _ in 0..descriptor_count {
            words.extend_from_slice(&[0, 0, 0x0010_0000, 0]);
        }
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, correct_length * 4);
        assert!(matches!(step, Step::Command(Command::DrawElements { .. })));
    }

    #[test]
    fn every_non_normal_format_bit_stays_legal_in_a_window_space_inline_draw() {
        let cases: [u32; 6] = [
            VertexFormat::COLOR,
            VertexFormat::TEXCOORD0,
            VertexFormat::TEXCOORD1,
            VertexFormat::TEXCOORD2,
            VertexFormat::TEXCOORD3,
            VertexFormat::FOGCOORD,
        ];
        for bit in cases {
            let format = VertexFormat((1 << VertexFormat::POS_COUNT_SHIFT) | bit);
            let vertex_words = format.vertex_words().unwrap();
            let correct_length = 1 + 3 + vertex_words;
            let mut words = vec![
                header(OP_DRAW_INLINE_WIN, correct_length as u16),
                4,
                format.0,
                1,
            ];
            for _ in 0..vertex_words {
                words.push(0);
            }
            let ring = ring_with(4096, 0, &words);
            let step = decode_one(&ring, 0, correct_length * 4);
            assert!(
                matches!(step, Step::Command(Command::DrawInlineWin { .. })),
                "bit {bit:#x} should stay legal in a window-space draw: {step:?}"
            );
        }
    }

    // -- Length-consistency (E_BAD_ARG) ------------------------------------

    #[test]
    fn a_fixed_payload_command_with_the_wrong_length_is_e_bad_arg() {
        let ring = ring_with(4096, 0, &[header(OP_FLUSH, 2), 0]);
        let step = decode_one(&ring, 0, 8);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    #[test]
    fn clear_rejects_a_non_zero_reserved_mask_bit() {
        let ring = ring_with(4096, 0, &[header(OP_CLEAR, 2), 0b100]);
        let step = decode_one(&ring, 0, 8);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    #[test]
    fn surface_define_rejects_a_non_zero_reserved_flags_bit() {
        let words = [
            header(OP_SURFACE_DEFINE, 8),
            1,    // id
            640,  // width
            480,  // height
            1280, // stride_bytes
            5,    // format: A8R8G8B8, supported by default
            0b10, // flags: bit 1 set (reserved)
            0,    // address
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 32);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    #[test]
    fn surface_define_with_the_guest_addr_flag_needs_the_capability() {
        let mut c = config(true);
        c.surface_guestaddr = false;
        let words = [
            header(OP_SURFACE_DEFINE, 8),
            1,
            640,
            480,
            1280,
            5,
            SURFACE_DEFINE_FLAG_GUEST_ADDR,
            0,
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 32, c);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    fn surface_define_words(
        id: u32,
        width: u32,
        height: u32,
        stride_bytes: u32,
        format: u32,
        flags: u32,
        address: u32,
    ) -> [u32; 8] {
        [
            header(OP_SURFACE_DEFINE, 8),
            id,
            width,
            height,
            stride_bytes,
            format,
            flags,
            address,
        ]
    }

    #[test]
    fn surface_define_with_correct_stride_and_extent_decodes() {
        // A8R8G8B8: 4 bytes/pixel, row_bytes = 64*4 = 256. Exact-fit
        // stride, address 0, extent (31*256+256=8192) well inside the
        // default 32 MiB aperture.
        let words = surface_define_words(1, 64, 32, 256, 5, 0, 0);
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 32);
        assert_eq!(
            step,
            Step::Command(Command::SurfaceDefine {
                id: 1,
                width: 64,
                height: 32,
                stride_bytes: 256,
                format: 5,
                flags: 0,
                address: 0,
            })
        );
    }

    #[test]
    fn surface_define_with_b8g8r8a8_decodes_under_the_default_config() {
        // B8G8R8A8 (format 6): RTG_COLOR_FORMAT_BGRA, the ZZ9000/z3660's
        // 32-bit RTG mode and glQuake's screen format -- a guest defining
        // a C3D surface matching that layout (for a direct blit/compose,
        // or a CAP_SURFACE_GUESTADDR target into the card's own VRAM)
        // must not hit E_UNSUPPORTED_FORMAT on Copperline's own default
        // config. Same 4 bytes/pixel layout as A8R8G8B8, so the same
        // stride/extent numbers apply.
        let words = surface_define_words(1, 64, 32, 256, 6, 0, 0);
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 32);
        assert_eq!(
            step,
            Step::Command(Command::SurfaceDefine {
                id: 1,
                width: 64,
                height: 32,
                stride_bytes: 256,
                format: 6,
                flags: 0,
                address: 0,
            })
        );
    }

    #[test]
    fn surface_define_stride_shorter_than_the_row_is_e_bad_arg() {
        // A8R8G8B8 needs 256 bytes/row; 200 is short.
        let words = surface_define_words(1, 64, 32, 200, 5, 0, 0);
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 32);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    #[test]
    fn surface_define_stride_equal_to_the_row_is_legal() {
        let words = surface_define_words(1, 64, 32, 256, 5, 0, 0);
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 32);
        assert!(matches!(step, Step::Command(Command::SurfaceDefine { .. })));
    }

    #[test]
    fn an_aperture_surface_whose_extent_exceeds_the_aperture_is_e_bad_ref() {
        let mut c = config(true);
        c.aperture_size = 8192;
        // width=64,height=33 (one row more than the "exactly fits" case):
        // extent = 32*256 + 256 = 8448 > 8192.
        let words = surface_define_words(1, 64, 33, 256, 5, 0, 0);
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 32, c);
        assert_eq!(step, err(ErrorCode::BadRef, 0, false));
    }

    #[test]
    fn an_aperture_surface_extent_exactly_at_the_aperture_bound_is_legal() {
        let mut c = config(true);
        c.aperture_size = 8192; // exactly the extent of the surface below
        let words = surface_define_words(1, 64, 32, 256, 5, 0, 0);
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 32, c);
        assert!(matches!(step, Step::Command(Command::SurfaceDefine { .. })));
    }

    #[test]
    fn an_aperture_surface_extent_that_overflows_u32_is_e_bad_ref() {
        // width=1, A8R8G8B8 -> row_bytes=4. height=2, stride=u32::MAX:
        // (height-1)*stride = u32::MAX, plus address(0) is still in
        // range, but adding row_bytes overflows.
        let words = surface_define_words(1, 1, 2, u32::MAX, 5, 0, 0);
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 32);
        assert_eq!(step, err(ErrorCode::BadRef, 0, false));
    }

    #[test]
    fn an_aperture_surface_address_near_u32_max_that_overflows_is_e_bad_ref() {
        // width=8, A8R8G8B8 -> row_bytes=32; height=1 so the extent is
        // just address+row_bytes, which overflows near u32::MAX.
        let words = surface_define_words(1, 8, 1, 32, 5, 0, 0xFFFF_FFF0);
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 32);
        assert_eq!(step, err(ErrorCode::BadRef, 0, false));
    }

    #[test]
    fn a_guest_address_surface_is_not_checked_against_the_aperture() {
        // Wildly out of any aperture's range, but CAP_SURFACE_GUESTADDR is
        // set and the guest-address flag is set, so it's unchecked here
        // (reachability of guest memory is a higher layer's concern).
        let words = surface_define_words(
            1,
            64,
            32,
            256,
            5,
            SURFACE_DEFINE_FLAG_GUEST_ADDR,
            0xFFFF_0000,
        );
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 32);
        assert!(matches!(step, Step::Command(Command::SurfaceDefine { .. })));
    }

    #[test]
    fn read_pixels_rejects_a_non_zero_reserved_flags_bit() {
        let words = [
            header(OP_READ_PIXELS, 10),
            0,
            0,
            4,
            4,           // x,y,w,h
            5,           // format
            16,          // row_bytes
            0b10,        // flags reserved bit
            0x0010_0000, // dest ref addr
            16,          // dest ref length
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 40);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    fn config_with_tex_palette_supported() -> DeviceConfig {
        DeviceConfig {
            texfmt_supported: DeviceConfig::default().texfmt_supported
                | (1 << proto::TextureFormat::I8Indexed as u32),
            ..DeviceConfig::default()
        }
    }

    #[test]
    fn tex_palette_rejects_more_than_256_entries() {
        let words = [
            header(OP_TEX_PALETTE, 5),
            1,   // id
            257, // entries
            0x0010_0000,
            0,
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 20, config_with_tex_palette_supported());
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    #[test]
    fn tex_palette_is_bad_opcode_without_texfmt_supported_bit_9() {
        // Otherwise entirely well-formed -- Copperline's own default
        // config (TEXFMT_SUPPORTED bits 0-7 only) doesn't set I8_INDEXED,
        // so the opcode itself doesn't exist per the spec, exactly like
        // an unknown opcode or a missing-tier one.
        let words = [
            header(OP_TEX_PALETTE, 5),
            1,   // id
            256, // entries
            0x0010_0000,
            1024, // ref length: 256 RGBA8 entries
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 20);
        assert_eq!(step, err(ErrorCode::BadOpcode, 0, false));
    }

    #[test]
    fn tex_palette_decodes_normally_when_texfmt_supported_bit_9_is_set() {
        let words = [
            header(OP_TEX_PALETTE, 5),
            1,   // id
            256, // entries
            0x0010_0000,
            1024, // ref length: 256 RGBA8 entries
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 20, config_with_tex_palette_supported());
        assert!(matches!(step, Step::Command(Command::TexPalette { .. })));
    }

    #[test]
    fn tex_image_rejects_non_power_of_two_dimensions() {
        let words = [
            header(OP_TEX_IMAGE, 9),
            1,   // id
            0,   // level
            0,   // format: RGBA8
            100, // width: not a power of two
            64,  // height
            0,   // row_bytes
            0x0010_0000,
            0,
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 36);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    // -- E_BAD_ID -----------------------------------------------------------

    #[test]
    fn surface_define_with_id_zero_is_e_bad_id() {
        let words = [
            header(OP_SURFACE_DEFINE, 8),
            0, // id: 0 is "none", not a legal id to define
            640,
            480,
            1280,
            5,
            0,
            0,
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 32);
        assert_eq!(step, err(ErrorCode::BadId, 0, false));
    }

    #[test]
    fn surface_define_with_an_id_above_max_surfaces_is_e_bad_id() {
        let mut c = config(true);
        c.max_surfaces = 4;
        let words = [
            header(OP_SURFACE_DEFINE, 8),
            5, // above max_surfaces
            640,
            480,
            1280,
            5,
            0,
            0,
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 32, c);
        assert_eq!(step, err(ErrorCode::BadId, 0, false));
    }

    #[test]
    fn set_draw_surface_with_id_zero_is_legal() {
        let ring = ring_with(4096, 0, &[header(OP_SET_DRAW_SURFACE, 2), 0]);
        let step = decode_one(&ring, 0, 8);
        assert_eq!(step, Step::Command(Command::SetDrawSurface { id: 0 }));
    }

    #[test]
    fn set_draw_surface_above_max_surfaces_is_e_bad_id() {
        let mut c = config(true);
        c.max_surfaces = 4;
        let ring = ring_with(4096, 0, &[header(OP_SET_DRAW_SURFACE, 2), 5]);
        let step = decode_one_with(&ring, 0, 8, c);
        assert_eq!(step, err(ErrorCode::BadId, 0, false));
    }

    #[test]
    fn tex_bind_with_id_zero_is_legal_null_texture() {
        let ring = ring_with(4096, 0, &[header(OP_TEX_BIND, 3), 0, 0]);
        let step = decode_one(&ring, 0, 12);
        assert_eq!(step, Step::Command(Command::TexBind { unit: 0, id: 0 }));
    }

    #[test]
    fn tex_create_with_id_zero_is_e_bad_id() {
        let ring = ring_with(4096, 0, &[header(OP_TEX_CREATE, 2), 0]);
        let step = decode_one(&ring, 0, 8);
        assert_eq!(step, err(ErrorCode::BadId, 0, false));
    }

    #[test]
    fn tex_create_above_max_textures_is_e_bad_id() {
        let mut c = config(true);
        c.max_textures = 4;
        let ring = ring_with(4096, 0, &[header(OP_TEX_CREATE, 2), 5]);
        let step = decode_one_with(&ring, 0, 8, c);
        assert_eq!(step, err(ErrorCode::BadId, 0, false));
    }

    // -- E_LIMIT --------------------------------------------------------

    #[test]
    fn active_unit_at_max_texture_units_is_e_limit() {
        let mut c = config(true);
        c.max_texture_units = 2;
        let ring = ring_with(4096, 0, &[header(OP_ACTIVE_UNIT, 2), 2]);
        let step = decode_one_with(&ring, 0, 8, c);
        assert_eq!(step, err(ErrorCode::Limit, 0, false));
    }

    #[test]
    fn active_unit_below_max_texture_units_decodes() {
        let mut c = config(true);
        c.max_texture_units = 2;
        let ring = ring_with(4096, 0, &[header(OP_ACTIVE_UNIT, 2), 1]);
        let step = decode_one_with(&ring, 0, 8, c);
        assert_eq!(step, Step::Command(Command::ActiveUnit { unit: 1 }));
    }

    #[test]
    fn a_non_zero_unit_without_multitexture_is_e_limit_not_e_bad_opcode() {
        let mut c = config(true);
        c.multitexture = false;
        c.max_texture_units = 1;
        let ring = ring_with(4096, 0, &[header(OP_ACTIVE_UNIT, 2), 1]);
        let step = decode_one_with(&ring, 0, 8, c);
        assert_eq!(step, err(ErrorCode::Limit, 0, false));
    }

    #[test]
    fn tex_bind_unit_at_or_above_max_texture_units_is_e_limit() {
        let mut c = config(true);
        c.max_texture_units = 1;
        let ring = ring_with(4096, 0, &[header(OP_TEX_BIND, 3), 1, 0]);
        let step = decode_one_with(&ring, 0, 12, c);
        assert_eq!(step, err(ErrorCode::Limit, 0, false));
    }

    #[test]
    fn current_texcoord_unit_at_or_above_max_texture_units_is_e_limit() {
        let mut c = config(true);
        c.max_texture_units = 1;
        let words = [header(OP_CURRENT_TEXCOORD, 4), 1, 0, 0];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 16, c);
        assert_eq!(step, err(ErrorCode::Limit, 0, false));
    }

    #[test]
    fn light_at_or_above_max_lights_is_e_limit() {
        let mut c = config(true);
        c.max_lights = 4;
        let words = [
            header(OP_LIGHT, 7),
            GL_LIGHT0 + 4, // at the limit
            0,
            0,
            0,
            0,
            0,
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 28, c);
        assert_eq!(step, err(ErrorCode::Limit, 0, false));
    }

    #[test]
    fn light_below_max_lights_decodes() {
        let words = [header(OP_LIGHT, 7), GL_LIGHT0, 0x1200, 0, 0, 0, 0];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 28);
        assert!(matches!(step, Step::Command(Command::Light { light, .. }) if light == GL_LIGHT0));
    }

    #[test]
    fn a_light_word_below_gl_light0_is_e_limit() {
        let words = [header(OP_LIGHT, 7), 0x1234, 0, 0, 0, 0, 0];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 28);
        assert_eq!(step, err(ErrorCode::Limit, 0, false));
    }

    #[test]
    fn clip_plane_at_or_above_max_clip_planes_is_e_limit() {
        let mut c = config(true);
        c.max_clip_planes = 2;
        let words = [header(OP_CLIP_PLANE, 6), GL_CLIP_PLANE0 + 2, 0, 0, 0, 0];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 24, c);
        assert_eq!(step, err(ErrorCode::Limit, 0, false));
    }

    #[test]
    fn tex_image_width_above_max_texture_size_is_e_limit() {
        let mut c = config(true);
        c.max_texture_size = 64;
        let words = [
            header(OP_TEX_IMAGE, 9),
            1,
            0,
            0,   // format
            128, // width: above the limit (still a power of two)
            64,
            0,
            0x0010_0000,
            0,
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 36, c);
        assert_eq!(step, err(ErrorCode::Limit, 0, false));
    }

    #[test]
    fn surface_define_width_above_max_surface_width_is_e_limit() {
        let mut c = config(true);
        c.max_surface_width = 640;
        let words = [
            header(OP_SURFACE_DEFINE, 8),
            1,
            1280, // width: above the limit
            480,
            2560,
            5,
            0,
            0,
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 32, c);
        assert_eq!(step, err(ErrorCode::Limit, 0, false));
    }

    // -- E_UNSUPPORTED_FORMAT -----------------------------------------------

    #[test]
    fn tex_image_with_an_unsupported_format_is_e_unsupported_format() {
        let mut c = config(true);
        c.texfmt_supported = 0; // nothing supported
        let words = [
            header(OP_TEX_IMAGE, 9),
            1,
            0,
            0, // format: RGBA8, not in the (empty) supported set
            64,
            64,
            0,
            0x0010_0000,
            0,
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 36, c);
        assert_eq!(step, err(ErrorCode::UnsupportedFormat, 0, false));
    }

    #[test]
    fn surface_define_with_an_unsupported_format_is_e_unsupported_format() {
        let mut c = config(true);
        c.surffmt_supported = 0;
        let words = [
            header(OP_SURFACE_DEFINE, 8),
            1,
            640,
            480,
            1280,
            5, // format: A8R8G8B8, not in the (empty) supported set
            0,
            0,
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 32, c);
        assert_eq!(step, err(ErrorCode::UnsupportedFormat, 0, false));
    }

    #[test]
    fn read_pixels_with_depth_format_is_always_allowed() {
        let words = [
            header(OP_READ_PIXELS, 10),
            0,
            0,
            4,
            4,
            255, // DEPTH
            16,
            0,
            0x0010_0000,
            16,
        ];
        let mut c = config(true);
        c.surffmt_supported = 0; // even with nothing else supported
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 40, c);
        assert!(matches!(
            step,
            Step::Command(Command::ReadPixels { format: 255, .. })
        ));
    }

    #[test]
    fn read_pixels_with_an_unsupported_non_depth_format_is_e_unsupported_format() {
        let mut c = config(true);
        c.surffmt_supported = 0;
        let words = [
            header(OP_READ_PIXELS, 10),
            0,
            0,
            4,
            4,
            5, // A8R8G8B8, not DEPTH, not supported
            16,
            0,
            0x0010_0000,
            16,
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 40, c);
        assert_eq!(step, err(ErrorCode::UnsupportedFormat, 0, false));
    }

    // -- Draw commands: length consistency per shape -----------------------

    fn simple_format() -> VertexFormat {
        // POS_COUNT=1 (3 words), no optional components: 3 words/vertex,
        // 1 descriptor.
        VertexFormat(1 << VertexFormat::POS_COUNT_SHIFT)
    }

    #[test]
    fn draw_inline_length_must_match_count_times_vertex_words() {
        let format = simple_format();
        let count = 2u32;
        let vertex_words = format.vertex_words().unwrap();
        let correct_length = 1 + 3 + count * vertex_words;
        let mut words = vec![
            header(OP_DRAW_INLINE, correct_length as u16),
            4, // prim = TRIANGLES... arbitrary
            format.0,
            count,
        ];
        for _ in 0..count * vertex_words {
            words.push(0);
        }
        let ring = ring_with(4096, 0, &words);
        let total_bytes = correct_length * 4;
        let step = decode_one(&ring, 0, total_bytes);
        match step {
            Step::Command(Command::DrawInline {
                prim,
                format: f,
                count: c,
                vertices,
            }) => {
                assert_eq!(prim, 4);
                assert_eq!(f, format);
                assert_eq!(c, count);
                assert_eq!(vertices.len() as u32, count * vertex_words * 4);
            }
            other => panic!("expected DrawInline, got {other:?}"),
        }

        // Now corrupt the declared length (one word short) and expect
        // E_BAD_ARG.
        let mut bad_words = words.clone();
        bad_words[0] = header(OP_DRAW_INLINE, (correct_length - 1) as u16);
        let ring = ring_with(4096, 0, &bad_words);
        let step = decode_one(&ring, 0, (correct_length - 1) * 4);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    #[test]
    fn draw_inline_win_length_must_match_count_times_vertex_words() {
        let format = simple_format();
        let count = 1u32;
        let vertex_words = format.vertex_words().unwrap();
        let correct_length = 1 + 3 + count * vertex_words;
        let mut words = vec![
            header(OP_DRAW_INLINE_WIN, correct_length as u16),
            0,
            format.0,
            count,
        ];
        for _ in 0..count * vertex_words {
            words.push(0);
        }
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, correct_length * 4);
        assert!(matches!(step, Step::Command(Command::DrawInlineWin { .. })));
    }

    #[test]
    fn draw_arrays_length_must_match_descriptor_count() {
        let format = VertexFormat(
            (1 << VertexFormat::POS_COUNT_SHIFT) | VertexFormat::COLOR | VertexFormat::TEXCOORD0,
        );
        let descriptor_count = format.descriptor_count().unwrap();
        assert_eq!(descriptor_count, 3); // pos + color + texcoord0
        let count = 10u32;
        let correct_length = 1 + 3 + descriptor_count * DESCRIPTOR_WORDS;
        let mut words = vec![
            header(OP_DRAW_ARRAYS, correct_length as u16),
            4,
            format.0,
            count,
        ];
        for _ in 0..descriptor_count {
            // type, stride_bytes, ref address, ref length
            words.extend_from_slice(&[0, 0, 0x0010_0000, 0]);
        }
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, correct_length * 4);
        match step {
            Step::Command(Command::DrawArrays {
                descriptors,
                count: c,
                ..
            }) => {
                assert_eq!(c, count);
                assert_eq!(
                    descriptors.len() as u32,
                    descriptor_count * DESCRIPTOR_WORDS * 4
                );
            }
            other => panic!("expected DrawArrays, got {other:?}"),
        }

        // Wrong length -> E_BAD_ARG.
        let mut bad = words.clone();
        bad[0] = header(OP_DRAW_ARRAYS, (correct_length + 4) as u16);
        // Need enough bytes present in the ring/tail for the *declared*
        // length to pass the framing check so we reach the arg check.
        let mut ring = ring_with(4096, 0, &bad);
        ring.resize(4096.max(((correct_length + 4) * 4) as usize + 16), 0);
        let step = decode_one(&ring, 0, (correct_length + 4) * 4);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    #[test]
    fn draw_arrays_win_decodes_like_draw_arrays() {
        let format = simple_format();
        let descriptor_count = format.descriptor_count().unwrap();
        let count = 4u32;
        let correct_length = 1 + 3 + descriptor_count * DESCRIPTOR_WORDS;
        let mut words = vec![
            header(OP_DRAW_ARRAYS_WIN, correct_length as u16),
            5,
            format.0,
            count,
        ];
        for _ in 0..descriptor_count {
            words.extend_from_slice(&[0, 0, 0x0010_0000, 0]);
        }
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, correct_length * 4);
        assert!(matches!(step, Step::Command(Command::DrawArraysWin { .. })));
    }

    #[test]
    fn draw_elements_length_must_match_fixed_part_plus_descriptors() {
        let format = simple_format();
        let descriptor_count = format.descriptor_count().unwrap();
        let correct_length = 9 + descriptor_count * DESCRIPTOR_WORDS;
        let mut words = vec![
            header(OP_DRAW_ELEMENTS, correct_length as u16),
            4,           // prim
            format.0,    // format
            6,           // count
            0,           // index_type (UNSIGNED_SHORT etc, raw)
            0,           // min_index
            5,           // max_index
            0x0010_0000, // index_ref addr
            12,          // index_ref length
        ];
        for _ in 0..descriptor_count {
            words.extend_from_slice(&[0, 0, 0x0010_0000, 0]);
        }
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, correct_length * 4);
        match step {
            Step::Command(Command::DrawElements {
                index_ref,
                descriptors,
                min_index,
                max_index,
                ..
            }) => {
                assert_eq!(min_index, 0);
                assert_eq!(max_index, 5);
                assert_eq!(index_ref.length, 12);
                assert_eq!(
                    descriptors.len() as u32,
                    descriptor_count * DESCRIPTOR_WORDS * 4
                );
            }
            other => panic!("expected DrawElements, got {other:?}"),
        }
    }

    #[test]
    fn draw_elements_win_with_wrong_length_is_e_bad_arg() {
        let format = simple_format();
        let descriptor_count = format.descriptor_count().unwrap();
        let correct_length = 9 + descriptor_count * DESCRIPTOR_WORDS;
        let mut words = vec![
            header(OP_DRAW_ELEMENTS_WIN, (correct_length - 4) as u16),
            0,
            format.0,
            0,
            0,
            0,
            0,
            0x0010_0000,
            0,
        ];
        for _ in 0..descriptor_count {
            words.extend_from_slice(&[0, 0, 0x0010_0000, 0]);
        }
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, (correct_length - 4) * 4);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    #[test]
    fn draw_inline_pos_count_2_3_4_each_change_the_expected_length() {
        for (field, pos_words) in [(0u32, 4u32), (1, 3), (2, 2)] {
            let format = VertexFormat(field << VertexFormat::POS_COUNT_SHIFT);
            assert_eq!(format.vertex_words(), Some(pos_words));
            let count = 1u32;
            let correct_length = 1 + 3 + count * pos_words;
            let mut words = vec![
                header(OP_DRAW_INLINE, correct_length as u16),
                4,
                format.0,
                count,
            ];
            for _ in 0..pos_words {
                words.push(0);
            }
            let ring = ring_with(4096, 0, &words);
            let step = decode_one(&ring, 0, correct_length * 4);
            assert!(
                matches!(step, Step::Command(Command::DrawInline { .. })),
                "POS_COUNT field {field} (pos_words {pos_words}) failed to decode: {step:?}"
            );
        }
    }

    #[test]
    fn draw_inline_pos_count_reserved_value_is_e_bad_arg() {
        let format = VertexFormat(0b11 << VertexFormat::POS_COUNT_SHIFT);
        let words = [header(OP_DRAW_INLINE, 4), 4, format.0, 0];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 16);
        assert_eq!(step, err(ErrorCode::BadArg, 0, false));
    }

    #[test]
    fn each_vertex_format_bit_changes_the_expected_inline_length() {
        let cases: [(u32, u32); 6] = [
            (VertexFormat::COLOR, 4),
            (VertexFormat::NORMAL, 3),
            (VertexFormat::TEXCOORD0, 2),
            (VertexFormat::TEXCOORD1, 2),
            (VertexFormat::FOGCOORD, 1),
            (VertexFormat::COLOR_PACKED, 1),
        ];
        for (bit, extra_words) in cases {
            let format = VertexFormat((1 << VertexFormat::POS_COUNT_SHIFT) | bit);
            let vertex_words = 3 + extra_words;
            assert_eq!(format.vertex_words(), Some(vertex_words));
            let count = 1u32;
            let correct_length = 1 + 3 + count * vertex_words;
            let mut words = vec![
                header(OP_DRAW_INLINE, correct_length as u16),
                4,
                format.0,
                count,
            ];
            for _ in 0..vertex_words {
                words.push(0);
            }
            let ring = ring_with(4096, 0, &words);
            let step = decode_one(&ring, 0, correct_length * 4);
            assert!(
                matches!(step, Step::Command(Command::DrawInline { .. })),
                "bit {bit:#x} failed: {step:?}"
            );
        }
    }

    // -- Multi-command submissions -----------------------------------------

    #[test]
    fn a_multi_command_submission_decodes_in_order() {
        let words = [
            header(OP_FLUSH, 1),
            header(OP_FENCE, 2),
            7,
            header(OP_CLEAR, 2),
            CLEAR_MASK_COLOR,
        ];
        let ring = ring_with(4096, 0, &words);
        let total = words.len() as u32 * 4;
        let (steps, head) = decode_all(&ring, 0, total, config(true));
        assert_eq!(
            steps,
            vec![
                Step::Command(Command::Flush),
                Step::Command(Command::Fence { id: 7 }),
                Step::Command(Command::Clear {
                    mask: CLEAR_MASK_COLOR
                }),
            ]
        );
        assert_eq!(head, total);
    }

    #[test]
    fn decoding_halts_at_the_first_halting_error_in_a_submission() {
        let words = [
            header(OP_FLUSH, 1),
            header(OP_FENCE, 0), // E_BAD_LENGTH: halts
            header(OP_FLUSH, 1), // never reached
        ];
        let ring = ring_with(4096, 0, &words);
        let (steps, head) = decode_all(&ring, 0, 12, config(true));
        assert_eq!(
            steps,
            vec![
                Step::Command(Command::Flush),
                err(ErrorCode::BadLength, 4, true),
            ]
        );
        assert_eq!(head, 4, "head stays at the offending command until ack");
    }

    // -- Refs ---------------------------------------------------------------

    #[test]
    fn a_ref_in_the_aperture_space_decodes_when_within_bounds() {
        let words = [
            header(OP_QUERY, 4),
            0x2000, // what (raw GL enum)
            0x0010_0000,
            64,
        ];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 16);
        match step {
            Step::Command(Command::Query { dest, .. }) => {
                assert_eq!(dest.space, RefSpace::Aperture);
                assert_eq!(dest.address, 0x0010_0000);
                assert_eq!(dest.length, 64);
            }
            other => panic!("expected Query, got {other:?}"),
        }
    }

    #[test]
    fn a_ref_in_the_guest_space_decodes_when_cap_guestmem_is_set() {
        let words = [header(OP_QUERY, 4), 0x2000, 0x0020_0000, 0x8000_0040];
        let ring = ring_with(4096, 0, &words);
        let (mut steps, _) = decode_all(&ring, 0, 16, config(true));
        match steps.remove(0) {
            Step::Command(Command::Query { dest, .. }) => {
                assert_eq!(dest.space, RefSpace::Guest);
                assert_eq!(dest.address, 0x0020_0000);
            }
            other => panic!("expected Query, got {other:?}"),
        }
    }

    #[test]
    fn a_guest_space_ref_is_e_bad_ref_without_cap_guestmem() {
        let words = [header(OP_QUERY, 4), 0x2000, 0x0020_0000, 0x8000_0040];
        let ring = ring_with(4096, 0, &words);
        let (mut steps, _) = decode_all(&ring, 0, 16, config(false));
        assert_eq!(steps.remove(0), err(ErrorCode::BadRef, 0, false));
    }

    #[test]
    fn an_aperture_ref_past_the_aperture_bound_is_e_bad_ref() {
        let mut c = config(true);
        c.aperture_size = 128;
        let words = [header(OP_QUERY, 4), 0x2000, 100, 64]; // 100+64 > 128
        let ring = ring_with(4096, 0, &words);
        let mut cursor = RingCursor::new(&ring, 0, 16, c);
        let step = cursor.step().unwrap();
        assert_eq!(step, err(ErrorCode::BadRef, 0, false));
    }

    #[test]
    fn a_misaligned_ref_address_is_e_bad_ref() {
        let words = [header(OP_QUERY, 4), 0x2000, 0x0010_0001, 64];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 16);
        assert_eq!(step, err(ErrorCode::BadRef, 0, false));
    }

    // -- CALL -----------------------------------------------------------

    #[test]
    fn call_decodes_to_the_right_command_with_an_aperture_ref() {
        let words = [header(OP_CALL, 3), 0x0010_0000, 256];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 12);
        match step {
            Step::Command(Command::Call { data }) => {
                assert_eq!(data.space, RefSpace::Aperture);
                assert_eq!(data.address, 0x0010_0000);
                assert_eq!(data.length, 256);
            }
            other => panic!("expected Call, got {other:?}"),
        }
    }

    #[test]
    fn call_decodes_with_a_guest_space_ref_under_cap_guestmem() {
        let words = [header(OP_CALL, 3), 0x0020_0000, 0x8000_0100];
        let ring = ring_with(4096, 0, &words);
        let (mut steps, _) = decode_all(&ring, 0, 12, config(true));
        match steps.remove(0) {
            Step::Command(Command::Call { data }) => {
                assert_eq!(data.space, RefSpace::Guest);
                assert_eq!(data.address, 0x0020_0000);
            }
            other => panic!("expected Call, got {other:?}"),
        }
    }

    #[test]
    fn call_with_a_guest_space_ref_is_e_bad_ref_without_cap_guestmem() {
        // A CALL to a space-1 buffer with CAP_GUESTMEM masked off must go
        // through the ordinary validate_ref path exactly like any other
        // ref -- not be silently accepted, which would bypass the
        // capability gate.
        let words = [header(OP_CALL, 3), 0x0020_0000, 0x8000_0100];
        let ring = ring_with(4096, 0, &words);
        let (mut steps, _) = decode_all(&ring, 0, 12, config(false));
        assert_eq!(steps.remove(0), err(ErrorCode::BadRef, 0, false));
    }

    #[test]
    fn call_with_a_misaligned_ref_is_e_bad_ref() {
        let words = [header(OP_CALL, 3), 0x0010_0001, 256];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one(&ring, 0, 12);
        assert_eq!(step, err(ErrorCode::BadRef, 0, false));
    }

    #[test]
    fn call_with_an_aperture_ref_past_the_bound_is_e_bad_ref() {
        let mut c = config(true);
        c.aperture_size = 128;
        let words = [header(OP_CALL, 3), 100, 64]; // 100+64 > 128
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 12, c);
        assert_eq!(step, err(ErrorCode::BadRef, 0, false));
    }

    #[test]
    fn call_is_not_gated_by_cap_transform_or_any_other_capability() {
        // Baseline tier: legal even on a device with every optional
        // capability clear.
        let mut c = config(true);
        c.transform = false;
        c.multitexture = false;
        c.surface_guestaddr = false;
        let words = [header(OP_CALL, 3), 0x0010_0000, 256];
        let ring = ring_with(4096, 0, &words);
        let step = decode_one_with(&ring, 0, 12, c);
        assert!(matches!(step, Step::Command(Command::Call { .. })));
    }

    // -- RingCursor::new_linear (a CALLed buffer's own decode) ---------------

    #[test]
    fn new_linear_decodes_a_buffer_filled_exactly_to_its_own_end() {
        // A buffer whose single command's length exactly consumes the
        // whole buffer must decode cleanly, not spuriously E_RING_OVERRUN
        // the way feeding the same head/tail/ring_size to the ordinary
        // circular `new` would (see `new_linear`'s doc comment).
        let words = [header(OP_FENCE, 2), 7];
        let buf = ring_with(8, 0, &words); // buffer is exactly 8 bytes
        let mut cursor = RingCursor::new_linear(&buf, config(true));
        assert_eq!(cursor.step(), Some(Step::Command(Command::Fence { id: 7 })));
        assert_eq!(cursor.head(), 8);
        assert!(!cursor.is_halted());
        // Exhausted: no further steps, and head stayed at the buffer's
        // own end rather than wrapping back to 0.
        assert_eq!(cursor.step(), None);
    }

    #[test]
    fn new_linear_reports_e_bad_length_for_a_command_past_the_buffers_end() {
        // The buffer declares a FENCE of length 2 (8 bytes) but is only 4
        // bytes long -- a framing error against the buffer's own end,
        // exactly as an ordinary ring overrun would be, per the spec's
        // "extends past the end of the referenced range ... E_BAD_LENGTH".
        let words = [header(OP_FENCE, 2)];
        let buf = ring_with(4, 0, &words);
        let mut cursor = RingCursor::new_linear(&buf, config(true));
        assert_eq!(cursor.step(), Some(err(ErrorCode::BadLength, 0, true)));
        assert!(cursor.is_halted());
    }
}
