// SPDX-License-Identifier: GPL-3.0-or-later

//! C3D wire-level constants and decoded command types.
//!
//! Everything in this file is a direct transcription of `docs/internals/
//! c3d.md` -- the device specification and the sole source of truth for
//! every numeric value here. Section references in the doc comments below
//! (`c3d-global-registers`, `c3d-cmd-draw`, ...) are that document's anchor
//! names. This module has no emulator or GPU dependency and performs no
//! decoding itself; [`super::ring`] is the decoder that turns ring bytes
//! into the [`Command`] values defined here.
//!
//! Raw GL enumerant values (blend factors, depth funcs, texenv modes,
//! primitive types, and so on) are carried through [`Command`] payloads as
//! plain `u32`s, never as typed Rust enums: deciding whether an enumerant is
//! legal is a *GL error*, which belongs to the per-context state machine
//! ([`super::state`]), not to this wire layer. The two format enumerations
//! that *are* typed here ([`SurfaceFormat`], [`TextureFormat`]) are the
//! device's own vocabulary, not GL's, and exist so callers don't have to
//! repeat the numbering from the spec's tables by hand.

/// `ID` register's magic value, ASCII `"C3D "` (see `c3d-global-registers`).
pub const ID_MAGIC: u32 = 0x4333_4420;

/// `VERSION` register reset value: `major << 16 | minor`. This is the
/// protocol draft Copperline implements (0.15 as of this writing, the
/// draft published at the specification's own repository), not the
/// eventual 1.0 release; see `c3d-versioning`. Bump this whenever a
/// change here catches Copperline up to a newer draft.
pub const PROTOCOL_VERSION: u32 = 0x0000_000F;

// ---------------------------------------------------------------------
// Global registers (`c3d-global-registers`). Offsets are bytes from the
// window base, within the first 64 KiB.
// ---------------------------------------------------------------------

pub const GLOBAL_ID: u32 = 0x000;
pub const GLOBAL_VERSION: u32 = 0x004;
pub const GLOBAL_CAPS0: u32 = 0x008;
pub const GLOBAL_CAPS1: u32 = 0x00C;
pub const GLOBAL_STATUS: u32 = 0x010;
pub const GLOBAL_CONTROL: u32 = 0x014;
pub const GLOBAL_IRQ_STATUS: u32 = 0x018;
pub const GLOBAL_IRQ_ENABLE: u32 = 0x01C;
pub const GLOBAL_APERTURE_OFFSET: u32 = 0x020;
pub const GLOBAL_APERTURE_SIZE: u32 = 0x024;
pub const GLOBAL_MAX_CONTEXTS: u32 = 0x028;
pub const GLOBAL_MAX_RING_SIZE: u32 = 0x02C;
pub const GLOBAL_MAX_TEXTURE_SIZE: u32 = 0x040;
pub const GLOBAL_MAX_TEXTURE_UNITS: u32 = 0x044;
pub const GLOBAL_MAX_TEXTURES: u32 = 0x048;
pub const GLOBAL_MAX_SURFACES: u32 = 0x04C;
pub const GLOBAL_MAX_LIGHTS: u32 = 0x050;
pub const GLOBAL_MAX_CLIP_PLANES: u32 = 0x054;
pub const GLOBAL_MAX_MATRIX_DEPTH_MV: u32 = 0x058;
pub const GLOBAL_MAX_MATRIX_DEPTH_PROJ: u32 = 0x05C;
pub const GLOBAL_MAX_MATRIX_DEPTH_TEX: u32 = 0x060;
pub const GLOBAL_TEXFMT_SUPPORTED: u32 = 0x064;
pub const GLOBAL_SURFFMT_SUPPORTED_LO: u32 = 0x068;
pub const GLOBAL_SURFFMT_SUPPORTED_HI: u32 = 0x06C;
pub const GLOBAL_MAX_SURFACE_WIDTH: u32 = 0x070;
pub const GLOBAL_MAX_SURFACE_HEIGHT: u32 = 0x074;

/// `APERTURE_OFFSET`'s fixed reset value (`c3d-zorro-identity`'s window
/// layout table): the data aperture always starts here.
pub const APERTURE_OFFSET_DEFAULT: u32 = 0x0010_0000;

/// The Z3 profile's default `APERTURE_SIZE` (`c3d-zorro-identity`'s bus
/// profile table: "one window, default 32 MiB"). Copperline ships this
/// profile; a Z2 board would default to 4 or 8 MiB instead.
pub const APERTURE_SIZE_DEFAULT: u32 = 32 * 1024 * 1024;

/// `STATUS` bits (`c3d-global-registers`).
pub const STATUS_READY: u32 = 1 << 0;
pub const STATUS_RESETTING: u32 = 1 << 1;
pub const STATUS_FATAL: u32 = 1 << 2;

/// `CONTROL` bits (`c3d-global-registers`).
pub const CONTROL_ENABLE: u32 = 1 << 0;
pub const CONTROL_RESET: u32 = 1 << 1;

/// `CAPS0` bits (`c3d-tiers`). `CAPS1` is reserved and always reads `0`.
pub const CAP_GUESTMEM: u32 = 1 << 0;
pub const CAP_IRQ: u32 = 1 << 1;
pub const CAP_TRANSFORM: u32 = 1 << 2;
pub const CAP_MULTITEXTURE: u32 = 1 << 3;
pub const CAP_SURFACE_GUESTADDR: u32 = 1 << 4;
// Bit 5 is reserved (withdrawn present-layer capability); always 0.
pub const CAP_REF_SYNC: u32 = 1 << 6;

/// The `IRQ_STATUS`/`IRQ_ENABLE` bit for context `n`'s fence interrupt.
pub const fn irq_fence_bit(context: u32) -> u32 {
    1 << context
}

/// The `IRQ_STATUS`/`IRQ_ENABLE` bit for context `n`'s latched protocol
/// error.
pub const fn irq_error_bit(context: u32) -> u32 {
    1 << (16 + context)
}

// ---------------------------------------------------------------------
// Context register pages (`c3d-context-registers`). Offsets are bytes
// within a 256-byte page at `0x0001_0000 + n * 0x100`.
// ---------------------------------------------------------------------

pub const CTX_CONTROL: u32 = 0x00;
pub const CTX_STATUS: u32 = 0x04;
pub const CTX_RING_BASE: u32 = 0x08;
pub const CTX_RING_SIZE: u32 = 0x0C;
pub const CTX_RING_TAIL: u32 = 0x10;
pub const CTX_RING_HEAD: u32 = 0x14;
pub const CTX_FENCE_COMPLETED: u32 = 0x18;
pub const CTX_ERROR_CODE: u32 = 0x1C;
pub const CTX_ERROR_OFFSET: u32 = 0x20;
pub const CTX_ERROR_ACK: u32 = 0x24;
pub const CTX_GL_ERROR: u32 = 0x28;
pub const CTX_GL_ERROR_ACK: u32 = 0x2C;
pub const CTX_FENCE_IRQ_TARGET: u32 = 0x30;

/// `CTX_CONTROL` bits.
pub const CTX_CONTROL_ALLOC: u32 = 1 << 0;
pub const CTX_CONTROL_ENABLE: u32 = 1 << 1;
pub const CTX_CONTROL_RESET: u32 = 1 << 2;

/// `CTX_STATUS` bits.
pub const CTX_STATUS_BUSY: u32 = 1 << 0;
pub const CTX_STATUS_HALTED: u32 = 1 << 1;
pub const CTX_STATUS_IDLE_RING: u32 = 1 << 2;

/// `RING_SIZE` bit 31: `RING_BASE` names a guest address rather than an
/// aperture offset (`CAP_GUESTMEM`). Bits 30:0 are the ring's byte size.
pub const RING_SIZE_GUEST_ADDR: u32 = 1 << 31;
pub const RING_SIZE_BYTES_MASK: u32 = 0x7FFF_FFFF;

/// `SURFACE_DEFINE`'s `flags` bit 0: `address` is a guest address
/// (`CAP_SURFACE_GUESTADDR`) rather than an aperture offset.
pub const SURFACE_DEFINE_FLAG_GUEST_ADDR: u32 = 1 << 0;

/// `READ_PIXELS`'s `flags` bit 0: the first row written is the rectangle's
/// bottom row.
pub const READ_PIXELS_FLAG_ROWS_BOTTOM_UP: u32 = 1 << 0;

/// `CLEAR`'s `mask` bits.
pub const CLEAR_MASK_COLOR: u32 = 1 << 0;
pub const CLEAR_MASK_DEPTH: u32 = 1 << 1;

// ---------------------------------------------------------------------
// Opcodes (`c3d-command-stream`'s opcode map). Each constant is the full
// 16-bit `opcode[31:16]` field of word 0, grouped exactly as the spec's
// section headings group them.
// ---------------------------------------------------------------------

// Control (0x00xx) -- `c3d-cmd-control`.
pub const OP_NOP: u16 = 0x0000;
pub const OP_FENCE: u16 = 0x0001;
pub const OP_FLUSH: u16 = 0x0002;
pub const OP_FINISH: u16 = 0x0003;
pub const OP_CTX_RESET_STATE: u16 = 0x0004;
/// `CALL` (draft 0.14): replay the commands found at a [`Ref`] as if they
/// appeared in the ring at the `CALL` site, then resume after it. Baseline
/// tier -- purely a decode-side addressing mode, not a rendering feature --
/// so it needs no capability bit; see [`super::dispatch`]'s nested-decode
/// handling and this crate's `docs/internals/c3d.md` for the one-level
/// nesting/error-offset rules `c3d-cmd-control` documents.
pub const OP_CALL: u16 = 0x0005;

// Surfaces (0x01xx) -- `c3d-cmd-surfaces`.
pub const OP_SURFACE_DEFINE: u16 = 0x0100;
pub const OP_SURFACE_DESTROY: u16 = 0x0101;
pub const OP_SET_DRAW_SURFACE: u16 = 0x0102;
pub const OP_SURFACE_UPLOAD: u16 = 0x0103;
pub const OP_SURFACE_READBACK: u16 = 0x0104;
pub const OP_CLEAR: u16 = 0x0105;

// Raster state (0x02xx) -- `c3d-cmd-raster`.
pub const OP_ENABLE: u16 = 0x0200;
pub const OP_DISABLE: u16 = 0x0201;
pub const OP_BLEND_FUNC: u16 = 0x0202;
pub const OP_DEPTH_FUNC: u16 = 0x0203;
pub const OP_DEPTH_MASK: u16 = 0x0204;
pub const OP_DEPTH_RANGE: u16 = 0x0205;
pub const OP_ALPHA_FUNC: u16 = 0x0206;
pub const OP_CULL_FACE: u16 = 0x0207;
pub const OP_FRONT_FACE: u16 = 0x0208;
pub const OP_SHADE_MODEL: u16 = 0x0209;
pub const OP_COLOR_MASK: u16 = 0x020A;
pub const OP_SCISSOR: u16 = 0x020B;
pub const OP_VIEWPORT: u16 = 0x020C;
pub const OP_POLYGON_OFFSET: u16 = 0x020D;
pub const OP_CLEAR_COLOR: u16 = 0x020E;
pub const OP_CLEAR_DEPTH: u16 = 0x020F;
pub const OP_FOG_MODE: u16 = 0x0210;
pub const OP_FOG_PARAMS: u16 = 0x0211;
pub const OP_FOG_COLOR: u16 = 0x0212;
pub const OP_HINT: u16 = 0x0213;
pub const OP_LINE_WIDTH: u16 = 0x0214;
pub const OP_POINT_SIZE: u16 = 0x0215;
/// Reserved: GL logic ops are not part of this device. `docs/internals/
/// c3d.md` states this opcode is `E_BAD_OPCODE` in version 1 unconditionally
/// (unlike a tier- or capability-gated opcode, which decodes structurally
/// and is rejected by [`super::state`]), so the decoder in [`super::ring`]
/// treats it exactly like an opcode outside the map: see that module's docs
/// for the reasoning.
pub const OP_LOGIC_OP: u16 = 0x0216;
pub const OP_POLYGON_MODE: u16 = 0x0217;
pub const OP_BLEND_EQUATION: u16 = 0x0218;
pub const OP_BLEND_FUNC_SEPARATE: u16 = 0x0219;

// Matrix (0x03xx) -- `c3d-cmd-matrix`, `CAP_TRANSFORM`.
pub const OP_MATRIX_MODE: u16 = 0x0300;
pub const OP_LOAD_MATRIX: u16 = 0x0301;
pub const OP_LOAD_IDENTITY: u16 = 0x0302;
pub const OP_MULT_MATRIX: u16 = 0x0303;
pub const OP_PUSH_MATRIX: u16 = 0x0304;
pub const OP_POP_MATRIX: u16 = 0x0305;
pub const OP_TRANSLATE: u16 = 0x0306;
pub const OP_ROTATE: u16 = 0x0307;
pub const OP_SCALE: u16 = 0x0308;
pub const OP_FRUSTUM: u16 = 0x0309;
pub const OP_ORTHO: u16 = 0x030A;

// Lighting, clipping and texgen (0x04xx) -- `c3d-cmd-lighting`,
// `CAP_TRANSFORM`.
pub const OP_LIGHT: u16 = 0x0400;
pub const OP_LIGHT_MODEL: u16 = 0x0401;
pub const OP_MATERIAL: u16 = 0x0402;
pub const OP_COLOR_MATERIAL: u16 = 0x0403;
pub const OP_CLIP_PLANE: u16 = 0x0404;
pub const OP_TEXGEN: u16 = 0x0405;
pub const OP_TEXGEN_PLANE: u16 = 0x0406;

// Textures (0x05xx) -- `c3d-cmd-textures`.
pub const OP_TEX_CREATE: u16 = 0x0500;
pub const OP_TEX_DESTROY: u16 = 0x0501;
pub const OP_TEX_BIND: u16 = 0x0502;
pub const OP_TEX_IMAGE: u16 = 0x0503;
pub const OP_TEX_SUBIMAGE: u16 = 0x0504;
pub const OP_TEX_PARAM: u16 = 0x0505;
pub const OP_TEX_ENV: u16 = 0x0506;
pub const OP_TEX_ENV_COLOR: u16 = 0x0507;
pub const OP_ACTIVE_UNIT: u16 = 0x0508;
pub const OP_TEX_PALETTE: u16 = 0x0509;
pub const OP_TEX_COPY_IMAGE: u16 = 0x050A;
pub const OP_TEX_COPY_SUBIMAGE: u16 = 0x050B;

/// `TEX_ENV`'s device-own `pname` for texcoord scaling (`c3d-cmd-textures`),
/// not a GL enumerant: `TEXCOORD_SPACE`, value `0` normalised / `1` texel.
pub const TEX_ENV_TEXCOORD_SPACE: u32 = 0x0001_0000;

// Current vertex state (0x06xx) -- `c3d-cmd-current`.
pub const OP_CURRENT_COLOR: u16 = 0x0600;
pub const OP_CURRENT_NORMAL: u16 = 0x0601;
pub const OP_CURRENT_TEXCOORD: u16 = 0x0602;
pub const OP_CURRENT_FOGCOORD: u16 = 0x0603;

// Draw (0x07xx) -- `c3d-cmd-draw`.
pub const OP_DRAW_INLINE: u16 = 0x0700;
pub const OP_DRAW_INLINE_WIN: u16 = 0x0701;
pub const OP_DRAW_ARRAYS: u16 = 0x0702;
pub const OP_DRAW_ARRAYS_WIN: u16 = 0x0703;
pub const OP_DRAW_ELEMENTS: u16 = 0x0704;
pub const OP_DRAW_ELEMENTS_WIN: u16 = 0x0705;

// Queries (0x08xx) -- `c3d-cmd-queries`.
pub const OP_QUERY: u16 = 0x0800;
pub const OP_READ_PIXELS: u16 = 0x0801;

// ---------------------------------------------------------------------
// Protocol errors (`c3d-errors`'s "Protocol errors" table).
// ---------------------------------------------------------------------

/// A protocol error's numeric code, exactly as latched in `ERROR_CODE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ErrorCode {
    /// `length` was `0`, or the command extends past `RING_TAIL` (or, in
    /// this decoder, past the physical ring end without a `NOP` -- see
    /// [`super::ring`]'s framing docs).
    BadLength = 1,
    /// `RING_TAIL` passed `RING_HEAD`.
    RingOverrun = 2,
    /// Unknown opcode, or an opcode of a tier the device lacks.
    BadOpcode = 3,
    /// A payload word outside its documented range, a `length`
    /// inconsistent with `count`/`format`, or a non-zero reserved field.
    BadArg = 4,
    /// A ref outside the aperture, outside reachable guest memory,
    /// misaligned, or in space `1` without `CAP_GUESTMEM`.
    BadRef = 5,
    /// A texture or surface ID of `0` or above its limit.
    BadId = 6,
    /// A draw, clear, readback or upload with no draw surface.
    NoSurface = 7,
    /// A rectangle outside its surface.
    BadRect = 8,
    /// A surface or texture format the device does not report.
    UnsupportedFormat = 9,
    /// Beyond a reported limit (texture size, unit, light, plane).
    Limit = 10,
}

impl ErrorCode {
    /// Whether this error halts the context (`CTX_STATUS.HALTED`, decoding
    /// stops until `ERROR_ACK`) rather than skipping the offending command.
    /// Per `c3d-errors`'s table, only the two framing errors halt.
    pub const fn halts(self) -> bool {
        matches!(self, ErrorCode::BadLength | ErrorCode::RingOverrun)
    }
}

// ---------------------------------------------------------------------
// Surface formats (`c3d-surface-formats`). The device's own enum, distinct
// from any window-system pixel format and from GL.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SurfaceFormat {
    R5G6B5 = 1,
    R5G6B5Le = 2,
    R5G5B5 = 3,
    R5G5B5Le = 4,
    A8R8G8B8 = 5,
    B8G8R8A8 = 6,
    R8G8B8A8 = 7,
    R8G8B8 = 8,
    B8G8R8 = 9,
    /// Reserved: 8-bit indexed, not implemented in version 1.
    Clut8 = 32,
    /// Only legal as a `READ_PIXELS` format: 32-bit unsigned depth.
    Depth = 255,
}

impl SurfaceFormat {
    /// Bytes per pixel. Used to validate `SURFACE_DEFINE`'s `stride_bytes`
    /// against the row's byte length and, for an aperture-backed surface,
    /// the whole backing extent against `APERTURE_SIZE` (`c3d-cmd-
    /// surfaces`'s `stride_bytes >= the row's byte length` and the
    /// aperture-bound rule `E_BAD_REF` shares with a `Ref`).
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            SurfaceFormat::R5G6B5
            | SurfaceFormat::R5G6B5Le
            | SurfaceFormat::R5G5B5
            | SurfaceFormat::R5G5B5Le => 2,
            SurfaceFormat::A8R8G8B8 | SurfaceFormat::B8G8R8A8 | SurfaceFormat::R8G8B8A8 => 4,
            SurfaceFormat::R8G8B8 | SurfaceFormat::B8G8R8 => 3,
            // Reserved 8-bit indexed format.
            SurfaceFormat::Clut8 => 1,
            // 32-bit unsigned depth; only legal as a READ_PIXELS format, so
            // this value never actually feeds a stride/extent check, but
            // the number is correct if it ever does.
            SurfaceFormat::Depth => 4,
        }
    }
}

impl TryFrom<u32> for SurfaceFormat {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            1 => SurfaceFormat::R5G6B5,
            2 => SurfaceFormat::R5G6B5Le,
            3 => SurfaceFormat::R5G5B5,
            4 => SurfaceFormat::R5G5B5Le,
            5 => SurfaceFormat::A8R8G8B8,
            6 => SurfaceFormat::B8G8R8A8,
            7 => SurfaceFormat::R8G8B8A8,
            8 => SurfaceFormat::R8G8B8,
            9 => SurfaceFormat::B8G8R8,
            32 => SurfaceFormat::Clut8,
            255 => SurfaceFormat::Depth,
            _ => return Err(()),
        })
    }
}

// ---------------------------------------------------------------------
// Texture formats (`c3d-texture-formats`).
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TextureFormat {
    Rgba8 = 0,
    Rgb8 = 1,
    Rgb565 = 2,
    Rgba4444 = 3,
    Rgba5551 = 4,
    L8 = 5,
    La8 = 6,
    A8 = 7,
    /// Intensity; sampled as indices into the bound palette when one is
    /// bound with `TEX_PALETTE`.
    I8 = 8,
    /// Not an image format: `TEXFMT_SUPPORTED` bit 9 reports whether
    /// `TEX_PALETTE` is implemented.
    I8Indexed = 9,
}

impl TryFrom<u32> for TextureFormat {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => TextureFormat::Rgba8,
            1 => TextureFormat::Rgb8,
            2 => TextureFormat::Rgb565,
            3 => TextureFormat::Rgba4444,
            4 => TextureFormat::Rgba5551,
            5 => TextureFormat::L8,
            6 => TextureFormat::La8,
            7 => TextureFormat::A8,
            8 => TextureFormat::I8,
            9 => TextureFormat::I8Indexed,
            _ => return Err(()),
        })
    }
}

// ---------------------------------------------------------------------
// References (`c3d-refs`).
// ---------------------------------------------------------------------

/// Which address space a [`Ref`] names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefSpace {
    /// An aperture offset (space `0`).
    Aperture,
    /// A guest address (space `1`); requires `CAP_GUESTMEM`.
    Guest,
}

/// A decoded two-word reference: `word A` (address) and `word B`
/// (`space[31] | length[30:0]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ref {
    pub address: u32,
    pub space: RefSpace,
    pub length: u32,
}

impl Ref {
    /// Decodes a ref from its two wire words, exactly as `c3d-refs` lays
    /// them out. This performs no validation -- see [`super::ring`] for
    /// the aperture-bound, reachability and alignment checks that turn an
    /// invalid ref into `E_BAD_REF`.
    pub const fn decode(word_a: u32, word_b: u32) -> Ref {
        let space = if word_b & 0x8000_0000 != 0 {
            RefSpace::Guest
        } else {
            RefSpace::Aperture
        };
        Ref {
            address: word_a,
            space,
            length: word_b & 0x7FFF_FFFF,
        }
    }
}

// ---------------------------------------------------------------------
// Vertex format (`c3d-cmd-draw`'s "Vertex format" table).
// ---------------------------------------------------------------------

/// A draw command's `format` word: which optional vertex components are
/// present, in the spec's fixed order, plus the 2-bit `POS_COUNT` field.
/// Bits 10 (`TEXCOORD_STQ`) and 11-31 are reserved and must be zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexFormat(pub u32);

impl VertexFormat {
    pub const COLOR: u32 = 1 << 0;
    pub const NORMAL: u32 = 1 << 1;
    pub const TEXCOORD0: u32 = 1 << 2;
    pub const TEXCOORD1: u32 = 1 << 3;
    pub const TEXCOORD2: u32 = 1 << 4;
    pub const TEXCOORD3: u32 = 1 << 5;
    pub const FOGCOORD: u32 = 1 << 6;
    pub const COLOR_PACKED: u32 = 1 << 7;
    pub const POS_COUNT_SHIFT: u32 = 8;
    pub const POS_COUNT_MASK: u32 = 0b11 << Self::POS_COUNT_SHIFT;
    /// Reserved: an explicit projective `q` per texcoord. Not in version 1
    /// -- a non-zero bit here is `E_BAD_ARG` like any other reserved field.
    pub const TEXCOORD_STQ: u32 = 1 << 10;
    /// Bits 10-31: `TEXCOORD_STQ` plus the fully reserved tail.
    const RESERVED_MASK: u32 = !0u32 << 10;

    /// Every optional per-vertex component bit that adds a fixed number of
    /// words, in format-table order. `COLOR` and `COLOR_PACKED` are
    /// mutually exclusive (checked separately) rather than both appearing
    /// here. `pub(crate)` (not private) so `render.rs`'s `DRAW_ARRAYS*`/
    /// `DRAW_ELEMENTS*` array-descriptor walk can share this exact order
    /// with `vertex_words`/`descriptor_count` rather than duplicating it
    /// and risking the two falling out of sync.
    pub(crate) const OPTIONAL_BITS: [u32; 8] = [
        Self::COLOR,
        Self::NORMAL,
        Self::TEXCOORD0,
        Self::TEXCOORD1,
        Self::TEXCOORD2,
        Self::TEXCOORD3,
        Self::FOGCOORD,
        Self::COLOR_PACKED,
    ];

    pub const fn has(self, bit: u32) -> bool {
        self.0 & bit != 0
    }

    /// The raw 2-bit `POS_COUNT` field (bits 9:8), before validating it's
    /// one of the three defined values.
    pub const fn pos_count_field(self) -> u32 {
        (self.0 & Self::POS_COUNT_MASK) >> Self::POS_COUNT_SHIFT
    }

    pub const fn reserved_bits_set(self) -> bool {
        self.0 & Self::RESERVED_MASK != 0
    }

    /// `POS`'s word count per `POS_COUNT`: `4`/`3`/`2`. `None` for the
    /// reserved field value `3`.
    pub const fn pos_words(self) -> Option<u32> {
        match self.pos_count_field() {
            0 => Some(4),
            1 => Some(3),
            2 => Some(2),
            _ => None,
        }
    }

    /// Structural well-formedness shared by every draw shape: no reserved
    /// bit set, `POS_COUNT` not the reserved value, and `COLOR`/
    /// `COLOR_PACKED` not both set.
    const fn is_well_formed(self) -> bool {
        !self.reserved_bits_set()
            && self.pos_count_field() != 0b11
            && !(self.has(Self::COLOR) && self.has(Self::COLOR_PACKED))
    }

    /// Total `f32`/`u32` words per vertex for `DRAW_INLINE`/
    /// `DRAW_INLINE_WIN`. `None` if the format is structurally invalid.
    pub const fn vertex_words(self) -> Option<u32> {
        if !self.is_well_formed() {
            return None;
        }
        // `pos_words` cannot be `None` here: `is_well_formed` already
        // excluded the reserved `POS_COUNT` value.
        let mut words = match self.pos_words() {
            Some(w) => w,
            None => return None,
        };
        let mut i = 0;
        while i < Self::OPTIONAL_BITS.len() {
            let bit = Self::OPTIONAL_BITS[i];
            let contribution = match bit {
                Self::COLOR => 4,
                Self::NORMAL => 3,
                Self::TEXCOORD0 | Self::TEXCOORD1 | Self::TEXCOORD2 | Self::TEXCOORD3 => 2,
                Self::FOGCOORD => 1,
                Self::COLOR_PACKED => 1,
                _ => 0,
            };
            if self.has(bit) {
                words += contribution;
            }
            i += 1;
        }
        Some(words)
    }

    /// Number of array descriptors for `DRAW_ARRAYS*`/`DRAW_ELEMENTS*`: the
    /// position descriptor (always first) plus one per set optional-bit.
    /// `None` if the format is structurally invalid.
    pub const fn descriptor_count(self) -> Option<u32> {
        if !self.is_well_formed() {
            return None;
        }
        let mut n = 1u32;
        let mut i = 0;
        while i < Self::OPTIONAL_BITS.len() {
            if self.has(Self::OPTIONAL_BITS[i]) {
                n += 1;
            }
            i += 1;
        }
        Some(n)
    }
}

/// One array descriptor from `DRAW_ARRAYS*`/`DRAW_ELEMENTS*`: `type,
/// stride_bytes, ref`. `array_type` is a raw GL enumerant (component count
/// in bits 31:28, element type in the low bits per `c3d-cmd-draw`) passed
/// through unvalidated, same as every other GL enumerant in this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrayDescriptor {
    pub array_type: u32,
    pub stride_bytes: u32,
    pub data: Ref,
}

/// Word count of one array descriptor (`type`, `stride_bytes`, `ref[2]`).
pub const DESCRIPTOR_WORDS: u32 = 4;

impl ArrayDescriptor {
    /// Decodes one descriptor from its 16 raw bytes, as packed
    /// `DESCRIPTOR_WORDS` (4) words apart in a `DrawArrays`/`DrawElements`
    /// command's `descriptors` slice. [`super::ring`] validates the
    /// slice's overall length against `format`'s descriptor count but
    /// leaves decoding the individual descriptors to the caller (typically
    /// [`super::state`], walking them in format-bit order); this is the
    /// shared decode step for that walk.
    pub fn decode(words: &[u8; 16]) -> ArrayDescriptor {
        let w = |i: usize| {
            u32::from_be_bytes([
                words[i * 4],
                words[i * 4 + 1],
                words[i * 4 + 2],
                words[i * 4 + 3],
            ])
        };
        ArrayDescriptor {
            array_type: w(0),
            stride_bytes: w(1),
            data: Ref::decode(w(2), w(3)),
        }
    }
}

// ---------------------------------------------------------------------
// Decoded commands (`c3d-command-stream` and every `c3dxxx` opcode group).
// ---------------------------------------------------------------------

/// One decoded command, borrowed from the ring bytes it came from. Draw
/// commands' bulk payload (vertices, array descriptors) is a borrowed byte
/// slice rather than a parsed-out `Vec`, so decoding a submission costs no
/// allocation; [`super::state`] walks that slice itself using the `format`/
/// `count` already decoded alongside it.
///
/// There is one variant per opcode (matching the spec's per-opcode payload
/// tables exactly) grouped by the spec's own section order, rather than one
/// variant per group: the payload shapes differ enough opcode to opcode
/// (fixed scalars here, a matrix there, a ref, a variable-length tail) that
/// a per-group wrapper would just push the same match back a level without
/// saving any code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Command<'a> {
    // Control (0x00xx).
    /// Padding or an ignored command; `length` may be anything >= 1.
    Nop,
    Fence {
        id: u32,
    },
    Flush,
    Finish,
    CtxResetState,
    /// `CALL`: replay the linear command buffer at `data` in place. See
    /// [`OP_CALL`] and [`super::dispatch`]'s nested-decode handling.
    Call {
        data: Ref,
    },

    // Surfaces (0x01xx).
    SurfaceDefine {
        id: u32,
        width: u32,
        height: u32,
        stride_bytes: u32,
        format: u32,
        flags: u32,
        address: u32,
    },
    SurfaceDestroy {
        id: u32,
    },
    SetDrawSurface {
        id: u32,
    },
    SurfaceUpload {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    },
    SurfaceReadback {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    },
    Clear {
        mask: u32,
    },

    // Raster state (0x02xx).
    Enable {
        cap: u32,
    },
    Disable {
        cap: u32,
    },
    BlendFunc {
        sfactor: u32,
        dfactor: u32,
    },
    DepthFunc {
        func: u32,
    },
    DepthMask {
        flag: u32,
    },
    DepthRange {
        near: f32,
        far: f32,
    },
    AlphaFunc {
        func: u32,
        reference: f32,
    },
    CullFace {
        mode: u32,
    },
    FrontFace {
        mode: u32,
    },
    ShadeModel {
        mode: u32,
    },
    ColorMask {
        r: u32,
        g: u32,
        b: u32,
        a: u32,
    },
    Scissor {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    },
    Viewport {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    },
    PolygonOffset {
        factor: f32,
        units: f32,
    },
    ClearColor {
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    },
    ClearDepth {
        depth: f32,
    },
    FogMode {
        mode: u32,
    },
    FogParams {
        density: f32,
        start: f32,
        end: f32,
    },
    FogColor {
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    },
    Hint {
        target: u32,
        mode: u32,
    },
    LineWidth {
        width: f32,
    },
    PointSize {
        size: f32,
    },
    PolygonMode {
        face: u32,
        mode: u32,
    },
    BlendEquation {
        mode: u32,
    },
    BlendFuncSeparate {
        src_rgb: u32,
        dst_rgb: u32,
        src_a: u32,
        dst_a: u32,
    },

    // Matrix (0x03xx) -- CAP_TRANSFORM.
    MatrixMode {
        mode: u32,
    },
    LoadMatrix {
        m: [f32; 16],
    },
    LoadIdentity,
    MultMatrix {
        m: [f32; 16],
    },
    PushMatrix,
    PopMatrix,
    Translate {
        x: f32,
        y: f32,
        z: f32,
    },
    Rotate {
        angle_deg: f32,
        x: f32,
        y: f32,
        z: f32,
    },
    Scale {
        x: f32,
        y: f32,
        z: f32,
    },
    Frustum {
        l: f32,
        r: f32,
        b: f32,
        t: f32,
        n: f32,
        f: f32,
    },
    Ortho {
        l: f32,
        r: f32,
        b: f32,
        t: f32,
        n: f32,
        f: f32,
    },

    // Lighting, clipping and texgen (0x04xx) -- CAP_TRANSFORM.
    Light {
        light: u32,
        pname: u32,
        v: [f32; 4],
    },
    LightModel {
        pname: u32,
        v: [f32; 4],
    },
    Material {
        face: u32,
        pname: u32,
        v: [f32; 4],
    },
    ColorMaterial {
        face: u32,
        mode: u32,
    },
    ClipPlane {
        plane: u32,
        eq: [f32; 4],
    },
    TexGen {
        unit: u32,
        coord: u32,
        mode: u32,
    },
    TexGenPlane {
        unit: u32,
        coord: u32,
        plane: u32,
        eq: [f32; 4],
    },

    // Textures (0x05xx).
    TexCreate {
        id: u32,
    },
    TexDestroy {
        id: u32,
    },
    TexBind {
        unit: u32,
        id: u32,
    },
    TexImage {
        id: u32,
        level: u32,
        format: u32,
        width: u32,
        height: u32,
        row_bytes: u32,
        data: Ref,
    },
    TexSubImage {
        id: u32,
        level: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        format: u32,
        row_bytes: u32,
        data: Ref,
    },
    TexParam {
        id: u32,
        pname: u32,
        value: u32,
    },
    TexEnv {
        unit: u32,
        pname: u32,
        value: u32,
    },
    TexEnvColor {
        unit: u32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    },
    ActiveUnit {
        unit: u32,
    },
    TexPalette {
        id: u32,
        entries: u32,
        data: Ref,
    },
    TexCopyImage {
        id: u32,
        level: u32,
        format: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    TexCopySubImage {
        id: u32,
        level: u32,
        xoff: u32,
        yoff: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },

    // Current vertex state (0x06xx).
    CurrentColor {
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    },
    CurrentNormal {
        x: f32,
        y: f32,
        z: f32,
    },
    CurrentTexCoord {
        unit: u32,
        s: f32,
        t: f32,
    },
    CurrentFogCoord {
        f: f32,
    },

    // Draw (0x07xx). `vertices`/`descriptors` are the command's raw
    // payload bytes after the fixed header, big-endian words, sized
    // exactly by `format`/`count` (for arrays/elements, index/array
    // descriptors only -- `arrays` in each case).
    DrawInline {
        prim: u32,
        format: VertexFormat,
        count: u32,
        vertices: &'a [u8],
    },
    DrawInlineWin {
        prim: u32,
        format: VertexFormat,
        count: u32,
        vertices: &'a [u8],
    },
    DrawArrays {
        prim: u32,
        format: VertexFormat,
        count: u32,
        descriptors: &'a [u8],
    },
    DrawArraysWin {
        prim: u32,
        format: VertexFormat,
        count: u32,
        descriptors: &'a [u8],
    },
    DrawElements {
        prim: u32,
        format: VertexFormat,
        count: u32,
        index_type: u32,
        min_index: u32,
        max_index: u32,
        index_ref: Ref,
        descriptors: &'a [u8],
    },
    DrawElementsWin {
        prim: u32,
        format: VertexFormat,
        count: u32,
        index_type: u32,
        min_index: u32,
        max_index: u32,
        index_ref: Ref,
        descriptors: &'a [u8],
    },

    // Queries (0x08xx).
    Query {
        what: u32,
        dest: Ref,
    },
    ReadPixels {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        format: u32,
        row_bytes: u32,
        flags: u32,
        dest: Ref,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_magic_spells_c3d_with_a_trailing_space() {
        assert_eq!(&ID_MAGIC.to_be_bytes(), b"C3D ");
    }

    #[test]
    fn error_codes_halt_only_for_the_two_framing_errors() {
        assert!(ErrorCode::BadLength.halts());
        assert!(ErrorCode::RingOverrun.halts());
        assert!(!ErrorCode::BadOpcode.halts());
        assert!(!ErrorCode::BadArg.halts());
        assert!(!ErrorCode::BadRef.halts());
        assert!(!ErrorCode::BadId.halts());
        assert!(!ErrorCode::NoSurface.halts());
        assert!(!ErrorCode::BadRect.halts());
        assert!(!ErrorCode::UnsupportedFormat.halts());
        assert!(!ErrorCode::Limit.halts());
    }

    #[test]
    fn ref_decode_separates_space_and_length_from_word_b() {
        let r = Ref::decode(0x0010_0000, 0x8000_0100);
        assert_eq!(r.address, 0x0010_0000);
        assert_eq!(r.space, RefSpace::Guest);
        assert_eq!(r.length, 0x0000_0100);

        let r = Ref::decode(0x0010_0000, 0x0000_0100);
        assert_eq!(r.space, RefSpace::Aperture);
        assert_eq!(r.length, 0x0000_0100);
    }

    #[test]
    fn vertex_format_pos_count_selects_the_documented_word_counts() {
        assert_eq!(VertexFormat(0b00 << 8).pos_words(), Some(4));
        assert_eq!(VertexFormat(0b01 << 8).pos_words(), Some(3));
        assert_eq!(VertexFormat(0b10 << 8).pos_words(), Some(2));
        assert_eq!(VertexFormat(0b11 << 8).pos_words(), None);
    }

    #[test]
    fn vertex_format_vertex_words_sums_every_optional_component() {
        // POS_COUNT=1 (3 words) + COLOR(4) + TEXCOORD0(2) + FOGCOORD(1).
        let fmt = VertexFormat(
            (1 << VertexFormat::POS_COUNT_SHIFT)
                | VertexFormat::COLOR
                | VertexFormat::TEXCOORD0
                | VertexFormat::FOGCOORD,
        );
        assert_eq!(fmt.vertex_words(), Some(3 + 4 + 2 + 1));
    }

    #[test]
    fn vertex_format_rejects_color_and_color_packed_together() {
        let fmt = VertexFormat(VertexFormat::COLOR | VertexFormat::COLOR_PACKED);
        assert_eq!(fmt.vertex_words(), None);
        assert_eq!(fmt.descriptor_count(), None);
    }

    #[test]
    fn vertex_format_rejects_a_non_zero_reserved_bit() {
        let fmt = VertexFormat(VertexFormat::TEXCOORD_STQ);
        assert_eq!(fmt.vertex_words(), None);
        let fmt = VertexFormat(1 << 31);
        assert_eq!(fmt.vertex_words(), None);
    }

    #[test]
    fn vertex_format_descriptor_count_is_position_plus_one_per_set_bit() {
        let fmt = VertexFormat(VertexFormat::COLOR | VertexFormat::TEXCOORD0);
        assert_eq!(fmt.descriptor_count(), Some(1 + 2));
        assert_eq!(VertexFormat(0).descriptor_count(), Some(1));
    }

    #[test]
    fn surface_format_try_from_accepts_documented_values_and_rejects_gaps() {
        assert_eq!(SurfaceFormat::try_from(1), Ok(SurfaceFormat::R5G6B5));
        assert_eq!(SurfaceFormat::try_from(255), Ok(SurfaceFormat::Depth));
        assert!(SurfaceFormat::try_from(0).is_err());
        assert!(SurfaceFormat::try_from(10).is_err());
    }

    #[test]
    fn surface_format_bytes_per_pixel_matches_each_layout() {
        assert_eq!(SurfaceFormat::R5G6B5.bytes_per_pixel(), 2);
        assert_eq!(SurfaceFormat::R5G6B5Le.bytes_per_pixel(), 2);
        assert_eq!(SurfaceFormat::R5G5B5.bytes_per_pixel(), 2);
        assert_eq!(SurfaceFormat::R5G5B5Le.bytes_per_pixel(), 2);
        assert_eq!(SurfaceFormat::A8R8G8B8.bytes_per_pixel(), 4);
        assert_eq!(SurfaceFormat::B8G8R8A8.bytes_per_pixel(), 4);
        assert_eq!(SurfaceFormat::R8G8B8A8.bytes_per_pixel(), 4);
        assert_eq!(SurfaceFormat::R8G8B8.bytes_per_pixel(), 3);
        assert_eq!(SurfaceFormat::B8G8R8.bytes_per_pixel(), 3);
        assert_eq!(SurfaceFormat::Clut8.bytes_per_pixel(), 1);
        assert_eq!(SurfaceFormat::Depth.bytes_per_pixel(), 4);
    }

    #[test]
    fn array_descriptor_decode_reads_type_stride_and_ref_in_order() {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&0x0000_1406u32.to_be_bytes()); // type: GL_FLOAT
        bytes[4..8].copy_from_slice(&12u32.to_be_bytes()); // stride_bytes
        bytes[8..12].copy_from_slice(&0x0010_0000u32.to_be_bytes()); // ref addr
        bytes[12..16].copy_from_slice(&64u32.to_be_bytes()); // ref length
        let d = ArrayDescriptor::decode(&bytes);
        assert_eq!(d.array_type, 0x0000_1406);
        assert_eq!(d.stride_bytes, 12);
        assert_eq!(d.data.space, RefSpace::Aperture);
        assert_eq!(d.data.address, 0x0010_0000);
        assert_eq!(d.data.length, 64);
    }

    #[test]
    fn texture_format_try_from_accepts_documented_values_and_rejects_gaps() {
        assert_eq!(TextureFormat::try_from(0), Ok(TextureFormat::Rgba8));
        assert_eq!(TextureFormat::try_from(9), Ok(TextureFormat::I8Indexed));
        assert!(TextureFormat::try_from(10).is_err());
    }
}
