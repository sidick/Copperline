// SPDX-License-Identifier: GPL-3.0-or-later

//! The per-context OpenGL 1.x state machine for C3D. See
//! `docs/internals/c3d.md` -- the contract this module implements exactly
//! -- in particular "Context register pages", "Raster state", "Matrix",
//! "Lighting, clipping and texgen", "Textures", "Current vertex state" and
//! "Surfaces". This module holds nothing but plain, typed device state: no
//! GPU types, no rasterisation, no pixel data, no ring decoding and no
//! wire parsing.
//!
//! ## Interface boundary
//!
//! `proto.rs` owns the wire encoding (opcodes, raw payload words);
//! `ring.rs` decodes commands off that wire and calls the typed setters
//! here. Every enumerant this module accepts is therefore a **typed**
//! value, not a raw `u32` -- the wire-to-type conversion lives in each
//! enum's `from_gl`/`from_wire` constructor, and a caller (`ring.rs`) turns
//! a `None` from one of those into `GL_INVALID_ENUM` itself by calling
//! [`State::raise_gl_error`]. This module never parses a command payload.
//! It imports exactly one thing from `proto`: [`proto::ErrorCode`], for
//! the two *protocol* errors (`E_NO_SURFACE`, `E_BAD_RECT`) that depend on
//! live context state only this module has -- see "Protocol errors this
//! module raises" below. It never imports `ring`.
//!
//! `from_gl` values are the public Khronos OpenGL registry's enumerants
//! (`GL_LESS` = `0x0203` and so on -- these are the wire values a real GL
//! driver would use, and are cited in comments next to each arm).
//! `from_wire` is used instead for enums that are this device's own
//! numbering (texture/surface pixel formats), which are not GL enumerants
//! at all.
//!
//! ## Protocol errors this module raises
//!
//! Almost every protocol error (`docs/internals/c3d.md`'s "Protocol
//! errors" table) is a framing or decode concern that belongs to
//! `ring.rs` alone. Two are not: `E_NO_SURFACE` (a draw, `CLEAR`,
//! `SURFACE_READBACK` or `SURFACE_UPLOAD` with no draw surface bound) and
//! `E_BAD_RECT` (a rectangle not wholly inside the surface it acts on, for
//! `SURFACE_UPLOAD`, `SURFACE_READBACK` and `READ_PIXELS`) both depend on
//! the current draw surface's existence and shape, which is exactly the
//! state this module owns and `ring.rs` does not. [`State::require_draw_surface`]
//! and [`State::check_rect`]/[`State::check_draw_surface_rect`] expose
//! those checks as a plain `Result<_, proto::ErrorCode>` -- a return path
//! entirely separate from the `GlError` first-error-wins latch, per the
//! spec's "Errors" section keeping the two classes apart (different
//! acknowledgement registers, `ERROR_ACK` vs `GL_ERROR_ACK`, and different
//! consumers). `SCISSOR`/`VIEWPORT` are deliberately not rectangle-checked
//! this way: GL clamps those instead of erroring.
//!
//! ## Fixed enumerant counts vs. board-configured limits
//!
//! GL 1.1 defines exactly eight lights (`GL_LIGHT0`..`GL_LIGHT7`) and six
//! user clip planes (`GL_CLIP_PLANE0`..`GL_CLIP_PLANE5`) as enumerants, so
//! [`LightId`] and [`ClipPlaneId`] (the wire-decoded selectors) cap out
//! there regardless of what a board advertises -- `GL_LIGHT0 + 8` is
//! simply not a value GL defines. The board's `MAX_LIGHTS`/
//! `MAX_CLIP_PLANES` registers are nonetheless genuine per-board limits on
//! how many of those are actually backed by storage (a device is free to
//! report fewer, though the spec requires at least 8/6 under
//! `CAP_TRANSFORM`), so [`Limits::max_lights`]/[`Limits::max_clip_planes`]
//! size [`Lights`]/[`ClipPlanes`] and [`Enables`]'s per-light/per-plane
//! vectors at construction; an index at or past the configured count is a
//! silent no-op here (the `E_LIMIT` protocol error for it is `ring.rs`'s
//! job, mirroring how out-of-range texture units are already handled).
//! Texture units and matrix stack depths are configured the same way via
//! [`Limits`], for the same reason: no GL enumerant caps them either.
//!
//! ## Determinism
//!
//! Every floating-point operation here (matrix multiply, `ROTATE`'s
//! trigonometry, plane transforms) uses plain `f32`/`f64` arithmetic with
//! no fused multiply-add (`mul_add` is not guaranteed to lower identically
//! across targets), and `ROTATE`'s `sin`/`cos` are computed in `f64` with
//! the whole rotation matrix narrowed to `f32` in a single step at the end
//! rather than per-term, so that most of a libm rounding difference is
//! absorbed before it can matter. This is **not a claim of bit-exact
//! `sin`/`cos` across targets**: they are libm calls, Rust uses the
//! system libm natively but its own implementation on
//! `wasm32-unknown-unknown`, and the two can differ in the last ulp. The
//! single narrowing step hides almost all of that difference but not
//! provably -- an `f64` result sitting exactly on an `f32` rounding
//! boundary can still round a different way per target. That is the
//! guarantee the specification states: floating-point results are
//! reproducible per implementation and per platform, not across
//! platforms, the same scope Copperline's MPEG audio decoder carries and
//! for the same reason (see `docs/internals/c3d.md`'s "Determinism and
//! timing" and `docs/internals/mhi.md`). An implementation wanting
//! cross-platform bit-exactness would have to supply its own
//! transcendentals; the specification does not require it.

use crate::c3d::proto::ErrorCode;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------
// GL errors (docs/internals/c3d.md, "GL errors")
// ---------------------------------------------------------------------

/// A GL-semantic error, tracked per context with first-error-wins
/// semantics (`GL_ERROR`/`GL_ERROR_ACK` in the spec's register table).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlError {
    InvalidEnum,
    InvalidValue,
    InvalidOperation,
    StackOverflow,
    StackUnderflow,
    OutOfMemory,
}

impl GlError {
    /// The GL enumerant `GL_ERROR` reports, per the spec's `GL_ERROR`
    /// register description.
    pub fn to_gl(self) -> u32 {
        match self {
            GlError::InvalidEnum => 0x0500,
            GlError::InvalidValue => 0x0501,
            GlError::InvalidOperation => 0x0502,
            GlError::StackOverflow => 0x0503,
            GlError::StackUnderflow => 0x0504,
            GlError::OutOfMemory => 0x0505,
        }
    }
}

/// `GL_NO_ERROR`.
pub const GL_NO_ERROR: u32 = 0x0000;

// ---------------------------------------------------------------------
// Small typed enums, one per GL enumerant group this device accepts.
// ---------------------------------------------------------------------

macro_rules! gl_enum {
    (
        $(#[$meta:meta])*
        $name:ident { $($variant:ident = $value:expr),+ $(,)? }
    ) => {
        $(#[$meta])*
        #[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        pub enum $name {
            $($variant),+
        }

        impl $name {
            /// Decode a wire value as a Khronos OpenGL registry enumerant.
            /// `None` is what a caller (`ring.rs`) turns into
            /// `GL_INVALID_ENUM`.
            pub fn from_gl(value: u32) -> Option<Self> {
                match value {
                    $($value => Some($name::$variant)),+,
                    _ => None,
                }
            }

            /// The wire value this variant decodes from.
            pub fn to_gl(self) -> u32 {
                match self {
                    $($name::$variant => $value),+
                }
            }
        }
    };
}

gl_enum! {
    /// `BLEND_FUNC`/`BLEND_FUNC_SEPARATE` factors.
    BlendFactor {
        Zero = 0x0000,                 // GL_ZERO
        One = 0x0001,                  // GL_ONE
        SrcColor = 0x0300,             // GL_SRC_COLOR
        OneMinusSrcColor = 0x0301,     // GL_ONE_MINUS_SRC_COLOR
        SrcAlpha = 0x0302,             // GL_SRC_ALPHA
        OneMinusSrcAlpha = 0x0303,     // GL_ONE_MINUS_SRC_ALPHA
        DstAlpha = 0x0304,             // GL_DST_ALPHA
        OneMinusDstAlpha = 0x0305,     // GL_ONE_MINUS_DST_ALPHA
        DstColor = 0x0306,             // GL_DST_COLOR
        OneMinusDstColor = 0x0307,     // GL_ONE_MINUS_DST_COLOR
        SrcAlphaSaturate = 0x0308,     // GL_SRC_ALPHA_SATURATE
    }
}

gl_enum! {
    /// `BLEND_EQUATION` modes.
    BlendEquation {
        FuncAdd = 0x8006,              // GL_FUNC_ADD
        Min = 0x8007,                  // GL_MIN
        Max = 0x8008,                  // GL_MAX
        FuncSubtract = 0x800A,         // GL_FUNC_SUBTRACT
        FuncReverseSubtract = 0x800B,  // GL_FUNC_REVERSE_SUBTRACT
    }
}

gl_enum! {
    /// `DEPTH_FUNC`/`ALPHA_FUNC` comparison functions.
    CompareFunc {
        Never = 0x0200,     // GL_NEVER
        Less = 0x0201,      // GL_LESS
        Equal = 0x0202,     // GL_EQUAL
        Lequal = 0x0203,    // GL_LEQUAL
        Greater = 0x0204,   // GL_GREATER
        Notequal = 0x0205,  // GL_NOTEQUAL
        Gequal = 0x0206,    // GL_GEQUAL
        Always = 0x0207,    // GL_ALWAYS
    }
}

/// `DEPTH_FUNC`'s argument type. A separate alias from [`CompareFunc`]
/// (which also serves `ALPHA_FUNC`) purely for call-site clarity; the wire
/// values and accepted set are identical.
pub type DepthFunc = CompareFunc;
/// `ALPHA_FUNC`'s argument type; see [`DepthFunc`].
pub type AlphaFunc = CompareFunc;

gl_enum! {
    /// `CULL_FACE` mode, and the `face` argument of `MATERIAL`,
    /// `COLOR_MATERIAL` and `POLYGON_MODE`.
    Face {
        Front = 0x0404,         // GL_FRONT
        Back = 0x0405,          // GL_BACK
        FrontAndBack = 0x0408,  // GL_FRONT_AND_BACK
    }
}

gl_enum! {
    /// `FRONT_FACE` winding.
    FrontFace {
        Cw = 0x0900,   // GL_CW
        Ccw = 0x0901,  // GL_CCW
    }
}

gl_enum! {
    /// `SHADE_MODEL`.
    ShadeModel {
        Flat = 0x1D00,    // GL_FLAT
        Smooth = 0x1D01,  // GL_SMOOTH
    }
}

gl_enum! {
    /// `FOG_MODE`.
    FogMode {
        Exp = 0x0800,     // GL_EXP
        Exp2 = 0x0801,    // GL_EXP2
        Linear = 0x2601,  // GL_LINEAR
    }
}

gl_enum! {
    /// `TEX_ENV`'s `TEXTURE_ENV_MODE` values.
    TexEnvMode {
        Add = 0x0104,       // GL_ADD
        Replace = 0x1E01,   // GL_REPLACE
        Modulate = 0x2100,  // GL_MODULATE
        Decal = 0x2101,     // GL_DECAL
        Blend = 0x2201,     // GL_BLEND
    }
}

gl_enum! {
    /// `TEX_PARAM`'s `TEXTURE_MIN_FILTER`/`TEXTURE_MAG_FILTER` values.
    /// The four mipmap modes are only meaningful for `MIN_FILTER`; a
    /// caller that receives one for `MAG_FILTER` treats it as this
    /// device's own error (the spec doesn't separate the two argument
    /// domains, matching GL, which also technically allows it and simply
    /// specifies mipmap `MAG_FILTER` values are never sampled that way).
    TexFilter {
        Nearest = 0x2600,               // GL_NEAREST
        Linear = 0x2601,                // GL_LINEAR
        NearestMipmapNearest = 0x2700,  // GL_NEAREST_MIPMAP_NEAREST
        LinearMipmapNearest = 0x2701,   // GL_LINEAR_MIPMAP_NEAREST
        NearestMipmapLinear = 0x2702,   // GL_NEAREST_MIPMAP_LINEAR
        LinearMipmapLinear = 0x2703,    // GL_LINEAR_MIPMAP_LINEAR
    }
}

gl_enum! {
    /// `TEX_PARAM`'s `TEXTURE_WRAP_S`/`TEXTURE_WRAP_T` values. `CLAMP` is
    /// accepted on the wire but always behaves as `CLAMP_TO_EDGE` (no
    /// border colour); see [`TexWrap::effective`].
    TexWrap {
        Clamp = 0x2900,         // GL_CLAMP
        Repeat = 0x2901,        // GL_REPEAT
        ClampToEdge = 0x812F,   // GL_CLAMP_TO_EDGE
    }
}

impl TexWrap {
    /// The wrap behaviour actually applied: the spec requires `CLAMP` to
    /// behave exactly as `CLAMP_TO_EDGE`, with no border colour ever
    /// introduced ("Textures", the paragraph after `TEX_PARAM`).
    pub fn effective(self) -> TexWrap {
        match self {
            TexWrap::Clamp => TexWrap::ClampToEdge,
            other => other,
        }
    }
}

gl_enum! {
    /// `POLYGON_MODE` mode.
    PolygonMode {
        Point = 0x1B00,  // GL_POINT
        Line = 0x1B01,   // GL_LINE
        Fill = 0x1B02,   // GL_FILL
    }
}

gl_enum! {
    /// `TEXGEN`'s mode argument.
    TexGenMode {
        EyeLinear = 0x2400,     // GL_EYE_LINEAR
        ObjectLinear = 0x2401,  // GL_OBJECT_LINEAR
        SphereMap = 0x2402,     // GL_SPHERE_MAP
    }
}

gl_enum! {
    /// `TEXGEN`/`TEXGEN_PLANE`'s `coord` argument.
    TexCoord {
        S = 0x2000,  // GL_S
        T = 0x2001,  // GL_T
    }
}

gl_enum! {
    /// `TEXGEN_PLANE`'s `plane` argument.
    TexGenPlaneKind {
        ObjectPlane = 0x2501,  // GL_OBJECT_PLANE
        EyePlane = 0x2502,     // GL_EYE_PLANE
    }
}

gl_enum! {
    /// `MATRIX_MODE`.
    MatrixMode {
        Modelview = 0x1700,   // GL_MODELVIEW
        Projection = 0x1701,  // GL_PROJECTION
        Texture = 0x1702,     // GL_TEXTURE
    }
}

gl_enum! {
    /// `LIGHT`'s `pname` argument.
    LightParam {
        Ambient = 0x1200,              // GL_AMBIENT
        Diffuse = 0x1201,              // GL_DIFFUSE
        Specular = 0x1202,             // GL_SPECULAR
        Position = 0x1203,             // GL_POSITION
        SpotDirection = 0x1204,        // GL_SPOT_DIRECTION
        SpotExponent = 0x1205,         // GL_SPOT_EXPONENT
        SpotCutoff = 0x1206,           // GL_SPOT_CUTOFF
        ConstantAttenuation = 0x1207,  // GL_CONSTANT_ATTENUATION
        LinearAttenuation = 0x1208,    // GL_LINEAR_ATTENUATION
        QuadraticAttenuation = 0x1209, // GL_QUADRATIC_ATTENUATION
    }
}

gl_enum! {
    /// `LIGHT`'s `light` argument: `GL_LIGHT0`..`GL_LIGHT7`.
    LightId {
        Light0 = 0x4000,
        Light1 = 0x4001,
        Light2 = 0x4002,
        Light3 = 0x4003,
        Light4 = 0x4004,
        Light5 = 0x4005,
        Light6 = 0x4006,
        Light7 = 0x4007,
    }
}

impl LightId {
    /// Index into [`Lights`]' fixed 8-element array.
    pub fn index(self) -> usize {
        (self.to_gl() - 0x4000) as usize
    }
}

gl_enum! {
    /// `CLIP_PLANE`'s `plane` argument: `GL_CLIP_PLANE0`..`GL_CLIP_PLANE5`.
    ClipPlaneId {
        ClipPlane0 = 0x3000,
        ClipPlane1 = 0x3001,
        ClipPlane2 = 0x3002,
        ClipPlane3 = 0x3003,
        ClipPlane4 = 0x3004,
        ClipPlane5 = 0x3005,
    }
}

impl ClipPlaneId {
    /// Index into [`ClipPlanes`]' fixed 6-element array.
    pub fn index(self) -> usize {
        (self.to_gl() - 0x3000) as usize
    }
}

gl_enum! {
    /// `LIGHT_MODEL`'s `pname` argument.
    LightModelParam {
        LocalViewer = 0x0B51,  // GL_LIGHT_MODEL_LOCAL_VIEWER
        TwoSide = 0x0B52,      // GL_LIGHT_MODEL_TWO_SIDE
        Ambient = 0x0B53,      // GL_LIGHT_MODEL_AMBIENT
    }
}

gl_enum! {
    /// `MATERIAL`'s `pname` argument.
    MaterialParam {
        Emission = 0x1600,          // GL_EMISSION
        Shininess = 0x1601,         // GL_SHININESS
        AmbientAndDiffuse = 0x1602, // GL_AMBIENT_AND_DIFFUSE
        Ambient = 0x1200,           // GL_AMBIENT
        Diffuse = 0x1201,           // GL_DIFFUSE
        Specular = 0x1202,          // GL_SPECULAR
    }
}

gl_enum! {
    /// `COLOR_MATERIAL`'s `mode` argument (the `MATERIAL` pnames that make
    /// sense as a tracked mode -- everything but `SHININESS`, which GL
    /// also excludes).
    ColorMaterialMode {
        Emission = 0x1600,          // GL_EMISSION
        AmbientAndDiffuse = 0x1602, // GL_AMBIENT_AND_DIFFUSE
        Ambient = 0x1200,           // GL_AMBIENT
        Diffuse = 0x1201,           // GL_DIFFUSE
        Specular = 0x1202,          // GL_SPECULAR
    }
}

gl_enum! {
    /// GL-space primitive type (`prim` on the draw opcodes). Not consumed
    /// by this module -- draw commands are `ring.rs`'s territory -- but
    /// defined here alongside the device's other typed wire enumerants.
    PrimitiveType {
        Points = 0x0000,        // GL_POINTS
        Lines = 0x0001,         // GL_LINES
        LineLoop = 0x0002,      // GL_LINE_LOOP
        LineStrip = 0x0003,     // GL_LINE_STRIP
        Triangles = 0x0004,     // GL_TRIANGLES
        TriangleStrip = 0x0005, // GL_TRIANGLE_STRIP
        TriangleFan = 0x0006,   // GL_TRIANGLE_FAN
        Quads = 0x0007,         // GL_QUADS
        QuadStrip = 0x0008,     // GL_QUAD_STRIP
        Polygon = 0x0009,       // GL_POLYGON
    }
}

/// `ENABLE`/`DISABLE` capabilities (`docs/internals/c3d.md`'s "Accepted
/// `ENABLE`/`DISABLE` caps" list). `Texture2D`/`TextureGenS`/
/// `TextureGenT` apply to whichever unit is currently `ACTIVE_UNIT` at the
/// time of the call, per the spec; `Light`/`ClipPlane` carry their own
/// index because the wire enumerant already encodes it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Capability {
    AlphaTest,
    Blend,
    CullFace,
    DepthTest,
    Dither,
    Fog,
    PolygonOffsetFill,
    ScissorTest,
    Texture2D,
    Lighting,
    Light(LightId),
    ColorMaterial,
    Normalize,
    RescaleNormal,
    ClipPlane(ClipPlaneId),
    TextureGenS,
    TextureGenT,
}

impl Capability {
    /// Decode `cap` as a Khronos OpenGL registry enumerant. `None` covers
    /// both a genuinely unknown enumerant and a legal-but-unsupported one
    /// such as `GL_STENCIL_TEST` (there is no stencil buffer in this
    /// protocol version) -- both are `GL_INVALID_ENUM` per the spec, so
    /// the caller need not distinguish them.
    pub fn from_gl(cap: u32) -> Option<Self> {
        Some(match cap {
            0x0BC0 => Capability::AlphaTest,         // GL_ALPHA_TEST
            0x0BE2 => Capability::Blend,             // GL_BLEND
            0x0B44 => Capability::CullFace,          // GL_CULL_FACE
            0x0B71 => Capability::DepthTest,         // GL_DEPTH_TEST
            0x0BD0 => Capability::Dither,            // GL_DITHER
            0x0B60 => Capability::Fog,               // GL_FOG
            0x8037 => Capability::PolygonOffsetFill, // GL_POLYGON_OFFSET_FILL
            0x0C11 => Capability::ScissorTest,       // GL_SCISSOR_TEST
            0x0DE1 => Capability::Texture2D,         // GL_TEXTURE_2D
            0x0B50 => Capability::Lighting,          // GL_LIGHTING
            0x0B57 => Capability::ColorMaterial,     // GL_COLOR_MATERIAL
            0x0BA1 => Capability::Normalize,         // GL_NORMALIZE
            0x803A => Capability::RescaleNormal,     // GL_RESCALE_NORMAL
            0x0C60 => Capability::TextureGenS,       // GL_TEXTURE_GEN_S
            0x0C61 => Capability::TextureGenT,       // GL_TEXTURE_GEN_T
            v @ 0x4000..=0x4007 => Capability::Light(LightId::from_gl(v)?),
            v @ 0x3000..=0x3005 => Capability::ClipPlane(ClipPlaneId::from_gl(v)?),
            _ => return None,
        })
    }
}

/// Texture pixel formats (`docs/internals/c3d.md`'s "Texture formats").
/// This is the device's own numbering, not a GL enumerant, so it decodes
/// with [`TexFormat::from_wire`] rather than `from_gl`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TexFormat {
    Rgba8,
    Rgb8,
    Rgb565,
    Rgba4444,
    Rgba5551,
    L8,
    La8,
    A8,
    I8,
}

impl TexFormat {
    pub fn from_wire(value: u32) -> Option<Self> {
        Some(match value {
            0 => TexFormat::Rgba8,
            1 => TexFormat::Rgb8,
            2 => TexFormat::Rgb565,
            3 => TexFormat::Rgba4444,
            4 => TexFormat::Rgba5551,
            5 => TexFormat::L8,
            6 => TexFormat::La8,
            7 => TexFormat::A8,
            8 => TexFormat::I8,
            _ => return None,
        })
    }
}

/// Surface pixel formats (`docs/internals/c3d.md`'s "Surface formats").
/// The device's own numbering; see [`TexFormat`]'s note. `Depth` is only
/// legal as a `READ_PIXELS` destination format, which this module does
/// not implement (pixel operations are out of scope), but the variant is
/// included for completeness of the wire enumeration.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurfaceFormat {
    R5g6b5,
    R5g6b5Le,
    R5g5b5,
    R5g5b5Le,
    A8r8g8b8,
    B8g8r8a8,
    R8g8b8a8,
    R8g8b8,
    B8g8r8,
    Depth,
}

impl SurfaceFormat {
    pub fn from_wire(value: u32) -> Option<Self> {
        Some(match value {
            1 => SurfaceFormat::R5g6b5,
            2 => SurfaceFormat::R5g6b5Le,
            3 => SurfaceFormat::R5g5b5,
            4 => SurfaceFormat::R5g5b5Le,
            5 => SurfaceFormat::A8r8g8b8,
            6 => SurfaceFormat::B8g8r8a8,
            7 => SurfaceFormat::R8g8b8a8,
            8 => SurfaceFormat::R8g8b8,
            9 => SurfaceFormat::B8g8r8,
            255 => SurfaceFormat::Depth,
            _ => return None,
        })
    }
}

/// The device's own `TEX_ENV` pname `TEXCOORD_SPACE` (`0x0001_0000`)
/// value: normalised (`0..1` spans the texture, GL's convention) or texel
/// (post-transform clients such as Warp3D). Not a GL enumerant, hence
/// `from_wire`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TexCoordSpace {
    Normalised,
    Texel,
}

impl TexCoordSpace {
    pub fn from_wire(value: u32) -> Option<Self> {
        match value {
            0 => Some(TexCoordSpace::Normalised),
            1 => Some(TexCoordSpace::Texel),
            _ => None,
        }
    }
}

/// A `TEX_PARAM` payload, already split into the typed pname/value pair
/// `ring.rs` decoded.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TexParam {
    MinFilter(TexFilter),
    MagFilter(TexFilter),
    WrapS(TexWrap),
    WrapT(TexWrap),
}

/// A `TEX_ENV` payload for the pnames this module tracks (`TEX_ENV_COLOR`
/// is its own setter, [`State::set_tex_env_color`], since it is not a
/// `pname, value` pair).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TexEnvParam {
    Mode(TexEnvMode),
    CoordSpace(TexCoordSpace),
}

/// Where a surface's (or, one day, a guest-address texture's) pixels
/// live: an aperture offset, or -- `CAP_SURFACE_GUESTADDR` -- a guest
/// address. Mirrors `SURFACE_DEFINE`'s `flags` bit 0.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backing {
    Aperture(u32),
    Guest(u32),
}

// ---------------------------------------------------------------------
// 4x4 matrices
// ---------------------------------------------------------------------

/// A GL-layout 4x4 matrix: 16 `f32` in column-major order, so `m[c * 4 +
/// r]` is row `r`, column `c` -- the same flat layout `LOAD_MATRIX` and
/// `QUERY`'s matrix results carry on the wire.
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mat4(pub [f32; 16]);

impl Mat4 {
    pub fn identity() -> Mat4 {
        #[rustfmt::skip]
        let m = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        Mat4(m)
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f32 {
        self.0[col * 4 + row]
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, v: f32) {
        self.0[col * 4 + row] = v;
    }

    /// `a * b`, GL's matrix multiplication order (column vectors,
    /// applying `b` first then `a`).
    pub fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
        let mut out = Mat4([0.0; 16]);
        for row in 0..4 {
            for col in 0..4 {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    sum += a.get(row, k) * b.get(k, col);
                }
                out.set(row, col, sum);
            }
        }
        out
    }

    /// `self * v`, treating `v` as a column vector.
    pub fn transform_point(&self, v: [f32; 4]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for row in 0..4 {
            let mut sum = 0.0f32;
            for col in 0..4 {
                sum += self.get(row, col) * v[col];
            }
            out[row] = sum;
        }
        out
    }

    /// The upper-left 3x3 applied to a direction, with no translation --
    /// what GL uses to transform `SPOT_DIRECTION`.
    pub fn transform_direction3(&self, v: [f32; 3]) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for row in 0..3 {
            let mut sum = 0.0f32;
            for col in 0..3 {
                sum += self.get(row, col) * v[col];
            }
            out[row] = sum;
        }
        out
    }

    /// Transforms a normal-space vector by `self`'s transpose, restricted
    /// to the upper-left 3x3 -- i.e. `transpose(self) * v` for `v`
    /// extended with an implied zero fourth component. Callers pass
    /// `modelview.inverse()` as `self`, which makes this
    /// `transpose(inverse(modelview)) * n`: GL 1.1's correct rule for
    /// transforming a normal into eye space (fixed-function pipeline,
    /// section 2.11), distinct from [`Mat4::transform_direction3`]'s
    /// plain upper-3x3 multiply -- the two only agree when the modelview's
    /// linear part is orthogonal (pure rotation, no scale). Under a
    /// non-uniform scale the two diverge; a normal transformed by the
    /// plain matrix (rather than the inverse-transpose) comes out
    /// non-perpendicular to a correspondingly-scaled surface.
    pub fn transform_normal3(&self, v: [f32; 3]) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for row in 0..3 {
            let mut sum = 0.0f32;
            for col in 0..3 {
                // `self.get(col, row)`, not `(row, col)`: this is the
                // transpose access pattern.
                sum += self.get(col, row) * v[col];
            }
            out[row] = sum;
        }
        out
    }

    /// The 4x4 inverse via Gauss-Jordan elimination with partial
    /// pivoting, or `None` if `self` is singular (GL leaves the result of
    /// transforming a clip plane or eye texgen plane by a singular
    /// modelview undefined; this module substitutes the identity so the
    /// operation is at least total -- see [`State::eye_transform`]).
    pub fn inverse(&self) -> Option<Mat4> {
        // Augmented [self | I], stored as 4 rows of 8 columns.
        let mut a = [[0.0f32; 8]; 4];
        for r in 0..4 {
            for c in 0..4 {
                a[r][c] = self.get(r, c);
            }
            a[r][4 + r] = 1.0;
        }
        for col in 0..4 {
            let mut pivot = col;
            let mut best = a[col][col].abs();
            for r in (col + 1)..4 {
                if a[r][col].abs() > best {
                    best = a[r][col].abs();
                    pivot = r;
                }
            }
            if best < 1e-12 {
                return None;
            }
            a.swap(col, pivot);
            let d = a[col][col];
            for k in 0..8 {
                a[col][k] /= d;
            }
            for r in 0..4 {
                if r == col {
                    continue;
                }
                let f = a[r][col];
                if f != 0.0 {
                    for k in 0..8 {
                        a[r][k] -= f * a[col][k];
                    }
                }
            }
        }
        let mut out = Mat4::identity();
        for r in 0..4 {
            for c in 0..4 {
                out.set(r, c, a[r][4 + c]);
            }
        }
        Some(out)
    }

    fn translation(x: f32, y: f32, z: f32) -> Mat4 {
        let mut m = Mat4::identity();
        m.set(0, 3, x);
        m.set(1, 3, y);
        m.set(2, 3, z);
        m
    }

    fn scaling(x: f32, y: f32, z: f32) -> Mat4 {
        let mut m = Mat4::identity();
        m.set(0, 0, x);
        m.set(1, 1, y);
        m.set(2, 2, z);
        m
    }

    /// The GL `glRotate` matrix for `angle_deg` about axis `(x, y, z)`
    /// (normalised internally; a zero-length axis is GL-undefined and is
    /// treated here as identity, i.e. no rotation, rather than producing
    /// NaNs). All trigonometry runs in `f64` and the whole matrix is
    /// narrowed to `f32` once at the end, per this module's determinism
    /// note.
    fn rotation(angle_deg: f32, x: f32, y: f32, z: f32) -> Mat4 {
        let len = ((x as f64).powi(2) + (y as f64).powi(2) + (z as f64).powi(2)).sqrt();
        if len < 1e-12 {
            return Mat4::identity();
        }
        let (x, y, z) = (x as f64 / len, y as f64 / len, z as f64 / len);
        let radians = (angle_deg as f64).to_radians();
        let c = radians.cos();
        let s = radians.sin();
        let t = 1.0 - c;

        #[rustfmt::skip]
        let m64: [f64; 16] = [
            t * x * x + c,       t * x * y + s * z,   t * x * z - s * y,   0.0,
            t * x * y - s * z,   t * y * y + c,        t * y * z + s * x,  0.0,
            t * x * z + s * y,   t * y * z - s * x,    t * z * z + c,      0.0,
            0.0,                 0.0,                  0.0,                1.0,
        ];
        let mut m32 = [0.0f32; 16];
        for i in 0..16 {
            m32[i] = m64[i] as f32;
        }
        Mat4(m32)
    }

    /// GL's `glFrustum` matrix.
    fn frustum(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4 {
        let mut m = Mat4([0.0; 16]);
        m.set(0, 0, (2.0 * n) / (r - l));
        m.set(1, 1, (2.0 * n) / (t - b));
        m.set(0, 2, (r + l) / (r - l));
        m.set(1, 2, (t + b) / (t - b));
        m.set(2, 2, -(f + n) / (f - n));
        m.set(3, 2, -1.0);
        m.set(2, 3, -(2.0 * f * n) / (f - n));
        m
    }

    /// GL's `glOrtho` matrix.
    fn ortho(l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) -> Mat4 {
        let mut m = Mat4::identity();
        m.set(0, 0, 2.0 / (r - l));
        m.set(1, 1, 2.0 / (t - b));
        m.set(2, 2, -2.0 / (f - n));
        m.set(0, 3, -(r + l) / (r - l));
        m.set(1, 3, -(t + b) / (t - b));
        m.set(2, 3, -(f + n) / (f - n));
        m
    }
}

/// One GL matrix stack: `PUSH`/`POP`/`LOAD`/`MULT` per GL 1.1, bounded to
/// `limit` entries. The stack always holds at least one matrix (the
/// current one); `POP` at depth 1 is `GL_STACK_UNDERFLOW` rather than
/// emptying the stack, matching GL.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MatrixStack {
    entries: Vec<Mat4>,
    limit: usize,
}

impl MatrixStack {
    pub fn new(limit: usize) -> MatrixStack {
        MatrixStack {
            entries: vec![Mat4::identity()],
            limit: limit.max(1),
        }
    }

    pub fn top(&self) -> &Mat4 {
        self.entries.last().expect("matrix stack is never empty")
    }

    fn top_mut(&mut self) -> &mut Mat4 {
        self.entries
            .last_mut()
            .expect("matrix stack is never empty")
    }

    pub fn depth(&self) -> usize {
        self.entries.len()
    }

    /// `PUSH_MATRIX`: duplicates the top entry. `GL_STACK_OVERFLOW`, stack
    /// unchanged, past the reported depth.
    pub fn push(&mut self) -> Result<(), GlError> {
        if self.entries.len() >= self.limit {
            return Err(GlError::StackOverflow);
        }
        let top = *self.top();
        self.entries.push(top);
        Ok(())
    }

    /// `POP_MATRIX`. `GL_STACK_UNDERFLOW`, stack unchanged, at depth 1.
    pub fn pop(&mut self) -> Result<(), GlError> {
        if self.entries.len() <= 1 {
            return Err(GlError::StackUnderflow);
        }
        self.entries.pop();
        Ok(())
    }

    pub fn load_identity(&mut self) {
        *self.top_mut() = Mat4::identity();
    }

    pub fn load(&mut self, m: Mat4) {
        *self.top_mut() = m;
    }

    /// `MULT_MATRIX`: post-multiplies (`top = top * m`), per GL and per
    /// the spec's "Matrix" section.
    pub fn mult(&mut self, m: &Mat4) {
        let top = *self.top();
        *self.top_mut() = Mat4::mul(&top, m);
    }
}

// ---------------------------------------------------------------------
// Board-configured limits
// ---------------------------------------------------------------------

/// The board-configured sizes [`State`] needs but that GL 1.1 does not fix
/// an enumerant count for -- texture units and matrix stack depths.
/// Lights and clip planes are *not* here; see this module's doc comment.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    /// `MAX_TEXTURE_UNITS`. At least 1.
    pub texture_units: usize,
    /// `MAX_MATRIX_DEPTH_MV`. The spec requires >= 32 under
    /// `CAP_TRANSFORM`.
    pub modelview_depth: usize,
    /// `MAX_MATRIX_DEPTH_PROJ`. The spec requires >= 2.
    pub projection_depth: usize,
    /// `MAX_MATRIX_DEPTH_TEX`. The spec requires >= 2.
    pub texture_depth: usize,
    /// `MAX_LIGHTS`. The spec requires >= 8 under `CAP_TRANSFORM`; GL 1.1
    /// itself never names a light past `GL_LIGHT7`, so a value above 8
    /// currently has no [`LightId`] able to reach it (see this module's
    /// doc comment).
    pub max_lights: usize,
    /// `MAX_CLIP_PLANES`. The spec requires >= 6 under `CAP_TRANSFORM`;
    /// see [`Limits::max_lights`]'s note -- the same reasoning applies via
    /// [`ClipPlaneId`].
    pub max_clip_planes: usize,
}

impl Default for Limits {
    /// The spec's stated minimums, with two texture units (Copperline's
    /// first-client target is two-unit multitexture; see
    /// `docs/internals/c3d.md`'s "Implementation order").
    fn default() -> Self {
        Limits {
            texture_units: 2,
            modelview_depth: 32,
            projection_depth: 2,
            // The spec's minimum is 2, but the first client's driver
            // (v27.2) carries a real 10-deep GL_TEXTURE stack it may map
            // 1:1 onto the board's -- 16 gives it headroom without the
            // guest library having to flatten pushes client-side.
            texture_depth: 16,
            max_lights: 8,
            max_clip_planes: 6,
        }
    }
}

// ---------------------------------------------------------------------
// Raster state
// ---------------------------------------------------------------------

/// Which capabilities are enabled. Every per-index vector here is sized by
/// [`Limits`] at construction (texture units for the texture-related
/// ones; [`Limits::max_lights`]/[`Limits::max_clip_planes`] for the
/// lighting ones).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Enables {
    pub alpha_test: bool,
    pub blend: bool,
    pub cull_face: bool,
    pub depth_test: bool,
    pub dither: bool,
    pub fog: bool,
    pub polygon_offset_fill: bool,
    pub scissor_test: bool,
    pub lighting: bool,
    pub color_material: bool,
    pub normalize: bool,
    pub rescale_normal: bool,
    /// Indexed by [`LightId::index`]; sized by [`Limits::max_lights`].
    pub light: Vec<bool>,
    /// Indexed by [`ClipPlaneId::index`]; sized by
    /// [`Limits::max_clip_planes`].
    pub clip_plane: Vec<bool>,
    /// Per texture unit.
    pub texture_2d: Vec<bool>,
    /// Per texture unit.
    pub texture_gen_s: Vec<bool>,
    /// Per texture unit.
    pub texture_gen_t: Vec<bool>,
}

impl Enables {
    fn new(texture_units: usize, max_lights: usize, max_clip_planes: usize) -> Enables {
        Enables {
            alpha_test: false,
            blend: false,
            cull_face: false,
            depth_test: false,
            dither: true, // spec: "everything disabled except DITHER"
            fog: false,
            polygon_offset_fill: false,
            scissor_test: false,
            lighting: false,
            color_material: false,
            normalize: false,
            rescale_normal: false,
            light: vec![false; max_lights],
            clip_plane: vec![false; max_clip_planes],
            texture_2d: vec![false; texture_units],
            texture_gen_s: vec![false; texture_units],
            texture_gen_t: vec![false; texture_units],
        }
    }
}

/// A GL rectangle in surface pixels, origin top-left (`SCISSOR`,
/// `VIEWPORT`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub const ZERO: Rect = Rect {
        x: 0,
        y: 0,
        w: 0,
        h: 0,
    };
}

/// `COLOR_MASK`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColorMask {
    pub r: bool,
    pub g: bool,
    pub b: bool,
    pub a: bool,
}

/// A raw RGBA colour, components in `0..1` per the spec's convention for
/// float colour payloads.
pub type Rgba = [f32; 4];

/// Everything in `docs/internals/c3d.md`'s "Raster state" section that
/// isn't an `Enables` flag.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RasterState {
    pub blend_src_rgb: BlendFactor,
    pub blend_dst_rgb: BlendFactor,
    pub blend_src_alpha: BlendFactor,
    pub blend_dst_alpha: BlendFactor,
    pub blend_equation: BlendEquation,
    pub depth_func: DepthFunc,
    pub depth_mask: bool,
    pub depth_range: (f32, f32),
    pub alpha_func: AlphaFunc,
    pub alpha_ref: f32,
    pub cull_face_mode: Face,
    pub front_face: FrontFace,
    pub shade_model: ShadeModel,
    pub color_mask: ColorMask,
    pub scissor: Rect,
    pub viewport: Rect,
    pub polygon_offset_factor: f32,
    pub polygon_offset_units: f32,
    pub clear_color: Rgba,
    pub clear_depth: f32,
    pub fog_mode: FogMode,
    pub fog_density: f32,
    pub fog_start: f32,
    pub fog_end: f32,
    pub fog_color: Rgba,
    pub line_width: f32,
    pub point_size: f32,
    pub polygon_mode_front: PolygonMode,
    pub polygon_mode_back: PolygonMode,
}

impl Default for RasterState {
    /// The spec's "Initial state is GL's" paragraph, plus plain GL 1.1
    /// defaults for the handful of fields it doesn't spell out
    /// (`POLYGON_OFFSET`, `LINE_WIDTH`, `POINT_SIZE`, `POLYGON_MODE`) and
    /// `BLEND_EQUATION`'s `FUNC_ADD`, confirmed against that paragraph's
    /// silence on it (added in draft 0.3 alongside `BLEND_FUNC_SEPARATE`,
    /// after the initial-state paragraph was written).
    fn default() -> Self {
        RasterState {
            blend_src_rgb: BlendFactor::One,
            blend_dst_rgb: BlendFactor::Zero,
            blend_src_alpha: BlendFactor::One,
            blend_dst_alpha: BlendFactor::Zero,
            blend_equation: BlendEquation::FuncAdd,
            depth_func: DepthFunc::Less,
            depth_mask: true,
            depth_range: (0.0, 1.0),
            alpha_func: AlphaFunc::Always,
            alpha_ref: 0.0,
            cull_face_mode: Face::Back,
            front_face: FrontFace::Ccw,
            shade_model: ShadeModel::Smooth,
            color_mask: ColorMask {
                r: true,
                g: true,
                b: true,
                a: true,
            },
            scissor: Rect::ZERO,
            viewport: Rect::ZERO,
            polygon_offset_factor: 0.0,
            polygon_offset_units: 0.0,
            clear_color: [0.0, 0.0, 0.0, 0.0],
            clear_depth: 1.0,
            fog_mode: FogMode::Exp,
            fog_density: 1.0,
            fog_start: 0.0,
            fog_end: 1.0,
            fog_color: [0.0, 0.0, 0.0, 0.0],
            line_width: 1.0,
            point_size: 1.0,
            polygon_mode_front: PolygonMode::Fill,
            polygon_mode_back: PolygonMode::Fill,
        }
    }
}

// ---------------------------------------------------------------------
// Lighting
// ---------------------------------------------------------------------

/// One light's parameters (`docs/internals/c3d.md`'s `LIGHT` opcode).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Light {
    pub ambient: Rgba,
    pub diffuse: Rgba,
    pub specular: Rgba,
    pub position: [f32; 4],
    pub spot_direction: [f32; 3],
    pub spot_exponent: f32,
    pub spot_cutoff: f32,
    pub constant_attenuation: f32,
    pub linear_attenuation: f32,
    pub quadratic_attenuation: f32,
}

impl Light {
    /// GL 1.1's default for `LIGHT1`..`LIGHT7`; `LIGHT0` overrides
    /// `diffuse`/`specular` to white (see [`Lights::new`]).
    fn default_non_zero() -> Light {
        Light {
            ambient: [0.0, 0.0, 0.0, 1.0],
            diffuse: [0.0, 0.0, 0.0, 0.0],
            specular: [0.0, 0.0, 0.0, 0.0],
            position: [0.0, 0.0, 1.0, 0.0],
            spot_direction: [0.0, 0.0, -1.0],
            spot_exponent: 0.0,
            spot_cutoff: 180.0,
            constant_attenuation: 1.0,
            linear_attenuation: 0.0,
            quadratic_attenuation: 0.0,
        }
    }
}

/// The board's lights, sized by [`Limits::max_lights`]; see this module's
/// doc comment on why the *count* is board-configured while [`LightId`]'s
/// wire range still stops at `GL_LIGHT7`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lights(pub Vec<Light>);

impl Lights {
    fn new(count: usize) -> Lights {
        let mut lights: Vec<Light> = (0..count).map(|_| Light::default_non_zero()).collect();
        // GL 1.1: LIGHT0's diffuse and specular default to white; every
        // other light defaults to black (i.e. contributes nothing until
        // configured).
        if let Some(light0) = lights.first_mut() {
            light0.diffuse = [1.0, 1.0, 1.0, 1.0];
            light0.specular = [1.0, 1.0, 1.0, 1.0];
        }
        Lights(lights)
    }
}

/// `LIGHT_MODEL`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LightModel {
    pub ambient: Rgba,
    pub local_viewer: bool,
    pub two_side: bool,
}

impl Default for LightModel {
    fn default() -> Self {
        LightModel {
            ambient: [0.2, 0.2, 0.2, 1.0],
            local_viewer: false,
            two_side: false,
        }
    }
}

/// One face's `MATERIAL` parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Material {
    pub ambient: Rgba,
    pub diffuse: Rgba,
    pub specular: Rgba,
    pub emission: Rgba,
    pub shininess: f32,
}

impl Default for Material {
    fn default() -> Self {
        Material {
            ambient: [0.2, 0.2, 0.2, 1.0],
            diffuse: [0.8, 0.8, 0.8, 1.0],
            specular: [0.0, 0.0, 0.0, 1.0],
            emission: [0.0, 0.0, 0.0, 1.0],
            shininess: 0.0,
        }
    }
}

/// `COLOR_MATERIAL`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorMaterial {
    pub face: Face,
    pub mode: ColorMaterialMode,
}

impl Default for ColorMaterial {
    fn default() -> Self {
        ColorMaterial {
            face: Face::FrontAndBack,
            mode: ColorMaterialMode::AmbientAndDiffuse,
        }
    }
}

/// The board's user clip planes, sized by [`Limits::max_clip_planes`];
/// see this module's doc comment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipPlanes(pub Vec<[f32; 4]>);

impl ClipPlanes {
    fn new(count: usize) -> ClipPlanes {
        ClipPlanes(vec![[0.0, 0.0, 0.0, 0.0]; count])
    }
}

/// One texture unit's `TEXGEN`/`TEXGEN_PLANE` state, for `S` and `T`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TexGenUnit {
    pub mode_s: TexGenMode,
    pub mode_t: TexGenMode,
    pub object_plane_s: [f32; 4],
    pub object_plane_t: [f32; 4],
    pub eye_plane_s: [f32; 4],
    pub eye_plane_t: [f32; 4],
}

impl Default for TexGenUnit {
    /// GL 1.1's default texgen state: `EYE_LINEAR` for both coordinates,
    /// and object/eye planes of `(1,0,0,0)` for `S` and `(0,1,0,0)` for
    /// `T` (identical at identity modelview, which is why both default to
    /// the same values).
    fn default() -> Self {
        TexGenUnit {
            mode_s: TexGenMode::EyeLinear,
            mode_t: TexGenMode::EyeLinear,
            object_plane_s: [1.0, 0.0, 0.0, 0.0],
            object_plane_t: [0.0, 1.0, 0.0, 0.0],
            eye_plane_s: [1.0, 0.0, 0.0, 0.0],
            eye_plane_t: [0.0, 1.0, 0.0, 0.0],
        }
    }
}

// ---------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------

/// One defined mip level's shape -- enough to describe the image's
/// existence for validation and queries; the pixel data itself lives on
/// the GPU side this module deliberately knows nothing about.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TexLevel {
    pub width: u32,
    pub height: u32,
    pub format: TexFormat,
}

/// A texture object's device-owned parameters (`docs/internals/c3d.md`'s
/// "Textures"). `TEX_IMAGE`/`TEX_SUBIMAGE`/`TEX_COPY_IMAGE`/
/// `TEX_COPY_SUBIMAGE`'s pixel payloads are out of this module's scope;
/// only the resulting level shape is recorded, via
/// [`State::define_tex_level`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureObject {
    pub min_filter: TexFilter,
    pub mag_filter: TexFilter,
    pub wrap_s: TexWrap,
    pub wrap_t: TexWrap,
    /// `TEX_PALETTE`'s bound palette entry count; `None` when unbound
    /// (`entries == 0` on the wire unbinds).
    pub palette_entries: Option<u32>,
    pub levels: BTreeMap<u32, TexLevel>,
}

impl Default for TextureObject {
    /// GL 1.1 defaults: `MIN_FILTER` is `NEAREST_MIPMAP_LINEAR`,
    /// `MAG_FILTER` is `LINEAR`, both wrap axes `REPEAT`.
    fn default() -> Self {
        TextureObject {
            min_filter: TexFilter::NearestMipmapLinear,
            mag_filter: TexFilter::Linear,
            wrap_s: TexWrap::Repeat,
            wrap_t: TexWrap::Repeat,
            palette_entries: None,
            levels: BTreeMap::new(),
        }
    }
}

/// One texture unit's `TEX_ENV` state and current binding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextureUnit {
    /// `0` = unbound (the null texture; texturing disabled for the unit
    /// regardless of `TEXTURE_2D`'s enable bit).
    pub bound_texture: u32,
    pub env_mode: TexEnvMode,
    pub env_color: Rgba,
    pub coord_space: TexCoordSpace,
}

impl Default for TextureUnit {
    fn default() -> Self {
        TextureUnit {
            bound_texture: 0,
            env_mode: TexEnvMode::Modulate,
            env_color: [0.0, 0.0, 0.0, 0.0],
            coord_space: TexCoordSpace::Normalised,
        }
    }
}

// ---------------------------------------------------------------------
// Current vertex state
// ---------------------------------------------------------------------

/// `docs/internals/c3d.md`'s "Current vertex state" -- the value any
/// vertex component a draw command's format omits takes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurrentVertex {
    pub color: Rgba,
    pub normal: [f32; 3],
    /// Per texture unit.
    pub texcoord: Vec<(f32, f32)>,
    pub fogcoord: f32,
}

impl CurrentVertex {
    fn new(texture_units: usize) -> CurrentVertex {
        CurrentVertex {
            color: [1.0, 1.0, 1.0, 1.0],
            normal: [0.0, 0.0, 1.0],
            texcoord: vec![(0.0, 0.0); texture_units],
            fogcoord: 0.0,
        }
    }
}

// ---------------------------------------------------------------------
// Surfaces
// ---------------------------------------------------------------------

/// A defined render target (`docs/internals/c3d.md`'s "Surfaces"). No
/// pixel storage -- just the shape and backing location a device needs to
/// validate rectangles and locate memory; rasterisation and the depth
/// buffer are the board/GPU layer's concern.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Surface {
    pub width: u32,
    pub height: u32,
    pub stride_bytes: u32,
    pub format: SurfaceFormat,
    pub backing: Backing,
}

// ---------------------------------------------------------------------
// The context state machine
// ---------------------------------------------------------------------

/// One context's full GL 1.x state: one independent state machine per the
/// spec's "one context = one command ring = one independent GL state
/// machine".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct State {
    #[serde(skip, default = "Limits::default")]
    limits: Limits,

    pub raster: RasterState,
    pub enables: Enables,

    matrix_mode: MatrixMode,
    modelview: MatrixStack,
    projection: MatrixStack,
    /// One stack per texture unit, selected by `active_unit` when
    /// `matrix_mode` is `Texture`.
    texture: Vec<MatrixStack>,

    pub lights: Lights,
    pub light_model: LightModel,
    pub material_front: Material,
    pub material_back: Material,
    pub color_material: ColorMaterial,
    pub clip_planes: ClipPlanes,
    /// Per texture unit.
    pub texgen: Vec<TexGenUnit>,

    textures: BTreeMap<u32, TextureObject>,
    pub texture_units: Vec<TextureUnit>,
    active_unit: u32,

    pub current: CurrentVertex,

    surfaces: BTreeMap<u32, Surface>,
    draw_surface: u32,

    pending_gl_error: Option<GlError>,
}

impl State {
    pub fn new(limits: Limits) -> State {
        let texture_units = limits.texture_units.max(1);
        State {
            limits,
            raster: RasterState::default(),
            enables: Enables::new(texture_units, limits.max_lights, limits.max_clip_planes),
            matrix_mode: MatrixMode::Modelview,
            modelview: MatrixStack::new(limits.modelview_depth),
            projection: MatrixStack::new(limits.projection_depth),
            texture: (0..texture_units)
                .map(|_| MatrixStack::new(limits.texture_depth))
                .collect(),
            lights: Lights::new(limits.max_lights),
            light_model: LightModel::default(),
            material_front: Material::default(),
            material_back: Material::default(),
            color_material: ColorMaterial::default(),
            clip_planes: ClipPlanes::new(limits.max_clip_planes),
            texgen: (0..texture_units).map(|_| TexGenUnit::default()).collect(),
            textures: BTreeMap::new(),
            texture_units: (0..texture_units).map(|_| TextureUnit::default()).collect(),
            active_unit: 0,
            current: CurrentVertex::new(texture_units),
            surfaces: BTreeMap::new(),
            draw_surface: 0,
            pending_gl_error: None,
        }
    }

    pub fn limits(&self) -> Limits {
        self.limits
    }

    // -- GL errors --------------------------------------------------

    /// Record `err` unless an error is already pending: GL's
    /// first-error-wins semantics. A caller (`ring.rs`) calls this both
    /// for errors it detects itself (an unknown enumerant, i.e. a `None`
    /// from a `from_gl`/`from_wire` call) and relies on the `Result`
    /// return of methods such as [`State::push_matrix`] for errors this
    /// module detects.
    pub fn raise_gl_error(&mut self, err: GlError) {
        if self.pending_gl_error.is_none() {
            self.pending_gl_error = Some(err);
        }
    }

    /// The context's `GL_ERROR` register value: the pending error's GL
    /// enumerant, or `GL_NO_ERROR`.
    pub fn gl_error(&self) -> u32 {
        self.pending_gl_error.map_or(GL_NO_ERROR, GlError::to_gl)
    }

    /// `GL_ERROR_ACK`: clears the pending error.
    pub fn ack_gl_error(&mut self) {
        self.pending_gl_error = None;
    }

    // -- Raster state (0x02xx) --------------------------------------

    pub fn enable(&mut self, cap: Capability) {
        self.set_capability(cap, true);
    }

    pub fn disable(&mut self, cap: Capability) {
        self.set_capability(cap, false);
    }

    pub fn is_enabled(&self, cap: Capability) -> bool {
        match cap {
            Capability::AlphaTest => self.enables.alpha_test,
            Capability::Blend => self.enables.blend,
            Capability::CullFace => self.enables.cull_face,
            Capability::DepthTest => self.enables.depth_test,
            Capability::Dither => self.enables.dither,
            Capability::Fog => self.enables.fog,
            Capability::PolygonOffsetFill => self.enables.polygon_offset_fill,
            Capability::ScissorTest => self.enables.scissor_test,
            Capability::Lighting => self.enables.lighting,
            Capability::ColorMaterial => self.enables.color_material,
            Capability::Normalize => self.enables.normalize,
            Capability::RescaleNormal => self.enables.rescale_normal,
            Capability::Light(id) => self.enables.light.get(id.index()).copied().unwrap_or(false),
            Capability::ClipPlane(id) => self
                .enables
                .clip_plane
                .get(id.index())
                .copied()
                .unwrap_or(false),
            Capability::Texture2D => self.per_unit_flag(&self.enables.texture_2d),
            Capability::TextureGenS => self.per_unit_flag(&self.enables.texture_gen_s),
            Capability::TextureGenT => self.per_unit_flag(&self.enables.texture_gen_t),
        }
    }

    fn per_unit_flag(&self, flags: &[bool]) -> bool {
        flags
            .get(self.active_unit as usize)
            .copied()
            .unwrap_or(false)
    }

    fn set_capability(&mut self, cap: Capability, value: bool) {
        let unit = self.active_unit as usize;
        match cap {
            Capability::AlphaTest => self.enables.alpha_test = value,
            Capability::Blend => self.enables.blend = value,
            Capability::CullFace => self.enables.cull_face = value,
            Capability::DepthTest => self.enables.depth_test = value,
            Capability::Dither => self.enables.dither = value,
            Capability::Fog => self.enables.fog = value,
            Capability::PolygonOffsetFill => self.enables.polygon_offset_fill = value,
            Capability::ScissorTest => self.enables.scissor_test = value,
            Capability::Lighting => self.enables.lighting = value,
            Capability::ColorMaterial => self.enables.color_material = value,
            Capability::Normalize => self.enables.normalize = value,
            Capability::RescaleNormal => self.enables.rescale_normal = value,
            Capability::Light(id) => {
                if let Some(f) = self.enables.light.get_mut(id.index()) {
                    *f = value;
                }
            }
            Capability::ClipPlane(id) => {
                if let Some(f) = self.enables.clip_plane.get_mut(id.index()) {
                    *f = value;
                }
            }
            Capability::Texture2D => {
                if let Some(f) = self.enables.texture_2d.get_mut(unit) {
                    *f = value;
                }
            }
            Capability::TextureGenS => {
                if let Some(f) = self.enables.texture_gen_s.get_mut(unit) {
                    *f = value;
                }
            }
            Capability::TextureGenT => {
                if let Some(f) = self.enables.texture_gen_t.get_mut(unit) {
                    *f = value;
                }
            }
        }
    }

    pub fn set_blend_func(&mut self, sfactor: BlendFactor, dfactor: BlendFactor) {
        self.raster.blend_src_rgb = sfactor;
        self.raster.blend_dst_rgb = dfactor;
        self.raster.blend_src_alpha = sfactor;
        self.raster.blend_dst_alpha = dfactor;
    }

    pub fn set_blend_func_separate(
        &mut self,
        src_rgb: BlendFactor,
        dst_rgb: BlendFactor,
        src_alpha: BlendFactor,
        dst_alpha: BlendFactor,
    ) {
        self.raster.blend_src_rgb = src_rgb;
        self.raster.blend_dst_rgb = dst_rgb;
        self.raster.blend_src_alpha = src_alpha;
        self.raster.blend_dst_alpha = dst_alpha;
    }

    pub fn set_blend_equation(&mut self, mode: BlendEquation) {
        self.raster.blend_equation = mode;
    }

    pub fn set_depth_func(&mut self, func: DepthFunc) {
        self.raster.depth_func = func;
    }

    pub fn set_depth_mask(&mut self, flag: bool) {
        self.raster.depth_mask = flag;
    }

    pub fn set_depth_range(&mut self, near: f32, far: f32) {
        self.raster.depth_range = (near, far);
    }

    pub fn set_alpha_func(&mut self, func: AlphaFunc, reference: f32) {
        self.raster.alpha_func = func;
        self.raster.alpha_ref = reference;
    }

    pub fn set_cull_face(&mut self, mode: Face) {
        self.raster.cull_face_mode = mode;
    }

    pub fn set_front_face(&mut self, mode: FrontFace) {
        self.raster.front_face = mode;
    }

    pub fn set_shade_model(&mut self, mode: ShadeModel) {
        self.raster.shade_model = mode;
    }

    pub fn set_color_mask(&mut self, r: bool, g: bool, b: bool, a: bool) {
        self.raster.color_mask = ColorMask { r, g, b, a };
    }

    pub fn set_scissor(&mut self, x: i32, y: i32, w: u32, h: u32) {
        self.raster.scissor = Rect { x, y, w, h };
    }

    pub fn set_viewport(&mut self, x: i32, y: i32, w: u32, h: u32) {
        self.raster.viewport = Rect { x, y, w, h };
    }

    pub fn set_polygon_offset(&mut self, factor: f32, units: f32) {
        self.raster.polygon_offset_factor = factor;
        self.raster.polygon_offset_units = units;
    }

    pub fn set_clear_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.raster.clear_color = [r, g, b, a];
    }

    pub fn set_clear_depth(&mut self, depth: f32) {
        self.raster.clear_depth = depth;
    }

    pub fn set_fog_mode(&mut self, mode: FogMode) {
        self.raster.fog_mode = mode;
    }

    pub fn set_fog_params(&mut self, density: f32, start: f32, end: f32) {
        self.raster.fog_density = density;
        self.raster.fog_start = start;
        self.raster.fog_end = end;
    }

    pub fn set_fog_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.raster.fog_color = [r, g, b, a];
    }

    pub fn set_line_width(&mut self, width: f32) {
        self.raster.line_width = width;
    }

    pub fn set_point_size(&mut self, size: f32) {
        self.raster.point_size = size;
    }

    pub fn set_polygon_mode(&mut self, face: Face, mode: PolygonMode) {
        match face {
            Face::Front => self.raster.polygon_mode_front = mode,
            Face::Back => self.raster.polygon_mode_back = mode,
            Face::FrontAndBack => {
                self.raster.polygon_mode_front = mode;
                self.raster.polygon_mode_back = mode;
            }
        }
    }

    // -- Matrix (0x03xx) ---------------------------------------------

    pub fn set_matrix_mode(&mut self, mode: MatrixMode) {
        self.matrix_mode = mode;
    }

    pub fn matrix_mode(&self) -> MatrixMode {
        self.matrix_mode
    }

    fn stack_mut(&mut self) -> Option<&mut MatrixStack> {
        match self.matrix_mode {
            MatrixMode::Modelview => Some(&mut self.modelview),
            MatrixMode::Projection => Some(&mut self.projection),
            MatrixMode::Texture => self.texture.get_mut(self.active_unit as usize),
        }
    }

    fn stack(&self, mode: MatrixMode) -> Option<&MatrixStack> {
        match mode {
            MatrixMode::Modelview => Some(&self.modelview),
            MatrixMode::Projection => Some(&self.projection),
            MatrixMode::Texture => self.texture.get(self.active_unit as usize),
        }
    }

    /// The top of `unit`'s texture matrix stack, regardless of which unit
    /// `ACTIVE_UNIT` currently selects -- [`MatrixMode::Texture`] via
    /// [`Self::top_matrix`] resolves through the *active* unit (the
    /// command-stream view), but the renderer needs every unit's matrix
    /// at draw time whatever unit happens to be active. `None` for a
    /// unit past the configured limit.
    pub fn texture_matrix(&self, unit: usize) -> Option<Mat4> {
        self.texture.get(unit).map(|s| *s.top())
    }

    /// The top of the current-mode matrix stack, i.e. what `QUERY` would
    /// read for `MODELVIEW_MATRIX`/`PROJECTION_MATRIX`/`TEXTURE_MATRIX`.
    pub fn top_matrix(&self, mode: MatrixMode) -> Option<Mat4> {
        self.stack(mode).map(|s| *s.top())
    }

    /// The modelview matrix in effect right now -- what `LIGHT`'s
    /// `POSITION`/`SPOT_DIRECTION` and `CLIP_PLANE`/`TEXGEN_PLANE`'s eye
    /// plane transform by, regardless of the current `MATRIX_MODE`.
    fn current_modelview(&self) -> Mat4 {
        *self.modelview.top()
    }

    pub fn load_matrix(&mut self, m: [f32; 16]) {
        if let Some(s) = self.stack_mut() {
            s.load(Mat4(m));
        }
    }

    pub fn load_identity(&mut self) {
        if let Some(s) = self.stack_mut() {
            s.load_identity();
        }
    }

    pub fn mult_matrix(&mut self, m: [f32; 16]) {
        if let Some(s) = self.stack_mut() {
            s.mult(&Mat4(m));
        }
    }

    pub fn push_matrix(&mut self) -> Result<(), GlError> {
        match self.stack_mut() {
            Some(s) => {
                let r = s.push();
                if let Err(e) = r {
                    self.raise_gl_error(e);
                }
                r
            }
            None => Ok(()),
        }
    }

    pub fn pop_matrix(&mut self) -> Result<(), GlError> {
        match self.stack_mut() {
            Some(s) => {
                let r = s.pop();
                if let Err(e) = r {
                    self.raise_gl_error(e);
                }
                r
            }
            None => Ok(()),
        }
    }

    pub fn translate(&mut self, x: f32, y: f32, z: f32) {
        if let Some(s) = self.stack_mut() {
            s.mult(&Mat4::translation(x, y, z));
        }
    }

    pub fn rotate(&mut self, angle_deg: f32, x: f32, y: f32, z: f32) {
        if let Some(s) = self.stack_mut() {
            s.mult(&Mat4::rotation(angle_deg, x, y, z));
        }
    }

    pub fn scale(&mut self, x: f32, y: f32, z: f32) {
        if let Some(s) = self.stack_mut() {
            s.mult(&Mat4::scaling(x, y, z));
        }
    }

    pub fn frustum(&mut self, l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) {
        if let Some(s) = self.stack_mut() {
            s.mult(&Mat4::frustum(l, r, b, t, n, f));
        }
    }

    pub fn ortho(&mut self, l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) {
        if let Some(s) = self.stack_mut() {
            s.mult(&Mat4::ortho(l, r, b, t, n, f));
        }
    }

    // -- Lighting, clipping and texgen (0x04xx) -----------------------

    pub fn set_light(&mut self, light: LightId, param: LightParam, v: [f32; 4]) {
        let modelview = self.current_modelview();
        // A no-op past the board's configured `MAX_LIGHTS`; `E_LIMIT` for
        // that is `ring.rs`'s job, mirroring out-of-range texture units.
        let Some(l) = self.lights.0.get_mut(light.index()) else {
            return;
        };
        match param {
            LightParam::Ambient => l.ambient = v,
            LightParam::Diffuse => l.diffuse = v,
            LightParam::Specular => l.specular = v,
            LightParam::Position => l.position = modelview.transform_point(v),
            LightParam::SpotDirection => {
                let dir3 = [v[0], v[1], v[2]];
                l.spot_direction = modelview.transform_direction3(dir3);
            }
            LightParam::SpotExponent => l.spot_exponent = v[0],
            LightParam::SpotCutoff => l.spot_cutoff = v[0],
            LightParam::ConstantAttenuation => l.constant_attenuation = v[0],
            LightParam::LinearAttenuation => l.linear_attenuation = v[0],
            LightParam::QuadraticAttenuation => l.quadratic_attenuation = v[0],
        }
    }

    pub fn set_light_model(&mut self, param: LightModelParam, v: [f32; 4]) {
        match param {
            LightModelParam::Ambient => self.light_model.ambient = v,
            LightModelParam::LocalViewer => self.light_model.local_viewer = v[0] != 0.0,
            LightModelParam::TwoSide => self.light_model.two_side = v[0] != 0.0,
        }
    }

    fn set_material_one(m: &mut Material, param: MaterialParam, v: [f32; 4]) {
        match param {
            MaterialParam::Ambient => m.ambient = v,
            MaterialParam::Diffuse => m.diffuse = v,
            MaterialParam::Specular => m.specular = v,
            MaterialParam::Emission => m.emission = v,
            MaterialParam::Shininess => m.shininess = v[0],
            MaterialParam::AmbientAndDiffuse => {
                m.ambient = v;
                m.diffuse = v;
            }
        }
    }

    pub fn set_material(&mut self, face: Face, param: MaterialParam, v: [f32; 4]) {
        match face {
            Face::Front => Self::set_material_one(&mut self.material_front, param, v),
            Face::Back => Self::set_material_one(&mut self.material_back, param, v),
            Face::FrontAndBack => {
                Self::set_material_one(&mut self.material_front, param, v);
                Self::set_material_one(&mut self.material_back, param, v);
            }
        }
    }

    pub fn set_color_material(&mut self, face: Face, mode: ColorMaterialMode) {
        self.color_material = ColorMaterial { face, mode };
    }

    /// A row-vector-by-matrix transform of a plane equation by `m`'s
    /// inverse: `plane * m^-1`, per GL's rule for transforming
    /// `CLIP_PLANE`/`TEXGEN_PLANE(EYE_PLANE)` equations by "the inverse
    /// modelview at the time of the command". A singular modelview (GL
    /// leaves this undefined) falls back to the identity rather than
    /// panicking or propagating a `NaN`-laden plane.
    fn eye_transform(m: &Mat4, plane: [f32; 4]) -> [f32; 4] {
        let inv = m.inverse().unwrap_or_else(Mat4::identity);
        let mut out = [0.0f32; 4];
        for col in 0..4 {
            let mut sum = 0.0f32;
            for row in 0..4 {
                sum += plane[row] * inv.get(row, col);
            }
            out[col] = sum;
        }
        out
    }

    pub fn set_clip_plane(&mut self, plane: ClipPlaneId, eq: [f32; 4]) {
        let transformed = Self::eye_transform(&self.current_modelview(), eq);
        // A no-op past the board's configured `MAX_CLIP_PLANES`; see
        // `set_light`'s note.
        if let Some(slot) = self.clip_planes.0.get_mut(plane.index()) {
            *slot = transformed;
        }
    }

    pub fn set_texgen_mode(&mut self, unit: u32, coord: TexCoord, mode: TexGenMode) {
        if let Some(u) = self.texgen.get_mut(unit as usize) {
            match coord {
                TexCoord::S => u.mode_s = mode,
                TexCoord::T => u.mode_t = mode,
            }
        }
    }

    pub fn set_texgen_plane(
        &mut self,
        unit: u32,
        coord: TexCoord,
        plane: TexGenPlaneKind,
        eq: [f32; 4],
    ) {
        let eq = match plane {
            TexGenPlaneKind::ObjectPlane => eq,
            TexGenPlaneKind::EyePlane => Self::eye_transform(&self.current_modelview(), eq),
        };
        if let Some(u) = self.texgen.get_mut(unit as usize) {
            match (coord, plane) {
                (TexCoord::S, TexGenPlaneKind::ObjectPlane) => u.object_plane_s = eq,
                (TexCoord::T, TexGenPlaneKind::ObjectPlane) => u.object_plane_t = eq,
                (TexCoord::S, TexGenPlaneKind::EyePlane) => u.eye_plane_s = eq,
                (TexCoord::T, TexGenPlaneKind::EyePlane) => u.eye_plane_t = eq,
            }
        }
    }

    // -- Textures (0x05xx) --------------------------------------------

    /// `TEX_CREATE`: creates an empty object with default parameters, or
    /// resets an existing one to defaults (per the spec, "Creating an
    /// existing ID resets it") without touching who has it bound.
    pub fn tex_create(&mut self, id: u32) {
        self.textures.insert(id, TextureObject::default());
    }

    /// `TEX_DESTROY`: releases the object and unbinds it from every unit.
    pub fn tex_destroy(&mut self, id: u32) {
        self.textures.remove(&id);
        for unit in &mut self.texture_units {
            if unit.bound_texture == id {
                unit.bound_texture = 0;
            }
        }
    }

    pub fn texture(&self, id: u32) -> Option<&TextureObject> {
        self.textures.get(&id)
    }

    /// `TEX_BIND`. A no-op if `unit` is out of range or `id` is not a
    /// created object -- both are protocol-level (`E_LIMIT`/`E_BAD_ID`)
    /// concerns the ring decoder is expected to have already rejected;
    /// this is a defensive fallback, not a second error path.
    pub fn tex_bind(&mut self, unit: u32, id: u32) {
        if id != 0 && !self.textures.contains_key(&id) {
            return;
        }
        if let Some(u) = self.texture_units.get_mut(unit as usize) {
            u.bound_texture = id;
        }
    }

    /// `TEX_PARAM`. `Err(GL_INVALID_OPERATION)` if `id` was never
    /// created (or has been destroyed).
    pub fn set_tex_param(&mut self, id: u32, param: TexParam) -> Result<(), GlError> {
        let Some(tex) = self.textures.get_mut(&id) else {
            self.raise_gl_error(GlError::InvalidOperation);
            return Err(GlError::InvalidOperation);
        };
        match param {
            TexParam::MinFilter(f) => tex.min_filter = f,
            TexParam::MagFilter(f) => tex.mag_filter = f,
            TexParam::WrapS(w) => tex.wrap_s = w,
            TexParam::WrapT(w) => tex.wrap_t = w,
        }
        Ok(())
    }

    /// Records `TEX_IMAGE`/`TEX_COPY_IMAGE`'s resulting level shape (the
    /// pixels themselves are the board/GPU layer's concern).
    pub fn define_tex_level(
        &mut self,
        id: u32,
        level: u32,
        format: TexFormat,
        width: u32,
        height: u32,
    ) -> Result<(), GlError> {
        let Some(tex) = self.textures.get_mut(&id) else {
            self.raise_gl_error(GlError::InvalidOperation);
            return Err(GlError::InvalidOperation);
        };
        tex.levels.insert(
            level,
            TexLevel {
                width,
                height,
                format,
            },
        );
        Ok(())
    }

    /// `TEX_PALETTE`. `entries == 0` unbinds, per the spec.
    pub fn set_tex_palette(&mut self, id: u32, entries: u32) -> Result<(), GlError> {
        let Some(tex) = self.textures.get_mut(&id) else {
            self.raise_gl_error(GlError::InvalidOperation);
            return Err(GlError::InvalidOperation);
        };
        tex.palette_entries = if entries == 0 { None } else { Some(entries) };
        Ok(())
    }

    pub fn set_tex_env(&mut self, unit: u32, param: TexEnvParam) {
        if let Some(u) = self.texture_units.get_mut(unit as usize) {
            match param {
                TexEnvParam::Mode(m) => u.env_mode = m,
                TexEnvParam::CoordSpace(s) => u.coord_space = s,
            }
        }
    }

    pub fn set_tex_env_color(&mut self, unit: u32, r: f32, g: f32, b: f32, a: f32) {
        if let Some(u) = self.texture_units.get_mut(unit as usize) {
            u.env_color = [r, g, b, a];
        }
    }

    pub fn set_active_unit(&mut self, unit: u32) {
        self.active_unit = unit;
    }

    pub fn active_unit(&self) -> u32 {
        self.active_unit
    }

    // -- Current vertex state (0x06xx) ---------------------------------

    pub fn set_current_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.current.color = [r, g, b, a];
    }

    pub fn set_current_normal(&mut self, x: f32, y: f32, z: f32) {
        self.current.normal = [x, y, z];
    }

    pub fn set_current_texcoord(&mut self, unit: u32, s: f32, t: f32) {
        if let Some(tc) = self.current.texcoord.get_mut(unit as usize) {
            *tc = (s, t);
        }
    }

    pub fn set_current_fogcoord(&mut self, f: f32) {
        self.current.fogcoord = f;
    }

    // -- Surfaces (0x01xx) ----------------------------------------------

    /// `SURFACE_DEFINE`: creates or redefines `id`.
    pub fn surface_define(&mut self, id: u32, surface: Surface) {
        self.surfaces.insert(id, surface);
    }

    /// `SURFACE_DESTROY`. If `id` was the draw surface, the draw surface
    /// becomes `0` (subsequent draws are `E_NO_SURFACE`, per the spec --
    /// enforced by the ring/board layer, not this module).
    pub fn surface_destroy(&mut self, id: u32) {
        self.surfaces.remove(&id);
        if self.draw_surface == id {
            self.draw_surface = 0;
        }
    }

    pub fn surface(&self, id: u32) -> Option<&Surface> {
        self.surfaces.get(&id)
    }

    /// `SET_DRAW_SURFACE`. When `id` names a defined surface, the scissor
    /// and viewport reset to the full surface, per the spec's initial
    /// raster state ("scissor and viewport the full draw surface at
    /// `SET_DRAW_SURFACE` time").
    pub fn set_draw_surface(&mut self, id: u32) {
        self.draw_surface = id;
        if let Some(s) = self.surfaces.get(&id) {
            let full = Rect {
                x: 0,
                y: 0,
                w: s.width,
                h: s.height,
            };
            self.raster.scissor = full;
            self.raster.viewport = full;
        }
    }

    pub fn draw_surface(&self) -> u32 {
        self.draw_surface
    }

    // -- Protocol errors this module is in a position to detect ---------
    //
    // Everything else in `docs/internals/c3d.md`'s "Protocol errors"
    // table is `ring.rs`'s job (framing, refs, opcodes, IDs). These two
    // depend on live context state -- the current draw surface's
    // existence and shape -- that only this module has, so `ring.rs`
    // calls back into these rather than duplicating that state. They
    // return `proto::ErrorCode` directly, deliberately not through
    // `GlError`/`raise_gl_error`: protocol and GL errors are two separate
    // latches with two separate acknowledgement registers (`ERROR_ACK`
    // vs `GL_ERROR_ACK`), and `ring.rs` owns the offset that goes with
    // `ERROR_CODE`/`ERROR_OFFSET`, which this module never sees.

    /// `E_NO_SURFACE`: whether a draw, `CLEAR`, `SURFACE_READBACK` or
    /// `SURFACE_UPLOAD` may proceed. Returns the bound surface on success
    /// so a caller that also needs its shape (for [`State::check_rect`])
    /// doesn't look it up twice.
    pub fn require_draw_surface(&self) -> Result<&Surface, ErrorCode> {
        self.surfaces
            .get(&self.draw_surface)
            .ok_or(ErrorCode::NoSurface)
    }

    /// `E_BAD_RECT`: whether rectangle `(x, y, w, h)` lies wholly inside a
    /// surface of `width` x `height` pixels. A negative origin, or a
    /// rectangle whose far edge overflows or exceeds the surface, is
    /// rejected; an empty rectangle (`w == 0` or `h == 0`) at an in-bounds
    /// origin is not an error -- it trivially lies within the surface,
    /// same as GL's own empty-rectangle calls. Deliberately not used for
    /// `SCISSOR`/`VIEWPORT`: GL clamps those rather than erroring.
    pub fn check_rect(
        width: u32,
        height: u32,
        x: i32,
        y: i32,
        w: u32,
        h: u32,
    ) -> Result<(), ErrorCode> {
        let (Ok(x), Ok(y)) = (u32::try_from(x), u32::try_from(y)) else {
            return Err(ErrorCode::BadRect);
        };
        match (x.checked_add(w), y.checked_add(h)) {
            (Some(right), Some(bottom)) if right <= width && bottom <= height => Ok(()),
            _ => Err(ErrorCode::BadRect),
        }
    }

    /// [`State::require_draw_surface`] followed by [`State::check_rect`]
    /// against it -- the combined check `SURFACE_UPLOAD`,
    /// `SURFACE_READBACK` and `READ_PIXELS` need.
    pub fn check_draw_surface_rect(&self, x: i32, y: i32, w: u32, h: u32) -> Result<(), ErrorCode> {
        let surface = self.require_draw_surface()?;
        Self::check_rect(surface.width, surface.height, x, y, w, h)
    }

    // -- Reset ------------------------------------------------------

    /// `CTX_RESET_STATE` (opcode `0x0004`): "return this context's GL
    /// state (not its objects, not its ring) to initial values". Texture
    /// objects, surfaces and the pending GL error are left untouched.
    /// Leaving `GL_ERROR` alone is deliberate and is what the spec
    /// requires of this opcode: it is a command in the stream, so
    /// discarding an error the guest has not yet read would lose it.
    /// `CTX_CONTROL.RESET` is the one that clears it -- see
    /// [`State::reset_all`].
    pub fn reset_state(&mut self) {
        let texture_units = self.limits.texture_units.max(1);
        self.raster = RasterState::default();
        self.enables = Enables::new(
            texture_units,
            self.limits.max_lights,
            self.limits.max_clip_planes,
        );
        self.matrix_mode = MatrixMode::Modelview;
        self.modelview = MatrixStack::new(self.limits.modelview_depth);
        self.projection = MatrixStack::new(self.limits.projection_depth);
        self.texture = (0..texture_units)
            .map(|_| MatrixStack::new(self.limits.texture_depth))
            .collect();
        self.lights = Lights::new(self.limits.max_lights);
        self.light_model = LightModel::default();
        self.material_front = Material::default();
        self.material_back = Material::default();
        self.color_material = ColorMaterial::default();
        self.clip_planes = ClipPlanes::new(self.limits.max_clip_planes);
        self.texgen = (0..texture_units).map(|_| TexGenUnit::default()).collect();
        self.texture_units = (0..texture_units).map(|_| TextureUnit::default()).collect();
        self.active_unit = 0;
        self.current = CurrentVertex::new(texture_units);
        self.draw_surface = 0;
    }

    /// `CTX_CONTROL.RESET` (the register bit, not the ring opcode): GL
    /// state plus every texture and surface object for this context, and
    /// the pending GL error with them. Clearing `GL_ERROR` here is the
    /// difference from [`State::reset_state`]: this is the guest asking
    /// for a clean slate through a register write, and an error surviving
    /// it would outlive the state that produced it. The protocol-error
    /// latch is cleared alongside it by the caller, which owns that
    /// latch.
    pub fn reset_all(&mut self) {
        self.reset_state();
        self.textures.clear();
        self.surfaces.clear();
        self.pending_gl_error = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> State {
        State::new(Limits::default())
    }

    // -- Initial state, field by field against the spec --------------

    #[test]
    fn initial_state_disables_everything_except_dither() {
        let s = state();
        assert!(!s.enables.alpha_test);
        assert!(!s.enables.blend);
        assert!(!s.enables.cull_face);
        assert!(!s.enables.depth_test);
        assert!(s.enables.dither);
        assert!(!s.enables.fog);
        assert!(!s.enables.polygon_offset_fill);
        assert!(!s.enables.scissor_test);
        assert!(!s.enables.lighting);
        assert!(!s.enables.color_material);
        assert!(!s.enables.normalize);
        assert!(!s.enables.rescale_normal);
        assert!(s.enables.light.iter().all(|&e| !e));
        assert!(s.enables.clip_plane.iter().all(|&e| !e));
        assert!(s.enables.texture_2d.iter().all(|&e| !e));
        assert!(s.enables.texture_gen_s.iter().all(|&e| !e));
        assert!(s.enables.texture_gen_t.iter().all(|&e| !e));
    }

    #[test]
    fn initial_state_matches_the_spec_raster_defaults() {
        let s = state();
        assert_eq!(s.raster.blend_src_rgb, BlendFactor::One);
        assert_eq!(s.raster.blend_dst_rgb, BlendFactor::Zero);
        assert_eq!(s.raster.blend_src_alpha, BlendFactor::One);
        assert_eq!(s.raster.blend_dst_alpha, BlendFactor::Zero);
        assert_eq!(s.raster.depth_func, DepthFunc::Less);
        assert!(s.raster.depth_mask);
        assert_eq!(s.raster.depth_range, (0.0, 1.0));
        assert_eq!(s.raster.alpha_func, AlphaFunc::Always);
        assert_eq!(s.raster.alpha_ref, 0.0);
        assert_eq!(s.raster.cull_face_mode, Face::Back);
        assert_eq!(s.raster.front_face, FrontFace::Ccw);
        assert_eq!(s.raster.shade_model, ShadeModel::Smooth);
        assert_eq!(
            s.raster.color_mask,
            ColorMask {
                r: true,
                g: true,
                b: true,
                a: true
            }
        );
        assert_eq!(s.raster.clear_color, [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(s.raster.clear_depth, 1.0);
        assert_eq!(s.raster.fog_mode, FogMode::Exp);
        assert_eq!(s.raster.fog_density, 1.0);
        assert_eq!(s.raster.fog_start, 0.0);
        assert_eq!(s.raster.fog_end, 1.0);
        assert_eq!(s.raster.fog_color, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn initial_scissor_and_viewport_are_zero_before_any_draw_surface_is_set() {
        let s = state();
        assert_eq!(s.raster.scissor, Rect::ZERO);
        assert_eq!(s.raster.viewport, Rect::ZERO);
    }

    #[test]
    fn set_draw_surface_resets_scissor_and_viewport_to_the_full_surface() {
        let mut s = state();
        s.set_scissor(1, 2, 3, 4);
        s.set_viewport(1, 2, 3, 4);
        s.surface_define(
            1,
            Surface {
                width: 640,
                height: 480,
                stride_bytes: 1280,
                format: SurfaceFormat::R5g6b5,
                backing: Backing::Aperture(0),
            },
        );
        s.set_draw_surface(1);
        assert_eq!(
            s.raster.scissor,
            Rect {
                x: 0,
                y: 0,
                w: 640,
                h: 480
            }
        );
        assert_eq!(
            s.raster.viewport,
            Rect {
                x: 0,
                y: 0,
                w: 640,
                h: 480
            }
        );
    }

    #[test]
    fn surface_destroy_of_the_draw_surface_clears_the_draw_surface_to_zero() {
        let mut s = state();
        s.surface_define(
            5,
            Surface {
                width: 4,
                height: 4,
                stride_bytes: 8,
                format: SurfaceFormat::R5g6b5,
                backing: Backing::Aperture(0),
            },
        );
        s.set_draw_surface(5);
        assert_eq!(s.draw_surface(), 5);
        s.surface_destroy(5);
        assert_eq!(s.draw_surface(), 0);
    }

    #[test]
    fn initial_current_vertex_state_matches_the_spec() {
        let s = state();
        assert_eq!(s.current.color, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(s.current.normal, [0.0, 0.0, 1.0]);
        assert!(s.current.texcoord.iter().all(|&tc| tc == (0.0, 0.0)));
        assert_eq!(s.current.fogcoord, 0.0);
    }

    #[test]
    fn initial_texture_object_defaults_match_gl() {
        let mut s = state();
        s.tex_create(1);
        let tex = s.texture(1).unwrap();
        assert_eq!(tex.min_filter, TexFilter::NearestMipmapLinear);
        assert_eq!(tex.mag_filter, TexFilter::Linear);
        assert_eq!(tex.wrap_s, TexWrap::Repeat);
        assert_eq!(tex.wrap_t, TexWrap::Repeat);
        assert_eq!(tex.palette_entries, None);
    }

    #[test]
    fn initial_texture_unit_state_matches_gl() {
        let s = state();
        for u in &s.texture_units {
            assert_eq!(u.bound_texture, 0);
            assert_eq!(u.env_mode, TexEnvMode::Modulate);
            assert_eq!(u.env_color, [0.0, 0.0, 0.0, 0.0]);
            assert_eq!(u.coord_space, TexCoordSpace::Normalised);
        }
    }

    #[test]
    fn initial_matrices_are_identity_on_every_stack() {
        let s = state();
        assert_eq!(s.top_matrix(MatrixMode::Modelview), Some(Mat4::identity()));
        assert_eq!(s.top_matrix(MatrixMode::Projection), Some(Mat4::identity()));
        assert_eq!(s.top_matrix(MatrixMode::Texture), Some(Mat4::identity()));
    }

    #[test]
    fn initial_matrix_mode_is_modelview() {
        let s = state();
        assert_eq!(s.matrix_mode(), MatrixMode::Modelview);
    }

    #[test]
    fn initial_light0_is_white_and_other_lights_are_black() {
        let s = state();
        assert_eq!(s.lights.0[0].diffuse, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(s.lights.0[0].specular, [1.0, 1.0, 1.0, 1.0]);
        for i in 1..8 {
            assert_eq!(s.lights.0[i].diffuse, [0.0, 0.0, 0.0, 0.0]);
            assert_eq!(s.lights.0[i].specular, [0.0, 0.0, 0.0, 0.0]);
        }
        for l in &s.lights.0 {
            assert_eq!(l.ambient, [0.0, 0.0, 0.0, 1.0]);
            assert_eq!(l.position, [0.0, 0.0, 1.0, 0.0]);
            assert_eq!(l.spot_direction, [0.0, 0.0, -1.0]);
            assert_eq!(l.spot_exponent, 0.0);
            assert_eq!(l.spot_cutoff, 180.0);
            assert_eq!(l.constant_attenuation, 1.0);
            assert_eq!(l.linear_attenuation, 0.0);
            assert_eq!(l.quadratic_attenuation, 0.0);
        }
    }

    #[test]
    fn initial_light_model_matches_gl() {
        let s = state();
        assert_eq!(s.light_model.ambient, [0.2, 0.2, 0.2, 1.0]);
        assert!(!s.light_model.local_viewer);
        assert!(!s.light_model.two_side);
    }

    #[test]
    fn initial_material_matches_gl_on_both_faces() {
        let s = state();
        for m in [&s.material_front, &s.material_back] {
            assert_eq!(m.ambient, [0.2, 0.2, 0.2, 1.0]);
            assert_eq!(m.diffuse, [0.8, 0.8, 0.8, 1.0]);
            assert_eq!(m.specular, [0.0, 0.0, 0.0, 1.0]);
            assert_eq!(m.emission, [0.0, 0.0, 0.0, 1.0]);
            assert_eq!(m.shininess, 0.0);
        }
    }

    #[test]
    fn initial_color_material_is_front_and_back_ambient_and_diffuse() {
        let s = state();
        assert_eq!(s.color_material.face, Face::FrontAndBack);
        assert_eq!(s.color_material.mode, ColorMaterialMode::AmbientAndDiffuse);
    }

    #[test]
    fn initial_clip_planes_are_all_zero() {
        let s = state();
        assert!(s.clip_planes.0.iter().all(|p| *p == [0.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn initial_texgen_matches_gl_defaults_per_unit() {
        let s = state();
        for u in &s.texgen {
            assert_eq!(u.mode_s, TexGenMode::EyeLinear);
            assert_eq!(u.mode_t, TexGenMode::EyeLinear);
            assert_eq!(u.object_plane_s, [1.0, 0.0, 0.0, 0.0]);
            assert_eq!(u.object_plane_t, [0.0, 1.0, 0.0, 0.0]);
            assert_eq!(u.eye_plane_s, [1.0, 0.0, 0.0, 0.0]);
            assert_eq!(u.eye_plane_t, [0.0, 1.0, 0.0, 0.0]);
        }
    }

    // -- Matrix ops, hand-computed -------------------------------------

    #[test]
    fn translate_produces_the_gl_translation_matrix() {
        let mut s = state();
        s.translate(1.0, 2.0, 3.0);
        let m = s.top_matrix(MatrixMode::Modelview).unwrap();
        #[rustfmt::skip]
        let expected = Mat4([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            1.0, 2.0, 3.0, 1.0,
        ]);
        assert_eq!(m, expected);
    }

    #[test]
    fn scale_produces_the_gl_scale_matrix() {
        let mut s = state();
        s.scale(2.0, 3.0, 4.0);
        let m = s.top_matrix(MatrixMode::Modelview).unwrap();
        #[rustfmt::skip]
        let expected = Mat4([
            2.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 4.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]);
        assert_eq!(m, expected);
    }

    #[test]
    fn rotate_90_degrees_about_z_maps_x_to_y() {
        let mut s = state();
        s.rotate(90.0, 0.0, 0.0, 1.0);
        let m = s.top_matrix(MatrixMode::Modelview).unwrap();
        let p = m.transform_point([1.0, 0.0, 0.0, 1.0]);
        assert!((p[0]).abs() < 1e-5);
        assert!((p[1] - 1.0).abs() < 1e-5);
        assert!((p[2]).abs() < 1e-5);
    }

    #[test]
    fn rotate_by_a_zero_length_axis_is_a_no_op() {
        let mut s = state();
        s.rotate(45.0, 0.0, 0.0, 0.0);
        assert_eq!(s.top_matrix(MatrixMode::Modelview), Some(Mat4::identity()));
    }

    #[test]
    fn mult_matrix_post_multiplies() {
        let mut s = state();
        // Load a scale of 2 as the base, then MULT_MATRIX a translation.
        // Post-multiplying means the translation happens in the
        // pre-scale (object) frame, so transforming (1,0,0,1) by the
        // result scales *after* translating: (1+1)*2 = 4.
        s.load_matrix(Mat4::scaling(2.0, 2.0, 2.0).0);
        s.mult_matrix(Mat4::translation(1.0, 0.0, 0.0).0);
        let m = s.top_matrix(MatrixMode::Modelview).unwrap();
        let p = m.transform_point([1.0, 0.0, 0.0, 1.0]);
        assert_eq!(p[0], 4.0);
    }

    #[test]
    fn frustum_matches_the_gl_frustum_matrix() {
        let mut s = state();
        s.set_matrix_mode(MatrixMode::Projection);
        s.frustum(-1.0, 1.0, -1.0, 1.0, 1.0, 10.0);
        let m = s.top_matrix(MatrixMode::Projection).unwrap();
        assert_eq!(m.get(0, 0), 1.0); // 2n/(r-l) = 2/2
        assert_eq!(m.get(1, 1), 1.0); // 2n/(t-b) = 2/2
        assert!((m.get(2, 2) - (-11.0 / 9.0)).abs() < 1e-6); // -(f+n)/(f-n)
        assert_eq!(m.get(3, 2), -1.0);
        assert!((m.get(2, 3) - (-20.0 / 9.0)).abs() < 1e-6); // -2fn/(f-n)
    }

    #[test]
    fn ortho_matches_the_gl_ortho_matrix() {
        let mut s = state();
        s.set_matrix_mode(MatrixMode::Projection);
        s.ortho(-2.0, 2.0, -1.0, 1.0, 1.0, 5.0);
        let m = s.top_matrix(MatrixMode::Projection).unwrap();
        assert_eq!(m.get(0, 0), 0.5); // 2/(r-l) = 2/4
        assert_eq!(m.get(1, 1), 1.0); // 2/(t-b) = 2/2
        assert_eq!(m.get(2, 2), -0.5); // -2/(f-n) = -2/4
        assert_eq!(m.get(0, 3), 0.0); // -(r+l)/(r-l) = 0
        assert_eq!(m.get(1, 3), 0.0); // -(t+b)/(t-b) = 0
        assert!((m.get(2, 3) - (-1.5)).abs() < 1e-6); // -(f+n)/(f-n)
    }

    #[test]
    fn load_identity_resets_the_top_of_the_current_stack_only() {
        let mut s = state();
        s.push_matrix().unwrap();
        s.translate(5.0, 0.0, 0.0);
        s.load_identity();
        assert_eq!(s.top_matrix(MatrixMode::Modelview), Some(Mat4::identity()));
        s.pop_matrix().unwrap();
        // The matrix below is untouched by the top's load_identity.
        assert_eq!(s.top_matrix(MatrixMode::Modelview), Some(Mat4::identity()));
    }

    // -- Stack overflow/underflow ---------------------------------------

    #[test]
    fn push_matrix_overflows_at_the_reported_depth_and_leaves_the_stack_unchanged() {
        let mut s = State::new(Limits {
            modelview_depth: 3,
            ..Limits::default()
        });
        s.translate(1.0, 0.0, 0.0);
        s.push_matrix().unwrap();
        s.push_matrix().unwrap();
        assert_eq!(s.top_matrix(MatrixMode::Modelview).unwrap().0[12], 1.0);
        // Depth is now 3 (1 base + 2 pushes), at the configured limit.
        assert_eq!(s.push_matrix(), Err(GlError::StackOverflow));
        assert_eq!(s.gl_error(), GlError::StackOverflow.to_gl());
        // Unchanged: still at the same depth and top value.
        assert_eq!(s.top_matrix(MatrixMode::Modelview).unwrap().0[12], 1.0);
    }

    #[test]
    fn pop_matrix_underflows_at_depth_one_and_leaves_the_stack_unchanged() {
        let mut s = state();
        s.translate(7.0, 0.0, 0.0);
        assert_eq!(s.pop_matrix(), Err(GlError::StackUnderflow));
        assert_eq!(s.gl_error(), GlError::StackUnderflow.to_gl());
        // Unchanged: the sole matrix is still there with its value intact.
        assert_eq!(s.top_matrix(MatrixMode::Modelview).unwrap().0[12], 7.0);
    }

    #[test]
    fn projection_stack_overflows_past_its_own_smaller_default_depth() {
        let mut s = state(); // default projection_depth is 2
        s.set_matrix_mode(MatrixMode::Projection);
        s.push_matrix().unwrap(); // depth 2, at the limit
        assert_eq!(s.push_matrix(), Err(GlError::StackOverflow));
    }

    #[test]
    fn texture_stack_depth_covers_the_first_clients_ten_deep_stack() {
        // The driver (v27.2) carries a real 10-deep GL_TEXTURE stack; the
        // board's default must accept at least that many levels so a
        // guest library can map its stack 1:1. Default is 16: 15 pushes
        // (16 levels) succeed, the 16th push overflows.
        let mut s = state();
        s.set_matrix_mode(MatrixMode::Texture);
        for _ in 0..15 {
            s.push_matrix().unwrap();
        }
        assert_eq!(s.push_matrix(), Err(GlError::StackOverflow));
    }

    #[test]
    fn texture_matrix_reads_any_units_stack_regardless_of_active_unit() {
        let mut s = state();
        s.set_matrix_mode(MatrixMode::Texture);
        s.set_active_unit(0);
        s.translate(1.0, 0.0, 0.0);
        s.set_active_unit(1);
        s.translate(2.0, 0.0, 0.0);
        // Active unit is 1, but the renderer-facing accessor must still
        // see unit 0's own matrix.
        assert_eq!(s.texture_matrix(0).unwrap().0[12], 1.0);
        assert_eq!(s.texture_matrix(1).unwrap().0[12], 2.0);
        assert_eq!(s.texture_matrix(99), None);
    }

    #[test]
    fn each_texture_unit_has_its_own_independent_stack() {
        let mut s = state();
        s.set_matrix_mode(MatrixMode::Texture);
        s.set_active_unit(0);
        s.translate(1.0, 0.0, 0.0);
        s.set_active_unit(1);
        s.translate(2.0, 0.0, 0.0);
        assert_eq!(s.top_matrix(MatrixMode::Texture).unwrap().0[12], 2.0);
        s.set_active_unit(0);
        assert_eq!(s.top_matrix(MatrixMode::Texture).unwrap().0[12], 1.0);
    }

    // -- from_gl: every legal value, and a bogus one -------------------

    #[test]
    fn blend_factor_from_gl_accepts_every_legal_value() {
        let all = [
            (0x0000, BlendFactor::Zero),
            (0x0001, BlendFactor::One),
            (0x0300, BlendFactor::SrcColor),
            (0x0301, BlendFactor::OneMinusSrcColor),
            (0x0302, BlendFactor::SrcAlpha),
            (0x0303, BlendFactor::OneMinusSrcAlpha),
            (0x0304, BlendFactor::DstAlpha),
            (0x0305, BlendFactor::OneMinusDstAlpha),
            (0x0306, BlendFactor::DstColor),
            (0x0307, BlendFactor::OneMinusDstColor),
            (0x0308, BlendFactor::SrcAlphaSaturate),
        ];
        for (wire, expected) in all {
            assert_eq!(BlendFactor::from_gl(wire), Some(expected));
            assert_eq!(expected.to_gl(), wire);
        }
        assert_eq!(BlendFactor::from_gl(0xDEAD_BEEF), None);
    }

    #[test]
    fn blend_equation_from_gl_accepts_every_legal_value_and_rejects_a_bogus_one() {
        for (wire, expected) in [
            (0x8006, BlendEquation::FuncAdd),
            (0x8007, BlendEquation::Min),
            (0x8008, BlendEquation::Max),
            (0x800A, BlendEquation::FuncSubtract),
            (0x800B, BlendEquation::FuncReverseSubtract),
        ] {
            assert_eq!(BlendEquation::from_gl(wire), Some(expected));
        }
        assert_eq!(BlendEquation::from_gl(0x1234), None);
    }

    #[test]
    fn compare_func_from_gl_accepts_never_through_always_and_rejects_a_bogus_one() {
        for (wire, expected) in [
            (0x0200, CompareFunc::Never),
            (0x0201, CompareFunc::Less),
            (0x0202, CompareFunc::Equal),
            (0x0203, CompareFunc::Lequal),
            (0x0204, CompareFunc::Greater),
            (0x0205, CompareFunc::Notequal),
            (0x0206, CompareFunc::Gequal),
            (0x0207, CompareFunc::Always),
        ] {
            assert_eq!(CompareFunc::from_gl(wire), Some(expected));
        }
        assert_eq!(CompareFunc::from_gl(0x0208), None);
    }

    #[test]
    fn face_from_gl_accepts_front_back_and_front_and_back_and_rejects_a_bogus_one() {
        assert_eq!(Face::from_gl(0x0404), Some(Face::Front));
        assert_eq!(Face::from_gl(0x0405), Some(Face::Back));
        assert_eq!(Face::from_gl(0x0408), Some(Face::FrontAndBack));
        assert_eq!(Face::from_gl(0x0406), None);
    }

    #[test]
    fn front_face_from_gl_accepts_cw_and_ccw_and_rejects_a_bogus_one() {
        assert_eq!(FrontFace::from_gl(0x0900), Some(FrontFace::Cw));
        assert_eq!(FrontFace::from_gl(0x0901), Some(FrontFace::Ccw));
        assert_eq!(FrontFace::from_gl(0x0902), None);
    }

    #[test]
    fn shade_model_from_gl_accepts_flat_and_smooth_and_rejects_a_bogus_one() {
        assert_eq!(ShadeModel::from_gl(0x1D00), Some(ShadeModel::Flat));
        assert_eq!(ShadeModel::from_gl(0x1D01), Some(ShadeModel::Smooth));
        assert_eq!(ShadeModel::from_gl(0x1D02), None);
    }

    #[test]
    fn fog_mode_from_gl_accepts_linear_exp_and_exp2_and_rejects_a_bogus_one() {
        assert_eq!(FogMode::from_gl(0x0800), Some(FogMode::Exp));
        assert_eq!(FogMode::from_gl(0x0801), Some(FogMode::Exp2));
        assert_eq!(FogMode::from_gl(0x2601), Some(FogMode::Linear));
        assert_eq!(FogMode::from_gl(0x0802), None);
    }

    #[test]
    fn tex_env_mode_from_gl_accepts_every_legal_value_and_rejects_a_bogus_one() {
        for (wire, expected) in [
            (0x2100, TexEnvMode::Modulate),
            (0x1E01, TexEnvMode::Replace),
            (0x2101, TexEnvMode::Decal),
            (0x2201, TexEnvMode::Blend),
            (0x0104, TexEnvMode::Add),
        ] {
            assert_eq!(TexEnvMode::from_gl(wire), Some(expected));
        }
        assert_eq!(TexEnvMode::from_gl(0x2102), None);
    }

    #[test]
    fn tex_filter_from_gl_accepts_every_legal_value_and_rejects_a_bogus_one() {
        for (wire, expected) in [
            (0x2600, TexFilter::Nearest),
            (0x2601, TexFilter::Linear),
            (0x2700, TexFilter::NearestMipmapNearest),
            (0x2701, TexFilter::LinearMipmapNearest),
            (0x2702, TexFilter::NearestMipmapLinear),
            (0x2703, TexFilter::LinearMipmapLinear),
        ] {
            assert_eq!(TexFilter::from_gl(wire), Some(expected));
        }
        assert_eq!(TexFilter::from_gl(0x2704), None);
    }

    #[test]
    fn tex_wrap_from_gl_accepts_repeat_clamp_and_clamp_to_edge_and_rejects_a_bogus_one() {
        assert_eq!(TexWrap::from_gl(0x2901), Some(TexWrap::Repeat));
        assert_eq!(TexWrap::from_gl(0x2900), Some(TexWrap::Clamp));
        assert_eq!(TexWrap::from_gl(0x812F), Some(TexWrap::ClampToEdge));
        assert_eq!(TexWrap::from_gl(0x2902), None);
    }

    #[test]
    fn tex_wrap_clamp_behaves_as_clamp_to_edge() {
        assert_eq!(TexWrap::Clamp.effective(), TexWrap::ClampToEdge);
        assert_eq!(TexWrap::ClampToEdge.effective(), TexWrap::ClampToEdge);
        assert_eq!(TexWrap::Repeat.effective(), TexWrap::Repeat);
    }

    #[test]
    fn polygon_mode_from_gl_accepts_point_line_fill_and_rejects_a_bogus_one() {
        assert_eq!(PolygonMode::from_gl(0x1B00), Some(PolygonMode::Point));
        assert_eq!(PolygonMode::from_gl(0x1B01), Some(PolygonMode::Line));
        assert_eq!(PolygonMode::from_gl(0x1B02), Some(PolygonMode::Fill));
        assert_eq!(PolygonMode::from_gl(0x1B03), None);
    }

    #[test]
    fn tex_gen_mode_from_gl_accepts_every_legal_value_and_rejects_a_bogus_one() {
        assert_eq!(TexGenMode::from_gl(0x2400), Some(TexGenMode::EyeLinear));
        assert_eq!(TexGenMode::from_gl(0x2401), Some(TexGenMode::ObjectLinear));
        assert_eq!(TexGenMode::from_gl(0x2402), Some(TexGenMode::SphereMap));
        assert_eq!(TexGenMode::from_gl(0x2403), None);
    }

    #[test]
    fn matrix_mode_from_gl_accepts_every_legal_value_and_rejects_a_bogus_one() {
        assert_eq!(MatrixMode::from_gl(0x1700), Some(MatrixMode::Modelview));
        assert_eq!(MatrixMode::from_gl(0x1701), Some(MatrixMode::Projection));
        assert_eq!(MatrixMode::from_gl(0x1702), Some(MatrixMode::Texture));
        assert_eq!(MatrixMode::from_gl(0x1703), None);
    }

    #[test]
    fn light_id_from_gl_accepts_light0_through_light7_and_rejects_a_bogus_one() {
        for n in 0..8u32 {
            assert_eq!(
                LightId::from_gl(0x4000 + n).map(|l| l.index()),
                Some(n as usize)
            );
        }
        assert_eq!(LightId::from_gl(0x4008), None);
    }

    #[test]
    fn clip_plane_id_from_gl_accepts_plane0_through_plane5_and_rejects_a_bogus_one() {
        for n in 0..6u32 {
            assert_eq!(
                ClipPlaneId::from_gl(0x3000 + n).map(|p| p.index()),
                Some(n as usize)
            );
        }
        assert_eq!(ClipPlaneId::from_gl(0x3006), None);
    }

    #[test]
    fn primitive_type_from_gl_accepts_points_through_polygon_and_rejects_a_bogus_one() {
        for (wire, expected) in [
            (0x0000, PrimitiveType::Points),
            (0x0001, PrimitiveType::Lines),
            (0x0002, PrimitiveType::LineLoop),
            (0x0003, PrimitiveType::LineStrip),
            (0x0004, PrimitiveType::Triangles),
            (0x0005, PrimitiveType::TriangleStrip),
            (0x0006, PrimitiveType::TriangleFan),
            (0x0007, PrimitiveType::Quads),
            (0x0008, PrimitiveType::QuadStrip),
            (0x0009, PrimitiveType::Polygon),
        ] {
            assert_eq!(PrimitiveType::from_gl(wire), Some(expected));
        }
        assert_eq!(PrimitiveType::from_gl(0x000A), None);
    }

    #[test]
    fn capability_from_gl_accepts_every_baseline_and_transform_cap() {
        assert_eq!(Capability::from_gl(0x0BC0), Some(Capability::AlphaTest));
        assert_eq!(Capability::from_gl(0x0BE2), Some(Capability::Blend));
        assert_eq!(Capability::from_gl(0x0B44), Some(Capability::CullFace));
        assert_eq!(Capability::from_gl(0x0B71), Some(Capability::DepthTest));
        assert_eq!(Capability::from_gl(0x0BD0), Some(Capability::Dither));
        assert_eq!(Capability::from_gl(0x0B60), Some(Capability::Fog));
        assert_eq!(
            Capability::from_gl(0x8037),
            Some(Capability::PolygonOffsetFill)
        );
        assert_eq!(Capability::from_gl(0x0C11), Some(Capability::ScissorTest));
        assert_eq!(Capability::from_gl(0x0DE1), Some(Capability::Texture2D));
        assert_eq!(Capability::from_gl(0x0B50), Some(Capability::Lighting));
        assert_eq!(
            Capability::from_gl(0x4003),
            Some(Capability::Light(LightId::from_gl(0x4003).unwrap()))
        );
        assert_eq!(
            Capability::from_gl(0x3002),
            Some(Capability::ClipPlane(ClipPlaneId::from_gl(0x3002).unwrap()))
        );
        assert_eq!(Capability::from_gl(0x0B57), Some(Capability::ColorMaterial));
        assert_eq!(Capability::from_gl(0x0BA1), Some(Capability::Normalize));
        assert_eq!(Capability::from_gl(0x803A), Some(Capability::RescaleNormal));
        assert_eq!(Capability::from_gl(0x0C60), Some(Capability::TextureGenS));
        assert_eq!(Capability::from_gl(0x0C61), Some(Capability::TextureGenT));
    }

    #[test]
    fn capability_from_gl_rejects_stencil_test_since_there_is_no_stencil_buffer() {
        // GL_STENCIL_TEST = 0x0B90; the spec explicitly calls this out as
        // GL_INVALID_ENUM, not merely "not yet supported".
        assert_eq!(Capability::from_gl(0x0B90), None);
    }

    #[test]
    fn tex_format_from_wire_accepts_every_legal_value_and_rejects_a_bogus_one() {
        for (wire, expected) in [
            (0, TexFormat::Rgba8),
            (1, TexFormat::Rgb8),
            (2, TexFormat::Rgb565),
            (3, TexFormat::Rgba4444),
            (4, TexFormat::Rgba5551),
            (5, TexFormat::L8),
            (6, TexFormat::La8),
            (7, TexFormat::A8),
            (8, TexFormat::I8),
        ] {
            assert_eq!(TexFormat::from_wire(wire), Some(expected));
        }
        assert_eq!(TexFormat::from_wire(9), None);
    }

    #[test]
    fn surface_format_from_wire_accepts_every_legal_value_and_rejects_a_bogus_one() {
        for (wire, expected) in [
            (1, SurfaceFormat::R5g6b5),
            (2, SurfaceFormat::R5g6b5Le),
            (3, SurfaceFormat::R5g5b5),
            (4, SurfaceFormat::R5g5b5Le),
            (5, SurfaceFormat::A8r8g8b8),
            (6, SurfaceFormat::B8g8r8a8),
            (7, SurfaceFormat::R8g8b8a8),
            (8, SurfaceFormat::R8g8b8),
            (9, SurfaceFormat::B8g8r8),
            (255, SurfaceFormat::Depth),
        ] {
            assert_eq!(SurfaceFormat::from_wire(wire), Some(expected));
        }
        assert_eq!(SurfaceFormat::from_wire(10), None);
        assert_eq!(SurfaceFormat::from_wire(0), None);
    }

    #[test]
    fn tex_coord_space_from_wire_accepts_zero_and_one_and_rejects_a_bogus_one() {
        assert_eq!(TexCoordSpace::from_wire(0), Some(TexCoordSpace::Normalised));
        assert_eq!(TexCoordSpace::from_wire(1), Some(TexCoordSpace::Texel));
        assert_eq!(TexCoordSpace::from_wire(2), None);
    }

    // -- GL error: first-error-wins --------------------------------------

    #[test]
    fn no_error_initially() {
        let s = state();
        assert_eq!(s.gl_error(), GL_NO_ERROR);
    }

    #[test]
    fn first_error_wins_until_acked() {
        let mut s = state();
        s.raise_gl_error(GlError::InvalidEnum);
        s.raise_gl_error(GlError::InvalidValue);
        assert_eq!(s.gl_error(), GlError::InvalidEnum.to_gl());
        s.ack_gl_error();
        assert_eq!(s.gl_error(), GL_NO_ERROR);
        s.raise_gl_error(GlError::OutOfMemory);
        assert_eq!(s.gl_error(), GlError::OutOfMemory.to_gl());
    }

    #[test]
    fn a_stack_overflow_is_reported_through_gl_error_and_does_not_clobber_an_earlier_error() {
        let mut s = State::new(Limits {
            modelview_depth: 1,
            ..Limits::default()
        });
        s.raise_gl_error(GlError::InvalidEnum);
        assert_eq!(s.push_matrix(), Err(GlError::StackOverflow));
        // The earlier InvalidEnum still wins.
        assert_eq!(s.gl_error(), GlError::InvalidEnum.to_gl());
    }

    // -- Independent S/T wrap, per-unit texenv and texgen ----------------

    #[test]
    fn wrap_s_and_wrap_t_are_independently_settable() {
        let mut s = state();
        s.tex_create(1);
        s.set_tex_param(1, TexParam::WrapS(TexWrap::Repeat))
            .unwrap();
        s.set_tex_param(1, TexParam::WrapT(TexWrap::Clamp)).unwrap();
        let tex = s.texture(1).unwrap();
        assert_eq!(tex.wrap_s, TexWrap::Repeat);
        assert_eq!(tex.wrap_t, TexWrap::Clamp);
    }

    #[test]
    fn tex_env_is_independent_per_unit() {
        let mut s = state();
        s.set_tex_env(0, TexEnvParam::Mode(TexEnvMode::Decal));
        s.set_tex_env(1, TexEnvParam::Mode(TexEnvMode::Replace));
        assert_eq!(s.texture_units[0].env_mode, TexEnvMode::Decal);
        assert_eq!(s.texture_units[1].env_mode, TexEnvMode::Replace);
    }

    #[test]
    fn tex_env_color_is_independent_per_unit() {
        let mut s = state();
        s.set_tex_env_color(0, 1.0, 0.0, 0.0, 1.0);
        s.set_tex_env_color(1, 0.0, 1.0, 0.0, 1.0);
        assert_eq!(s.texture_units[0].env_color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(s.texture_units[1].env_color, [0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn texcoord_space_is_independent_per_unit() {
        let mut s = state();
        s.set_tex_env(0, TexEnvParam::CoordSpace(TexCoordSpace::Texel));
        assert_eq!(s.texture_units[0].coord_space, TexCoordSpace::Texel);
        assert_eq!(s.texture_units[1].coord_space, TexCoordSpace::Normalised);
    }

    #[test]
    fn texgen_mode_is_independent_per_unit_and_per_coordinate() {
        let mut s = state();
        s.set_texgen_mode(0, TexCoord::S, TexGenMode::ObjectLinear);
        s.set_texgen_mode(0, TexCoord::T, TexGenMode::SphereMap);
        s.set_texgen_mode(1, TexCoord::S, TexGenMode::SphereMap);
        assert_eq!(s.texgen[0].mode_s, TexGenMode::ObjectLinear);
        assert_eq!(s.texgen[0].mode_t, TexGenMode::SphereMap);
        assert_eq!(s.texgen[1].mode_s, TexGenMode::SphereMap);
        // Unit 1's T coordinate is untouched.
        assert_eq!(s.texgen[1].mode_t, TexGenMode::EyeLinear);
    }

    #[test]
    fn texture_2d_enable_applies_to_the_active_unit_only() {
        let mut s = state();
        s.set_active_unit(0);
        s.enable(Capability::Texture2D);
        s.set_active_unit(1);
        assert!(!s.is_enabled(Capability::Texture2D));
        s.set_active_unit(0);
        assert!(s.is_enabled(Capability::Texture2D));
    }

    // -- Lighting/clipping transforms ------------------------------------

    #[test]
    fn light_position_is_transformed_by_the_modelview_at_the_time_of_the_command() {
        let mut s = state();
        s.translate(1.0, 0.0, 0.0);
        s.set_light(LightId::Light0, LightParam::Position, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(s.lights.0[0].position, [1.0, 0.0, 0.0, 1.0]);
        // A later modelview change does not retroactively move it.
        s.translate(5.0, 0.0, 0.0);
        assert_eq!(s.lights.0[0].position, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn spot_direction_ignores_translation_but_applies_rotation() {
        let mut s = state();
        s.translate(9.0, 9.0, 9.0);
        s.rotate(90.0, 0.0, 0.0, 1.0);
        s.set_light(
            LightId::Light0,
            LightParam::SpotDirection,
            [1.0, 0.0, 0.0, 0.0],
        );
        let d = s.lights.0[0].spot_direction;
        assert!(d[0].abs() < 1e-5);
        assert!((d[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn clip_plane_at_identity_modelview_is_unchanged() {
        let mut s = state();
        s.set_clip_plane(ClipPlaneId::ClipPlane0, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.clip_planes.0[0], [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn clip_plane_is_transformed_by_the_inverse_modelview() {
        let mut s = state();
        s.translate(1.0, 0.0, 0.0);
        // Plane x = 0 in object space, at object-space translation +1.
        s.set_clip_plane(ClipPlaneId::ClipPlane0, [1.0, 0.0, 0.0, 0.0]);
        let eq = s.clip_planes.0[0];
        // The eye-space plane must still pass through the transformed
        // origin (1,0,0): a*1 + b*0 + c*0 + d == 0.
        assert!((eq[0] * 1.0 + eq[3]).abs() < 1e-5);
    }

    // -- Reset semantics --------------------------------------------------

    #[test]
    fn reset_state_restores_gl_state_but_keeps_texture_and_surface_objects() {
        let mut s = state();
        s.tex_create(1);
        s.surface_define(
            1,
            Surface {
                width: 4,
                height: 4,
                stride_bytes: 8,
                format: SurfaceFormat::R5g6b5,
                backing: Backing::Aperture(0),
            },
        );
        s.translate(1.0, 0.0, 0.0);
        s.enable(Capability::Blend);
        s.raise_gl_error(GlError::InvalidEnum);

        s.reset_state();

        assert_eq!(s.top_matrix(MatrixMode::Modelview), Some(Mat4::identity()));
        assert!(!s.enables.blend);
        assert!(s.texture(1).is_some());
        assert!(s.surface(1).is_some());
        // GL_ERROR is documented as its own independent mechanism, never
        // touched by CTX_RESET_STATE.
        assert_eq!(s.gl_error(), GlError::InvalidEnum.to_gl());
    }

    #[test]
    fn reset_all_also_destroys_every_texture_and_surface_object() {
        let mut s = state();
        s.tex_create(1);
        s.surface_define(
            1,
            Surface {
                width: 4,
                height: 4,
                stride_bytes: 8,
                format: SurfaceFormat::R5g6b5,
                backing: Backing::Aperture(0),
            },
        );
        s.reset_all();
        assert!(s.texture(1).is_none());
        assert!(s.surface(1).is_none());
    }

    // -- Textures: creation, destruction, binding -------------------------

    #[test]
    fn tex_destroy_unbinds_the_texture_from_every_unit() {
        let mut s = state();
        s.tex_create(1);
        s.tex_bind(0, 1);
        s.tex_bind(1, 1);
        s.tex_destroy(1);
        assert_eq!(s.texture_units[0].bound_texture, 0);
        assert_eq!(s.texture_units[1].bound_texture, 0);
        assert!(s.texture(1).is_none());
    }

    #[test]
    fn tex_create_on_an_existing_id_resets_its_parameters() {
        let mut s = state();
        s.tex_create(1);
        s.set_tex_param(1, TexParam::WrapS(TexWrap::Clamp)).unwrap();
        s.tex_create(1);
        assert_eq!(s.texture(1).unwrap().wrap_s, TexWrap::Repeat);
    }

    #[test]
    fn set_tex_param_on_an_undefined_texture_is_invalid_operation() {
        let mut s = state();
        assert_eq!(
            s.set_tex_param(42, TexParam::WrapS(TexWrap::Repeat)),
            Err(GlError::InvalidOperation)
        );
        assert_eq!(s.gl_error(), GlError::InvalidOperation.to_gl());
    }

    #[test]
    fn tex_palette_of_zero_entries_unbinds() {
        let mut s = state();
        s.tex_create(1);
        s.set_tex_palette(1, 256).unwrap();
        assert_eq!(s.texture(1).unwrap().palette_entries, Some(256));
        s.set_tex_palette(1, 0).unwrap();
        assert_eq!(s.texture(1).unwrap().palette_entries, None);
    }

    #[test]
    fn define_tex_level_records_shape_for_later_levels_independently() {
        let mut s = state();
        s.tex_create(1);
        s.define_tex_level(1, 0, TexFormat::Rgba8, 64, 64).unwrap();
        s.define_tex_level(1, 1, TexFormat::Rgba8, 32, 32).unwrap();
        let tex = s.texture(1).unwrap();
        assert_eq!(
            tex.levels[&0],
            TexLevel {
                width: 64,
                height: 64,
                format: TexFormat::Rgba8
            }
        );
        assert_eq!(
            tex.levels[&1],
            TexLevel {
                width: 32,
                height: 32,
                format: TexFormat::Rgba8
            }
        );
    }

    // -- Determinism: 4x4 inverse round trip -----------------------------

    #[test]
    fn matrix_inverse_of_a_composed_transform_round_trips_to_identity() {
        let mut s = state();
        s.translate(3.0, -2.0, 1.0);
        s.rotate(37.0, 0.3, 0.6, 0.7);
        s.scale(2.0, 0.5, 3.0);
        let m = s.top_matrix(MatrixMode::Modelview).unwrap();
        let inv = m.inverse().unwrap();
        let identity = Mat4::mul(&m, &inv);
        for i in 0..16 {
            let expected = Mat4::identity().0[i];
            assert!(
                (identity.0[i] - expected).abs() < 1e-4,
                "index {i}: {identity:?}"
            );
        }
    }

    #[test]
    fn singular_matrix_has_no_inverse() {
        let m = Mat4([0.0; 16]);
        assert_eq!(m.inverse(), None);
    }

    /// `Mat4::transform_normal3` must apply the *inverse-transpose*, not
    /// the plain matrix -- GL 1.1's correct rule for transforming a
    /// normal (section 2.11), which only coincides with the plain matrix
    /// when the linear part is orthogonal (pure rotation, no scale). A
    /// non-uniform scale of `2x` in `x` (and `1x` elsewhere) is the
    /// simplest case where the two rules diverge and diverge in opposite
    /// *directions*: the correct inverse-transpose scales an `x`-aligned
    /// normal by `0.5` (the object got wider in `x`, so its surface
    /// normal must lean *less* in `x` to stay perpendicular), while the
    /// plain matrix -- the naive, wrong approach a render-side caller
    /// could accidentally use instead -- would scale it by `2.0`, the
    /// same direction a position transforms.
    #[test]
    fn transform_normal3_uses_inverse_transpose_not_the_plain_matrix() {
        #[rustfmt::skip]
        let modelview = Mat4([
            2.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]);
        let inv = modelview.inverse().unwrap();

        let correct = inv.transform_normal3([1.0, 0.0, 0.0]);
        assert_eq!(correct, [0.5, 0.0, 0.0]);

        let wrong = modelview.transform_direction3([1.0, 0.0, 0.0]);
        assert_eq!(wrong, [2.0, 0.0, 0.0]);
        assert_ne!(
            correct, wrong,
            "the inverse-transpose and the plain matrix must diverge under non-uniform scale"
        );
    }

    // -- max_lights/max_clip_planes as construction parameters ----------

    #[test]
    fn light_and_clip_plane_counts_follow_limits_not_a_hardcoded_eight_and_six() {
        let s = State::new(Limits {
            max_lights: 3,
            max_clip_planes: 2,
            ..Limits::default()
        });
        assert_eq!(s.lights.0.len(), 3);
        assert_eq!(s.clip_planes.0.len(), 2);
        assert_eq!(s.enables.light.len(), 3);
        assert_eq!(s.enables.clip_plane.len(), 2);
        assert_eq!(s.limits().max_lights, 3);
        assert_eq!(s.limits().max_clip_planes, 2);
    }

    #[test]
    fn a_light_or_clip_plane_past_the_configured_count_is_a_silent_no_op() {
        let mut s = State::new(Limits {
            max_lights: 1,
            max_clip_planes: 1,
            ..Limits::default()
        });
        // LIGHT1 and CLIP_PLANE1 both exist as wire enumerants (GL 1.1
        // always defines LIGHT0..7/CLIP_PLANE0..5) but this board only
        // backs the first of each.
        s.set_light(
            LightId::Light1,
            LightParam::SpotExponent,
            [7.0, 0.0, 0.0, 0.0],
        );
        s.enable(Capability::Light(LightId::Light1));
        s.set_clip_plane(ClipPlaneId::ClipPlane1, [1.0, 1.0, 1.0, 1.0]);
        s.enable(Capability::ClipPlane(ClipPlaneId::ClipPlane1));
        // No panic, and nothing observable changed for the in-range ones.
        assert_eq!(s.lights.0.len(), 1);
        assert_eq!(s.clip_planes.0.len(), 1);
        assert!(!s.is_enabled(Capability::Light(LightId::Light1)));
        assert!(!s.is_enabled(Capability::ClipPlane(ClipPlaneId::ClipPlane1)));
    }

    #[test]
    fn light0_is_still_white_when_max_lights_is_reduced() {
        let s = State::new(Limits {
            max_lights: 1,
            ..Limits::default()
        });
        assert_eq!(s.lights.0[0].diffuse, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(s.lights.0[0].specular, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn a_board_with_zero_lights_has_no_light_and_does_not_panic() {
        let s = State::new(Limits {
            max_lights: 0,
            max_clip_planes: 0,
            ..Limits::default()
        });
        assert!(s.lights.0.is_empty());
        assert!(s.clip_planes.0.is_empty());
    }

    // -- E_NO_SURFACE / E_BAD_RECT: the protocol errors state.rs raises --

    fn defined_surface(s: &mut State, id: u32, width: u32, height: u32) {
        s.surface_define(
            id,
            Surface {
                width,
                height,
                stride_bytes: width * 2,
                format: SurfaceFormat::R5g6b5,
                backing: Backing::Aperture(0),
            },
        );
    }

    #[test]
    fn require_draw_surface_is_no_surface_when_none_is_bound() {
        let s = state();
        assert_eq!(s.require_draw_surface().err(), Some(ErrorCode::NoSurface));
    }

    #[test]
    fn require_draw_surface_is_no_surface_after_the_bound_surface_is_destroyed() {
        let mut s = state();
        defined_surface(&mut s, 1, 4, 4);
        s.set_draw_surface(1);
        assert!(s.require_draw_surface().is_ok());
        s.surface_destroy(1);
        assert_eq!(s.require_draw_surface().err(), Some(ErrorCode::NoSurface));
    }

    #[test]
    fn require_draw_surface_succeeds_once_a_surface_is_bound() {
        let mut s = state();
        defined_surface(&mut s, 1, 4, 4);
        s.set_draw_surface(1);
        assert_eq!(s.require_draw_surface().unwrap().width, 4);
    }

    #[test]
    fn check_rect_accepts_a_rectangle_wholly_inside_the_surface() {
        assert_eq!(State::check_rect(640, 480, 0, 0, 640, 480), Ok(()));
        assert_eq!(State::check_rect(640, 480, 100, 200, 50, 50), Ok(()));
    }

    #[test]
    fn check_rect_accepts_an_empty_rectangle_at_an_in_bounds_origin() {
        assert_eq!(State::check_rect(640, 480, 640, 480, 0, 0), Ok(()));
    }

    #[test]
    fn check_rect_rejects_a_negative_origin() {
        assert_eq!(
            State::check_rect(640, 480, -1, 0, 10, 10),
            Err(ErrorCode::BadRect)
        );
        assert_eq!(
            State::check_rect(640, 480, 0, -1, 10, 10),
            Err(ErrorCode::BadRect)
        );
    }

    #[test]
    fn check_rect_rejects_a_rectangle_extending_past_the_surface() {
        assert_eq!(
            State::check_rect(640, 480, 630, 0, 20, 10),
            Err(ErrorCode::BadRect)
        );
        assert_eq!(
            State::check_rect(640, 480, 0, 470, 10, 20),
            Err(ErrorCode::BadRect)
        );
    }

    #[test]
    fn check_rect_rejects_an_overflowing_width_or_height() {
        assert_eq!(
            State::check_rect(640, 480, 10, 0, u32::MAX, 10),
            Err(ErrorCode::BadRect)
        );
        assert_eq!(
            State::check_rect(640, 480, 0, 10, 10, u32::MAX),
            Err(ErrorCode::BadRect)
        );
    }

    #[test]
    fn check_draw_surface_rect_is_no_surface_before_bad_rect_when_both_would_apply() {
        // No draw surface at all: must report NoSurface, not BadRect,
        // even though the rectangle given is also nonsensical.
        let s = state();
        assert_eq!(
            s.check_draw_surface_rect(-5, -5, 1_000_000, 1_000_000),
            Err(ErrorCode::NoSurface)
        );
    }

    #[test]
    fn check_draw_surface_rect_reports_bad_rect_against_the_bound_surfaces_own_shape() {
        let mut s = state();
        defined_surface(&mut s, 1, 64, 32);
        s.set_draw_surface(1);
        assert_eq!(s.check_draw_surface_rect(0, 0, 64, 32), Ok(()));
        assert_eq!(
            s.check_draw_surface_rect(0, 0, 65, 32),
            Err(ErrorCode::BadRect)
        );
    }

    #[test]
    fn scissor_and_viewport_are_not_rect_checked_gl_clamps_them_instead() {
        // SCISSOR/VIEWPORT accept any values, in or out of surface
        // bounds; only SURFACE_UPLOAD/SURFACE_READBACK/READ_PIXELS go
        // through check_rect. This test documents that set_scissor and
        // set_viewport never consult the draw surface at all.
        let mut s = state();
        s.set_scissor(-100, -100, 1_000_000, 1_000_000);
        s.set_viewport(-100, -100, 1_000_000, 1_000_000);
        assert_eq!(
            s.raster.scissor,
            Rect {
                x: -100,
                y: -100,
                w: 1_000_000,
                h: 1_000_000
            }
        );
        assert_eq!(
            s.raster.viewport,
            Rect {
                x: -100,
                y: -100,
                w: 1_000_000,
                h: 1_000_000
            }
        );
    }
}
