// SPDX-License-Identifier: GPL-3.0-or-later

//! The C3D wgpu backend: turns [`super::dispatch::RenderOp`]s into pixels.
//!
//! `docs/internals/c3d.md` is the device specification this module renders
//! against; where this module and that spec disagree, the spec wins. This
//! is the one part of `src/c3d/` that owns a GPU device (cargo feature
//! `c3d`, off by default -- see `src/c3d/mod.rs`), and it owns its *own*
//! headless `wgpu` device: it never assumes a window or a surface exists,
//! because neither the conformance-trace runner (`src/bin/c3d-trace.rs`)
//! nor a headless capture has one. [`Renderer::new`] blocks (via
//! `pollster`, since there is no async runtime here and none is wanted) on
//! requesting a hardware adapter, falling back to a software one -- CI has
//! no GPU, and that is the expected case there, not a failure.
//!
//! ## M1 scope
//!
//! This milestone ("a native trace runner rendering hand-written traces
//! (clear, triangle, textured quad) to PNG") proves the pipeline end to
//! end, not the full fixed-function feature set (that is M3). Implemented:
//!
//! - [`RenderOp::Clear`] honouring `CLEAR_COLOR`/`CLEAR_DEPTH`, the colour
//!   mask and the scissor -- a full-rect, full-mask clear uses `wgpu`'s
//!   own `LoadOp::Clear`; a scissored or partial-colour-mask clear falls
//!   back to a degenerate scissored draw instead
//!   ([`ClearRectPipelineKey`]), since neither restriction is something
//!   `LoadOp::Clear` alone can express.
//! - [`RenderOp::Draw`] for the **window-space** draw shapes
//!   (`DRAW_INLINE_WIN` only; `DRAW_ARRAYS_WIN`/`DRAW_ELEMENTS_WIN` are
//!   deferred -- see [`RenderError::Unimplemented`]), `POS_COUNT` 4/3/2,
//!   `COLOR`, `COLOR_PACKED` and `TEXCOORD0`, correct `rhw` handling,
//!   every primitive type with the spec's normative quad/polygon
//!   triangulation and provoking vertex, flat and smooth shading -- and
//!   `DRAW_INLINE` (**GL-space**, `CAP_TRANSFORM`), **position transform
//!   only**: object-space `(x, y, z, w)` transformed by `modelview` then
//!   `projection` ([`Renderer::op_draw_inline_gl`]), then `VIEWPORT` and
//!   `DEPTH_RANGE`, per GL 1.1's fixed-function pipeline exactly as the
//!   spec requires. Colour and `TEXCOORD0` pass through unlit and
//!   untexgen'd, same as the window-space path; `NORMAL` is parsed (legal
//!   here, unlike window-space) but not yet used. `DRAW_ARRAYS`/
//!   `DRAW_ELEMENTS` (GL-space), lighting, texgen and `CLIP_PLANE` remain
//!   deferred -- see the updated list below.
//! - [`RenderOp::TexImage`]/[`RenderOp::TexSubImage`] for the conforming
//!   minimum texture formats, `TEXTURE_2D` sampling with the filter and
//!   wrap modes (`CLAMP` behaving as `CLAMP_TO_EDGE`, never a border
//!   colour), `TEXCOORD_SPACE` normalised (texel space is deferred).
//! - [`RenderOp::SurfaceReadback`] resolving the rendered surface into a
//!   CPU buffer in the surface's own pixel format, and
//!   [`RenderOp::SurfaceUpload`], its mirror -- both row by row, each row
//!   at its own `address + (y + row) * stride_bytes + x * bpp` offset in
//!   backing memory, never the rectangle's own tightly-packed offset.
//! - [`RenderOp::ReadPixels`], including the `Depth` format
//!   `SurfaceReadback`/`encode_pixel` do not accept, scaled from wgpu's
//!   own `[0, 1]` depth range to the spec's 32-bit unsigned one, and
//!   `ROWS_BOTTOM_UP`.
//! - [`RenderOp::TexCopyImage`]/[`RenderOp::TexCopySubImage`] as a
//!   GPU-side texture-to-texture copy from the draw surface -- no CPU
//!   round trip, since the surface's internal colour format already
//!   matches every texture object's.
//! - Depth test and mask, blend func/func-separate/equation, polygon
//!   offset, alpha test, cull face and front face (POINTS/LINES-family
//!   primitives are never culled, per GL), scissor, viewport (window-space
//!   draws ignore it, per the spec -- viewport is transform-tier), colour
//!   mask, fog (linear/exp/exp2; per-vertex `FOGCOORD` when the format
//!   carries it, else derived from `1 / rhw`, per the window-space rule).
//!
//! Deferred to M3: `DRAW_ARRAYS`/`DRAW_ELEMENTS` (GL-space; the window-
//! space array/indexed equivalents are a separate, parallel effort),
//! lighting (`LIGHT`/`LIGHT_MODEL`/`MATERIAL`/`COLOR_MATERIAL`), texgen
//! (`TEXGEN`/`TEXGEN_PLANE`), `CLIP_PLANE` user clipping, multitexture
//! (`TEXCOORD1`-`3`), texel-space texture coordinates. Every one of these
//! returns [`RenderError::Unimplemented`] rather than panicking.
//! (`TEX_PALETTE` is not in this list: it's spec-optional and Copperline
//! correctly never advertises it, so `ring.rs` rejects it as
//! `E_BAD_OPCODE` before it ever reaches this module -- see `ring.rs`'s
//! own doc comment.)
//!
//! ## Two traps the spec calls out
//!
//! - **Clip-space depth convention.** GL's clip space is `z in [-1, 1]`;
//!   wgpu's is `z in [0, 1]`. Window-space vertices already carry final
//!   `[0, 1]` surface depth (the spec: "`z` is depth in `0..1`"), so no
//!   fixup is needed on that path. [`gl_clip_z_to_wgpu`] is the fixup
//!   [`Renderer::op_draw_inline_gl`] needs (generalised to an arbitrary
//!   `DEPTH_RANGE` right there, rather than adding a second helper --
//!   see that function's own doc comment).
//! - **`CLAMP` has no border colour.** [`tex_wrap_to_address_mode`] maps
//!   both `CLAMP` and `CLAMP_TO_EDGE` to `wgpu::AddressMode::ClampToEdge`
//!   and never to `ClampToBorder`, via [`state::TexWrap::effective`].
//!
//! ## Structure
//!
//! One WGSL shader template ([`SHADER_TEMPLATE`]) specialised by
//! [`PipelineKey`] (texturing, texenv mode, alpha test, shading), with
//! pipelines cached by that key plus the blend/depth/cull state and the
//! target surface format ([`Renderer::pipeline_for`]). The fixed-function
//! content is thin in M1 (no lighting, no fog) but the specialisation
//! mechanism is the M3-ready part: a later milestone adds WGSL fragments
//! and key fields, not a new architecture.
//!
//! Vertex parsing, triangulation (including the spec's normative quad/
//! polygon/strip/fan decomposition and provoking-vertex selection),
//! colour unpacking and format/wrap/filter mapping are pure functions with
//! no GPU dependency, tested directly. Anything that needs a live device
//! (pipeline creation, an actual draw) is gated behind
//! [`Renderer::for_testing`] and skips cleanly with no adapter, per
//! `tests/README.md`'s asset-gated pattern.

use super::dispatch::{DrawVertices, QueryResult, RenderOp};
use super::proto::{self, Ref, RefSpace, VertexFormat};
use super::state::{
    self, BlendEquation, BlendFactor, CompareFunc, Face, FogMode, FrontFace, Mat4, MatrixMode,
    PrimitiveType, ShadeModel, State, Surface, SurfaceFormat as WireSurfaceFormat, TexEnvMode,
    TexFilter, TexFormat, TexWrap,
};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

/// Everything that can go wrong executing a [`RenderOp`] or setting up the
/// renderer itself. Never raised via `panic!`/`unwrap`/`expect` on
/// anything derived from trace or wire data.
#[derive(Debug)]
pub enum RenderError {
    /// No adapter at all -- neither hardware nor software -- was
    /// available. Distinct from a normal "software only" CI run, which is
    /// not an error (see the module doc comment).
    NoAdapter,
    /// `Adapter::request_device` failed.
    DeviceRequest(String),
    /// This op, or this particular shape of it, is not implemented in this
    /// milestone. Carries a short reason so a runner can print something
    /// more useful than "unsupported".
    Unimplemented(&'static str),
    /// A ref or backing location this renderer was asked to read/write
    /// could not be resolved by the caller's [`Memory`] implementation.
    BadMemory(String),
    /// The op referenced a surface or texture id that either does not
    /// exist in `state` or was never defined on the GPU side.
    UnknownObject(&'static str, u32),
    /// A vertex or texel payload was shorter than its format/dimensions
    /// require.
    ShortPayload(&'static str),
    /// A GPU-side operation (texture/buffer creation, mapping) failed.
    Gpu(String),
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderError::NoAdapter => write!(f, "no wgpu adapter available (hardware or software)"),
            RenderError::DeviceRequest(e) => write!(f, "wgpu device request failed: {e}"),
            RenderError::Unimplemented(what) => {
                write!(f, "not implemented in this milestone: {what}")
            }
            RenderError::BadMemory(e) => write!(f, "memory access failed: {e}"),
            RenderError::UnknownObject(kind, id) => write!(f, "unknown {kind} id {id}"),
            RenderError::ShortPayload(what) => write!(f, "payload too short for {what}"),
            RenderError::Gpu(e) => write!(f, "GPU error: {e}"),
        }
    }
}

impl std::error::Error for RenderError {}

// ---------------------------------------------------------------------
// Memory: how the renderer reaches ref'd and surface-backing bytes
// ---------------------------------------------------------------------

/// Where a byte range named by a [`Ref`] or a [`state::Backing`] actually
/// lives. The renderer never touches the aperture or guest memory model
/// itself (it has none -- that is the board/runner's concern), so every
/// read or write of guest-visible memory goes through this trait, which
/// the caller (the conformance-trace runner, or eventually the board
/// layer) implements over whatever it uses to model those address spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemLoc {
    Aperture(u32),
    Guest(u32),
}

impl MemLoc {
    pub fn from_ref(r: Ref) -> MemLoc {
        match r.space {
            RefSpace::Aperture => MemLoc::Aperture(r.address),
            RefSpace::Guest => MemLoc::Guest(r.address),
        }
    }

    pub fn from_backing(b: state::Backing) -> MemLoc {
        match b {
            state::Backing::Aperture(a) => MemLoc::Aperture(a),
            state::Backing::Guest(a) => MemLoc::Guest(a),
        }
    }
}

/// A surface's backing location, offset by `extra` bytes -- one row of a
/// [`RenderOp::SurfaceReadback`] or [`RenderOp::SurfaceUpload`] rectangle
/// that does not start at the surface's own first row/column, per
/// `def.stride_bytes` and the rectangle's own `x`/`y`.
fn offset_backing(b: state::Backing, extra: u32) -> MemLoc {
    match b {
        state::Backing::Aperture(a) => MemLoc::Aperture(a + extra),
        state::Backing::Guest(a) => MemLoc::Guest(a + extra),
    }
}

/// The same shift as [`offset_backing`], for a [`Ref`] destination
/// instead of a surface's own backing -- [`RenderOp::ReadPixels`]'s
/// `dest` (an arbitrary caller-chosen buffer, not the surface itself),
/// written one row at a time at its own caller-specified `row_bytes`
/// stride.
fn offset_ref(r: Ref, extra: u32) -> MemLoc {
    match r.space {
        RefSpace::Aperture => MemLoc::Aperture(r.address + extra),
        RefSpace::Guest => MemLoc::Guest(r.address + extra),
    }
}

/// The renderer's view of guest-visible memory: read for texture/array
/// data, write for a surface readback landing back in backing memory.
pub trait Memory {
    fn read(&self, loc: MemLoc, len: usize) -> Option<&[u8]>;
    fn write(&mut self, loc: MemLoc, data: &[u8]) -> bool;
}

// ---------------------------------------------------------------------
// Pure logic: colour, vertex decode, triangulation, format mapping
// ---------------------------------------------------------------------

/// Unpacks `COLOR_PACKED`'s `0xRRGGBBAA` word (unnormalised bytes) to
/// `0..1` floats, per the spec's "the device scales to `0..1`".
pub fn unpack_color(word: u32) -> [f32; 4] {
    let r = ((word >> 24) & 0xFF) as f32 / 255.0;
    let g = ((word >> 16) & 0xFF) as f32 / 255.0;
    let b = ((word >> 8) & 0xFF) as f32 / 255.0;
    let a = (word & 0xFF) as f32 / 255.0;
    [r, g, b, a]
}

/// The fixup a GL-space (`CAP_TRANSFORM`, M3) vertex pipeline needs at the
/// clip-space boundary: GL's `z in [-1, 1]` mapped to wgpu's `z in [0,
/// 1]`, given the clip-space `w` the division still has to happen against.
/// `clip_z_gl` and `clip_w` are pre-perspective-divide clip-space values;
/// the result is the `z` component wgpu expects in the same (still
/// undivided) clip space, i.e. `clip_z_wgpu = (clip_z_gl + clip_w) / 2`,
/// which divides to `(ndc_z_gl + 1) / 2 in [0, 1]` exactly as OpenGL's own
/// `glDepthRange`-less mapping would.
///
/// Window-space draws (this milestone's whole scope) never call this: the
/// spec hands over `z` already in `[0, 1]`, the surface-pixel convention,
/// with no clip-space step at all. It is included and tested now so the
/// GL-space vertex path M3 adds does not have to rediscover it.
pub fn gl_clip_z_to_wgpu(clip_z_gl: f32, clip_w: f32) -> f32 {
    (clip_z_gl + clip_w) * 0.5
}

/// `CLAMP`/`CLAMP_TO_EDGE` both map to wgpu's edge clamp; `CLAMP` never
/// gets a border colour (the spec: "this is the behaviour period software
/// expects and a device must not introduce a border").
pub fn tex_wrap_to_address_mode(wrap: TexWrap) -> wgpu::AddressMode {
    match wrap.effective() {
        TexWrap::Repeat => wgpu::AddressMode::Repeat,
        TexWrap::ClampToEdge | TexWrap::Clamp => wgpu::AddressMode::ClampToEdge,
    }
}

/// `TEXTURE_MIN_FILTER`/`TEXTURE_MAG_FILTER`. The four mipmap modes are
/// collapsed to their base filter -- M1 uploads level 0 only (mipmap
/// generation is M3), so `min_filter`'s mipmap component has nothing to
/// select between yet.
pub fn tex_filter_to_wgpu(filter: TexFilter) -> wgpu::FilterMode {
    match filter {
        TexFilter::Nearest | TexFilter::NearestMipmapNearest | TexFilter::NearestMipmapLinear => {
            wgpu::FilterMode::Nearest
        }
        TexFilter::Linear | TexFilter::LinearMipmapNearest | TexFilter::LinearMipmapLinear => {
            wgpu::FilterMode::Linear
        }
    }
}

/// `docs/internals/c3d.md`'s "Surface formats" mapped to a wgpu render
/// target format. The device's own chunky formats that aren't a wgpu
/// native format (`R5G6B5` and friends) are not attachment formats here:
/// the renderer's internal render target is always `Rgba8Unorm`, and
/// conversion to/from a surface's declared wire format happens at
/// readback/upload time ([`convert_from_rgba8`]/[`convert_to_rgba8`]),
/// not by trying to render directly into a 16-bit target.
pub const INTERNAL_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const INTERNAL_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Converts one RGBA8 (straight, `0..255` per channel) pixel to the bytes
/// of `fmt`, for [`RenderOp::SurfaceReadback`]. `Depth` is not a colour
/// format and has no encoding here (`READ_PIXELS` -- which does read
/// depth -- is deferred to M3).
pub fn encode_pixel(fmt: WireSurfaceFormat, rgba: [u8; 4]) -> Result<Vec<u8>, RenderError> {
    let [r, g, b, a] = rgba;
    Ok(match fmt {
        WireSurfaceFormat::A8r8g8b8 => vec![a, r, g, b],
        WireSurfaceFormat::B8g8r8a8 => vec![b, g, r, a],
        WireSurfaceFormat::R8g8b8a8 => vec![r, g, b, a],
        WireSurfaceFormat::R8g8b8 => vec![r, g, b],
        WireSurfaceFormat::B8g8r8 => vec![b, g, r],
        WireSurfaceFormat::R5g6b5 => {
            let v = pack565(r, g, b);
            v.to_be_bytes().to_vec()
        }
        WireSurfaceFormat::R5g6b5Le => {
            let v = pack565(r, g, b);
            v.to_le_bytes().to_vec()
        }
        WireSurfaceFormat::R5g5b5 => {
            let v = pack555(r, g, b, a);
            v.to_be_bytes().to_vec()
        }
        WireSurfaceFormat::R5g5b5Le => {
            let v = pack555(r, g, b, a);
            v.to_le_bytes().to_vec()
        }
        WireSurfaceFormat::Depth => {
            return Err(RenderError::Unimplemented("DEPTH surface readback"));
        }
    })
}

/// Bytes per pixel of a wire [`WireSurfaceFormat`], mirroring
/// [`proto::SurfaceFormat::bytes_per_pixel`] (that method lives on the
/// wire-level enum this module doesn't use; `state.rs`'s own
/// [`WireSurfaceFormat`] has no equivalent, so this module supplies one).
pub fn surface_format_bytes_per_pixel(fmt: WireSurfaceFormat) -> u32 {
    match fmt {
        WireSurfaceFormat::R5g6b5
        | WireSurfaceFormat::R5g6b5Le
        | WireSurfaceFormat::R5g5b5
        | WireSurfaceFormat::R5g5b5Le => 2,
        WireSurfaceFormat::A8r8g8b8 | WireSurfaceFormat::B8g8r8a8 | WireSurfaceFormat::R8g8b8a8 => {
            4
        }
        WireSurfaceFormat::R8g8b8 | WireSurfaceFormat::B8g8r8 => 3,
        WireSurfaceFormat::Depth => 4,
    }
}

fn pack565(r: u8, g: u8, b: u8) -> u16 {
    let r5 = (r as u16 * 31 + 127) / 255;
    let g6 = (g as u16 * 63 + 127) / 255;
    let b5 = (b as u16 * 31 + 127) / 255;
    (r5 << 11) | (g6 << 5) | b5
}

fn pack555(r: u8, g: u8, b: u8, a: u8) -> u16 {
    let r5 = (r as u16 * 31 + 127) / 255;
    let g5 = (g as u16 * 31 + 127) / 255;
    let b5 = (b as u16 * 31 + 127) / 255;
    let a1 = if a >= 128 { 1u16 } else { 0 };
    (r5 << 10) | (g5 << 5) | b5 | a1
}

fn unpack565(v: u16) -> (u8, u8, u8) {
    let r5 = (v >> 11) & 0x1F;
    let g6 = (v >> 5) & 0x3F;
    let b5 = v & 0x1F;
    let r = ((r5 << 3) | (r5 >> 2)) as u8;
    let g = ((g6 << 2) | (g6 >> 4)) as u8;
    let b = ((b5 << 3) | (b5 >> 2)) as u8;
    (r, g, b)
}

fn unpack555(v: u16) -> (u8, u8, u8, u8) {
    let r5 = (v >> 10) & 0x1F;
    let g5 = (v >> 5) & 0x1F;
    let b5 = v & 0x1F;
    let a1 = v & 0x1;
    let r = ((r5 << 3) | (r5 >> 2)) as u8;
    let g = ((g5 << 3) | (g5 >> 2)) as u8;
    let b = ((b5 << 3) | (b5 >> 2)) as u8;
    let a = if a1 != 0 { 255u8 } else { 0u8 };
    (r, g, b, a)
}

/// The exact inverse of [`encode_pixel`], for [`RenderOp::SurfaceUpload`]:
/// converts one pixel of `fmt` at `src` back to a straight RGBA8 pixel.
/// `None` if `src` is too short for one pixel of `fmt`, or for `Depth`,
/// which is not a colour format `SURFACE_UPLOAD` ever targets (surfaces
/// are colour render targets; `Depth` only ever appears as a
/// `READ_PIXELS` output format).
pub fn decode_pixel(fmt: WireSurfaceFormat, src: &[u8]) -> Option<[u8; 4]> {
    Some(match fmt {
        WireSurfaceFormat::A8r8g8b8 => {
            let [a, r, g, b] = *<&[u8; 4]>::try_from(src.get(..4)?).ok()?;
            [r, g, b, a]
        }
        WireSurfaceFormat::B8g8r8a8 => {
            let [b, g, r, a] = *<&[u8; 4]>::try_from(src.get(..4)?).ok()?;
            [r, g, b, a]
        }
        WireSurfaceFormat::R8g8b8a8 => {
            let [r, g, b, a] = *<&[u8; 4]>::try_from(src.get(..4)?).ok()?;
            [r, g, b, a]
        }
        WireSurfaceFormat::R8g8b8 => {
            let [r, g, b] = *<&[u8; 3]>::try_from(src.get(..3)?).ok()?;
            [r, g, b, 255]
        }
        WireSurfaceFormat::B8g8r8 => {
            let [b, g, r] = *<&[u8; 3]>::try_from(src.get(..3)?).ok()?;
            [r, g, b, 255]
        }
        WireSurfaceFormat::R5g6b5 => {
            let v = u16::from_be_bytes(*<&[u8; 2]>::try_from(src.get(..2)?).ok()?);
            let (r, g, b) = unpack565(v);
            [r, g, b, 255]
        }
        WireSurfaceFormat::R5g6b5Le => {
            let v = u16::from_le_bytes(*<&[u8; 2]>::try_from(src.get(..2)?).ok()?);
            let (r, g, b) = unpack565(v);
            [r, g, b, 255]
        }
        WireSurfaceFormat::R5g5b5 => {
            let v = u16::from_be_bytes(*<&[u8; 2]>::try_from(src.get(..2)?).ok()?);
            let (r, g, b, a) = unpack555(v);
            [r, g, b, a]
        }
        WireSurfaceFormat::R5g5b5Le => {
            let v = u16::from_le_bytes(*<&[u8; 2]>::try_from(src.get(..2)?).ok()?);
            let (r, g, b, a) = unpack555(v);
            [r, g, b, a]
        }
        WireSurfaceFormat::Depth => return None,
    })
}

/// Converts a texel of `fmt` at `src` into a straight RGBA8 pixel, for
/// texture upload into the renderer's always-RGBA8 sampling textures.
/// `None` if `src` is too short for one texel of `fmt`.
pub fn convert_texel_to_rgba8(fmt: TexFormat, src: &[u8]) -> Option<[u8; 4]> {
    Some(match fmt {
        TexFormat::Rgba8 => {
            if src.len() < 4 {
                return None;
            }
            [src[0], src[1], src[2], src[3]]
        }
        TexFormat::Rgb8 => {
            if src.len() < 3 {
                return None;
            }
            [src[0], src[1], src[2], 255]
        }
        TexFormat::Rgb565 => {
            if src.len() < 2 {
                return None;
            }
            let v = u16::from_be_bytes([src[0], src[1]]);
            let r5 = (v >> 11) & 0x1F;
            let g6 = (v >> 5) & 0x3F;
            let b5 = v & 0x1F;
            [expand5(r5), expand6(g6), expand5(b5), 255]
        }
        TexFormat::Rgba4444 => {
            if src.len() < 2 {
                return None;
            }
            let v = u16::from_be_bytes([src[0], src[1]]);
            let r4 = (v >> 12) & 0xF;
            let g4 = (v >> 8) & 0xF;
            let b4 = (v >> 4) & 0xF;
            let a4 = v & 0xF;
            [expand4(r4), expand4(g4), expand4(b4), expand4(a4)]
        }
        TexFormat::Rgba5551 => {
            if src.len() < 2 {
                return None;
            }
            let v = u16::from_be_bytes([src[0], src[1]]);
            let r5 = (v >> 11) & 0x1F;
            let g5 = (v >> 6) & 0x1F;
            let b5 = (v >> 1) & 0x1F;
            let a1 = v & 0x1;
            [
                expand5(r5),
                expand5(g5),
                expand5(b5),
                if a1 != 0 { 255 } else { 0 },
            ]
        }
        TexFormat::L8 => {
            if src.is_empty() {
                return None;
            }
            [src[0], src[0], src[0], 255]
        }
        TexFormat::La8 => {
            if src.len() < 2 {
                return None;
            }
            [src[0], src[0], src[0], src[1]]
        }
        TexFormat::A8 => {
            if src.is_empty() {
                return None;
            }
            [0, 0, 0, src[0]]
        }
        TexFormat::I8 => {
            // Unindexed (no palette bound in this milestone -- TEX_PALETTE
            // is deferred): I8 is intensity, R=G=B=A=I, per the spec.
            if src.is_empty() {
                return None;
            }
            [src[0], src[0], src[0], src[0]]
        }
    })
}

fn expand5(v: u16) -> u8 {
    ((v * 255 + 15) / 31) as u8
}
fn expand6(v: u16) -> u8 {
    ((v * 255 + 31) / 63) as u8
}
fn expand4(v: u16) -> u8 {
    ((v * 255 + 7) / 15) as u8
}

/// Bytes per texel of `fmt`, matching [`convert_texel_to_rgba8`]'s reads.
pub fn tex_format_bytes_per_pixel(fmt: TexFormat) -> usize {
    match fmt {
        TexFormat::Rgba8 => 4,
        TexFormat::Rgb8 => 3,
        TexFormat::Rgb565 | TexFormat::Rgba4444 | TexFormat::Rgba5551 | TexFormat::La8 => 2,
        TexFormat::L8 | TexFormat::A8 | TexFormat::I8 => 1,
    }
}

/// One parsed window-space vertex: everything a fragment might need,
/// already carrying its `rhw` in the position's fourth component exactly
/// as the wire does, so the vertex shader (not this code) performs the
/// perspective-correct interpolation setup. `x`/`y` are surface pixels,
/// `z` is `0..1`, `w` is `rhw`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WinVertex {
    pub pos: [f32; 4], // x, y, z, rhw
    pub color: [f32; 4],
    pub texcoord0: [f32; 2],
    /// `FOGCOORD` if the format carries it, else `CURRENT_FOGCOORD` --
    /// the ordinary "omitted component takes the current value" rule.
    /// Whether this or `1 / rhw` is what actually drives the fog
    /// equation is decided by the caller, per the spec's window-space
    /// rule ("fog distance is `FOGCOORD` if present, else derived from
    /// `rhw`") -- a *format-level* decision, not a per-vertex one, so it
    /// isn't made here.
    pub fogcoord: f32,
}

fn read_f32(bytes: &[u8], at: usize) -> Option<f32> {
    let w: [u8; 4] = bytes.get(at..at + 4)?.try_into().ok()?;
    Some(f32::from_be_bytes(w))
}

fn read_u32_be(bytes: &[u8], at: usize) -> Option<u32> {
    let w: [u8; 4] = bytes.get(at..at + 4)?.try_into().ok()?;
    Some(u32::from_be_bytes(w))
}

/// The byte layout `DRAW_INLINE` (GL-space) and `DRAW_INLINE_WIN`
/// (window-space) share exactly, per `docs/internals/c3d.md`'s "Vertex
/// format" table: only `POS`'s four components' *semantics* differ
/// between the two opcodes (object-space to be transformed vs. window
/// pixels + `rhw`), which is entirely a matter of how a caller
/// interprets `pos`, not how the bytes are walked. [`parse_window_vertices`]
/// and [`parse_gl_vertices`] are both thin wrappers over this shared
/// walk, returning [`WinVertex`]/[`GlVertex`] respectively -- the two
/// structs are structurally identical, but kept distinct types so a
/// caller can never accidentally feed one draw kind's vertices into the
/// other's transform path.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RawVertex {
    pos: [f32; 4],
    color: [f32; 4],
    texcoord0: [f32; 2],
    fogcoord: f32,
}

/// Parses `count` vertices out of `data`, `format`-interleaved, exactly
/// as `DRAW_INLINE`/`DRAW_INLINE_WIN` lay them out. Every optional
/// component in the format is walked and its words consumed --
/// including ones this milestone doesn't render (`NORMAL`, `TEXCOORD1`-
/// `3`) -- so a later vertex in the same command parses at the right
/// offset even though this milestone only *uses*
/// `COLOR`/`COLOR_PACKED`/`TEXCOORD0`/`FOGCOORD`. `NORMAL`'s "illegal in
/// a window-space draw" rule is enforced at the dispatch/decode layer
/// (`ring.rs`'s `check_window_space_format`), not here: both draw kinds
/// must still walk its bytes to keep later components at the right
/// offset. `current` supplies the value for any component the format
/// omits. `Err` if `data` runs out before `count` vertices are read.
fn parse_vertices_raw(
    data: &[u8],
    format: VertexFormat,
    count: u32,
    current: &state::CurrentVertex,
) -> Result<Vec<RawVertex>, RenderError> {
    let Some(pos_words) = format.pos_words() else {
        return Err(RenderError::ShortPayload("POS_COUNT"));
    };
    let mut out = Vec::with_capacity(count as usize);
    let mut at = 0usize;
    let bad = || RenderError::ShortPayload("vertex");

    for _ in 0..count {
        let mut pos = [0.0f32, 0.0, 0.0, 1.0];
        for (i, slot) in pos.iter_mut().enumerate().take(pos_words as usize) {
            *slot = read_f32(data, at).ok_or_else(bad)?;
            at += 4;
            let _ = i;
        }

        let mut color = current.color;
        if format.has(VertexFormat::COLOR) {
            for c in color.iter_mut() {
                *c = read_f32(data, at).ok_or_else(bad)?;
                at += 4;
            }
        } else if format.has(VertexFormat::COLOR_PACKED) {
            let word = read_u32_be(data, at).ok_or_else(bad)?;
            at += 4;
            color = unpack_color(word);
        }

        if format.has(VertexFormat::NORMAL) {
            // Illegal in a window-space draw per the spec (E_BAD_ARG); the
            // decode layer is expected to have rejected this already, but
            // this parser still has to consume the words if asked to keep
            // decoding, so it treats it as an ordinary (unused) component.
            at += 4 * 3;
            if at > data.len() {
                return Err(bad());
            }
        }

        let mut texcoord0 = current.texcoord.first().copied().unwrap_or((0.0, 0.0));
        if format.has(VertexFormat::TEXCOORD0) {
            let s = read_f32(data, at).ok_or_else(bad)?;
            let t = read_f32(data, at + 4).ok_or_else(bad)?;
            at += 8;
            texcoord0 = (s, t);
        }
        for bit in [
            VertexFormat::TEXCOORD1,
            VertexFormat::TEXCOORD2,
            VertexFormat::TEXCOORD3,
        ] {
            if format.has(bit) {
                at += 8;
                if at > data.len() {
                    return Err(bad());
                }
            }
        }
        let mut fogcoord = current.fogcoord;
        if format.has(VertexFormat::FOGCOORD) {
            fogcoord = read_f32(data, at).ok_or_else(bad)?;
            at += 4;
        }

        out.push(RawVertex {
            pos,
            color,
            texcoord0: [texcoord0.0, texcoord0.1],
            fogcoord,
        });
    }
    Ok(out)
}

/// Parses `count` window-space vertices out of `data`, `format`-
/// interleaved, exactly as `DRAW_INLINE_WIN` lays them out -- see
/// [`parse_vertices_raw`], which this wraps.
pub fn parse_window_vertices(
    data: &[u8],
    format: VertexFormat,
    count: u32,
    current: &state::CurrentVertex,
) -> Result<Vec<WinVertex>, RenderError> {
    Ok(parse_vertices_raw(data, format, count, current)?
        .into_iter()
        .map(|r| WinVertex {
            pos: r.pos,
            color: r.color,
            texcoord0: r.texcoord0,
            fogcoord: r.fogcoord,
        })
        .collect())
}

/// One parsed GL-space (object-space, `CAP_TRANSFORM`) vertex: `pos` is
/// `(x, y, z, w)` in object space, to be transformed by the current
/// modelview and projection matrices, then mapped through `VIEWPORT` and
/// `DEPTH_RANGE` -- unlike [`WinVertex::pos`], which already carries
/// final window pixels and `rhw`. `color`/`texcoord0`/`fogcoord` are the
/// same wire semantics as [`WinVertex`]'s (this milestone applies no
/// lighting or texgen, so they pass through exactly as parsed, same as
/// the window-space path).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlVertex {
    pub pos: [f32; 4], // object-space x, y, z, w
    pub color: [f32; 4],
    pub texcoord0: [f32; 2],
    pub fogcoord: f32,
}

/// Parses `count` GL-space vertices out of `data`, `format`-interleaved,
/// exactly as `DRAW_INLINE` lays them out -- see [`parse_vertices_raw`],
/// which this wraps. `NORMAL` is legal in this format (unlike
/// [`parse_window_vertices`]'s draw kind) but is not yet used: no
/// lighting is implemented this milestone, so its words are walked (to
/// keep later components at the right offset) and discarded, exactly
/// like `TEXCOORD1`-`3` already are.
pub fn parse_gl_vertices(
    data: &[u8],
    format: VertexFormat,
    count: u32,
    current: &state::CurrentVertex,
) -> Result<Vec<GlVertex>, RenderError> {
    Ok(parse_vertices_raw(data, format, count, current)?
        .into_iter()
        .map(|r| GlVertex {
            pos: r.pos,
            color: r.color,
            texcoord0: r.texcoord0,
            fogcoord: r.fogcoord,
        })
        .collect())
}

/// One triangle's three vertex indices into the vertex list a draw
/// produced, already **cyclically rotated so the provoking vertex is
/// first**. A cyclic rotation (not a swap) preserves winding, which is
/// what lets flat shading (wgpu/WGSL's `@interpolate(flat)`, which always
/// takes the *first* vertex of the primitive) line up with the spec's GL
/// provoking-vertex rule (the *last* vertex of each generated triangle,
/// except `TRIANGLES` itself, whose provoking vertex is the third) without
/// perturbing [`Capability::CullFace`]/[`state::FrontFace`] results, which
/// depend only on the cyclic order, not on which vertex is listed first.
pub type Triangle = [usize; 3];

/// Implements `docs/internals/c3d.md`'s "Primitive types" triangulation
/// table exactly, including the quad/quad-strip/polygon decomposition
/// order the spec calls **normative**. `POINTS`/`LINES`/`LINE_LOOP`/
/// `LINE_STRIP` have no triangle form and return
/// `Err(RenderError::Unimplemented)` -- assembling those primitives is
/// [`primitive_index_order`]'s job, which never calls this function for
/// them.
pub fn triangulate(prim: PrimitiveType, vertex_count: usize) -> Result<Vec<Triangle>, RenderError> {
    let n = vertex_count;
    match prim {
        PrimitiveType::Points
        | PrimitiveType::Lines
        | PrimitiveType::LineLoop
        | PrimitiveType::LineStrip => Err(RenderError::Unimplemented("point/line primitives")),
        PrimitiveType::Triangles => {
            let mut tris = Vec::new();
            let mut i = 0;
            while i + 3 <= n {
                // TRIANGLES: provoking vertex is the third; already first
                // in the sense that no rotation is needed relative to
                // itself -- rotate so v2 leads: (v2, v0, v1).
                tris.push([i + 2, i, i + 1]);
                i += 3;
            }
            Ok(tris)
        }
        PrimitiveType::TriangleStrip => {
            let mut tris = Vec::new();
            let mut i = 0;
            while i + 3 <= n {
                // Alternating winding; provoking vertex is the last of
                // each generated triangle (v(i+2)), so rotate it first.
                if (i % 2) == 0 {
                    tris.push([i + 2, i, i + 1]);
                } else {
                    tris.push([i + 2, i + 1, i]);
                }
                i += 1;
            }
            Ok(tris)
        }
        PrimitiveType::TriangleFan => {
            let mut tris = Vec::new();
            for i in 1..n.saturating_sub(1) {
                // Fan from v0; provoking vertex is the last (v(i+1)).
                tris.push([i + 1, 0, i]);
            }
            Ok(tris)
        }
        PrimitiveType::Quads => {
            let mut tris = Vec::new();
            let mut i = 0;
            while i + 4 <= n {
                let (v0, v1, v2, v3) = (i, i + 1, i + 2, i + 3);
                // (v0 v1 v2), (v0 v2 v3); flat colour of both is v3's.
                tris.push([v3, v0, v1]); // rotate v0,v1,v2 so v3... wait: provoking is v3, not in this tri.
                tris.push([v3, v0, v2]);
                i += 4;
            }
            Ok(tris)
        }
        PrimitiveType::QuadStrip => {
            let mut tris = Vec::new();
            let mut i = 0;
            while i + 4 <= n {
                let (v0, v1, v2, v3) = (i, i + 1, i + 2, i + 3);
                // (v0 v1 v3), (v0 v3 v2) per quad; flat colour from v3.
                tris.push([v3, v0, v1]);
                tris.push([v3, v0, v2]);
                i += 2;
            }
            Ok(tris)
        }
        PrimitiveType::Polygon => {
            let mut tris = Vec::new();
            for i in 1..n.saturating_sub(1) {
                // Fan from v0; flat colour from vn (the last vertex of the
                // whole polygon, not of each generated triangle).
                tris.push([n - 1, 0, i]);
            }
            Ok(tris)
        }
    }
}

/// Top-level primitive assembly: dispatches `TRIANGLES`-family primitives
/// to [`triangulate`], and expands `POINTS`/`LINES`/`LINE_LOOP`/
/// `LINE_STRIP` into the flat, already-duplicated vertex-index order the
/// chosen `wgpu` *List* topology draws directly -- no primitive-restart
/// index buffer, matching [`PrimTopologyKey`]'s doc comment. `LINE_STRIP`/
/// `LINE_LOOP` decompose into `LineList` segments the same normative way
/// `triangulate` decomposes quads/fans (`LINE_LOOP` additionally closes
/// with a segment back to vertex 0); `POINTS` maps to `PointList`
/// one-for-one. Each line segment's two indices are emitted **second
/// vertex first**, so `wgpu`/WGSL's `@interpolate(flat)` (which always
/// takes the primitive's first vertex) lines up with GL's provoking-
/// vertex rule for line primitives (the second vertex of each segment) --
/// the same reasoning [`Triangle`]'s own doc comment gives for triangles.
/// A count that does not complete the last primitive drops the
/// incomplete one, per GL (matching `triangulate`'s own rule).
pub fn primitive_index_order(
    prim: PrimitiveType,
    vertex_count: usize,
) -> Result<(PrimTopologyKey, Vec<usize>), RenderError> {
    let n = vertex_count;
    match prim {
        PrimitiveType::Points => Ok((PrimTopologyKey::Points, (0..n).collect())),
        PrimitiveType::Lines => {
            let mut idx = Vec::new();
            let mut i = 0;
            while i + 2 <= n {
                idx.push(i + 1);
                idx.push(i);
                i += 2;
            }
            Ok((PrimTopologyKey::Lines, idx))
        }
        PrimitiveType::LineStrip => {
            let mut idx = Vec::new();
            for i in 0..n.saturating_sub(1) {
                idx.push(i + 1);
                idx.push(i);
            }
            Ok((PrimTopologyKey::Lines, idx))
        }
        PrimitiveType::LineLoop => {
            let mut idx = Vec::new();
            for i in 0..n.saturating_sub(1) {
                idx.push(i + 1);
                idx.push(i);
            }
            if n >= 2 {
                idx.push(0);
                idx.push(n - 1);
            }
            Ok((PrimTopologyKey::Lines, idx))
        }
        _ => Ok((
            PrimTopologyKey::Triangles,
            triangulate(prim, n)?.into_iter().flatten().collect(),
        )),
    }
}

// ---------------------------------------------------------------------
// Pipeline specialisation
// ---------------------------------------------------------------------

/// The state a render pipeline is specialised over. Two draws with
/// identical keys share a pipeline; [`Renderer::pipeline_for`] builds and
/// caches one lazily per distinct key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub textured: bool,
    pub tex_env: TexEnvModeKey,
    pub flat_shading: bool,
    pub alpha_test: Option<CompareFuncKey>,
    pub depth_test: Option<CompareFuncKey>,
    pub depth_write: bool,
    pub blend: bool,
    pub blend_src: BlendFactorKey,
    pub blend_dst: BlendFactorKey,
    pub blend_src_alpha: BlendFactorKey,
    pub blend_dst_alpha: BlendFactorKey,
    pub blend_equation: BlendEquationKey,
    pub cull: Option<FaceKey>,
    pub front_face_ccw: bool,
    pub color_write: (bool, bool, bool, bool),
    pub surface_format: TextureFormatKey,
    pub polygon_offset: Option<PolygonOffsetKey>,
    pub topology: PrimTopologyKey,
    pub fog: Option<FogKey>,
}

// wgpu's own enums don't implement `Hash`/`Eq` in a way this module wants
// to depend on staying stable, and `state.rs`'s typed enums are `Copy` but
// not `Hash`; small local mirrors keep `PipelineKey` cheaply hashable
// without either dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TexEnvModeKey {
    Modulate,
    Replace,
    Decal,
    Blend,
    Add,
}
impl From<TexEnvMode> for TexEnvModeKey {
    fn from(m: TexEnvMode) -> Self {
        match m {
            TexEnvMode::Modulate => Self::Modulate,
            TexEnvMode::Replace => Self::Replace,
            TexEnvMode::Decal => Self::Decal,
            TexEnvMode::Blend => Self::Blend,
            TexEnvMode::Add => Self::Add,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareFuncKey {
    Never,
    Less,
    Equal,
    Lequal,
    Greater,
    Notequal,
    Gequal,
    Always,
}
impl From<CompareFunc> for CompareFuncKey {
    fn from(f: CompareFunc) -> Self {
        match f {
            CompareFunc::Never => Self::Never,
            CompareFunc::Less => Self::Less,
            CompareFunc::Equal => Self::Equal,
            CompareFunc::Lequal => Self::Lequal,
            CompareFunc::Greater => Self::Greater,
            CompareFunc::Notequal => Self::Notequal,
            CompareFunc::Gequal => Self::Gequal,
            CompareFunc::Always => Self::Always,
        }
    }
}
impl CompareFuncKey {
    fn to_wgpu(self) -> wgpu::CompareFunction {
        match self {
            Self::Never => wgpu::CompareFunction::Never,
            Self::Less => wgpu::CompareFunction::Less,
            Self::Equal => wgpu::CompareFunction::Equal,
            Self::Lequal => wgpu::CompareFunction::LessEqual,
            Self::Greater => wgpu::CompareFunction::Greater,
            Self::Notequal => wgpu::CompareFunction::NotEqual,
            Self::Gequal => wgpu::CompareFunction::GreaterEqual,
            Self::Always => wgpu::CompareFunction::Always,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendFactorKey {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    DstColor,
    OneMinusDstColor,
    SrcAlphaSaturate,
}
impl From<BlendFactor> for BlendFactorKey {
    fn from(f: BlendFactor) -> Self {
        match f {
            BlendFactor::Zero => Self::Zero,
            BlendFactor::One => Self::One,
            BlendFactor::SrcColor => Self::SrcColor,
            BlendFactor::OneMinusSrcColor => Self::OneMinusSrcColor,
            BlendFactor::SrcAlpha => Self::SrcAlpha,
            BlendFactor::OneMinusSrcAlpha => Self::OneMinusSrcAlpha,
            BlendFactor::DstAlpha => Self::DstAlpha,
            BlendFactor::OneMinusDstAlpha => Self::OneMinusDstAlpha,
            BlendFactor::DstColor => Self::DstColor,
            BlendFactor::OneMinusDstColor => Self::OneMinusDstColor,
            BlendFactor::SrcAlphaSaturate => Self::SrcAlphaSaturate,
        }
    }
}
impl BlendFactorKey {
    fn to_wgpu(self) -> wgpu::BlendFactor {
        match self {
            Self::Zero => wgpu::BlendFactor::Zero,
            Self::One => wgpu::BlendFactor::One,
            Self::SrcColor => wgpu::BlendFactor::Src,
            Self::OneMinusSrcColor => wgpu::BlendFactor::OneMinusSrc,
            Self::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
            Self::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
            Self::DstAlpha => wgpu::BlendFactor::DstAlpha,
            Self::OneMinusDstAlpha => wgpu::BlendFactor::OneMinusDstAlpha,
            Self::DstColor => wgpu::BlendFactor::Dst,
            Self::OneMinusDstColor => wgpu::BlendFactor::OneMinusDst,
            Self::SrcAlphaSaturate => wgpu::BlendFactor::SrcAlphaSaturated,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendEquationKey {
    FuncAdd,
    Min,
    Max,
    FuncSubtract,
    FuncReverseSubtract,
}
impl From<BlendEquation> for BlendEquationKey {
    fn from(e: BlendEquation) -> Self {
        match e {
            BlendEquation::FuncAdd => Self::FuncAdd,
            BlendEquation::Min => Self::Min,
            BlendEquation::Max => Self::Max,
            BlendEquation::FuncSubtract => Self::FuncSubtract,
            BlendEquation::FuncReverseSubtract => Self::FuncReverseSubtract,
        }
    }
}
impl BlendEquationKey {
    fn to_wgpu(self) -> wgpu::BlendOperation {
        match self {
            Self::FuncAdd => wgpu::BlendOperation::Add,
            Self::Min => wgpu::BlendOperation::Min,
            Self::Max => wgpu::BlendOperation::Max,
            Self::FuncSubtract => wgpu::BlendOperation::Subtract,
            Self::FuncReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        }
    }
}

/// `POLYGON_OFFSET`'s (factor, units), stored as bit patterns so the key
/// stays `Eq`/`Hash` without pulling in an ordered-float dependency --
/// these values are only ever compared for exact pipeline-cache identity,
/// never ordered or arithmetic-compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PolygonOffsetKey {
    factor_bits: u32,
    units_bits: u32,
}
impl PolygonOffsetKey {
    fn new(factor: f32, units: f32) -> Self {
        Self {
            factor_bits: factor.to_bits(),
            units_bits: units.to_bits(),
        }
    }
    /// The direct GL -> wgpu mapping every GL-on-Vulkan/D3D layer uses:
    /// `factor` becomes the slope-scaled term, `units` becomes the
    /// constant term in depth-format basic units. GL leaves the constant
    /// term's absolute scale implementation-defined, so this is a
    /// reasonable rendering of the spec's intent, not an exact formula.
    fn to_wgpu(self) -> wgpu::DepthBiasState {
        wgpu::DepthBiasState {
            constant: f32::from_bits(self.units_bits).round() as i32,
            slope_scale: f32::from_bits(self.factor_bits),
            clamp: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FogModeKey {
    Linear,
    Exp,
    Exp2,
}
impl From<FogMode> for FogModeKey {
    fn from(m: FogMode) -> Self {
        match m {
            FogMode::Linear => Self::Linear,
            FogMode::Exp => Self::Exp,
            FogMode::Exp2 => Self::Exp2,
        }
    }
}

/// `FOG_MODE`/`FOG_PARAMS`/`FOG_COLOR`, baked as WGSL literals (see
/// [`VERTEX_SHADER`]'s doc comment for why) -- every field is stored as a
/// bit pattern purely so this stays `Eq`/`Hash` for the pipeline cache
/// key, the same reason [`PolygonOffsetKey`] does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FogKey {
    mode: FogModeKey,
    density_bits: u32,
    start_bits: u32,
    end_bits: u32,
    color_bits: [u32; 3], // r, g, b only -- fog never touches alpha
}
impl FogKey {
    fn new(mode: FogModeKey, density: f32, start: f32, end: f32, color: [f32; 4]) -> Self {
        Self {
            mode,
            density_bits: density.to_bits(),
            start_bits: start.to_bits(),
            end_bits: end.to_bits(),
            color_bits: [color[0].to_bits(), color[1].to_bits(), color[2].to_bits()],
        }
    }
}

/// The three `wgpu` "list" topologies this module ever draws with --
/// strip/loop primitives are decomposed to one of these (the same
/// normative-decomposition approach [`triangulate`] already takes for
/// quads/fans) rather than using `wgpu`'s native strip topologies, so no
/// primitive-restart index buffer is ever needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrimTopologyKey {
    #[default]
    Triangles,
    Lines,
    Points,
}
impl PrimTopologyKey {
    fn to_wgpu(self) -> wgpu::PrimitiveTopology {
        match self {
            Self::Triangles => wgpu::PrimitiveTopology::TriangleList,
            Self::Lines => wgpu::PrimitiveTopology::LineList,
            Self::Points => wgpu::PrimitiveTopology::PointList,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaceKey {
    Front,
    Back,
    FrontAndBack,
}
impl From<Face> for FaceKey {
    fn from(f: Face) -> Self {
        match f {
            Face::Front => Self::Front,
            Face::Back => Self::Back,
            Face::FrontAndBack => Self::FrontAndBack,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureFormatKey(pub u32);
impl From<wgpu::TextureFormat> for TextureFormatKey {
    fn from(f: wgpu::TextureFormat) -> Self {
        // A textual discriminant is enough to key a cache; wgpu's format
        // enum doesn't implement `Hash`.
        TextureFormatKey(format!("{f:?}").len() as u32 ^ format_tag(f))
    }
}
fn format_tag(f: wgpu::TextureFormat) -> u32 {
    // Cheap, stable-enough-for-a-cache-key fold of the debug string.
    format!("{f:?}")
        .bytes()
        .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32))
}

/// `FRONT_FACE`'s window-space meaning: the spec evaluates winding "in
/// window space with y down", and this maps across **unchanged**.
///
/// It is tempting to invert it, because the vertex stage does flip y when
/// it maps y-down window space to wgpu's y-up NDC. But WebGPU decides
/// facing from the signed area of the triangle in **framebuffer** space,
/// not in NDC, and the viewport transform flips y a second time on the
/// way there (`fb_y = (1 - ndc_y) / 2 * height`). The two flips cancel:
/// a window-space pixel coordinate lands at the identical framebuffer
/// coordinate, so a triangle's winding in the spec's y-down window space
/// *is* its winding in framebuffer space. An inverted mapping culls
/// exactly the faces it should keep, and does so silently -- nothing
/// renders differently until something enables culling.
///
/// `a_front_facing_triangle_survives_back_face_culling` pins this down
/// on a real device rather than by argument, because the reasoning above
/// is easy to get wrong in either direction.
fn front_face_to_wgpu(f: FrontFace) -> wgpu::FrontFace {
    match f {
        FrontFace::Ccw => wgpu::FrontFace::Ccw,
        FrontFace::Cw => wgpu::FrontFace::Cw,
    }
}

fn cull_to_wgpu(face: Option<FaceKey>) -> Option<wgpu::Face> {
    match face? {
        FaceKey::Front => Some(wgpu::Face::Front),
        FaceKey::Back => Some(wgpu::Face::Back),
        // wgpu has no "cull both"; the caller (draw submission) skips the
        // draw entirely in that case instead of relying on this mapping.
        FaceKey::FrontAndBack => None,
    }
}

// ---------------------------------------------------------------------
// WGSL shader template
// ---------------------------------------------------------------------

/// The shared shader source every [`PipelineKey`] specialises via string
/// substitution of `__FLAT__` (either empty or `@interpolate(flat)`) and
/// `__FRAGMENT_BODY__` (the texenv/alpha-test/fog combination). M1's
/// fixed-function content is deliberately thin (no lighting -- see the
/// module doc comment); the specialisation mechanism is what carries
/// forward to M3, not the amount of WGSL below it. Fog's own parameters
/// (mode, density, start, end, colour) are baked as WGSL literals per
/// [`FogKey`] rather than passed through a uniform, the same way
/// `alpha_test`'s comparison *operator* is baked while only its
/// threshold value gets a uniform -- unlike the threshold, a guest
/// changing fog parameters mid-scene is rare enough that the resulting
/// pipeline-cache churn is an acceptable simplification for now.
const VERTEX_SHADER: &str = r#"
struct VsIn {
    @location(0) clip_pos: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) texcoord0: vec2<f32>,
    @location(3) fog_distance: f32,
};
struct VsOut {
    @builtin(position) position: vec4<f32>,
    __FLAT__ @location(0) color: vec4<f32>,
    @location(1) texcoord0: vec2<f32>,
    @location(2) fog_distance: f32,
};
@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.position = in.clip_pos;
    out.color = in.color;
    out.texcoord0 = in.texcoord0;
    out.fog_distance = in.fog_distance;
    return out;
}
"#;

const FRAGMENT_SHADER_HEADER: &str = r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
    __FLAT__ @location(0) color: vec4<f32>,
    @location(1) texcoord0: vec2<f32>,
    @location(2) fog_distance: f32,
};
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
"#;

/// Builds the fragment shader body for one [`PipelineKey`]: samples the
/// bound texture (if `textured`) and combines it with the interpolated
/// colour per `tex_env`, then applies the alpha test (if any) via
/// `discard`.
fn fragment_shader_source(key: &PipelineKey) -> String {
    let sample = if key.textured {
        "let texel = textureSample(tex, samp, in.texcoord0);"
    } else {
        "let texel = vec4<f32>(1.0, 1.0, 1.0, 1.0);"
    };
    let combine = if !key.textured {
        "var outc = in.color;"
    } else {
        match key.tex_env {
            TexEnvModeKey::Replace => "var outc = texel;",
            TexEnvModeKey::Decal => {
                "var outc = vec4<f32>(mix(in.color.rgb, texel.rgb, texel.a), in.color.a);"
            }
            TexEnvModeKey::Blend => {
                "var outc = vec4<f32>(in.color.rgb * (1.0 - texel.rgb), in.color.a * texel.a);"
            }
            TexEnvModeKey::Add => {
                "var outc = vec4<f32>(in.color.rgb + texel.rgb, in.color.a * texel.a);"
            }
            TexEnvModeKey::Modulate => "var outc = in.color * texel;",
        }
    };
    let alpha_test = match key.alpha_test {
        None => String::new(),
        Some(func) => {
            let op = match func {
                CompareFuncKey::Never => "false",
                CompareFuncKey::Less => "outc.a < alpha_ref",
                CompareFuncKey::Equal => "outc.a == alpha_ref",
                CompareFuncKey::Lequal => "outc.a <= alpha_ref",
                CompareFuncKey::Greater => "outc.a > alpha_ref",
                CompareFuncKey::Notequal => "outc.a != alpha_ref",
                CompareFuncKey::Gequal => "outc.a >= alpha_ref",
                CompareFuncKey::Always => "true",
            };
            format!("if (!({op})) {{ discard; }}\n",)
        }
    };
    let alpha_ref_decl = if key.alpha_test.is_some() {
        "@group(0) @binding(2) var<uniform> alpha_ref: f32;\n"
    } else {
        ""
    };
    // GL's real fixed-function order (texturing, then alpha test, then
    // fog) is followed here too, though it's only a correctness
    // difference for a discarded fragment -- fog never touches alpha, so
    // it can't change the alpha test's own outcome either way.
    let fog = match key.fog {
        None => String::new(),
        Some(fk) => {
            let density = f32::from_bits(fk.density_bits);
            let start = f32::from_bits(fk.start_bits);
            let end = f32::from_bits(fk.end_bits);
            let [r, g, b] = fk.color_bits.map(f32::from_bits);
            let factor = match fk.mode {
                FogModeKey::Linear => {
                    format!("clamp(({end:?} - dist) / ({end:?} - {start:?}), 0.0, 1.0)")
                }
                FogModeKey::Exp => format!("clamp(exp(-{density:?} * dist), 0.0, 1.0)"),
                FogModeKey::Exp2 => {
                    format!("clamp(exp(-pow({density:?} * dist, 2.0)), 0.0, 1.0)")
                }
            };
            format!(
                "let dist = in.fog_distance;\n    let fog_f = {factor};\n    outc = vec4<f32>(mix(vec3<f32>({r:?}, {g:?}, {b:?}), outc.rgb, fog_f), outc.a);\n"
            )
        }
    };
    let flat = if key.flat_shading {
        "@interpolate(flat)"
    } else {
        ""
    };
    let header = FRAGMENT_SHADER_HEADER.replace("__FLAT__", flat);
    format!(
        "{header}\n{alpha_ref_decl}@fragment\nfn fs_main(in: VsOut) -> @location(0) vec4<f32> {{\n    {sample}\n    {combine}\n    {alpha_test}    {fog}    return outc;\n}}\n",
    )
}

fn vertex_shader_source(flat: bool) -> String {
    VERTEX_SHADER.replace("__FLAT__", if flat { "@interpolate(flat)" } else { "" })
}

// ---------------------------------------------------------------------
// GPU-side objects
// ---------------------------------------------------------------------

struct GpuSurface {
    def: Surface,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    // Kept alive for `depth_view`'s sake (a `TextureView` borrows its
    // texture at the API level even though wgpu's handle is `Clone`);
    // never read directly, since every depth op goes through the view.
    #[allow(dead_code)]
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

struct GpuTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// The wgpu renderer. Owns its own headless device -- never a window's.
pub struct Renderer {
    _instance: wgpu::Instance,
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sampler_cache: HashMap<(wgpu::AddressMode, wgpu::AddressMode, wgpu::FilterMode), wgpu::Sampler>,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    bind_group_layout_textured: wgpu::BindGroupLayout,
    bind_group_layout_textured_alpha: wgpu::BindGroupLayout,
    /// One uniform (the clear colour + depth) -- see
    /// [`ClearRectPipelineKey`] and [`Renderer::op_clear`].
    bind_group_layout_clear_rect: wgpu::BindGroupLayout,
    surfaces: HashMap<u32, GpuSurface>,
    textures: HashMap<u32, GpuTexture>,
    clear_rect_pipelines: HashMap<ClearRectPipelineKey, wgpu::RenderPipeline>,
}

/// Ranks a [`wgpu::AdapterInfo`]'s device type for adapter selection:
/// lower is preferred. Hardware first, software (CI's normal case) last
/// but still accepted -- never treated as a failure.
fn adapter_rank(info: &wgpu::AdapterInfo) -> u8 {
    match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    }
}

/// A scissored/partial-colour-mask `CLEAR` is done as a degenerate draw
/// (a full-viewport triangle, clipped by the render pass's own scissor
/// rect) rather than `wgpu`'s pass-wide `LoadOp::Clear`, which always
/// covers the whole attachment and cannot be masked per channel either.
/// This is a separate, much smaller pipeline family from [`PipelineKey`]'s
/// -- no texturing, no vertex buffer, no blending, a fixed depth compare
/// -- specialised only over what a clear actually varies: which colour
/// channels land, whether depth is written, and the surface's format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ClearRectPipelineKey {
    color_write: (bool, bool, bool, bool),
    depth_write: bool,
    surface_format: TextureFormatKey,
}

/// Uniform bytes for the clear-rect shader: the clear colour and the
/// clear depth, packed to a 16-byte-aligned 32-byte struct matching
/// `CLEAR_RECT_SHADER`'s `ClearUniforms` exactly (`vec4<f32>` colour,
/// `f32` depth, 12 bytes of tail padding -- a uniform buffer's total size
/// must be a multiple of its largest member's alignment, 16 here).
fn clear_rect_uniform_bytes(color: [f32; 4], depth: f32) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, c) in color.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&c.to_le_bytes());
    }
    out[16..20].copy_from_slice(&depth.to_le_bytes());
    out
}

/// A full-viewport triangle from `@builtin(vertex_index)` alone (no
/// vertex buffer needed), the same construction
/// `src/video/window/rtg_texture.rs`'s own fullscreen pass uses. The
/// render pass's scissor rect (set by the caller to either the whole
/// surface or the active `SCISSOR` rectangle) is what actually restricts
/// which pixels this touches; the triangle itself always covers the
/// whole clip-space square. `u.depth` is written as clip-space `z` with
/// `w = 1`, so it lands unchanged as the fragment's depth -- this shader
/// works entirely in wgpu's own `[0, 1]` depth convention, unlike the
/// GL-space (M3 transform tier) path, which has an extra `[-1, 1]` fixup
/// to do first (see [`gl_clip_z_to_wgpu`]).
const CLEAR_RECT_SHADER: &str = r#"
struct ClearUniforms {
    color: vec4<f32>,
    depth: f32,
};
@group(0) @binding(0) var<uniform> u: ClearUniforms;

struct VOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VOut {
    let tc = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VOut;
    let ndc = tc * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
    out.pos = vec4<f32>(ndc, u.depth, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return u.color;
}
"#;

impl Renderer {
    /// Creates a headless renderer: no window, no surface. Prefers a
    /// hardware adapter, falls back to software (lavapipe on Linux, WARP
    /// on Windows) -- the expected case in CI, never an error on its own.
    /// Only [`RenderError::NoAdapter`]/[`RenderError::DeviceRequest`] fail
    /// this, never a panic.
    pub fn new() -> Result<Renderer, RenderError> {
        let instance = wgpu::Instance::default();

        let mut adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        adapters.sort_by_key(|a| adapter_rank(&a.get_info()));
        let adapter = if let Some(a) = adapters.into_iter().next() {
            a
        } else {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::None,
                force_fallback_adapter: true,
                compatible_surface: None,
            }))
            .map_err(|_| RenderError::NoAdapter)?
        };

        let adapter_info = adapter.get_info();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("c3d headless device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            ..Default::default()
        }))
        .map_err(|e| RenderError::DeviceRequest(e.to_string()))?;

        let bind_group_layout_textured =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("c3d textured bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let bind_group_layout_textured_alpha =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("c3d textured+alpha-test bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group_layout_clear_rect =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("c3d clear-rect bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        Ok(Renderer {
            _instance: instance,
            adapter_info,
            device,
            queue,
            sampler_cache: HashMap::new(),
            pipelines: HashMap::new(),
            bind_group_layout_textured,
            bind_group_layout_textured_alpha,
            bind_group_layout_clear_rect,
            surfaces: HashMap::new(),
            textures: HashMap::new(),
            clear_rect_pipelines: HashMap::new(),
        })
    }

    /// A one-line description of the adapter this renderer picked --
    /// name, backend and whether it is hardware or software -- for the
    /// trace runner to print.
    pub fn describe(&self) -> String {
        let kind = match self.adapter_info.device_type {
            wgpu::DeviceType::Cpu => "software",
            wgpu::DeviceType::Other => "unknown",
            _ => "hardware",
        };
        format!(
            "{} ({:?} backend, {}, {:?})",
            self.adapter_info.name, self.adapter_info.backend, kind, self.adapter_info.device_type
        )
    }

    /// Whether the selected adapter is a software one (WARP, lavapipe,
    /// `llvmpipe`, ...). Purely informative.
    pub fn is_software(&self) -> bool {
        self.adapter_info.device_type == wgpu::DeviceType::Cpu
    }

    fn ensure_surface(&mut self, id: u32, def: &Surface) -> Result<&mut GpuSurface, RenderError> {
        let needs_create = match self.surfaces.get(&id) {
            Some(existing) => existing.def.width != def.width || existing.def.height != def.height,
            None => true,
        };
        if needs_create {
            let size = wgpu::Extent3d {
                width: def.width.max(1),
                height: def.height.max(1),
                depth_or_array_layers: 1,
            };
            let color = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("c3d surface color"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: INTERNAL_COLOR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
            let depth = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("c3d surface depth"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: INTERNAL_DEPTH_FORMAT,
                // COPY_SRC: READ_PIXELS(DEPTH) reads this back via
                // copy_texture_to_cpu, exactly like the colour texture's
                // own COPY_SRC above.
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
            self.surfaces.insert(
                id,
                GpuSurface {
                    def: *def,
                    color,
                    color_view,
                    depth,
                    depth_view,
                },
            );
        } else if let Some(existing) = self.surfaces.get_mut(&id) {
            existing.def = *def;
        }
        Ok(self
            .surfaces
            .get_mut(&id)
            .expect("just inserted or present"))
    }

    fn sampler(
        &mut self,
        wrap_s: wgpu::AddressMode,
        wrap_t: wgpu::AddressMode,
        filter: wgpu::FilterMode,
    ) -> &wgpu::Sampler {
        let key = (wrap_s, wrap_t, filter);
        self.sampler_cache.entry(key).or_insert_with(|| {
            self.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("c3d sampler"),
                address_mode_u: wrap_s,
                address_mode_v: wrap_t,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: filter,
                min_filter: filter,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                ..Default::default()
            })
        })
    }

    fn pipeline_for(&mut self, key: &PipelineKey) -> &wgpu::RenderPipeline {
        if !self.pipelines.contains_key(key) {
            let pipeline = self.build_pipeline(key);
            self.pipelines.insert(key.clone(), pipeline);
        }
        self.pipelines.get(key).expect("just inserted")
    }

    fn build_pipeline(&self, key: &PipelineKey) -> wgpu::RenderPipeline {
        let vs_src = vertex_shader_source(key.flat_shading);
        let fs_src = fragment_shader_source(key);
        let vs_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("c3d vertex shader"),
                source: wgpu::ShaderSource::Wgsl(vs_src.into()),
            });
        let fs_module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("c3d fragment shader"),
                source: wgpu::ShaderSource::Wgsl(fs_src.into()),
            });

        let layout = if key.textured && key.alpha_test.is_some() {
            Some(&self.bind_group_layout_textured_alpha)
        } else if key.textured {
            Some(&self.bind_group_layout_textured)
        } else {
            None
        };
        let layouts: &[Option<&wgpu::BindGroupLayout>] = match layout {
            Some(l) => &[Some(l)],
            None => &[],
        };
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("c3d pipeline layout"),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 32,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 40,
                    shader_location: 3,
                },
            ],
        };

        let color_write = {
            let (r, g, b, a) = key.color_write;
            let mut mask = wgpu::ColorWrites::empty();
            if r {
                mask |= wgpu::ColorWrites::RED;
            }
            if g {
                mask |= wgpu::ColorWrites::GREEN;
            }
            if b {
                mask |= wgpu::ColorWrites::BLUE;
            }
            if a {
                mask |= wgpu::ColorWrites::ALPHA;
            }
            mask
        };
        let blend = if key.blend {
            let operation = key.blend_equation.to_wgpu();
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: key.blend_src.to_wgpu(),
                    dst_factor: key.blend_dst.to_wgpu(),
                    operation,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: key.blend_src_alpha.to_wgpu(),
                    dst_factor: key.blend_dst_alpha.to_wgpu(),
                    operation,
                },
            })
        } else {
            None
        };

        let surface_format = wgpu::TextureFormat::Rgba8Unorm; // internal target format
        let _ = key.surface_format; // reserved: keyed for a future non-uniform internal format

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("c3d pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vs_module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[vertex_layout],
                },
                primitive: wgpu::PrimitiveState {
                    topology: key.topology.to_wgpu(),
                    strip_index_format: None,
                    front_face: front_face_to_wgpu(if key.front_face_ccw {
                        FrontFace::Ccw
                    } else {
                        FrontFace::Cw
                    }),
                    // GL only culls polygons: face culling never applies
                    // to POINTS/LINES-family primitives, whose two (or
                    // one) vertices don't have a meaningful winding.
                    cull_mode: if key.topology == PrimTopologyKey::Triangles {
                        cull_to_wgpu(key.cull)
                    } else {
                        None
                    },
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: INTERNAL_DEPTH_FORMAT,
                    depth_write_enabled: Some(key.depth_write),
                    depth_compare: Some(
                        key.depth_test
                            .map(CompareFuncKey::to_wgpu)
                            .unwrap_or(wgpu::CompareFunction::Always),
                    ),
                    stencil: wgpu::StencilState::default(),
                    bias: key
                        .polygon_offset
                        .map(PolygonOffsetKey::to_wgpu)
                        .unwrap_or_default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &fs_module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend,
                        write_mask: color_write,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    fn clear_rect_pipeline_for(&mut self, key: &ClearRectPipelineKey) -> &wgpu::RenderPipeline {
        if !self.clear_rect_pipelines.contains_key(key) {
            let pipeline = self.build_clear_rect_pipeline(key);
            self.clear_rect_pipelines.insert(*key, pipeline);
        }
        self.clear_rect_pipelines.get(key).expect("just inserted")
    }

    fn build_clear_rect_pipeline(&self, key: &ClearRectPipelineKey) -> wgpu::RenderPipeline {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("c3d clear-rect shader"),
                source: wgpu::ShaderSource::Wgsl(CLEAR_RECT_SHADER.into()),
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("c3d clear-rect pipeline layout"),
                bind_group_layouts: &[Some(&self.bind_group_layout_clear_rect)],
                immediate_size: 0,
            });

        let color_write = {
            let (r, g, b, a) = key.color_write;
            let mut mask = wgpu::ColorWrites::empty();
            if r {
                mask |= wgpu::ColorWrites::RED;
            }
            if g {
                mask |= wgpu::ColorWrites::GREEN;
            }
            if b {
                mask |= wgpu::ColorWrites::BLUE;
            }
            if a {
                mask |= wgpu::ColorWrites::ALPHA;
            }
            mask
        };
        let surface_format = wgpu::TextureFormat::Rgba8Unorm; // internal target format
        let _ = key.surface_format; // reserved, matches PipelineKey's own note

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("c3d clear-rect pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[], // no vertex buffer -- see CLEAR_RECT_SHADER's doc comment
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: INTERNAL_DEPTH_FORMAT,
                    depth_write_enabled: Some(key.depth_write),
                    // Always: a clear unconditionally overwrites whatever
                    // depth is already there within the scissor rect,
                    // never depth-tested against it.
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: surface_format,
                        blend: None,
                        write_mask: color_write,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    fn pipeline_key_from_state(
        &self,
        state: &State,
        unit0_bound: bool,
        surface_format: wgpu::TextureFormat,
        topology: PrimTopologyKey,
    ) -> PipelineKey {
        let raster = &state.raster;
        let alpha_test = if state.is_enabled(state::Capability::AlphaTest) {
            Some(raster.alpha_func.into())
        } else {
            None
        };
        let depth_test = if state.is_enabled(state::Capability::DepthTest) {
            Some(raster.depth_func.into())
        } else {
            None
        };
        let cull = if state.is_enabled(state::Capability::CullFace) {
            Some(raster.cull_face_mode.into())
        } else {
            None
        };
        let tex_env = state
            .texture_units
            .first()
            .map(|u| u.env_mode.into())
            .unwrap_or(TexEnvModeKey::Modulate);
        PipelineKey {
            textured: unit0_bound && state.enables.texture_2d.first().copied().unwrap_or(false),
            tex_env,
            flat_shading: raster.shade_model == ShadeModel::Flat,
            alpha_test,
            depth_test,
            depth_write: raster.depth_mask,
            blend: state.is_enabled(state::Capability::Blend),
            blend_src: raster.blend_src_rgb.into(),
            blend_dst: raster.blend_dst_rgb.into(),
            blend_src_alpha: raster.blend_src_alpha.into(),
            blend_dst_alpha: raster.blend_dst_alpha.into(),
            blend_equation: raster.blend_equation.into(),
            cull,
            front_face_ccw: raster.front_face == FrontFace::Ccw,
            color_write: (
                raster.color_mask.r,
                raster.color_mask.g,
                raster.color_mask.b,
                raster.color_mask.a,
            ),
            surface_format: surface_format.into(),
            polygon_offset: if state.is_enabled(state::Capability::PolygonOffsetFill) {
                Some(PolygonOffsetKey::new(
                    raster.polygon_offset_factor,
                    raster.polygon_offset_units,
                ))
            } else {
                None
            },
            topology,
            fog: if state.is_enabled(state::Capability::Fog) {
                Some(FogKey::new(
                    raster.fog_mode.into(),
                    raster.fog_density,
                    raster.fog_start,
                    raster.fog_end,
                    raster.fog_color,
                ))
            } else {
                None
            },
        }
    }

    // -- Op execution -------------------------------------------------

    /// Executes every op in `ops` in order against `state`, using `mem` to
    /// resolve refs and surface backing memory. Returns the errors any
    /// individual op raised (an unimplemented op, a bad ref, ...);
    /// execution continues past an error exactly like the spec's "skip"
    /// error actions, since a conformance run wants to see every failure
    /// in one pass rather than stopping at the first.
    pub fn execute(
        &mut self,
        ops: &[RenderOp<'_>],
        state: &State,
        mem: &mut dyn Memory,
    ) -> Vec<RenderError> {
        let mut errors = Vec::new();
        for op in ops {
            if let Err(e) = self.execute_op(op, state, mem) {
                errors.push(e);
            }
        }
        errors
    }

    fn execute_op(
        &mut self,
        op: &RenderOp<'_>,
        state: &State,
        mem: &mut dyn Memory,
    ) -> Result<(), RenderError> {
        match op {
            RenderOp::Fence { .. } => Ok(()), // no GPU work; the caller tracks completion
            RenderOp::Clear { mask } => self.op_clear(*mask, state),
            RenderOp::Draw {
                window_space,
                prim,
                format,
                count,
                vertices,
            } => {
                let DrawVertices::Inline(bytes) = vertices else {
                    return Err(RenderError::Unimplemented(
                        "DRAW_ARRAYS(_WIN)/DRAW_ELEMENTS(_WIN) (M3)",
                    ));
                };
                if *window_space {
                    self.op_draw_inline_win(*prim, *format, *count, bytes, state)
                } else {
                    self.op_draw_inline_gl(*prim, *format, *count, bytes, state)
                }
            }
            RenderOp::TexImage {
                id,
                level,
                format,
                width,
                height,
                row_bytes,
                data,
            } => self.op_tex_image(
                *id, *level, *format, *width, *height, *row_bytes, *data, mem,
            ),
            RenderOp::TexSubImage {
                id,
                level,
                x,
                y,
                width,
                height,
                format,
                row_bytes,
                data,
            } => self.op_tex_subimage(
                *id, *level, *x, *y, *width, *height, *format, *row_bytes, *data, mem,
            ),
            RenderOp::SurfaceReadback { x, y, w, h } => {
                self.op_surface_readback(*x, *y, *w, *h, state, mem)
            }
            RenderOp::SurfaceUpload { x, y, w, h } => {
                self.op_surface_upload(*x, *y, *w, *h, state, mem)
            }
            RenderOp::TexCopyImage {
                id,
                level,
                format: _,
                x,
                y,
                width,
                height,
            } => self.op_tex_copy_image(*id, *level, *x, *y, *width, *height, state),
            RenderOp::TexCopySubImage {
                id,
                level,
                xoff,
                yoff,
                x,
                y,
                width,
                height,
            } => {
                self.op_tex_copy_subimage(*id, *level, *xoff, *yoff, *x, *y, *width, *height, state)
            }
            RenderOp::TexPalette { .. } => Err(RenderError::Unimplemented("TEX_PALETTE (M3)")),
            RenderOp::ReadPixels {
                x,
                y,
                w,
                h,
                format,
                row_bytes,
                flags,
                dest,
            } => self.op_read_pixels(
                *x, *y, *w, *h, *format, *row_bytes, *flags, *dest, state, mem,
            ),
            RenderOp::Query { dest, result } => self.op_query(*dest, *result, mem),
        }
    }

    fn op_clear(&mut self, mask: u32, state: &State) -> Result<(), RenderError> {
        let id = state.draw_surface();
        let def = *state
            .surface(id)
            .ok_or(RenderError::UnknownObject("surface", id))?;
        let gpu = self.ensure_surface(id, &def)?;
        let color_view = gpu.color_view.clone();
        let depth_view = gpu.depth_view.clone();

        let mask_full = state.raster.color_mask.r
            && state.raster.color_mask.g
            && state.raster.color_mask.b
            && state.raster.color_mask.a;
        // Real GL only lets SCISSOR restrict a clear when GL_SCISSOR_TEST
        // is enabled, and only within the surface's own bounds --
        // `clamped_scissor_rect` returns `None` for exactly the cases
        // that need no restriction (test off, or a rect that already
        // covers everything -- `set_draw_surface` resets scissor to the
        // whole surface, so an untouched scissor is "full" too).
        let scissor_rect = Self::clamped_scissor_rect(state, def.width, def.height);

        // `wgpu::LoadOp::Clear` covers the whole attachment and cannot be
        // masked per colour channel, so anything a scissor or a partial
        // colour mask would need to *not* touch is instead handled by a
        // second, scissored draw pass below (`ClearRectPipelineKey`) --
        // colour and depth independently, since a scissor without a
        // colour mask still wants the cheap whole-attachment depth clear
        // whenever there is no scissor to restrict it by.
        let color_needs_draw =
            mask & proto::CLEAR_MASK_COLOR != 0 && (scissor_rect.is_some() || !mask_full);
        let depth_needs_draw = mask & proto::CLEAR_MASK_DEPTH != 0 && scissor_rect.is_some();

        let [r, g, b, a] = state.raster.clear_color;
        let load_color = if mask & proto::CLEAR_MASK_COLOR != 0 && !color_needs_draw {
            wgpu::LoadOp::Clear(wgpu::Color {
                r: r as f64,
                g: g as f64,
                b: b as f64,
                a: a as f64,
            })
        } else {
            wgpu::LoadOp::Load
        };
        let load_depth = if mask & proto::CLEAR_MASK_DEPTH != 0 && !depth_needs_draw {
            wgpu::LoadOp::Clear(state.raster.clear_depth)
        } else {
            wgpu::LoadOp::Load
        };

        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("c3d clear"),
                });
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("c3d clear pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: load_color,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: load_depth,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            self.queue.submit(Some(encoder.finish()));
        }

        if color_needs_draw || depth_needs_draw {
            let key = ClearRectPipelineKey {
                color_write: if color_needs_draw {
                    (
                        state.raster.color_mask.r,
                        state.raster.color_mask.g,
                        state.raster.color_mask.b,
                        state.raster.color_mask.a,
                    )
                } else {
                    (false, false, false, false)
                },
                depth_write: depth_needs_draw,
                surface_format: wgpu::TextureFormat::Rgba8Unorm.into(),
            };
            self.clear_rect_pipeline_for(&key);
            let pipeline = self.clear_rect_pipelines.get(&key).expect("just ensured");

            let uniform_bytes = clear_rect_uniform_bytes(
                [r, g, b, a],
                if depth_needs_draw {
                    state.raster.clear_depth
                } else {
                    0.0
                },
            );
            let uniform_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("c3d clear-rect uniforms"),
                        contents: &uniform_bytes,
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("c3d clear-rect bind group"),
                layout: &self.bind_group_layout_clear_rect,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("c3d clear-rect"),
                });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("c3d clear-rect pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(pipeline);
                if let Some((x, y, sw, sh)) = scissor_rect {
                    pass.set_scissor_rect(x, y, sw, sh);
                }
                pass.set_bind_group(0, &bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            self.queue.submit(Some(encoder.finish()));
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// The active `SCISSOR` rect, clamped to the surface and to
    /// non-negative device coordinates, or `None` when `SCISSOR_TEST` is
    /// off or the rect is empty -- in either case nothing should be
    /// scissored, i.e. the caller draws (or clears) unrestricted. Shared
    /// by ordinary scissor-tested draws and [`Renderer::op_clear`]'s
    /// scissored-clear path, so the two can never clamp differently.
    fn clamped_scissor_rect(
        state: &State,
        width: u32,
        height: u32,
    ) -> Option<(u32, u32, u32, u32)> {
        let scissor = state.raster.scissor;
        if scissor.w == 0 || scissor.h == 0 || !state.is_enabled(state::Capability::ScissorTest) {
            return None;
        }
        let x = scissor.x.max(0) as u32;
        let y = scissor.y.max(0) as u32;
        let sw = scissor.w.min(width.saturating_sub(x));
        let sh = scissor.h.min(height.saturating_sub(y));
        (sw > 0 && sh > 0).then_some((x, y, sw, sh))
    }

    fn op_draw_inline_win(
        &mut self,
        prim: PrimitiveType,
        format: VertexFormat,
        count: u32,
        bytes: &[u8],
        state: &State,
    ) -> Result<(), RenderError> {
        let id = state.draw_surface();
        let def = *state
            .surface(id)
            .ok_or(RenderError::UnknownObject("surface", id))?;

        let verts = parse_window_vertices(bytes, format, count, &state.current)?;
        let (topology, order) = primitive_index_order(prim, verts.len())?;
        if order.is_empty() {
            return Ok(());
        }

        let (w, h) = (def.width as f32, def.height as f32);
        // Fog distance is a format-wide decision, not a per-vertex one
        // (the format bits are the same for every vertex in a draw): the
        // spec's window-space rule is "FOGCOORD if present, else derived
        // from rhw" -- CURRENT_FOGCOORD's own "omitted takes the current
        // value" fallback (already resolved into `v.fogcoord` by
        // `parse_window_vertices`) only actually applies to fog distance
        // when the format DOES carry FOGCOORD (an omitted vertex within
        // such a draw still gets `current.fogcoord`, per that rule); an
        // entirely FOGCOORD-less draw ignores it altogether and always
        // derives distance from `1 / rhw` instead.
        let has_fogcoord = format.has(VertexFormat::FOGCOORD);
        let to_gpu = |v: &WinVertex| -> GpuVertex {
            let rhw = if v.pos[3].abs() < 1e-8 { 1.0 } else { v.pos[3] };
            let clip_w = 1.0 / rhw;
            let ndc_x = (v.pos[0] / w) * 2.0 - 1.0;
            let ndc_y = 1.0 - (v.pos[1] / h) * 2.0;
            let ndc_z = v.pos[2];
            GpuVertex {
                clip_pos: [ndc_x * clip_w, ndc_y * clip_w, ndc_z * clip_w, clip_w],
                color: v.color,
                texcoord0: v.texcoord0,
                fog_distance: if has_fogcoord { v.fogcoord } else { clip_w },
            }
        };

        let mut vbuf: Vec<GpuVertex> = Vec::with_capacity(order.len());
        for idx in order {
            let Some(v) = verts.get(idx) else {
                return Err(RenderError::ShortPayload("primitive vertex index"));
            };
            vbuf.push(to_gpu(v));
        }

        self.submit_draw(state, id, def, topology, vbuf)
    }

    /// The GL-space (object-space, `CAP_TRANSFORM`) counterpart of
    /// [`Self::op_draw_inline_win`]: **position transform only** this
    /// milestone -- no lighting, texgen or user clipping (see the module
    /// doc comment's M3 scope). Each vertex's `(x, y, z, w)` object-space
    /// position is transformed by `modelview` then `projection`
    /// (`docs/internals/c3d.md`: "follow the OpenGL 1.1 specification's
    /// fixed-function pipeline exactly, including the `[-1, 1]` clip-space
    /// depth range mapped through `DEPTH_RANGE`"), then mapped through
    /// `VIEWPORT` and `DEPTH_RANGE` -- both spec-mandated transform-tier
    /// steps a window-space draw skips entirely (window-space vertices
    /// already carry final surface pixels and `[0, 1]` depth, and
    /// deliberately ignore `VIEWPORT`, per the module doc comment).
    /// Colour and `TEXCOORD0` pass through exactly as parsed (or the
    /// current-state fallback), same as the window-space path -- no
    /// lighting or texgen is applied to them here.
    fn op_draw_inline_gl(
        &mut self,
        prim: PrimitiveType,
        format: VertexFormat,
        count: u32,
        bytes: &[u8],
        state: &State,
    ) -> Result<(), RenderError> {
        let id = state.draw_surface();
        let def = *state
            .surface(id)
            .ok_or(RenderError::UnknownObject("surface", id))?;

        let verts = parse_gl_vertices(bytes, format, count, &state.current)?;
        let (topology, order) = primitive_index_order(prim, verts.len())?;
        if order.is_empty() {
            return Ok(());
        }

        // `top_matrix` always returns `Some` for `Modelview`/`Projection`
        // (only a `Texture` stack past `MAX_TEXTURES` can be `None`; see
        // `state.rs`'s `stack`), but the fallback keeps this total rather
        // than relying on that invariant never changing.
        let modelview = state
            .top_matrix(MatrixMode::Modelview)
            .unwrap_or_else(Mat4::identity);
        let projection = state
            .top_matrix(MatrixMode::Projection)
            .unwrap_or_else(Mat4::identity);
        // `VIEWPORT`/`DEPTH_RANGE`: both transform-tier-only raster state
        // (`docs/internals/c3d.md`'s raster-state table), origin top-left
        // like every other rectangle this spec defines (`state.rs`'s
        // `Rect` doc comment). A window-space draw skips this whole step
        // (see the module doc comment: "window-space draws ignore
        // [VIEWPORT], per the spec"); `set_draw_surface` resets it to the
        // full surface, so an untouched viewport still covers everything.
        let vp = state.raster.viewport;
        let (near, far) = state.raster.depth_range;
        let (w, h) = (def.width as f32, def.height as f32);

        // Fog distance: the spec's "`FOGCOORD` if present, else derived
        // from `rhw`" rule is stated only for the window-space vertex
        // format, which has no `w` in the GL-space sense at all. For a
        // GL-space vertex without `FOGCOORD`, this falls back to GL 1.1's
        // own fixed-function default fog coordinate: the eye-space
        // distance `-eye.z` (positive in front of the viewer), computed
        // from the modelview-only transform before projection.
        let has_fogcoord = format.has(VertexFormat::FOGCOORD);
        let to_gpu = |v: &GlVertex| -> GpuVertex {
            let eye = modelview.transform_point(v.pos);
            let clip = projection.transform_point(eye);
            // A vertex behind the eye at exactly `w == 0` is GL-undefined
            // (it doesn't correspond to any real point, and the wire
            // format legally allows an explicit `w = 0`); guarded the
            // same way the window-space path guards a zero `rhw` --
            // `safe_w`, not the raw `clip[3]`, is used for *every* use of
            // `w` below, including the one written into `clip_pos.w`
            // itself. (An earlier version of this guard only protected
            // the `ndc_x`/`ndc_y` division here and left the unguarded
            // `clip_w` in `clip_pos.w`/`depth_clip_z`, which still let
            // wgpu's own GPU-side perspective divide hit a real `0.0`
            // there -- `0/0 = NaN`. `draw_inline_survives_a_zero_clip_w`
            // pins this down.)
            let safe_w = if clip[3].abs() < 1e-8 { 1e-8 } else { clip[3] };
            let ndc_x = clip[0] / safe_w;
            let ndc_y = clip[1] / safe_w;

            // GL's viewport transform (origin top-left in this device's
            // window space, unlike native GL's bottom-left -- see this
            // function's doc comment) maps NDC into the `VIEWPORT` sub-
            // rectangle of the surface; wgpu's own implicit viewport
            // always covers the *whole* surface (nothing here ever calls
            // `set_viewport`, matching the window-space path), so that
            // sub-rectangle mapping is folded into a second remap back to
            // "as if the whole surface were the viewport" -- the same NDC
            // convention `op_draw_inline_win` produces from raw pixels.
            let device_x = vp.x as f32 + (ndc_x * 0.5 + 0.5) * vp.w as f32;
            let device_y = vp.y as f32 + (1.0 - (ndc_y * 0.5 + 0.5)) * vp.h as f32;
            let final_ndc_x = (device_x / w) * 2.0 - 1.0;
            let final_ndc_y = 1.0 - (device_y / h) * 2.0;

            // `DEPTH_RANGE`: GL maps clip-space `z` (after divide, in
            // `[-1, 1]`) to `[near, far]`. `gl_clip_z_to_wgpu` already
            // computes the `near = 0, far = 1` case
            // (`(clip_z_gl + clip_w) / 2`, i.e. `clip_w * (ndc_z + 1) /
            // 2`); scaling that by `(far - near)` and adding `near *
            // clip_w` generalises it to an arbitrary `DEPTH_RANGE` while
            // keeping the same undivided-clip-space convention
            // `clip_pos.z` needs (the fixture this helper was written and
            // unit-tested for -- see the module doc comment's "two
            // traps").
            let depth_clip_z = near * safe_w + (far - near) * gl_clip_z_to_wgpu(clip[2], safe_w);

            GpuVertex {
                clip_pos: [
                    final_ndc_x * safe_w,
                    final_ndc_y * safe_w,
                    depth_clip_z,
                    safe_w,
                ],
                color: v.color,
                texcoord0: v.texcoord0,
                fog_distance: if has_fogcoord { v.fogcoord } else { -eye[2] },
            }
        };

        let mut vbuf: Vec<GpuVertex> = Vec::with_capacity(order.len());
        for idx in order {
            let Some(v) = verts.get(idx) else {
                return Err(RenderError::ShortPayload("primitive vertex index"));
            };
            vbuf.push(to_gpu(v));
        }

        self.submit_draw(state, id, def, topology, vbuf)
    }

    /// The shared tail of [`Self::op_draw_inline_win`]/
    /// [`Self::op_draw_inline_gl`]: given an already fully-transformed
    /// [`GpuVertex`] list (in the primitive's [`primitive_index_order`]),
    /// builds/looks up the pipeline, binds the texture (if any), and
    /// submits the draw. Neither caller's vertex-transform math belongs
    /// here -- this only knows about the GPU-side shape both draw kinds
    /// converge on.
    fn submit_draw(
        &mut self,
        state: &State,
        id: u32,
        def: Surface,
        topology: PrimTopologyKey,
        vbuf: Vec<GpuVertex>,
    ) -> Result<(), RenderError> {
        let unit0 = state.texture_units.first();
        let bound_tex = unit0.map(|u| u.bound_texture).unwrap_or(0);
        let unit0_bound = bound_tex != 0 && self.textures.contains_key(&bound_tex);
        let surface_format = INTERNAL_COLOR_FORMAT;
        let key = self.pipeline_key_from_state(state, unit0_bound, surface_format, topology);

        if topology == PrimTopologyKey::Triangles && key.cull == Some(FaceKey::FrontAndBack) {
            return Ok(()); // culls every triangle: nothing to draw
        }

        let (color_view, depth_view) = {
            let gpu = self.ensure_surface(id, &def)?;
            (gpu.color_view.clone(), gpu.depth_view.clone())
        };

        let vertex_bytes = gpu_vertices_to_bytes(&vbuf);
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("c3d vertex buffer"),
                contents: &vertex_bytes,
                usage: wgpu::BufferUsages::VERTEX,
            });

        // Ensure the pipeline and (if textured) sampler exist first --
        // `pipeline_for`/`sampler` need `&mut self` to build-and-cache --
        // then look both back up immutably afterwards: the bind group and
        // render pass only need shared borrows of `self.device`/the
        // caches, and doing every mutable ensure up front (rather than
        // interleaving) keeps that split straightforward for the borrow
        // checker.
        let sampler_key = if key.textured {
            let obj = state.texture(bound_tex);
            let (ws, wt, min) = obj.map(|t| (t.wrap_s, t.wrap_t, t.min_filter)).unwrap_or((
                TexWrap::Repeat,
                TexWrap::Repeat,
                TexFilter::Linear,
            ));
            let k = (
                tex_wrap_to_address_mode(ws),
                tex_wrap_to_address_mode(wt),
                tex_filter_to_wgpu(min),
            );
            self.sampler(k.0, k.1, k.2);
            Some(k)
        } else {
            None
        };
        self.pipeline_for(&key);
        let pipeline = self.pipelines.get(&key).expect("just ensured");

        let bind_group = if key.textured {
            let tex = &self.textures[&bound_tex];
            let sampler = self
                .sampler_cache
                .get(&sampler_key.expect("textured implies a sampler key"))
                .expect("just ensured");
            if key.alpha_test.is_some() {
                let alpha_ref_bytes = state.raster.alpha_ref.to_le_bytes();
                let alpha_ref = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("c3d alpha ref"),
                        contents: &alpha_ref_bytes,
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("c3d textured+alpha bind group"),
                    layout: &self.bind_group_layout_textured_alpha,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&tex.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: alpha_ref.as_entire_binding(),
                        },
                    ],
                }))
            } else {
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("c3d textured bind group"),
                    layout: &self.bind_group_layout_textured,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&tex.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                }))
            }
        } else {
            None
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("c3d draw"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("c3d draw pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            if let Some((x, y, sw, sh)) = Self::clamped_scissor_rect(state, def.width, def.height) {
                pass.set_scissor_rect(x, y, sw, sh);
            }
            if let Some(bg) = &bind_group {
                pass.set_bind_group(0, bg, &[]);
            }
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.draw(0..vbuf.len() as u32, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn op_tex_image(
        &mut self,
        id: u32,
        level: u32,
        format: TexFormat,
        width: u32,
        height: u32,
        row_bytes: u32,
        data: Ref,
        mem: &mut dyn Memory,
    ) -> Result<(), RenderError> {
        if level != 0 {
            return Err(RenderError::Unimplemented("mipmap levels above 0 (M3)"));
        }
        let bpp = tex_format_bytes_per_pixel(format) as u32;
        let needed = (row_bytes.max(bpp * width) as usize) * height as usize;
        let loc = MemLoc::from_ref(data);
        let src = mem
            .read(loc, needed)
            .ok_or_else(|| RenderError::BadMemory(format!("TEX_IMAGE ref at {loc:?}")))?;

        let mut rgba = vec![0u8; (width * height * 4) as usize];
        for row in 0..height {
            let row_start = row as usize * row_bytes as usize;
            for col in 0..width {
                let texel_start = row_start + col as usize * bpp as usize;
                let texel = src
                    .get(texel_start..)
                    .and_then(|s| convert_texel_to_rgba8(format, s))
                    .ok_or(RenderError::ShortPayload("TEX_IMAGE texel"))?;
                let out = ((row * width + col) * 4) as usize;
                rgba[out..out + 4].copy_from_slice(&texel);
            }
        }

        let size = wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("c3d texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.textures.insert(
            id,
            GpuTexture {
                texture,
                view,
                width,
                height,
            },
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn op_tex_subimage(
        &mut self,
        id: u32,
        level: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        format: TexFormat,
        row_bytes: u32,
        data: Ref,
        mem: &mut dyn Memory,
    ) -> Result<(), RenderError> {
        if level != 0 {
            return Err(RenderError::Unimplemented("mipmap levels above 0 (M3)"));
        }
        let Some(existing) = self.textures.get(&id) else {
            return Err(RenderError::UnknownObject("texture", id));
        };
        if x + width > existing.width || y + height > existing.height {
            return Err(RenderError::ShortPayload("TEX_SUBIMAGE rectangle"));
        }

        let bpp = tex_format_bytes_per_pixel(format) as u32;
        let needed = (row_bytes.max(bpp * width) as usize) * height as usize;
        let loc = MemLoc::from_ref(data);
        let src = mem
            .read(loc, needed)
            .ok_or_else(|| RenderError::BadMemory(format!("TEX_SUBIMAGE ref at {loc:?}")))?;

        let mut rgba = vec![0u8; (width * height * 4) as usize];
        for row in 0..height {
            let row_start = row as usize * row_bytes as usize;
            for col in 0..width {
                let texel_start = row_start + col as usize * bpp as usize;
                let texel = src
                    .get(texel_start..)
                    .and_then(|s| convert_texel_to_rgba8(format, s))
                    .ok_or(RenderError::ShortPayload("TEX_SUBIMAGE texel"))?;
                let out = ((row * width + col) * 4) as usize;
                rgba[out..out + 4].copy_from_slice(&texel);
            }
        }

        let texture = &self.textures[&id].texture;
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    /// `TEX_COPY_IMAGE` (`glCopyTexImage2D`): defines a fresh level 0 from
    /// the draw surface's rectangle with a GPU-side texture-to-texture
    /// copy -- no guest round trip, no CPU readback. The draw surface's
    /// internal colour format is already [`INTERNAL_COLOR_FORMAT`], the
    /// same format every texture object is stored in, so this is a
    /// straight copy with no conversion step; `format` names only the
    /// *wire* layout `TEX_IMAGE`/`TEX_SUBIMAGE` would decode from guest
    /// bytes; it has no guest bytes to decode here, so it goes unused
    /// (the spec permits a device to store at higher precision than a
    /// requested format -- "Texture formats" -- which is what every
    /// texture object already does).
    fn op_tex_copy_image(
        &mut self,
        id: u32,
        level: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        state: &State,
    ) -> Result<(), RenderError> {
        if level != 0 {
            return Err(RenderError::Unimplemented("mipmap levels above 0 (M3)"));
        }
        let surface_id = state.draw_surface();
        let def = *state
            .surface(surface_id)
            .ok_or(RenderError::UnknownObject("surface", surface_id))?;
        let color = self.ensure_surface(surface_id, &def)?.color.clone();

        let size = wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("c3d texture (TEX_COPY_IMAGE)"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: INTERNAL_COLOR_FORMAT,
            // COPY_SRC alongside the usual TEXTURE_BINDING|COPY_DST: a
            // render-to-texture result is a natural source for a later
            // `TEX_COPY_SUBIMAGE`-into-another-texture or mip-generation
            // step, unlike a `TEX_IMAGE` upload from guest bytes.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("c3d tex-copy-image encoder"),
            });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            size,
        );
        self.queue.submit(Some(encoder.finish()));

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.textures.insert(
            id,
            GpuTexture {
                texture,
                view,
                width,
                height,
            },
        );
        Ok(())
    }

    /// `TEX_COPY_SUBIMAGE` (`glCopyTexSubImage2D`): the in-place mirror of
    /// [`Self::op_tex_copy_image`] -- same GPU-side copy, but into an
    /// existing level's `(xoff, yoff)` rather than creating a new texture.
    /// The destination rectangle is checked against the *existing* level's
    /// own dimensions here, exactly as [`Self::op_tex_subimage`] already
    /// does for `TEX_SUBIMAGE`, since dispatch only ever validated the
    /// *source* rectangle against the live draw surface.
    #[allow(clippy::too_many_arguments)]
    fn op_tex_copy_subimage(
        &mut self,
        id: u32,
        level: u32,
        xoff: u32,
        yoff: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        state: &State,
    ) -> Result<(), RenderError> {
        if level != 0 {
            return Err(RenderError::Unimplemented("mipmap levels above 0 (M3)"));
        }
        let Some(existing) = self.textures.get(&id) else {
            return Err(RenderError::UnknownObject("texture", id));
        };
        if xoff + width > existing.width || yoff + height > existing.height {
            return Err(RenderError::ShortPayload("TEX_COPY_SUBIMAGE rectangle"));
        }

        let surface_id = state.draw_surface();
        let def = *state
            .surface(surface_id)
            .ok_or(RenderError::UnknownObject("surface", surface_id))?;
        let color = self.ensure_surface(surface_id, &def)?.color.clone();
        let dest = &self.textures[&id].texture;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("c3d tex-copy-subimage encoder"),
            });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: dest,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: xoff,
                    y: yoff,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// `SURFACE_UPLOAD`: the mirror of [`Self::op_surface_readback`] --
    /// reads the rectangle from the surface's own backing memory (the
    /// guest has drawn 2D into the bitmap and wants 3D composited over
    /// it) and uploads it into the render target, one row at a time for
    /// exactly the same reason the readback direction needs it: a row's
    /// position in backing memory is `address + (y + row) * stride_bytes +
    /// x * bpp`, never simply the rectangle's own tightly-packed offset.
    fn op_surface_upload(
        &mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        state: &State,
        mem: &mut dyn Memory,
    ) -> Result<(), RenderError> {
        let id = state.draw_surface();
        let def = *state
            .surface(id)
            .ok_or(RenderError::UnknownObject("surface", id))?;
        let color = self.ensure_surface(id, &def)?.color.clone();

        let bpp = surface_format_bytes_per_pixel(def.format);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for row in 0..h {
            let row_offset = (y + row) * def.stride_bytes + x * bpp;
            let loc = offset_backing(def.backing, row_offset);
            let needed = (w * bpp) as usize;
            let src = mem.read(loc, needed).ok_or_else(|| {
                RenderError::BadMemory(format!("SURFACE_UPLOAD read from {loc:?} (row {row})"))
            })?;
            for col in 0..w {
                let texel_at = (col * bpp) as usize;
                let pixel = src
                    .get(texel_at..)
                    .and_then(|s| decode_pixel(def.format, s))
                    .ok_or(RenderError::ShortPayload("SURFACE_UPLOAD pixel"))?;
                let out_at = ((row * w + col) * 4) as usize;
                rgba[out_at..out_at + 4].copy_from_slice(&pixel);
            }
        }

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    /// `READ_PIXELS`: like [`Self::op_surface_readback`], reads the draw
    /// surface's rectangle back to the CPU, but into an arbitrary `dest`
    /// ref at the caller's own `row_bytes` stride rather than the
    /// surface's own backing memory and its own `stride_bytes` --
    /// `SurfaceReadback` and `ReadPixels` differ in *where* the pixels
    /// land and, for `ReadPixels`, that `Depth` is a legal target format;
    /// the row-by-row addressing the fix to `SurfaceReadback`/
    /// `SurfaceUpload` already established applies identically to the
    /// destination side here, just keyed by `row_bytes` instead of
    /// `stride_bytes`. `flags` bit 0 (`ROWS_BOTTOM_UP`) reverses which
    /// *output* row a source row lands at -- row values are unchanged,
    /// only their order -- so a client with a bottom-left image origin
    /// (`glReadPixels`) needs no row-reversal of its own.
    #[allow(clippy::too_many_arguments)]
    fn op_read_pixels(
        &mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        format: WireSurfaceFormat,
        row_bytes: u32,
        flags: u32,
        dest: Ref,
        state: &State,
        mem: &mut dyn Memory,
    ) -> Result<(), RenderError> {
        let id = state.draw_surface();
        let def = *state
            .surface(id)
            .ok_or(RenderError::UnknownObject("surface", id))?;
        let gpu = self
            .surfaces
            .get(&id)
            .ok_or(RenderError::UnknownObject("surface", id))?;

        // Both branches produce one f32-per-channel-or-depth value per
        // source texel, straight from `copy_texture_to_cpu`'s raw bytes
        // (4 bytes/texel either way: RGBA8 four channel bytes, or one
        // little-endian f32 for Depth32Float -- a raw GPU-to-CPU memory
        // copy, not a serialised value, so it is read back in the same
        // little-endian convention every other GPU buffer in this module
        // uses, e.g. `gpu_vertices_to_bytes`).
        let bpp = if format == WireSurfaceFormat::Depth {
            4
        } else {
            surface_format_bytes_per_pixel(format)
        };
        let encode_at = |raw: &[u8], sx: u32, sy: u32| -> Result<Vec<u8>, RenderError> {
            let at = ((sy * def.width + sx) * 4) as usize;
            if format == WireSurfaceFormat::Depth {
                let depth = f32::from_le_bytes(
                    *<&[u8; 4]>::try_from(&raw[at..at + 4]).expect("4-byte slice"),
                );
                let scaled = (depth.clamp(0.0, 1.0) as f64 * u32::MAX as f64).round() as u32;
                Ok(scaled.to_be_bytes().to_vec())
            } else {
                let pixel = [raw[at], raw[at + 1], raw[at + 2], raw[at + 3]];
                encode_pixel(format, pixel)
            }
        };
        let raw = if format == WireSurfaceFormat::Depth {
            let depth = gpu.depth.clone();
            self.copy_texture_to_cpu(&depth, def.width, def.height)?
        } else {
            let color = gpu.color.clone();
            self.copy_texture_to_cpu(&color, def.width, def.height)?
        };

        let bottom_up = flags & proto::READ_PIXELS_FLAG_ROWS_BOTTOM_UP != 0;
        for out_row in 0..h {
            let src_row = if bottom_up { h - 1 - out_row } else { out_row };
            let sy = y + src_row;
            let mut row_out = vec![0u8; (w * bpp) as usize];
            for col in 0..w {
                let sx = x + col;
                if sx >= def.width || sy >= def.height {
                    continue;
                }
                let encoded = encode_at(&raw, sx, sy)?;
                let at = (col * bpp) as usize;
                row_out[at..at + encoded.len()].copy_from_slice(&encoded);
            }
            let loc = offset_ref(dest, out_row * row_bytes);
            if !mem.write(loc, &row_out) {
                return Err(RenderError::BadMemory(format!(
                    "READ_PIXELS write to {loc:?} (row {out_row})"
                )));
            }
        }
        Ok(())
    }

    fn op_surface_readback(
        &mut self,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        state: &State,
        mem: &mut dyn Memory,
    ) -> Result<(), RenderError> {
        let id = state.draw_surface();
        let def = *state
            .surface(id)
            .ok_or(RenderError::UnknownObject("surface", id))?;
        let gpu = self
            .surfaces
            .get(&id)
            .ok_or(RenderError::UnknownObject("surface", id))?;

        let rgba = self.copy_texture_to_cpu(&gpu.color, def.width, def.height)?;

        let bpp = surface_format_bytes_per_pixel(def.format);
        // Each row lands at its own offset inside the surface's backing
        // memory -- address + (y + row) * stride_bytes + x * bpp -- one
        // `mem.write` per row, not the whole rectangle packed at the
        // surface's own base address regardless of x/y. A rectangle away
        // from the origin, or a surface whose stride_bytes carries
        // padding past width * bpp, both need this: the rect's own w * bpp
        // packing is only ever the row *length*, never where the row
        // starts.
        for row in 0..h {
            let mut out_row = vec![0u8; (w * bpp) as usize];
            for col in 0..w {
                let sx = x + col;
                let sy = y + row;
                if sx >= def.width || sy >= def.height {
                    continue;
                }
                let src_at = ((sy * def.width + sx) * 4) as usize;
                let pixel = [
                    rgba[src_at],
                    rgba[src_at + 1],
                    rgba[src_at + 2],
                    rgba[src_at + 3],
                ];
                let encoded = encode_pixel(def.format, pixel)?;
                let dst_at = (col * bpp) as usize;
                out_row[dst_at..dst_at + encoded.len()].copy_from_slice(&encoded);
            }
            let row_offset = (y + row) * def.stride_bytes + x * bpp;
            let loc = offset_backing(def.backing, row_offset);
            if !mem.write(loc, &out_row) {
                return Err(RenderError::BadMemory(format!(
                    "SURFACE_READBACK write to {loc:?} (row {row})"
                )));
            }
        }
        Ok(())
    }

    /// `QUERY`: the spec's whole contract is "write `result` to `dest`,
    /// as big-endian `f32`s, matrices in column-major order" -- `dispatch`
    /// has already resolved *which* state `what` meant (`resolve_query`),
    /// so there is no GL semantics left to apply here, only the guest
    /// memory write, the same big-endian convention every other
    /// guest-visible value on this wire uses (`docs/internals/c3d.md`'s
    /// "IEEE-754 single precision, big-endian").
    fn op_query(
        &mut self,
        dest: Ref,
        result: QueryResult,
        mem: &mut dyn Memory,
    ) -> Result<(), RenderError> {
        let bytes: Vec<u8> = result.values[..result.count as usize]
            .iter()
            .flat_map(|f| f.to_be_bytes())
            .collect();
        let loc = MemLoc::from_ref(dest);
        if !mem.write(loc, &bytes) {
            return Err(RenderError::BadMemory(format!("QUERY write to {loc:?}")));
        }
        Ok(())
    }

    /// Copies a colour texture's whole contents to a CPU-side RGBA8
    /// buffer, blocking on the map. Used by [`Self::op_surface_readback`]
    /// and by callers (the trace runner, `selftest`) that want to inspect
    /// or golden-compare a rendered surface directly.
    pub fn copy_texture_to_cpu(
        &self,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, RenderError> {
        let bytes_per_row_unaligned = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = bytes_per_row_unaligned.div_ceil(align) * align;

        let buffer_size = (padded_bytes_per_row * height) as wgpu::BufferAddress;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("c3d readback buffer"),
            size: buffer_size.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("c3d readback encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| RenderError::Gpu(e.to_string()))?;
        rx.recv()
            .map_err(|e| RenderError::Gpu(e.to_string()))?
            .map_err(|e| RenderError::Gpu(e.to_string()))?;

        let data = slice.get_mapped_range();
        let mut out = vec![0u8; (width * height * 4) as usize];
        for row in 0..height as usize {
            let src_start = row * padded_bytes_per_row as usize;
            let dst_start = row * width as usize * 4;
            out[dst_start..dst_start + width as usize * 4]
                .copy_from_slice(&data[src_start..src_start + width as usize * 4]);
        }
        drop(data);
        buffer.unmap();
        Ok(out)
    }

    /// Reads back surface `id`'s current colour contents as RGBA8, for a
    /// runner's `GOLD` comparison or `selftest`. Not a [`RenderOp`] --
    /// `SurfaceReadback` writes through `Memory` in the surface's own wire
    /// format instead -- this is the convenience path a runner uses
    /// directly against the renderer.
    pub fn read_surface_rgba8(&self, id: u32) -> Result<(u32, u32, Vec<u8>), RenderError> {
        let gpu = self
            .surfaces
            .get(&id)
            .ok_or(RenderError::UnknownObject("surface", id))?;
        let pixels = self.copy_texture_to_cpu(&gpu.color, gpu.def.width, gpu.def.height)?;
        Ok((gpu.def.width, gpu.def.height, pixels))
    }
}

/// The GPU-side vertex layout: `clip_pos` is already fully perspective-
/// ready clip-space `(x*w, y*w, z*w, w)`, `color`/`texcoord0`/
/// `fog_distance` are the undivided attribute values the spec describes --
/// the hardware's own perspective-correct interpolation (driven by
/// `clip_pos.w`) does exactly the "interpolate `attr * rhw`, recover
/// `attr` per fragment" the spec asks for, with no per-fragment division
/// written by this module. `fog_distance` is carried unconditionally
/// (like `texcoord0` for an untextured draw) even on a pipeline with fog
/// disabled, where the fragment shader simply never reads it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuVertex {
    clip_pos: [f32; 4],
    color: [f32; 4],
    texcoord0: [f32; 2],
    fog_distance: f32,
}

/// Serialises a vertex list to the plain little-endian bytes a wgpu vertex
/// buffer expects. No `bytemuck` dependency: this crate does not declare
/// one directly (it is only ever pulled in transitively), so this module
/// writes the bytes itself rather than reaching for `cast_slice`.
fn gpu_vertices_to_bytes(vertices: &[GpuVertex]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(vertices));
    for v in vertices {
        for f in v.clip_pos {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for f in v.color {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for f in v.texcoord0 {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.extend_from_slice(&v.fog_distance.to_le_bytes());
    }
    out
}

use wgpu::util::DeviceExt;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c3d::state::CurrentVertex;

    fn current() -> CurrentVertex {
        CurrentVertex {
            color: [1.0, 1.0, 1.0, 1.0],
            normal: [0.0, 0.0, 1.0],
            texcoord: vec![(0.0, 0.0), (0.0, 0.0)],
            fogcoord: 0.0,
        }
    }

    fn f32be(v: f32) -> [u8; 4] {
        v.to_be_bytes()
    }

    // -- unpack_color / gl_clip_z_to_wgpu / wrap trap --------------------

    #[test]
    fn unpack_color_scales_bytes_to_zero_one() {
        assert_eq!(
            unpack_color(0xFF80_4000),
            [1.0, 128.0 / 255.0, 64.0 / 255.0, 0.0]
        );
        assert_eq!(unpack_color(0x0000_00FF), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn gl_clip_z_to_wgpu_maps_the_documented_endpoints() {
        // GL NDC z=-1 (near) with w=1 -> wgpu clip z=0 (dividing by w=1).
        assert_eq!(gl_clip_z_to_wgpu(-1.0, 1.0), 0.0);
        // GL NDC z=+1 (far) with w=1 -> wgpu clip z=1.
        assert_eq!(gl_clip_z_to_wgpu(1.0, 1.0), 1.0);
        // GL NDC z=0 (mid) with w=1 -> wgpu clip z=0.5.
        assert_eq!(gl_clip_z_to_wgpu(0.0, 1.0), 0.5);
        // Scales correctly with a non-unit w (still pre-perspective-divide).
        assert_eq!(gl_clip_z_to_wgpu(-2.0, 2.0), 0.0);
        assert_eq!(gl_clip_z_to_wgpu(2.0, 2.0), 2.0);
    }

    #[test]
    fn clamp_wraps_to_edge_never_to_a_border() {
        assert_eq!(
            tex_wrap_to_address_mode(TexWrap::Clamp),
            wgpu::AddressMode::ClampToEdge
        );
        assert_eq!(
            tex_wrap_to_address_mode(TexWrap::ClampToEdge),
            wgpu::AddressMode::ClampToEdge
        );
        assert_eq!(
            tex_wrap_to_address_mode(TexWrap::Repeat),
            wgpu::AddressMode::Repeat
        );
    }

    // -- vertex parsing ---------------------------------------------------

    #[test]
    fn parses_a_pos_only_window_vertex_with_default_pos_count() {
        // POS_COUNT=0 (4 words): x,y,z,rhw.
        let mut bytes = Vec::new();
        for v in [1.0f32, 2.0, 0.25, 0.5] {
            bytes.extend_from_slice(&f32be(v));
        }
        let format = VertexFormat(0);
        let verts = parse_window_vertices(&bytes, format, 1, &current()).unwrap();
        assert_eq!(verts.len(), 1);
        assert_eq!(verts[0].pos, [1.0, 2.0, 0.25, 0.5]);
        assert_eq!(verts[0].color, [1.0, 1.0, 1.0, 1.0]); // from `current`
    }

    #[test]
    fn parses_pos3_with_default_rhw_of_one() {
        let mut bytes = Vec::new();
        for v in [10.0f32, 20.0, 0.75] {
            bytes.extend_from_slice(&f32be(v));
        }
        let format = VertexFormat(1 << VertexFormat::POS_COUNT_SHIFT);
        let verts = parse_window_vertices(&bytes, format, 1, &current()).unwrap();
        assert_eq!(verts[0].pos, [10.0, 20.0, 0.75, 1.0]);
    }

    #[test]
    fn parses_color_packed_and_texcoord0() {
        let mut bytes = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0] {
            bytes.extend_from_slice(&f32be(v)); // POS_COUNT=0
        }
        bytes.extend_from_slice(&0xFF804000u32.to_be_bytes()); // COLOR_PACKED
        bytes.extend_from_slice(&f32be(0.25)); // TEXCOORD0.s
        bytes.extend_from_slice(&f32be(0.75)); // TEXCOORD0.t
        let format = VertexFormat(VertexFormat::COLOR_PACKED | VertexFormat::TEXCOORD0);
        let verts = parse_window_vertices(&bytes, format, 1, &current()).unwrap();
        assert_eq!(verts[0].color, unpack_color(0xFF804000));
        assert_eq!(verts[0].texcoord0, [0.25, 0.75]);
    }

    #[test]
    fn a_truncated_vertex_buffer_is_a_short_payload_error() {
        let bytes = [0u8; 4]; // one word short of even one POS4 vertex
        let format = VertexFormat(0);
        let err = parse_window_vertices(&bytes, format, 1, &current()).unwrap_err();
        assert!(matches!(err, RenderError::ShortPayload(_)));
    }

    #[test]
    fn skips_unused_components_words_correctly_to_reach_the_next_vertex_and_reads_fogcoord() {
        // Two POS-only vertices, each with its own FOGCOORD word.
        let format = VertexFormat(1 << VertexFormat::POS_COUNT_SHIFT | VertexFormat::FOGCOORD);
        let mut bytes = Vec::new();
        for vertex in [[1.0f32, 2.0, 0.0], [3.0, 4.0, 0.0]] {
            for v in vertex {
                bytes.extend_from_slice(&f32be(v));
            }
            bytes.extend_from_slice(&f32be(9.0)); // fogcoord
        }
        let verts = parse_window_vertices(&bytes, format, 2, &current()).unwrap();
        assert_eq!(verts.len(), 2);
        assert_eq!(verts[0].pos[0..2], [1.0, 2.0]);
        assert_eq!(verts[1].pos[0..2], [3.0, 4.0]);
        assert_eq!(verts[0].fogcoord, 9.0);
        assert_eq!(verts[1].fogcoord, 9.0);
    }

    #[test]
    fn an_omitted_fogcoord_takes_the_current_value() {
        let format = VertexFormat(1 << VertexFormat::POS_COUNT_SHIFT); // no FOGCOORD bit
        let bytes = [f32be(1.0), f32be(2.0), f32be(0.0)].concat();
        let mut cur = current();
        cur.fogcoord = 4.5;
        let verts = parse_window_vertices(&bytes, format, 1, &cur).unwrap();
        assert_eq!(verts[0].fogcoord, 4.5);
    }

    // -- triangulation ------------------------------------------------

    #[test]
    fn triangles_provoking_vertex_is_rotated_to_the_third_vertex() {
        let tris = triangulate(PrimitiveType::Triangles, 3).unwrap();
        assert_eq!(tris, vec![[2, 0, 1]]);
    }

    #[test]
    fn quads_split_normatively_with_v3_as_the_provoking_vertex() {
        let tris = triangulate(PrimitiveType::Quads, 4).unwrap();
        // (v0 v1 v2), (v0 v2 v3), flat colour v3 for both -> v3 first in
        // both, cyclically rotated (winding-preserving).
        assert_eq!(tris, vec![[3, 0, 1], [3, 0, 2]]);
    }

    #[test]
    fn quad_strip_splits_normatively() {
        // One quad: v0 v1 v2 v3 -> (v0 v1 v3),(v0 v3 v2), provoking v3.
        let tris = triangulate(PrimitiveType::QuadStrip, 4).unwrap();
        assert_eq!(tris, vec![[3, 0, 1], [3, 0, 2]]);
    }

    #[test]
    fn triangle_fan_provoking_vertex_is_the_last_of_each_triangle() {
        let tris = triangulate(PrimitiveType::TriangleFan, 4).unwrap();
        // Triangles (v0 v1 v2), (v0 v2 v3); provoking = last = v2, v3.
        assert_eq!(tris, vec![[2, 0, 1], [3, 0, 2]]);
    }

    #[test]
    fn polygon_takes_its_flat_colour_from_the_polygons_own_last_vertex() {
        let tris = triangulate(PrimitiveType::Polygon, 5).unwrap();
        // Fan from v0, but every triangle's provoking vertex is v4 (n-1),
        // not each triangle's own last vertex (contrast with a fan).
        for tri in &tris {
            assert_eq!(tri[0], 4);
        }
        assert_eq!(tris.len(), 3);
    }

    #[test]
    fn triangle_strip_alternates_winding_with_a_rotated_provoking_vertex() {
        let tris = triangulate(PrimitiveType::TriangleStrip, 5).unwrap();
        assert_eq!(tris.len(), 3);
        // Odd-indexed generated triangles swap two indices relative to
        // even ones (GL's alternating strip winding), while every
        // triangle's provoking vertex (its own last) leads.
        assert_eq!(tris[0], [2, 0, 1]);
        assert_eq!(tris[1], [3, 2, 1]);
        assert_eq!(tris[2], [4, 2, 3]);
    }

    #[test]
    fn an_incomplete_final_primitive_is_silently_dropped() {
        // 4 vertices for TRIANGLES: only one full triangle, remainder
        // dropped, per GL.
        let tris = triangulate(PrimitiveType::Triangles, 4).unwrap();
        assert_eq!(tris.len(), 1);
    }

    #[test]
    fn point_and_line_primitives_are_reported_unimplemented_not_panicking() {
        assert!(matches!(
            triangulate(PrimitiveType::Points, 3),
            Err(RenderError::Unimplemented(_))
        ));
        assert!(matches!(
            triangulate(PrimitiveType::Lines, 4),
            Err(RenderError::Unimplemented(_))
        ));
    }

    #[test]
    fn primitive_index_order_maps_points_one_to_one() {
        let (topology, idx) = primitive_index_order(PrimitiveType::Points, 3).unwrap();
        assert_eq!(topology, PrimTopologyKey::Points);
        assert_eq!(idx, vec![0, 1, 2]);
    }

    #[test]
    fn primitive_index_order_pairs_lines_with_the_provoking_vertex_first() {
        // LINES (v0 v1 v2 v3) -> segments (v0 v1), (v2 v3); GL's
        // provoking vertex is each segment's second, emitted first to
        // line up with wgpu/WGSL's first-vertex flat interpolation.
        let (topology, idx) = primitive_index_order(PrimitiveType::Lines, 4).unwrap();
        assert_eq!(topology, PrimTopologyKey::Lines);
        assert_eq!(idx, vec![1, 0, 3, 2]);
    }

    #[test]
    fn primitive_index_order_drops_an_incomplete_trailing_line() {
        let (_, idx) = primitive_index_order(PrimitiveType::Lines, 3).unwrap();
        assert_eq!(idx, vec![1, 0]);
    }

    #[test]
    fn primitive_index_order_decomposes_line_strip_into_segments() {
        // LINE_STRIP (v0 v1 v2) -> (v0 v1), (v1 v2).
        let (topology, idx) = primitive_index_order(PrimitiveType::LineStrip, 3).unwrap();
        assert_eq!(topology, PrimTopologyKey::Lines);
        assert_eq!(idx, vec![1, 0, 2, 1]);
    }

    #[test]
    fn primitive_index_order_closes_line_loop_back_to_the_first_vertex() {
        // LINE_LOOP (v0 v1 v2) -> (v0 v1), (v1 v2), (v2 v0).
        let (topology, idx) = primitive_index_order(PrimitiveType::LineLoop, 3).unwrap();
        assert_eq!(topology, PrimTopologyKey::Lines);
        assert_eq!(idx, vec![1, 0, 2, 1, 0, 2]);
    }

    #[test]
    fn primitive_index_order_still_delegates_triangle_family_to_triangulate() {
        let (topology, idx) = primitive_index_order(PrimitiveType::Triangles, 3).unwrap();
        assert_eq!(topology, PrimTopologyKey::Triangles);
        assert_eq!(idx, vec![2, 0, 1]);
    }

    // -- format conversion ---------------------------------------------

    #[test]
    fn rgba8_texel_round_trips_exactly() {
        let px = convert_texel_to_rgba8(TexFormat::Rgba8, &[10, 20, 30, 40]).unwrap();
        assert_eq!(px, [10, 20, 30, 40]);
    }

    #[test]
    fn rgb565_texel_expands_to_full_range_white_and_black() {
        let white = convert_texel_to_rgba8(TexFormat::Rgb565, &0xFFFFu16.to_be_bytes()).unwrap();
        assert_eq!(white, [255, 255, 255, 255]);
        let black = convert_texel_to_rgba8(TexFormat::Rgb565, &0x0000u16.to_be_bytes()).unwrap();
        assert_eq!(black, [0, 0, 0, 255]);
    }

    #[test]
    fn a8_texel_carries_alpha_only() {
        let px = convert_texel_to_rgba8(TexFormat::A8, &[0x80]).unwrap();
        assert_eq!(px, [0, 0, 0, 0x80]);
    }

    #[test]
    fn a_short_texel_source_is_none_not_a_panic() {
        assert_eq!(convert_texel_to_rgba8(TexFormat::Rgba8, &[1, 2]), None);
        assert_eq!(convert_texel_to_rgba8(TexFormat::L8, &[]), None);
    }

    #[test]
    fn encode_pixel_packs_a8r8g8b8_in_the_documented_byte_order() {
        let bytes = encode_pixel(WireSurfaceFormat::A8r8g8b8, [0x11, 0x22, 0x33, 0xFF]).unwrap();
        assert_eq!(bytes, vec![0xFF, 0x11, 0x22, 0x33]);
    }

    #[test]
    fn encode_pixel_packs_b8g8r8a8_in_the_documented_byte_order() {
        let bytes = encode_pixel(WireSurfaceFormat::B8g8r8a8, [0x11, 0x22, 0x33, 0xFF]).unwrap();
        assert_eq!(bytes, vec![0x33, 0x22, 0x11, 0xFF]);
    }

    #[test]
    fn encode_pixel_rejects_depth_as_unimplemented() {
        assert!(matches!(
            encode_pixel(WireSurfaceFormat::Depth, [0, 0, 0, 0]),
            Err(RenderError::Unimplemented(_))
        ));
    }

    // -- GPU-dependent tests: gated on adapter availability ---------------
    //
    // Per `tests/README.md`'s asset-gated pattern: skip cleanly (print and
    // return) rather than fail when no adapter -- hardware or software --
    // is available at all, which is a real, if rare, environment (a
    // container with no Vulkan/Metal/DX12 loader and no llvmpipe/WARP
    // installed) rather than a bug.

    #[test]
    fn renderer_new_either_succeeds_or_reports_no_adapter_cleanly() {
        match Renderer::new() {
            Ok(r) => {
                // A real adapter was found; describe() must not panic.
                let _ = r.describe();
            }
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available in this environment");
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        }
    }

    struct TestMemory {
        aperture: Vec<u8>,
    }
    impl TestMemory {
        fn new(size: usize) -> Self {
            TestMemory {
                aperture: vec![0u8; size],
            }
        }
    }
    impl Memory for TestMemory {
        fn read(&self, loc: MemLoc, len: usize) -> Option<&[u8]> {
            match loc {
                MemLoc::Aperture(addr) => {
                    let start = addr as usize;
                    self.aperture.get(start..start + len)
                }
                MemLoc::Guest(_) => None,
            }
        }
        fn write(&mut self, loc: MemLoc, data: &[u8]) -> bool {
            match loc {
                MemLoc::Aperture(addr) => {
                    let start = addr as usize;
                    if start + data.len() > self.aperture.len() {
                        return false;
                    }
                    self.aperture[start..start + data.len()].copy_from_slice(data);
                    true
                }
                MemLoc::Guest(_) => false,
            }
        }
    }

    /// Renders a clear then reads it back, checking the surface is
    /// uniformly the clear colour -- the M1 selftest's own check, pinned
    /// here as a unit test too. Skips cleanly with no adapter.
    #[test]
    fn a_clear_produces_a_uniformly_coloured_surface() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: 8,
                height: 8,
                stride_bytes: 32,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);
        state.set_clear_color(0.0, 1.0, 0.0, 1.0);

        let mut mem = TestMemory::new(4096);
        let errs = renderer.execute(
            &[RenderOp::Clear {
                mask: proto::CLEAR_MASK_COLOR | proto::CLEAR_MASK_DEPTH,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        let (w, h, pixels) = renderer.read_surface_rgba8(1).unwrap();
        assert_eq!((w, h), (8, 8));
        for px in pixels.chunks(4) {
            assert_eq!(px, [0, 255, 0, 255]);
        }
    }

    /// A scissored `CLEAR` must touch only the rectangle inside
    /// `SCISSOR`, leaving everything outside it exactly as it was --
    /// this is the case `wgpu::LoadOp::Clear` alone cannot express
    /// (see `Renderer::op_clear`'s doc comment), so it is the one worth
    /// checking pixel-by-pixel rather than trusting "no error".
    #[test]
    fn a_scissored_clear_touches_only_the_scissor_rectangle() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: 8,
                height: 8,
                stride_bytes: 32,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);

        let mut mem = TestMemory::new(4096);
        // First, an ordinary full clear to a known background (blue), so
        // "untouched" has an unambiguous value to check against.
        state.set_clear_color(0.0, 0.0, 1.0, 1.0);
        let errs = renderer.execute(
            &[RenderOp::Clear {
                mask: proto::CLEAR_MASK_COLOR,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        // Then a scissored clear to red, restricted to the top-left 3x2
        // rectangle only.
        state.set_clear_color(1.0, 0.0, 0.0, 1.0);
        state.enable(state::Capability::ScissorTest);
        state.set_scissor(0, 0, 3, 2);
        let errs = renderer.execute(
            &[RenderOp::Clear {
                mask: proto::CLEAR_MASK_COLOR,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        let (_, _, pixels) = renderer.read_surface_rgba8(1).unwrap();
        for y in 0..8u32 {
            for x in 0..8u32 {
                let at = ((y * 8 + x) * 4) as usize;
                let px = &pixels[at..at + 4];
                if x < 3 && y < 2 {
                    assert_eq!(
                        px,
                        [255, 0, 0, 255],
                        "({x}, {y}) should be red (inside scissor)"
                    );
                } else {
                    assert_eq!(
                        px,
                        [0, 0, 255, 255],
                        "({x}, {y}) should still be blue (outside scissor)"
                    );
                }
            }
        }
    }

    /// A `CLEAR` with a partial colour mask must leave the masked-off
    /// channels exactly as they were, even for a full-surface clear with
    /// no scissor -- the same `wgpu::LoadOp::Clear` limitation as the
    /// scissored case, for a different reason (per-channel masking
    /// rather than a sub-rectangle).
    #[test]
    fn a_clear_with_a_partial_colour_mask_leaves_the_masked_channels_alone() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: 4,
                height: 4,
                stride_bytes: 16,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);
        let mut mem = TestMemory::new(4096);

        // Background: opaque white, so every channel starts at 255 and a
        // masked-off channel staying at 255 is unambiguous.
        state.set_clear_color(1.0, 1.0, 1.0, 1.0);
        let errs = renderer.execute(
            &[RenderOp::Clear {
                mask: proto::CLEAR_MASK_COLOR,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        // Clear to black with only the red channel writable.
        state.set_clear_color(0.0, 0.0, 0.0, 1.0);
        state.set_color_mask(true, false, false, false);
        let errs = renderer.execute(
            &[RenderOp::Clear {
                mask: proto::CLEAR_MASK_COLOR,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        let (_, _, pixels) = renderer.read_surface_rgba8(1).unwrap();
        for px in pixels.chunks(4) {
            assert_eq!(
                px,
                [0, 255, 255, 255],
                "red should have cleared to 0; green/blue/alpha must stay untouched at 255"
            );
        }
    }

    /// The bug `SURFACE_UPLOAD`'s own implementation would otherwise have
    /// repeated: a rectangle away from the surface's origin, and a
    /// surface whose `stride_bytes` carries padding past `width * bpp`,
    /// both need each row placed at `address + (y + row) * stride_bytes
    /// + x * bpp` in backing memory -- never the readback rectangle's own
    /// tightly-packed offset, which is only ever a row's *length*. This
    /// clears a padded, wider-than-tall surface to a known colour, reads
    /// back a 2x2 rectangle away from the origin, and checks the bytes
    /// landed at the correct strided offset -- with the padding bytes
    /// (and everything outside the rectangle) left exactly as they
    /// started, proving this does not merely "work" by writing more than
    /// it should.
    #[test]
    fn a_surface_readback_of_a_subrectangle_lands_at_the_correct_strided_offset() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        const WIDTH: u32 = 8;
        const HEIGHT: u32 = 8;
        const BPP: u32 = 4; // A8R8G8B8
        const STRIDE: u32 = WIDTH * BPP + 16; // deliberate padding past width * bpp
        const BACKING_LEN: usize = (STRIDE * HEIGHT) as usize;

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: WIDTH,
                height: HEIGHT,
                stride_bytes: STRIDE,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);
        state.set_clear_color(0.0, 1.0, 0.0, 1.0); // opaque green

        let mut mem = TestMemory::new(BACKING_LEN);
        // Mark the whole backing buffer with a sentinel first, so
        // "untouched" is unambiguous -- a real readback bug that writes
        // zeros where it shouldn't would otherwise be indistinguishable
        // from a freshly zeroed test buffer.
        mem.aperture.fill(0xAA);

        let errs = renderer.execute(
            &[
                RenderOp::Clear {
                    mask: proto::CLEAR_MASK_COLOR,
                },
                RenderOp::SurfaceReadback {
                    x: 3,
                    y: 2,
                    w: 2,
                    h: 2,
                },
            ],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        // A R G B for opaque green.
        let green = [0xFFu8, 0x00, 0xFF, 0x00];
        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                let at = (row * STRIDE + col * BPP) as usize;
                let in_rect = (3..5).contains(&col) && (2..4).contains(&row);
                if in_rect {
                    assert_eq!(
                        &mem.aperture[at..at + 4],
                        &green,
                        "({col}, {row}) is inside the rectangle and should be green"
                    );
                } else {
                    assert_eq!(
                        &mem.aperture[at..at + 4],
                        &[0xAA, 0xAA, 0xAA, 0xAA],
                        "({col}, {row}) is outside the rectangle and must be untouched"
                    );
                }
            }
            // Stride padding for this row must also be untouched.
            let pad_at = (row * STRIDE + WIDTH * BPP) as usize;
            assert_eq!(
                &mem.aperture[pad_at..pad_at + 16],
                &[0xAAu8; 16][..],
                "row {row}'s stride padding must be untouched"
            );
        }
    }

    /// The mirror of the readback test above, same surface shape: an
    /// upload from backing memory must read each row from its own
    /// strided offset too, not the rectangle's tightly-packed offset.
    #[test]
    fn a_surface_upload_reads_a_subrectangle_from_the_correct_strided_offset() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        const WIDTH: u32 = 8;
        const HEIGHT: u32 = 8;
        const BPP: u32 = 4;
        const STRIDE: u32 = WIDTH * BPP + 16;
        const BACKING_LEN: usize = (STRIDE * HEIGHT) as usize;

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: WIDTH,
                height: HEIGHT,
                stride_bytes: STRIDE,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);
        state.set_clear_color(0.0, 0.0, 0.0, 1.0); // black background

        let mut mem = TestMemory::new(BACKING_LEN);
        // Paint just the 2x2 rectangle at (3, 2) opaque blue (A R G B) in
        // backing memory, at its correct strided position; everything
        // else stays the zeroed TestMemory default.
        let blue = [0xFFu8, 0x00, 0x00, 0xFF];
        for row in 2..4u32 {
            for col in 3..5u32 {
                let at = (row * STRIDE + col * BPP) as usize;
                mem.aperture[at..at + 4].copy_from_slice(&blue);
            }
        }

        let errs = renderer.execute(
            &[
                RenderOp::Clear {
                    mask: proto::CLEAR_MASK_COLOR,
                },
                RenderOp::SurfaceUpload {
                    x: 3,
                    y: 2,
                    w: 2,
                    h: 2,
                },
            ],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        let (_, _, pixels) = renderer.read_surface_rgba8(1).unwrap();
        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                let at = ((row * WIDTH + col) * 4) as usize;
                let in_rect = (3..5).contains(&col) && (2..4).contains(&row);
                if in_rect {
                    assert_eq!(
                        &pixels[at..at + 4],
                        &[0, 0, 255, 255],
                        "({col}, {row}) should be blue (uploaded)"
                    );
                } else {
                    assert_eq!(
                        &pixels[at..at + 4],
                        &[0, 0, 0, 255],
                        "({col}, {row}) should still be black (never uploaded)"
                    );
                }
            }
        }
    }

    /// A colour `READ_PIXELS` with `ROWS_BOTTOM_UP` set must land the
    /// surface's own top row *last* in the destination -- the row
    /// *values* are unchanged, only which output row each lands at, so a
    /// two-colour surface (top red, bottom green) proves the flip
    /// unambiguously in a way a uniform colour never could.
    #[test]
    fn read_pixels_with_rows_bottom_up_reverses_output_row_order_not_values() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: 2,
                height: 2,
                stride_bytes: 8,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);

        let mut mem = TestMemory::new(64);
        // Top row (y=0) red via a scissored clear, bottom row (y=1) green.
        state.set_clear_color(1.0, 0.0, 0.0, 1.0);
        state.enable(state::Capability::ScissorTest);
        state.set_scissor(0, 0, 2, 1);
        let errs = renderer.execute(
            &[RenderOp::Clear {
                mask: proto::CLEAR_MASK_COLOR,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        state.set_clear_color(0.0, 1.0, 0.0, 1.0);
        state.set_scissor(0, 1, 2, 1);
        let errs = renderer.execute(
            &[RenderOp::Clear {
                mask: proto::CLEAR_MASK_COLOR,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        let dest = Ref {
            address: 32,
            space: RefSpace::Aperture,
            length: 16,
        };
        let errs = renderer.execute(
            &[RenderOp::ReadPixels {
                x: 0,
                y: 0,
                w: 2,
                h: 2,
                format: WireSurfaceFormat::A8r8g8b8,
                row_bytes: 8,
                flags: proto::READ_PIXELS_FLAG_ROWS_BOTTOM_UP,
                dest,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        // A R G B. Bottom-up: output row 0 = source row 1 (green), output
        // row 1 = source row 0 (red) -- the reverse of plain top-down.
        let green = [0xFFu8, 0x00, 0xFF, 0x00];
        let red = [0xFFu8, 0xFF, 0x00, 0x00];
        assert_eq!(
            &mem.aperture[32..36],
            &green,
            "output row 0 must be the source's bottom row"
        );
        assert_eq!(
            &mem.aperture[40..44],
            &red,
            "output row 1 must be the source's top row"
        );
    }

    /// `READ_PIXELS` with `format = Depth` reads the depth buffer, not
    /// the colour one, as a 32-bit unsigned value scaled from wgpu's own
    /// `[0, 1]` depth range -- `0` at the near plane, `0xFFFFFFFF` at the
    /// far plane, per the spec. A `CLEAR_DEPTH` to a known value is the
    /// simplest way to put a known depth in the buffer without a real
    /// draw.
    #[test]
    fn read_pixels_of_depth_reads_the_depth_buffer_as_a_scaled_u32() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: 2,
                height: 2,
                stride_bytes: 8,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);
        state.set_clear_depth(0.5);
        let mut mem = TestMemory::new(64);
        let errs = renderer.execute(
            &[RenderOp::Clear {
                mask: proto::CLEAR_MASK_DEPTH,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        let dest = Ref {
            address: 0,
            space: RefSpace::Aperture,
            length: 16,
        };
        let errs = renderer.execute(
            &[RenderOp::ReadPixels {
                x: 0,
                y: 0,
                w: 2,
                h: 2,
                format: WireSurfaceFormat::Depth,
                row_bytes: 8,
                flags: 0,
                dest,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        let expected = ((0.5f64) * u32::MAX as f64).round() as u32;
        for chunk in mem.aperture[0..16].chunks_exact(4) {
            let value = u32::from_be_bytes(chunk.try_into().unwrap());
            // Within 1 of the expected scaled value: acceptable f32
            // rounding either side of the exact midpoint.
            assert!(
                (value as i64 - expected as i64).abs() <= 1,
                "expected roughly {expected}, got {value}"
            );
        }
    }

    /// Draws a single fully-covering triangle at a fixed window-space
    /// depth of 0.5 onto a freshly-cleared depth buffer, then reads that
    /// depth back at a pixel known to be inside it. `set_extra_state` sets
    /// up `POLYGON_OFFSET`/`Capability::PolygonOffsetFill` (or leaves them
    /// at their default-off state, for the baseline run).
    fn drawn_depth_at_center(set_extra_state: impl FnOnce(&mut State)) -> Option<u32> {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return None;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: 16,
                height: 16,
                stride_bytes: 64,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);
        state.set_clear_depth(0.0);
        set_extra_state(&mut state);

        // A large triangle fully covering the whole 16x16 surface, at
        // window-space z = 0.5.
        let tri = [(-16.0f32, -16.0f32), (-16.0, 48.0), (48.0, -16.0)];
        let mut verts = Vec::new();
        for (x, y) in tri {
            for w in [x, y, 0.5f32, 1.0f32] {
                verts.extend_from_slice(&w.to_bits().to_be_bytes());
            }
        }
        let mut mem = TestMemory::new(65536);
        let errs = renderer.execute(
            &[
                RenderOp::Clear {
                    mask: proto::CLEAR_MASK_DEPTH,
                },
                RenderOp::Draw {
                    prim: PrimitiveType::Triangles,
                    format: VertexFormat(0),
                    count: 3,
                    window_space: true,
                    vertices: DrawVertices::Inline(&verts),
                },
            ],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        let dest = Ref {
            address: 0,
            space: RefSpace::Aperture,
            length: 4,
        };
        let errs = renderer.execute(
            &[RenderOp::ReadPixels {
                x: 8,
                y: 8,
                w: 1,
                h: 1,
                format: WireSurfaceFormat::Depth,
                row_bytes: 4,
                flags: 0,
                dest,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        Some(u32::from_be_bytes(mem.aperture[0..4].try_into().unwrap()))
    }

    /// `POLYGON_OFFSET` must actually reach the pipeline's depth bias --
    /// before this test, `set_polygon_offset`'s state was decoded but
    /// `build_pipeline` always used `DepthBiasState::default()`, so no
    /// amount of offset changed a thing. The exact GL-to-hardware scale
    /// for `units` is implementation-defined (the spec itself leaves it
    /// that way), so this only checks that a large offset moves the
    /// written depth by something far larger than ordinary f32 rounding
    /// noise -- not a specific formula.
    #[test]
    fn polygon_offset_shifts_the_written_depth_value() {
        let Some(baseline) = drawn_depth_at_center(|_| {}) else {
            return; // no adapter
        };
        let Some(offset) = drawn_depth_at_center(|s| {
            s.enable(state::Capability::PolygonOffsetFill);
            s.set_polygon_offset(0.0, 100_000.0);
        }) else {
            return; // no adapter
        };
        assert!(
            (offset as i64 - baseline as i64).abs() > 1_000_000,
            "a large POLYGON_OFFSET units value should shift the written \
             depth well past f32 rounding noise: baseline={baseline}, offset={offset}"
        );
    }

    /// Clears `surface` id `1` (16x16) to `top` for rows `0..8` and
    /// `bottom` for rows `8..16`, via a plain clear plus one scissored
    /// clear -- shared by the two `TEX_COPY_IMAGE`/`TEX_COPY_SUBIMAGE`
    /// tests below, which need a surface whose two halves are
    /// distinguishable so a copy's `(x, y)` origin is actually load-
    /// bearing, not just "some colour landed in the texture".
    fn two_tone_surface(
        renderer: &mut Renderer,
        state: &mut State,
        top: [f32; 4],
        bottom: [f32; 4],
    ) {
        state.surface_define(
            1,
            Surface {
                width: 16,
                height: 16,
                stride_bytes: 64,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);
        let mut mem = TestMemory::new(65536);
        state.set_clear_color(bottom[0], bottom[1], bottom[2], bottom[3]);
        let errs = renderer.execute(
            &[RenderOp::Clear {
                mask: proto::CLEAR_MASK_COLOR,
            }],
            state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        state.set_clear_color(top[0], top[1], top[2], top[3]);
        state.enable(state::Capability::ScissorTest);
        state.set_scissor(0, 0, 16, 8);
        let errs = renderer.execute(
            &[RenderOp::Clear {
                mask: proto::CLEAR_MASK_COLOR,
            }],
            state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        state.disable(state::Capability::ScissorTest);
    }

    /// `TEX_COPY_IMAGE` must capture the draw surface's *rectangle*, not
    /// just some pixels from it -- copies the bottom (blue) half of a
    /// two-tone surface and checks every texel came out blue, not the top
    /// half's red. A wrong `origin` on the GPU-side
    /// `copy_texture_to_texture` call (the easy mistake: it takes a
    /// separate origin per side, source and destination) would silently
    /// copy from the wrong place without erroring.
    #[test]
    fn tex_copy_image_captures_the_draw_surfaces_rectangle() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };
        let mut state = State::new(state::Limits::default());
        two_tone_surface(
            &mut renderer,
            &mut state,
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        );

        let mut mem = TestMemory::new(65536);
        let errs = renderer.execute(
            &[RenderOp::TexCopyImage {
                id: 1,
                level: 0,
                format: TexFormat::Rgba8,
                x: 0,
                y: 8,
                width: 16,
                height: 8,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        let tex = renderer
            .textures
            .get(&1)
            .expect("TEX_COPY_IMAGE created it");
        assert_eq!((tex.width, tex.height), (16, 8));
        let pixels = renderer
            .copy_texture_to_cpu(&tex.texture, tex.width, tex.height)
            .unwrap();
        for (i, px) in pixels.chunks(4).enumerate() {
            assert_eq!(
                px,
                [0, 0, 255, 255],
                "texel {i} should be the bottom half's blue, not the top half's red"
            );
        }
    }

    /// `TEX_COPY_SUBIMAGE` overwrites an existing level *in place* --
    /// captures the top (red) half with `TEX_COPY_IMAGE`, then replaces
    /// the whole thing with the bottom (blue) half via
    /// `TEX_COPY_SUBIMAGE`, and checks the texture actually changed.
    #[test]
    fn tex_copy_subimage_overwrites_the_existing_level_in_place() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };
        let mut state = State::new(state::Limits::default());
        two_tone_surface(
            &mut renderer,
            &mut state,
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        );

        let mut mem = TestMemory::new(65536);
        let errs = renderer.execute(
            &[RenderOp::TexCopyImage {
                id: 1,
                level: 0,
                format: TexFormat::Rgba8,
                x: 0,
                y: 0,
                width: 16,
                height: 8,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        let tex = renderer.textures.get(&1).unwrap();
        let before = renderer
            .copy_texture_to_cpu(&tex.texture, tex.width, tex.height)
            .unwrap();
        assert_eq!(
            &before[..4],
            [255, 0, 0, 255],
            "sanity: captured the red top half first"
        );

        let errs = renderer.execute(
            &[RenderOp::TexCopySubImage {
                id: 1,
                level: 0,
                xoff: 0,
                yoff: 0,
                x: 0,
                y: 8,
                width: 16,
                height: 8,
            }],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        let tex = renderer.textures.get(&1).unwrap();
        let after = renderer
            .copy_texture_to_cpu(&tex.texture, tex.width, tex.height)
            .unwrap();
        for (i, px) in after.chunks(4).enumerate() {
            assert_eq!(
                px,
                [0, 0, 255, 255],
                "texel {i} should now be the bottom half's blue after TEX_COPY_SUBIMAGE"
            );
        }
    }

    /// Draws a single fully-covering triangle at window-space `rhw` (so
    /// fog distance -- `1 / rhw` when the format carries no `FOGCOORD` --
    /// is under the caller's control) with `Capability::Fog` enabled, and
    /// returns the RGBA8 bytes at a pixel known to be inside it.
    fn draw_fogged_covering_triangle(
        rhw: f32,
        current_color: [f32; 4],
        set_fog_state: impl FnOnce(&mut State),
    ) -> Option<[u8; 4]> {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return None;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: 16,
                height: 16,
                stride_bytes: 64,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);
        state.set_clear_color(0.0, 0.0, 0.0, 1.0);
        let [r, g, b, a] = current_color;
        state.set_current_color(r, g, b, a);
        state.enable(state::Capability::Fog);
        set_fog_state(&mut state);

        let tri = [(-16.0f32, -16.0f32), (-16.0, 48.0), (48.0, -16.0)];
        let mut verts = Vec::new();
        for (x, y) in tri {
            for w in [x, y, 0.5f32, rhw] {
                verts.extend_from_slice(&w.to_bits().to_be_bytes());
            }
        }
        let mut mem = TestMemory::new(65536);
        let errs = renderer.execute(
            &[
                RenderOp::Clear {
                    mask: proto::CLEAR_MASK_COLOR,
                },
                RenderOp::Draw {
                    prim: PrimitiveType::Triangles,
                    format: VertexFormat(0),
                    count: 3,
                    window_space: true,
                    vertices: DrawVertices::Inline(&verts),
                },
            ],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        let (_, _, pixels) = renderer.read_surface_rgba8(1).unwrap();
        let px = &pixels[(8 * 16 + 8) * 4..][..4];
        Some([px[0], px[1], px[2], px[3]])
    }

    /// `FOG` must actually reach the fragment shader -- picks `rhw` so
    /// `1 / rhw` (the fog distance a `FOGCOORD`-less format derives, per
    /// the window-space rule) lands exactly on `fog_start` in one draw and
    /// exactly on `fog_end` in another, giving `LINEAR` fog factors of
    /// exactly `1.0` and `0.0` -- clean byte-exact endpoints with no
    /// rounding tolerance needed, unlike an arbitrary mid-fog distance.
    #[test]
    fn linear_fog_blends_toward_the_fog_colour_based_on_distance_from_1_over_rhw() {
        let sample = |rhw: f32| {
            draw_fogged_covering_triangle(rhw, [1.0, 1.0, 1.0, 1.0], |s| {
                s.set_fog_mode(FogMode::Linear);
                s.set_fog_params(0.0, 1.0, 5.0); // density unused by LINEAR
                s.set_fog_color(0.0, 0.0, 0.0, 1.0);
            })
        };

        let Some(at_start) = sample(1.0) else {
            return; // no adapter; distance = 1/1.0 = fog_start
        };
        assert_eq!(
            at_start,
            [255, 255, 255, 255],
            "at fog_start, colour should pass through completely unfogged: {at_start:?}"
        );

        let Some(at_end) = sample(0.2) else {
            return; // distance = 1/0.2 = fog_end
        };
        assert_eq!(
            at_end,
            [0, 0, 0, 255],
            "at fog_end, colour should be fully replaced by the fog colour: {at_end:?}"
        );
    }

    /// `FOGCOORD`, when the vertex format carries it, must override the
    /// `1 / rhw` default entirely -- an `rhw` that alone would land
    /// exactly on `fog_start` (no fog) still ends up fully fogged once an
    /// explicit per-vertex `FOGCOORD` names `fog_end` instead.
    #[test]
    fn an_explicit_fogcoord_overrides_the_rhw_derived_distance() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: 16,
                height: 16,
                stride_bytes: 64,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);
        state.set_clear_color(0.0, 0.0, 0.0, 1.0);
        state.enable(state::Capability::Fog);
        state.set_fog_mode(FogMode::Linear);
        state.set_fog_params(0.0, 1.0, 5.0);
        state.set_fog_color(0.0, 0.0, 0.0, 1.0);

        // rhw = 1.0 (distance would be fog_start, i.e. unfogged) but every
        // vertex carries an explicit FOGCOORD of fog_end instead.
        // POS(4) + COLOR(4) + FOGCOORD(1) per vertex.
        let tri = [(-16.0f32, -16.0f32), (-16.0, 48.0), (48.0, -16.0)];
        let mut verts = Vec::new();
        for (x, y) in tri {
            for w in [x, y, 0.5f32, 1.0f32] {
                verts.extend_from_slice(&w.to_bits().to_be_bytes());
            }
            for c in [1.0f32, 1.0, 1.0, 1.0] {
                verts.extend_from_slice(&c.to_bits().to_be_bytes());
            }
            verts.extend_from_slice(&5.0f32.to_bits().to_be_bytes()); // FOGCOORD = fog_end
        }

        let mut mem = TestMemory::new(65536);
        let errs = renderer.execute(
            &[
                RenderOp::Clear {
                    mask: proto::CLEAR_MASK_COLOR,
                },
                RenderOp::Draw {
                    prim: PrimitiveType::Triangles,
                    format: VertexFormat(VertexFormat::COLOR | VertexFormat::FOGCOORD),
                    count: 3,
                    window_space: true,
                    vertices: DrawVertices::Inline(&verts),
                },
            ],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        let (_, _, pixels) = renderer.read_surface_rgba8(1).unwrap();
        let px = &pixels[(8 * 16 + 8) * 4..][..4];
        assert_eq!(
            &px[..3],
            &[0, 0, 0],
            "an explicit FOGCOORD of fog_end should fully fog the pixel \
             even though 1/rhw alone would have meant no fog at all: {px:?}"
        );
    }

    // -- DRAW_INLINE (GL-space, CAP_TRANSFORM, position transform only) --

    /// Builds and executes a `RenderOp::Draw { window_space: false, .. }`
    /// against a fresh 16x16 surface cleared to black, drawing `verts`
    /// (already-encoded GL-space vertex bytes, `POS(4) + COLOR(4)`) as one
    /// `TRIANGLES` primitive, and returns the rendered RGBA8 pixels (or
    /// `None` if no adapter is available, the usual skip-cleanly path).
    fn draw_gl_space_triangle(
        set_extra_state: impl FnOnce(&mut State),
        verts: &[u8],
    ) -> Option<Vec<u8>> {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return None;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: 16,
                height: 16,
                stride_bytes: 64,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);
        state.set_clear_color(0.0, 0.0, 0.0, 1.0);
        set_extra_state(&mut state);

        let mut mem = TestMemory::new(65536);
        let errs = renderer.execute(
            &[
                RenderOp::Clear {
                    mask: proto::CLEAR_MASK_COLOR,
                },
                RenderOp::Draw {
                    prim: PrimitiveType::Triangles,
                    format: VertexFormat(VertexFormat::COLOR),
                    count: 3,
                    window_space: false,
                    vertices: DrawVertices::Inline(verts),
                },
            ],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        let (_, _, pixels) = renderer.read_surface_rgba8(1).unwrap();
        Some(pixels)
    }

    fn encode_gl_vertex(pos: [f32; 4], color: [f32; 4], out: &mut Vec<u8>) {
        for w in pos {
            out.extend_from_slice(&w.to_bits().to_be_bytes());
        }
        for c in color {
            out.extend_from_slice(&c.to_bits().to_be_bytes());
        }
    }

    /// Proves `DRAW_INLINE` transforms object-space positions by
    /// `modelview` *then* `projection` -- not the other order, and not
    /// both folded into one matrix -- by putting a rotation on the
    /// `MODELVIEW` stack and a translation on the (separate) `PROJECTION`
    /// stack. `ROTATE(90, 0, 0, 1)` and `TRANSLATE(1, 0, 0)` do not
    /// commute, so applying them in the wrong order sends the triangle
    /// somewhere else entirely, not just to a slightly different pixel --
    /// a test that folded both onto one stack (or used two commuting
    /// operations) would only re-check `state.rs`'s own within-stack
    /// `MULT_MATRIX` ordering (already covered by
    /// `mult_matrix_post_multiplies` there), not this module's
    /// modelview-before-projection composition.
    ///
    /// The triangle's object-space anchor `(0, 1, 0)` is chosen so the
    /// *correct* order (`eye = modelview.transform_point(obj)`, `clip =
    /// projection.transform_point(eye)`, i.e. `clip = TRANSLATE(ROTATE(obj))`)
    /// lands exactly on NDC `(0, 0)` -- the surface's centre pixel, `(8,
    /// 8)` on a 16x16 surface -- while the wrong order (`ROTATE(TRANSLATE(obj))`)
    /// lands near NDC `(-0.7, 0.7)` to `(-1.3, 1.0)`, nowhere near the
    /// centre. Radius `0.3` keeps the triangle comfortably larger than a
    /// pixel (a rotation and a translation are both isometries in `x`/
    /// `y`, so nothing here shrinks it), avoiding the subpixel-coverage
    /// ambiguity a scaled-down test triangle would risk.
    ///
    /// Verified per this session's practice: temporarily swapping the two
    /// `transform_point` calls in `op_draw_inline_gl` (applying
    /// `projection` before `modelview`) was confirmed to fail this
    /// assertion (the centre pixel stayed the clear colour) before this
    /// fix was restored.
    #[test]
    fn draw_inline_transforms_by_modelview_then_projection() {
        let anchor = (0.0f32, 1.0f32);
        let tri = [
            (anchor.0 - 0.3, anchor.1 - 0.3),
            (anchor.0 + 0.3, anchor.1 - 0.3),
            (anchor.0, anchor.1 + 0.3),
        ];
        let mut verts = Vec::new();
        for (x, y) in tri {
            encode_gl_vertex([x, y, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], &mut verts);
        }

        let Some(pixels) = draw_gl_space_triangle(
            |s| {
                s.set_matrix_mode(MatrixMode::Modelview);
                s.rotate(90.0, 0.0, 0.0, 1.0);
                s.set_matrix_mode(MatrixMode::Projection);
                s.translate(1.0, 0.0, 0.0);
            },
            &verts,
        ) else {
            return; // no adapter
        };

        let center = &pixels[(8 * 16 + 8) * 4..][..4];
        assert_eq!(
            &center[..3],
            &[255, 255, 255],
            "the correctly-ordered modelview*projection transform should \
             land the triangle's centre on the surface's centre pixel: {center:?}"
        );
    }

    /// `VIEWPORT` (transform-tier-only, unlike `DRAW_INLINE_WIN` which
    /// ignores it per the spec) must be folded into the NDC mapping: a
    /// full-NDC-covering triangle (`(-1,-1)`, `(3,-1)`, `(-1,3)`, which
    /// over-covers `[-1, 1]^2` the same way `drawn_depth_at_center`'s
    /// window-space triangle over-covers its surface) restricted to
    /// `VIEWPORT(4, 4, 8, 8)` on a 16x16 surface should colour only that
    /// 8x8 sub-rectangle, leaving every pixel outside it at the clear
    /// colour.
    #[test]
    fn draw_inline_folds_viewport_into_the_full_surface_ndc_mapping() {
        let tri = [(-1.0f32, -1.0), (3.0, -1.0), (-1.0, 3.0)];
        let mut verts = Vec::new();
        for (x, y) in tri {
            encode_gl_vertex([x, y, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], &mut verts);
        }

        let Some(pixels) = draw_gl_space_triangle(
            |s| {
                s.set_viewport(4, 4, 8, 8);
            },
            &verts,
        ) else {
            return; // no adapter
        };

        // Inside the viewport: coloured white.
        let inside = &pixels[(8 * 16 + 8) * 4..][..4];
        assert_eq!(
            &inside[..3],
            &[255, 255, 255],
            "(8, 8) is inside VIEWPORT(4, 4, 8, 8) and should be coloured: {inside:?}"
        );
        // Outside the viewport (top-left corner): still the clear colour.
        let outside = &pixels[(1 * 16 + 1) * 4..][..4];
        assert_eq!(
            &outside[..3],
            &[0, 0, 0],
            "(1, 1) is outside VIEWPORT(4, 4, 8, 8) and must stay the clear colour: {outside:?}"
        );
    }

    /// `DEPTH_RANGE(near, far)` maps GL clip-space `z` (post-divide, `[-1,
    /// 1]`) onto `[near, far]`, per the spec's "OpenGL 1.1 ... including
    /// the `[-1, 1]` clip-space depth range mapped through `DEPTH_RANGE`".
    /// With identity modelview/projection, an object-space `z` of exactly
    /// `-1.0`/`1.0` *is* the post-divide clip `z` (`w = 1`), so the
    /// written device depth should land exactly on `near`/`far` -- clean
    /// endpoints, avoiding the float-rounding ambiguity a mid-range value
    /// would risk (the same reasoning `POLYGON_OFFSET`'s and the LINEAR
    /// fog test's endpoint choices followed this session).
    #[test]
    fn draw_inline_maps_depth_range_onto_near_and_far() {
        let sample = |z: f32| -> Option<u32> {
            let mut renderer = match Renderer::new() {
                Ok(r) => r,
                Err(RenderError::NoAdapter) => {
                    eprintln!("skipping: no wgpu adapter available");
                    return None;
                }
                Err(e) => panic!("unexpected renderer error: {e}"),
            };
            let mut state = State::new(state::Limits::default());
            state.surface_define(
                1,
                Surface {
                    width: 16,
                    height: 16,
                    stride_bytes: 64,
                    format: WireSurfaceFormat::A8r8g8b8,
                    backing: state::Backing::Aperture(0),
                },
            );
            state.set_draw_surface(1);
            state.set_clear_depth(0.0);
            state.set_depth_range(0.25, 0.75);

            let tri = [(-16.0f32, -16.0), (-16.0, 48.0), (48.0, -16.0)];
            let mut verts = Vec::new();
            for (x, y) in tri {
                encode_gl_vertex([x, y, z, 1.0], [1.0, 1.0, 1.0, 1.0], &mut verts);
            }

            let mut mem = TestMemory::new(65536);
            let errs = renderer.execute(
                &[
                    RenderOp::Clear {
                        mask: proto::CLEAR_MASK_DEPTH,
                    },
                    RenderOp::Draw {
                        prim: PrimitiveType::Triangles,
                        format: VertexFormat(VertexFormat::COLOR),
                        count: 3,
                        window_space: false,
                        vertices: DrawVertices::Inline(&verts),
                    },
                ],
                &state,
                &mut mem,
            );
            assert!(errs.is_empty(), "unexpected errors: {errs:?}");

            let dest = Ref {
                address: 0,
                space: RefSpace::Aperture,
                length: 4,
            };
            let errs = renderer.execute(
                &[RenderOp::ReadPixels {
                    x: 8,
                    y: 8,
                    w: 1,
                    h: 1,
                    format: WireSurfaceFormat::Depth,
                    row_bytes: 4,
                    flags: 0,
                    dest,
                }],
                &state,
                &mut mem,
            );
            assert!(errs.is_empty(), "unexpected errors: {errs:?}");
            Some(u32::from_be_bytes(mem.aperture[0..4].try_into().unwrap()))
        };

        let Some(at_near) = sample(-1.0) else {
            return; // no adapter
        };
        let expected_near = (0.25f64 * u32::MAX as f64).round() as u32;
        assert!(
            (at_near as i64 - expected_near as i64).abs() <= 1,
            "object z = -1.0 should map to DEPTH_RANGE's near (0.25): expected ~{expected_near}, got {at_near}"
        );

        let Some(at_far) = sample(1.0) else {
            return; // no adapter
        };
        let expected_far = (0.75f64 * u32::MAX as f64).round() as u32;
        assert!(
            (at_far as i64 - expected_far as i64).abs() <= 1,
            "object z = 1.0 should map to DEPTH_RANGE's far (0.75): expected ~{expected_far}, got {at_far}"
        );
    }

    /// A vertex whose clip-space `w` is exactly `0.0` -- legal on the
    /// wire (an explicit `POS` `w` of `0.0`, or, under a real `FRUSTUM`/
    /// `PERSPECTIVE` projection, any object point with eye-space `z ==
    /// 0`) -- must not turn into a GPU-side `NaN`. `clip_pos.w` is what
    /// wgpu's own hardware perspective divide (`clip_pos.xyz /
    /// clip_pos.w`) uses to recover NDC; if the *raw* (unguarded)
    /// `clip[3]` is written there while `clip_pos.xyz` was built from a
    /// `safe_w`-divided NDC and then re-multiplied by that same raw,
    /// zero `clip_w`, the result is `clip_pos = [0, 0, z, 0]` and the
    /// GPU computes `0 / 0 = NaN` -- silently, with no `RenderError` at
    /// all, since nothing on the Rust side ever divides by the raw zero.
    ///
    /// This constructs the same over-sized covering triangle
    /// `drawn_depth_at_center`/`draw_inline_maps_depth_range_onto_near_and_far`
    /// use, but gives its third vertex an explicit `w = 0.0` (identity
    /// modelview/projection, so `clip == obj` exactly and this vertex's
    /// `clip[3]` really is `0.0`, not just close to it). With the fix
    /// (`safe_w` used consistently for every occurrence of `w` in
    /// `clip_pos`, including the last component), this vertex becomes an
    /// enormous but finite point in the same direction its `w = 1.0`
    /// self would have been, and the triangle -- built from two ordinary
    /// vertices plus this one -- still covers the surface's centre pixel.
    ///
    /// Verified per this session's practice: reverting the fix (using
    /// the raw `clip[3]`/`clip_w` for `clip_pos`'s `x`/`y`/`z`/`w`
    /// components, as the first version of this function did) was
    /// confirmed to fail this assertion (the centre pixel stayed the
    /// clear colour -- the degenerate `NaN` vertex collapses the
    /// triangle instead of rendering it) before the fix was restored.
    #[test]
    fn draw_inline_survives_a_zero_clip_w() {
        let mut verts = Vec::new();
        for (x, y) in [(-16.0f32, -16.0), (-16.0, 48.0)] {
            encode_gl_vertex([x, y, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], &mut verts);
        }
        // The third vertex: same direction as an ordinary covering
        // triangle's corner, but with an explicit w = 0.0.
        encode_gl_vertex([48.0, -16.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], &mut verts);

        let Some(pixels) = draw_gl_space_triangle(|_| {}, &verts) else {
            return; // no adapter
        };

        let center = &pixels[(8 * 16 + 8) * 4..][..4];
        assert_eq!(
            &center[..3],
            &[255, 255, 255],
            "a vertex with clip w = 0.0 must not collapse the triangle to \
             nothing via a GPU-side NaN divide; centre pixel: {center:?}"
        );
    }

    /// `parse_gl_vertices` must accept `NORMAL` (illegal only in a
    /// window-space draw, per `ring.rs`'s `check_window_space_format`) and
    /// still walk its three words so a following component lands at the
    /// right offset -- mirroring `parse_window_vertices`'s existing
    /// coverage of `NORMAL` as an "unused but consumed" component.
    #[test]
    fn parse_gl_vertices_accepts_and_skips_normal() {
        let format = VertexFormat(
            (1 << VertexFormat::POS_COUNT_SHIFT) | VertexFormat::NORMAL | VertexFormat::COLOR,
        );
        let mut bytes = Vec::new();
        for w in [1.0f32, 2.0, 3.0] {
            // POS (3 words: POS_COUNT = 1)
            bytes.extend_from_slice(&w.to_bits().to_be_bytes());
        }
        // The wire's fixed component order is POS, then bit order (COLOR
        // is bit 0, NORMAL is bit 1), so COLOR precedes NORMAL here --
        // not the format table's row order, which merely lists bits.
        for w in [0.5f32, 0.25, 0.75, 1.0] {
            // COLOR
            bytes.extend_from_slice(&w.to_bits().to_be_bytes());
        }
        for w in [0.0f32, 0.0, 1.0] {
            // NORMAL (skipped)
            bytes.extend_from_slice(&w.to_bits().to_be_bytes());
        }

        let verts = parse_gl_vertices(&bytes, format, 1, &current()).unwrap();
        assert_eq!(verts.len(), 1);
        assert_eq!(verts[0].pos, [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(verts[0].color, [0.5, 0.25, 0.75, 1.0]);
    }

    /// `QUERY` needs no GPU work at all -- `dispatch::Context` has
    /// already resolved the state into a `QueryResult`, so this checks
    /// only that `op_query` writes the right bytes, in the wire's
    /// big-endian convention, to the right place. A `Renderer` is still
    /// needed to call `execute` on (device creation itself may skip with
    /// no adapter, even though this op never touches it).
    #[test]
    fn a_query_writes_its_result_as_big_endian_f32s_to_the_dest_ref() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let state = State::new(state::Limits::default());
        let mut mem = TestMemory::new(256);

        // A vec4-shaped result (as CURRENT_COLOR/VIEWPORT/etc. all are),
        // written to aperture offset 16 so a non-zero base offset is
        // exercised too.
        let result = QueryResult {
            values: [
                1.5, -2.25, 3.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            count: 4,
        };
        let dest = Ref {
            address: 16,
            space: RefSpace::Aperture,
            length: 16,
        };

        let errs = renderer.execute(&[RenderOp::Query { dest, result }], &state, &mut mem);
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");

        let written = mem.aperture[16..32].to_vec();
        let expected: Vec<u8> = [1.5f32, -2.25, 3.0, 0.5]
            .iter()
            .flat_map(|f| f.to_be_bytes())
            .collect();
        assert_eq!(
            written, expected,
            "QUERY must write big-endian f32s, matching every other guest-visible value on this wire"
        );
    }

    /// A `count` shorter than 16 (a vec4/vec2 result, not a matrix) must
    /// write only that many floats -- nothing past the result's own
    /// length, so a small destination buffer is never over-run and a
    /// caller reading exactly `count` floats back sees nothing stray
    /// after them.
    #[test]
    fn a_query_writes_exactly_count_floats_not_the_full_16_word_buffer() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let state = State::new(state::Limits::default());
        let mut mem = TestMemory::new(256);

        let result = QueryResult {
            values: [9.0; 16],
            count: 2, // e.g. CURRENT_TEXTURE_COORDS (s, t)
        };
        let dest = Ref {
            address: 0,
            space: RefSpace::Aperture,
            length: 8,
        };
        let errs = renderer.execute(&[RenderOp::Query { dest, result }], &state, &mut mem);
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        assert_eq!(mem.aperture[0..8].to_vec(), 9.0f32.to_be_bytes().repeat(2));
    }

    /// Culling is the one piece of window-space rasterisation whose
    /// orientation cannot be reasoned about safely: the vertex stage maps
    /// y-down window space to wgpu's y-up NDC, and wgpu's viewport
    /// transform maps NDC back to a y-down framebuffer, so the two flips
    /// cancel and a triangle's winding in framebuffer space -- which is
    /// where WebGPU decides facing -- is its winding in window space. A
    /// triangle wound counter-clockwise in the spec's y-down window space,
    /// with `FRONT_FACE` `CCW` and `CULL_FACE` `BACK`, must therefore
    /// survive. This renders it and asserts pixels actually arrived.
    #[test]
    fn a_front_facing_triangle_survives_back_face_culling() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        // Counter-clockwise with y pointing DOWN: (1,1) -> (1,15) -> (15,1)
        // sweeps anticlockwise on screen.
        let ccw = [(1.0f32, 1.0f32), (1.0, 15.0), (15.0, 1.0)];
        let cw = [(1.0f32, 1.0f32), (15.0, 1.0), (1.0, 15.0)];

        let covered = |renderer: &mut Renderer, tri: [(f32, f32); 3]| -> usize {
            let mut state = State::new(state::Limits::default());
            state.surface_define(
                1,
                Surface {
                    width: 16,
                    height: 16,
                    stride_bytes: 64,
                    format: WireSurfaceFormat::A8r8g8b8,
                    backing: state::Backing::Aperture(0),
                },
            );
            state.set_draw_surface(1);
            state.set_clear_color(0.0, 0.0, 0.0, 1.0);
            state.enable(state::Capability::CullFace);
            state.set_cull_face(Face::Back);
            state.set_front_face(FrontFace::Ccw);

            let mut verts = Vec::new();
            for (x, y) in tri {
                for w in [x, y, 0.5f32, 1.0f32] {
                    verts.extend_from_slice(&w.to_bits().to_be_bytes());
                }
            }
            let mut mem = TestMemory::new(65536);
            let errs = renderer.execute(
                &[
                    RenderOp::Clear {
                        mask: proto::CLEAR_MASK_COLOR | proto::CLEAR_MASK_DEPTH,
                    },
                    RenderOp::Draw {
                        prim: PrimitiveType::Triangles,
                        format: VertexFormat(0),
                        count: 3,
                        window_space: true,
                        vertices: DrawVertices::Inline(&verts),
                    },
                ],
                &state,
                &mut mem,
            );
            assert!(errs.is_empty(), "unexpected errors: {errs:?}");
            let (_, _, pixels) = renderer.read_surface_rgba8(1).unwrap();
            pixels.chunks(4).filter(|px| px[..3] != [0, 0, 0]).count()
        };

        let front = covered(&mut renderer, ccw);
        let back = covered(&mut renderer, cw);
        assert!(
            front > 0,
            "a front-facing (CCW in y-down window space) triangle was culled"
        );
        assert_eq!(
            back, 0,
            "a back-facing (CW in y-down window space) triangle was drawn"
        );
    }

    /// GL applies `CullFace` to polygons only -- a `LINES` draw must land
    /// pixels even with `Face::FrontAndBack` enabled (which discards every
    /// triangle unconditionally). Before `op_draw_inline_win`'s own
    /// "cull == FrontAndBack -> nothing to draw" short-circuit was made
    /// topology-aware, it fired for every primitive type, silently
    /// dropping every line/point draw whenever a client legitimately left
    /// FrontAndBack culling enabled between polygon and line draws.
    #[test]
    fn cull_face_front_and_back_never_discards_a_line_draw() {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: 16,
                height: 16,
                stride_bytes: 64,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);
        state.set_clear_color(0.0, 0.0, 0.0, 1.0);
        state.set_current_color(1.0, 1.0, 1.0, 1.0);
        state.enable(state::Capability::CullFace);
        state.set_cull_face(Face::FrontAndBack);

        // A horizontal line straight through the pixel this test samples
        // -- 8.5, not 8.0, to sit on row 8's pixel *center* rather than
        // the ambiguous boundary between rows 7 and 8.
        let line = [(2.0f32, 8.5f32), (14.0, 8.5)];
        let mut verts = Vec::new();
        for (x, y) in line {
            for w in [x, y, 0.5f32, 1.0f32] {
                verts.extend_from_slice(&w.to_bits().to_be_bytes());
            }
        }
        let mut mem = TestMemory::new(65536);
        let errs = renderer.execute(
            &[
                RenderOp::Clear {
                    mask: proto::CLEAR_MASK_COLOR,
                },
                RenderOp::Draw {
                    prim: PrimitiveType::Lines,
                    format: VertexFormat(0),
                    count: 2,
                    window_space: true,
                    vertices: DrawVertices::Inline(&verts),
                },
            ],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        let (_, _, pixels) = renderer.read_surface_rgba8(1).unwrap();
        let px = &pixels[(8 * 16 + 8) * 4..][..4];
        assert_eq!(
            &px[..3],
            &[255, 255, 255],
            "the line should have lit this pixel regardless of FrontAndBack culling: {px:?}"
        );
    }

    /// Draws a single fully-covering opaque-red triangle over a cleared
    /// surface with `Blend` enabled, and returns the RGBA8 bytes at a
    /// pixel known to be inside it. Shared by the two `BLEND_FUNC_SEPARATE`/
    /// `BLEND_EQUATION` tests below, which only differ in the blend state
    /// they set before drawing.
    fn draw_blended_covering_triangle(
        clear_color: (f32, f32, f32, f32),
        current_color: (f32, f32, f32, f32),
        set_blend_state: impl FnOnce(&mut State),
    ) -> Option<[u8; 4]> {
        let mut renderer = match Renderer::new() {
            Ok(r) => r,
            Err(RenderError::NoAdapter) => {
                eprintln!("skipping: no wgpu adapter available");
                return None;
            }
            Err(e) => panic!("unexpected renderer error: {e}"),
        };

        let mut state = State::new(state::Limits::default());
        state.surface_define(
            1,
            Surface {
                width: 16,
                height: 16,
                stride_bytes: 64,
                format: WireSurfaceFormat::A8r8g8b8,
                backing: state::Backing::Aperture(0),
            },
        );
        state.set_draw_surface(1);
        let (cr, cg, cb, ca) = clear_color;
        state.set_clear_color(cr, cg, cb, ca);
        let (vr, vg, vb, va) = current_color;
        state.set_current_color(vr, vg, vb, va);
        state.enable(state::Capability::Blend);
        set_blend_state(&mut state);

        // A large triangle fully covering the whole 16x16 surface.
        let tri = [(-16.0f32, -16.0f32), (-16.0, 48.0), (48.0, -16.0)];
        let mut verts = Vec::new();
        for (x, y) in tri {
            for w in [x, y, 0.5f32, 1.0f32] {
                verts.extend_from_slice(&w.to_bits().to_be_bytes());
            }
        }
        let mut mem = TestMemory::new(65536);
        let errs = renderer.execute(
            &[
                RenderOp::Clear {
                    mask: proto::CLEAR_MASK_COLOR,
                },
                RenderOp::Draw {
                    prim: PrimitiveType::Triangles,
                    format: VertexFormat(0),
                    count: 3,
                    window_space: true,
                    vertices: DrawVertices::Inline(&verts),
                },
            ],
            &state,
            &mut mem,
        );
        assert!(errs.is_empty(), "unexpected errors: {errs:?}");
        let (_, _, pixels) = renderer.read_surface_rgba8(1).unwrap();
        let px = &pixels[(8 * 16 + 8) * 4..][..4];
        Some([px[0], px[1], px[2], px[3]])
    }

    /// `BLEND_FUNC_SEPARATE` must apply its RGB factors to colour and its
    /// alpha factors to alpha independently -- not share one factor pair
    /// across both, which is what the pipeline did before this test
    /// existed. Colour uses (One, Zero) so it comes through unchanged from
    /// the fragment; alpha uses (Zero, One) so it comes through unchanged
    /// from the destination instead -- a shared-factor implementation
    /// would give alpha the fragment's value (1.0) rather than the
    /// destination's (0.4).
    #[test]
    fn blend_func_separate_applies_independent_rgb_and_alpha_factors() {
        let Some(px) =
            draw_blended_covering_triangle((0.0, 0.0, 0.0, 0.4), (1.0, 0.0, 0.0, 1.0), |s| {
                s.set_blend_func_separate(
                    BlendFactor::One,
                    BlendFactor::Zero,
                    BlendFactor::Zero,
                    BlendFactor::One,
                );
            })
        else {
            return; // no adapter
        };
        assert_eq!(
            &px[..3],
            &[255, 0, 0],
            "colour factors (One, Zero) should pass the fragment's RGB through unchanged: {px:?}"
        );
        assert!(
            (px[3] as i32 - 102).abs() <= 2,
            "alpha factors (Zero, One) should pass the destination's alpha (0.4 ~= 102) through, not the fragment's: {px:?}"
        );
    }

    /// `BLEND_EQUATION` must select the combine operation, not just the
    /// factors -- `FuncSubtract` with (One, One) computes `src - dst`
    /// rather than the default `FuncAdd`'s `src + dst`.
    #[test]
    fn blend_equation_selects_the_combine_operation() {
        let Some(px) = draw_blended_covering_triangle(
            (50.0 / 255.0, 0.0, 0.0, 1.0),
            (200.0 / 255.0, 0.0, 0.0, 1.0),
            |s| {
                s.set_blend_func_separate(
                    BlendFactor::One,
                    BlendFactor::One,
                    BlendFactor::One,
                    BlendFactor::One,
                );
                s.set_blend_equation(BlendEquation::FuncSubtract);
            },
        ) else {
            return; // no adapter
        };
        assert!(
            (px[0] as i32 - 150).abs() <= 2,
            "FuncSubtract with (One, One) should compute src(200) - dst(50) = 150: {px:?}"
        );
    }
}
