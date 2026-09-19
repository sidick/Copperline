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
//!   (`DRAW_INLINE_WIN` only; `DRAW_ARRAYS_WIN`/`DRAW_ELEMENTS_WIN` and
//!   every GL-space shape are deferred -- see [`RenderError::Unimplemented`]),
//!   `POS_COUNT` 4/3/2, `COLOR`, `COLOR_PACKED` and `TEXCOORD0`, correct
//!   `rhw` handling, every primitive type with the spec's normative quad/
//!   polygon triangulation and provoking vertex, flat and smooth shading.
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
//! - Depth test and mask, blend func (not yet separate/equation), alpha
//!   test, cull face and front face, scissor, viewport (window-space draws
//!   ignore it, per the spec -- viewport is transform-tier), colour mask.
//!
//! Deferred to M3: GL-space transform/lighting draws, `DRAW_ARRAYS*`/
//! `DRAW_ELEMENTS*`, multitexture (`TEXCOORD1`-`3`), texel-space texture
//! coordinates, `TEX_COPY_IMAGE`/`TEX_COPY_SUBIMAGE`/`TEX_PALETTE`, fog,
//! polygon offset, line/point primitives
//! (only `TRIANGLES`-family and `POINTS`/`LINES`-as-degenerate-triangles
//! are not special-cased -- see [`triangulate`]). Every one of these
//! returns [`RenderError::Unimplemented`] rather than panicking.
//!
//! ## Two traps the spec calls out
//!
//! - **Clip-space depth convention.** GL's clip space is `z in [-1, 1]`;
//!   wgpu's is `z in [0, 1]`. Window-space vertices already carry final
//!   `[0, 1]` surface depth (the spec: "`z` is depth in `0..1`"), so no
//!   fixup is needed on the path this milestone renders. The fixup a
//!   GL-space (M3) vertex pipeline will need is [`gl_clip_z_to_wgpu`],
//!   implemented and unit-tested here now so M3 has it ready.
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
    self, BlendEquation, BlendFactor, CompareFunc, Face, FrontFace, PrimitiveType, ShadeModel,
    State, Surface, SurfaceFormat as WireSurfaceFormat, TexEnvMode, TexFilter, TexFormat, TexWrap,
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
}

fn read_f32(bytes: &[u8], at: usize) -> Option<f32> {
    let w: [u8; 4] = bytes.get(at..at + 4)?.try_into().ok()?;
    Some(f32::from_be_bytes(w))
}

fn read_u32_be(bytes: &[u8], at: usize) -> Option<u32> {
    let w: [u8; 4] = bytes.get(at..at + 4)?.try_into().ok()?;
    Some(u32::from_be_bytes(w))
}

/// Parses `count` window-space vertices out of `data`, `format`-
/// interleaved, exactly as `DRAW_INLINE_WIN` lays them out
/// (`docs/internals/c3d.md`'s "Vertex format"). Every optional component
/// in the format is walked and its words consumed -- including ones this
/// milestone doesn't render (`NORMAL`, `TEXCOORD1`-`3`, `FOGCOORD`) -- so
/// a later vertex in the same command parses at the right offset even
/// though this milestone only *uses* `COLOR`/`COLOR_PACKED`/`TEXCOORD0`.
/// `current` supplies the value for any component the format omits.
/// `Err` if `data` runs out before `count` vertices are read.
pub fn parse_window_vertices(
    data: &[u8],
    format: VertexFormat,
    count: u32,
    current: &state::CurrentVertex,
) -> Result<Vec<WinVertex>, RenderError> {
    let Some(pos_words) = format.pos_words() else {
        return Err(RenderError::ShortPayload("POS_COUNT"));
    };
    let mut out = Vec::with_capacity(count as usize);
    let mut at = 0usize;
    let bad = || RenderError::ShortPayload("window-space vertex");

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
        if format.has(VertexFormat::FOGCOORD) {
            at += 4;
            if at > data.len() {
                return Err(bad());
            }
        }

        out.push(WinVertex {
            pos,
            color,
            texcoord0: [texcoord0.0, texcoord0.1],
        });
    }
    Ok(out)
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
/// `Err(RenderError::Unimplemented)` -- M1's traces are triangle/quad
/// shaped, and point/line rasterisation is deferred, per the module doc
/// comment.
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
/// `__FRAGMENT_BODY__` (the texenv/alpha-test combination). M1's
/// fixed-function content is deliberately thin (no lighting, no fog --
/// see the module doc comment); the specialisation mechanism is what
/// carries forward to M3, not the amount of WGSL below it.
const VERTEX_SHADER: &str = r#"
struct VsIn {
    @location(0) clip_pos: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) texcoord0: vec2<f32>,
};
struct VsOut {
    @builtin(position) position: vec4<f32>,
    __FLAT__ @location(0) color: vec4<f32>,
    @location(1) texcoord0: vec2<f32>,
};
@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.position = in.clip_pos;
    out.color = in.color;
    out.texcoord0 = in.texcoord0;
    return out;
}
"#;

const FRAGMENT_SHADER_HEADER: &str = r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
    __FLAT__ @location(0) color: vec4<f32>,
    @location(1) texcoord0: vec2<f32>,
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
    let flat = if key.flat_shading {
        "@interpolate(flat)"
    } else {
        ""
    };
    let header = FRAGMENT_SHADER_HEADER.replace("__FLAT__", flat);
    format!(
        "{header}\n{alpha_ref_decl}@fragment\nfn fs_main(in: VsOut) -> @location(0) vec4<f32> {{\n    {sample}\n    {combine}\n    {alpha_test}    return outc;\n}}\n",
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
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: front_face_to_wgpu(if key.front_face_ccw {
                        FrontFace::Ccw
                    } else {
                        FrontFace::Cw
                    }),
                    cull_mode: cull_to_wgpu(key.cull),
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
                    bias: wgpu::DepthBiasState::default(),
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
                if !*window_space {
                    return Err(RenderError::Unimplemented(
                        "GL-space draws (CAP_TRANSFORM, M3)",
                    ));
                }
                let DrawVertices::Inline(bytes) = vertices else {
                    return Err(RenderError::Unimplemented(
                        "DRAW_ARRAYS_WIN/DRAW_ELEMENTS_WIN (M3)",
                    ));
                };
                self.op_draw_inline_win(*prim, *format, *count, bytes, state)
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
            RenderOp::TexCopyImage { .. } => Err(RenderError::Unimplemented("TEX_COPY_IMAGE (M3)")),
            RenderOp::TexCopySubImage { .. } => {
                Err(RenderError::Unimplemented("TEX_COPY_SUBIMAGE (M3)"))
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
        let tris = triangulate(prim, verts.len())?;
        if tris.is_empty() {
            return Ok(());
        }

        let (w, h) = (def.width as f32, def.height as f32);
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
            }
        };

        let mut vbuf: Vec<GpuVertex> = Vec::with_capacity(tris.len() * 3);
        for tri in &tris {
            for &idx in tri {
                let Some(v) = verts.get(idx) else {
                    return Err(RenderError::ShortPayload("triangulated vertex index"));
                };
                vbuf.push(to_gpu(v));
            }
        }

        let unit0 = state.texture_units.first();
        let bound_tex = unit0.map(|u| u.bound_texture).unwrap_or(0);
        let unit0_bound = bound_tex != 0 && self.textures.contains_key(&bound_tex);
        let surface_format = INTERNAL_COLOR_FORMAT;
        let key = self.pipeline_key_from_state(state, unit0_bound, surface_format);

        if key.cull == Some(FaceKey::FrontAndBack) {
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
/// ready clip-space `(x*w, y*w, z*w, w)`, `color`/`texcoord0` are the
/// undivided attribute values the spec describes -- the hardware's own
/// perspective-correct interpolation (driven by `clip_pos.w`) does exactly
/// the "interpolate `attr * rhw`, recover `attr` per fragment" the spec
/// asks for, with no per-fragment division written by this module.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GpuVertex {
    clip_pos: [f32; 4],
    color: [f32; 4],
    texcoord0: [f32; 2],
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
    fn skips_unused_components_words_correctly_to_reach_the_next_vertex() {
        // Two POS-only vertices with FOGCOORD in between to skip.
        let format = VertexFormat(1 << VertexFormat::POS_COUNT_SHIFT | VertexFormat::FOGCOORD);
        let mut bytes = Vec::new();
        for vertex in [[1.0f32, 2.0, 0.0], [3.0, 4.0, 0.0]] {
            for v in vertex {
                bytes.extend_from_slice(&f32be(v));
            }
            bytes.extend_from_slice(&f32be(9.0)); // fogcoord, unused by this parser
        }
        let verts = parse_window_vertices(&bytes, format, 2, &current()).unwrap();
        assert_eq!(verts.len(), 2);
        assert_eq!(verts[0].pos[0..2], [1.0, 2.0]);
        assert_eq!(verts[1].pos[0..2], [3.0, 4.0]);
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
