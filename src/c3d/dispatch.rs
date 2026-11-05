// SPDX-License-Identifier: GPL-3.0-or-later

//! The C3D dispatch layer: the connection `docs/internals/c3d.md` calls the
//! emulation-thread half of the "Threading" note -- consuming a submission
//! and "produc[ing] a stream of renderer commands". [`ring`] turns ring
//! bytes into [`proto::Command`]s; [`state`] holds one context's GL state
//! machine; nothing before this module ties the two together. [`Context`]
//! is that tie: one C3D context's registers plus its [`state::State`], and
//! [`Context::submit`] is the doorbell -- the single entry point a board
//! layer calls when the guest writes `RING_TAIL`.
//!
//! ## What this module does *not* do
//!
//! No rendering, no rasterisation, no pixel data, no GPU types of any
//! kind: [`RenderOp`] *describes* work for a renderer to perform, it never
//! performs any. No register decode either -- `submit` is hand a `ring`
//! slice and a `new_tail` value already resolved from `RING_BASE`/
//! `RING_SIZE`/`RING_TAIL`, exactly as [`ring::RingCursor::new`] expects;
//! decoding `CTX_CONTROL`/`CTX_STATUS` and the rest of the context register
//! page is a later milestone's board layer.
//!
//! A command whose bulk payload is named by a [`proto::Ref`] (a
//! [`RenderOp::TexImage`]'s pixels, a [`RenderOp::Query`]'s destination)
//! stays a `Ref` all the way through this module: `submit` only ever sees
//! the ring's own bytes, never the aperture or guest memory the `Ref`
//! points into, so it has no way to resolve one even if it wanted to. Only
//! a command's payload that is genuinely inline in the ring -- a
//! `DRAW_INLINE`'s vertices, a `DRAW_ARRAYS`'s array descriptors -- can be
//! (and is) handed onward as actual bytes; see [`RenderOp`]'s doc comment
//! for why those stay borrowed rather than being copied.
//!
//! ## Error routing
//!
//! Two latches, kept apart exactly as `docs/internals/c3d.md`'s "Errors"
//! section keeps them apart, each first-error-wins and each independent of
//! the other:
//!
//! | Source | Latch | Effect |
//! |---|---|---|
//! | [`ring::Step::Error`] | [`Context::error_code`]/[`Context::error_offset`] | skip, or halt if [`proto::ErrorCode::halts`] |
//! | [`state::State::require_draw_surface`]/[`state::State::check_rect`]/[`state::State::check_draw_surface_rect`] returning `Err` | the same protocol latch | skip, never halts |
//! | a `from_gl`/`from_wire` conversion returning `None` | [`state::State`]'s own `GL_ERROR` latch | skip, keep decoding |
//! | [`state::State::push_matrix`]/[`state::State::pop_matrix`] over/underflow | the `GL_ERROR` latch (raised inside those methods themselves) | skip, keep decoding |
//!
//! `state.rs`'s own module doc comment names this module as the "caller"
//! that turns a `from_gl`/`from_wire` `None` into a GL error by calling
//! [`state::State::raise_gl_error`] -- `ring.rs` cannot do it, since it
//! never imports `state.rs`'s typed enums at all. This module applies that
//! rule uniformly to every typed enum `state.rs` exposes, `from_wire`
//! (device-own vocabulary: [`state::TexFormat`], [`state::SurfaceFormat`],
//! [`state::TexCoordSpace`]) included, not only `from_gl`: `state.rs`'s
//! comment draws no distinction between the two constructors for this
//! purpose, and treating a bad device-format value as anything other than
//! a GL error would need a code path the spec's error table has no slot
//! for.
//!
//! `E_NO_SURFACE`/`E_BAD_RECT` are applied at exactly the opcodes
//! `docs/internals/c3d.md` and this milestone's brief name: `CLEAR` and
//! every draw get [`state::State::require_draw_surface`] alone (neither
//! carries a rectangle payload); `SURFACE_UPLOAD`/`SURFACE_READBACK` get
//! [`state::State::check_draw_surface_rect`] (surface *and* rectangle,
//! always the draw surface -- neither is framebuffer-aware). `READ_
//! PIXELS` and the copy-to-texture commands (`TEX_COPY_IMAGE`/`TEX_COPY_
//! SUBIMAGE` and their `_FACE` siblings) get [`state::State::check_read_
//! source_rect`] instead: the draw surface, or (`CAP_PROGRAMMABLE`, a
//! non-default read framebuffer bound) that framebuffer's `READ_BUFFER`-
//! selected attachment's actual size -- see that method's doc comment.
//! `SCISSOR`/`VIEWPORT` are never rectangle-checked, per `state.rs`'s own
//! note and its test that pins that behaviour.

use super::proto::{self, Command, ErrorCode, Ref};
use super::ring::{self, DeviceConfig, RingCursor};
use super::state::{self, GlError, State};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------
// Renderer operations
// ---------------------------------------------------------------------

/// One instruction for a renderer to carry out, in the order `submit`
/// produced it. Everything that is pure state -- capabilities, blend and
/// depth modes, matrices, lights, texture parameters, which texture is
/// bound where, surface *definitions* -- never appears here: a renderer
/// reads that straight out of the [`Context`]'s [`state::State`], exactly
/// as `docs/internals/c3d.md`'s "Design principle" describes the device
/// (the guest streams changes cheaply; the device -- here, the renderer --
/// holds the state). A `RenderOp` carries only what a renderer needs *in
/// addition to* that state to execute one particular unit of work: a
/// draw's per-call primitive/format/vertex data, an upload's pixel
/// reference, a query's already-resolved result, a fence's id.
///
/// ## Why some variants also carry a `State` snapshot
///
/// "A renderer reads [state] straight out of the `Context`" above needs
/// one correction: for any op whose GPU effect actually depends on
/// time-varying state (`Clear`, every `Draw`/`DrawArraysProg`/
/// `DrawElementsProg`, `SurfaceUpload`/`SurfaceReadback`, `TexCopyImage`/
/// `TexCopySubImage`, `ReadPixels`), reading `Context::state` live at
/// *execute* time is wrong, not just an optimisation left on the table.
/// `Context::submit` decodes and applies an entire ring submission
/// before any of the `RenderOp`s it produced are executed (`execute` is
/// a separate pass over the whole batch, by every caller today -- see
/// `board.rs`'s doorbell and `c3d-trace.rs`'s replay loop) -- so a bare
/// `&State` handed to `execute` reflects whatever the state happens to
/// be at the *end* of that submission, not what it was when each
/// individual command actually ran. A guest doing "bind framebuffer A,
/// draw, bind framebuffer B, draw" or "set scissor, draw, clear
/// scissor, draw" in one doorbell write would get both draws seeing
/// framebuffer B, or both seeing no scissor -- silently wrong, not an
/// error. These variants instead carry their own `State`, cloned in
/// [`Context::apply`] at the exact moment the command is applied (every
/// earlier command in the batch has already mutated `self.state` by
/// then, and no later one has yet), exactly extending the pattern
/// `Query`'s `result` and `FramebufferStatus`'s `status` already use for
/// a single resolved value rather than a whole state snapshot. `State`
/// is a plain `Clone`, and a clone per draw/clear/readback is
/// negligible next to the GPU work it accompanies.
///
/// ## Borrowed vs. owned bulk data
///
/// [`RenderOp::Draw`]'s inline vertices and array descriptors are borrowed
/// (`&'a [u8]`, tied to the `ring` slice `Context::submit` was called
/// with) rather than copied into an owned `Vec<u8>`. Three reasons:
///
/// 1. It is the zero-copy property `ring.rs` was explicitly built to keep
///    (see that module's doc comment: "allocates nothing in the decode
///    path"), and this module's whole job is to sit downstream of it
///    without giving that property away by default.
/// 2. It is the *only* bulk payload this module ever sees as bytes at
///    all -- everything else that isn't inline in the command (texture
///    images, array vertex data proper, query destinations) is named by a
///    [`Ref`] into the aperture or guest memory, which `submit` cannot
///    reach and so passes through unresolved regardless of this choice.
/// 3. `docs/internals/c3d.md`'s "Threading" note already puts a real
///    thread boundary between decode (this module, on the emulation
///    thread) and the renderer (a separate worker); crossing that
///    boundary needs owned (`'static`) data no matter what a pure
///    decode-layer type does, so the copy such a hand-off requires happens
///    exactly once, at the actual boundary, in whatever glue owns that
///    channel -- not redundantly in here as well.
///
/// The consequence for a caller: the `RenderOp`s a `submit` call produces
/// borrow from the `ring` slice passed to that same call, so they must be
/// fully consumed (rendered, or copied into an owned queue) before that
/// slice's backing bytes can legitimately be considered free -- in
/// practice, before the next command overwriting that ring region is
/// allowed to be decoded. A caller driving everything on one thread (the
/// native conformance-trace runner `docs/internals/c3d.md`'s "Conformance"
/// section describes) can simply drain `out` synchronously after each
/// `submit` and never pay a copy at all.
#[derive(Debug, Clone, PartialEq)]
pub enum RenderOp<'a> {
    /// `CLEAR`. `mask` is `CLEAR_MASK_COLOR`/`CLEAR_MASK_DEPTH`/
    /// `CLEAR_MASK_STENCIL` as decoded; the colour/depth/stencil values,
    /// and whether scissor/colour-mask apply, come from `state.raster`
    /// -- a full [`state::State`] snapshot taken *at the moment this
    /// command was applied* (see this type's doc comment on why every
    /// op below that depends on time-varying state carries one).
    Clear { mask: u32, state: State },
    /// `DRAW_INLINE(_WIN)`/`DRAW_ARRAYS(_WIN)`/`DRAW_ELEMENTS(_WIN)`.
    /// `window_space` distinguishes the `_WIN` opcodes (already-transformed
    /// vertices) from the GL-space ones, which is the one piece of
    /// [`proto::Command`] shape that isn't otherwise recoverable from
    /// `prim`/`format`/`vertices` alone.
    Draw {
        window_space: bool,
        prim: state::PrimitiveType,
        format: proto::VertexFormat,
        count: u32,
        vertices: DrawVertices<'a>,
        state: State,
    },
    /// `SURFACE_UPLOAD`. Rectangle in surface pixels, already validated
    /// against the draw surface's bounds.
    SurfaceUpload { x: u32, y: u32, w: u32, h: u32, state: State },
    /// `SURFACE_READBACK`. See the [readback contract](
    /// https://docs.internals/c3d.md#c3d-readback) for when its effects
    /// must be visible.
    SurfaceReadback { x: u32, y: u32, w: u32, h: u32, state: State },
    /// `TEX_IMAGE`. Emitted only once [`state::State::define_tex_level`]
    /// has recorded the level's shape against an existing texture object.
    TexImage {
        id: u32,
        level: u32,
        format: state::TexFormat,
        width: u32,
        height: u32,
        row_bytes: u32,
        data: Ref,
    },
    /// `TEX_SUBIMAGE`. Emitted only once this module has confirmed `id`
    /// has a level `level` already defined with a matching `format`, per
    /// the spec's "`format` must match the level's".
    TexSubImage {
        id: u32,
        level: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        format: state::TexFormat,
        row_bytes: u32,
        data: Ref,
    },
    /// `TEX_COPY_IMAGE` (`glCopyTexImage2D`): defines a level from the
    /// draw surface's rectangle.
    TexCopyImage {
        id: u32,
        level: u32,
        format: state::TexFormat,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        state: State,
    },
    /// `TEX_COPY_SUBIMAGE` (`glCopyTexSubImage2D`). Emitted only once this
    /// module has confirmed `id` has a level `level` already defined (its
    /// format is inherited, unlike `TEX_SUBIMAGE`, since the wire command
    /// carries none).
    TexCopySubImage {
        id: u32,
        level: u32,
        xoff: u32,
        yoff: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        state: State,
    },
    /// `TEX_IMAGE_FACE`. As `TexImage`, for one face of a cube-map object.
    TexImageFace {
        id: u32,
        face: u32,
        level: u32,
        format: state::TexFormat,
        width: u32,
        height: u32,
        row_bytes: u32,
        data: Ref,
    },
    /// `TEX_SUBIMAGE_FACE`. As `TexSubImage`, for one face.
    TexSubImageFace {
        id: u32,
        face: u32,
        level: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        format: state::TexFormat,
        row_bytes: u32,
        data: Ref,
    },
    /// `TEX_COPY_IMAGE_FACE`. As `TexCopyImage`, for one face.
    TexCopyImageFace {
        id: u32,
        face: u32,
        level: u32,
        format: state::TexFormat,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        state: State,
    },
    /// `TEX_COPY_SUBIMAGE_FACE`. As `TexCopySubImage`, for one face.
    TexCopySubImageFace {
        id: u32,
        face: u32,
        level: u32,
        xoff: u32,
        yoff: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        state: State,
    },
    /// `TEX_PALETTE`. Emitted only once
    /// [`state::State::set_tex_palette`] has bound it against an existing
    /// texture object.
    TexPalette { id: u32, entries: u32, data: Ref },
    /// `TEX_GENERATE_MIPS`. No `state` needed -- it operates purely on
    /// an existing texture object's own GPU-side mip levels, independent
    /// of any other context state.
    TexGenerateMips { id: u32 },
    /// `READ_PIXELS`. Rectangle already validated against the draw
    /// surface's bounds.
    ReadPixels {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
        format: state::SurfaceFormat,
        row_bytes: u32,
        flags: u32,
        dest: Ref,
        state: State,
    },
    /// `QUERY`. `result` is resolved from [`state::State`] *at the moment
    /// this command was applied* -- matching `glGet*`'s immediate-read
    /// semantics -- rather than deferred to whenever a renderer eventually
    /// executes this op, since a later matrix operation in the same or a
    /// later submission must not retroactively change an earlier query's
    /// answer. All that is left for a renderer to do is write `result` to
    /// `dest`.
    Query { dest: Ref, result: QueryResult },
    /// `FENCE`. The renderer calls [`Context::complete_fence`] once every
    /// preceding op (including a readback's or query's memory effects)
    /// has actually taken effect; `submit` never advances
    /// `FENCE_COMPLETED` itself.
    Fence { id: u32 },

    // -- CAP_PROGRAMMABLE: shaders and programs. See this module's doc
    // comment above `RenderOp` -- these carry refs (source text, an
    // attribute name) or need the renderer's own compiled-shader/program
    // tables (link, validate), which `state.rs` has no access to, exactly
    // as `TexImage`'s pixel bytes never reach `state.rs`.
    /// `SHADER_SOURCE`. Replaces the shader's source text; the renderer
    /// also clears any previous compile result, matching the spec's
    /// "replacing any previous source".
    ShaderSource { id: u32, source: Ref },
    /// `SHADER_COMPILE`. `kind` is read from `state` at dispatch time (see
    /// [`Context::apply`]) so the renderer doesn't need a `state`
    /// parameter for this op.
    ShaderCompile { id: u32, kind: state::ShaderKind },
    /// `PROGRAM_BIND_ATTRIB_LOCATION`. `name` is resolved by the renderer
    /// and kept in its own program table -- it only ever feeds
    /// `ProgramLink`, never read back by `state.rs`.
    ProgramBindAttribLocation { program: u32, index: u32, name: Ref },
    /// `PROGRAM_LINK`. `vertex`/`fragment` are the program's attached
    /// shader ids, read from `state` at dispatch time (`None` if nothing
    /// of that kind is attached, which the renderer treats as a link
    /// failure -- a program needs both stages).
    ProgramLink {
        program: u32,
        vertex: Option<u32>,
        fragment: Option<u32>,
    },
    /// `PROGRAM_VALIDATE`.
    ProgramValidate { program: u32 },
    /// `SHADER_STATUS`/`PROGRAM_STATUS` (`0x0D00`/`0x0D01`). Both write a
    /// fixed-shape status record (and the two leave their own variant
    /// rather than sharing `Query`'s `QueryResult`, which is a fixed `f32`
    /// vector -- these records are `u32` status words plus a
    /// variable-length UTF-8 log, a different shape entirely).
    ShaderStatus { shader: u32, dest: Ref },
    ProgramStatus { program: u32, dest: Ref },

    // -- CAP_PROGRAMMABLE: buffers. `BUFFER_CREATE`/`_DESTROY` and every
    // `VERTEX_ATTRIB_*` opcode carry no ref and are pure `state.rs`
    // mutations with no `RenderOp` at all -- only the two that move bytes
    // (`BUFFER_DATA`/`BUFFER_SUBDATA`) need the renderer.
    /// `BUFFER_DATA`: (re)allocates the buffer's storage from `data`.
    BufferData {
        id: u32,
        size: u32,
        usage: u32,
        data: Ref,
    },
    /// `BUFFER_SUBDATA`: replaces `size` bytes at `offset` from `data`.
    BufferSubData {
        id: u32,
        offset: u32,
        size: u32,
        data: Ref,
    },

    // -- CAP_PROGRAMMABLE: programmable draw (0x0Cxx). Unlike the
    // fixed-function `Draw` op, these carry no vertex data at all --
    // everything they draw from (the current program, the enabled
    // vertex-attribute slots, the bound draw framebuffer) comes from
    // their own `state` snapshot (see this type's doc comment on why).
    // `E_NO_PROGRAM` is caught at dispatch time (`Context::apply`), so
    // by the time either op reaches the renderer a current program is
    // guaranteed to exist.
    DrawArraysProg {
        prim: state::PrimitiveType,
        first: u32,
        count: u32,
        state: State,
    },
    DrawElementsProg {
        prim: state::PrimitiveType,
        count: u32,
        index_type: u32,
        index_ref: Ref,
        state: State,
    },

    // -- CAP_PROGRAMMABLE: uniforms. Pushed unconditionally (no `state`
    // gate, unlike most other programmable-tier ops): whether `program`
    // exists, is linked, and whether `location` is one of its active
    // uniforms is render.rs-resident knowledge (the linked interface,
    // computed from the compiled naga modules at PROGRAM_LINK), which
    // dispatch/state cannot see -- see ProgramSlot's doc comment. An
    // unknown program/location is therefore a silent no-op in the
    // renderer, not a GL_INVALID_OPERATION the spec would prefer: a
    // known, documented gap (the same one PROGRAM_BIND_ATTRIB_LOCATION's
    // doc comment already notes) rather than a retrofitted error path.
    Uniform {
        program: u32,
        location: u32,
        type_: u32,
        count: u32,
        values: &'a [u8],
    },
    UniformSampler {
        program: u32,
        location: u32,
        unit: u32,
    },
    /// `PROGRAM_INTERFACE`.
    ProgramInterface { program: u32, dest: Ref },

    // -- CAP_PROGRAMMABLE: framebuffers and renderbuffers. `FRAMEBUFFER_
    // CREATE`/`_DESTROY`, `BIND_FRAMEBUFFER`, `RENDERBUFFER_CREATE`/
    // `_DESTROY`/`_STORAGE_*` and the two `FRAMEBUFFER_ATTACH_*` opcodes
    // carry no ref and are pure `state.rs` mutations with no `RenderOp`
    // at all -- only `FRAMEBUFFER_STATUS` needs the renderer, to write
    // its result to `dest` (needs `mem` access this module doesn't
    // have). `status` is resolved from `state` immediately, exactly
    // like `Query`'s `result` -- completeness is pure state, so there is
    // nothing a renderer needs to re-derive later.
    FramebufferStatus { dest: Ref, status: u32 },
}

/// A draw command's vertex/index source, exactly as `docs/internals/
/// c3d.md`'s "Draw" section shapes it. Array and index data are named by
/// [`Ref`]s the renderer resolves itself (see [`RenderOp`]'s doc comment
/// on why this module cannot); only `Inline`'s vertices and the array
/// descriptors themselves are ring-inline bytes this module can hand over
/// directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DrawVertices<'a> {
    /// `DRAW_INLINE(_WIN)`: `count` vertices, `format`-interleaved,
    /// written straight into the ring.
    Inline(&'a [u8]),
    /// `DRAW_ARRAYS(_WIN)`: one [`proto::ArrayDescriptor`] (`type,
    /// stride_bytes, ref`) per set `format` bit, position first, packed
    /// [`proto::DESCRIPTOR_WORDS`] words apart -- undecoded, since
    /// decoding them needs no context this module has that the renderer
    /// lacks.
    Arrays(&'a [u8]),
    /// `DRAW_ELEMENTS(_WIN)`: indices at `index_ref` plus the same array
    /// descriptor bytes as `Arrays`.
    Elements {
        index_type: u32,
        min_index: u32,
        max_index: u32,
        index_ref: Ref,
        descriptors: &'a [u8],
    },
}

/// A `QUERY`'s already-resolved result: up to 16 `f32`s (a `4x4` matrix is
/// the largest), column-major where the source is a matrix, in wire order
/// otherwise. `count` says how many of `values` are meaningful.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryResult {
    pub values: [f32; 16],
    pub count: u8,
}

impl QueryResult {
    fn matrix(m: state::Mat4) -> QueryResult {
        QueryResult {
            values: m.0,
            count: 16,
        }
    }

    fn vec4(v: [f32; 4]) -> QueryResult {
        let mut values = [0.0f32; 16];
        values[..4].copy_from_slice(&v);
        QueryResult { values, count: 4 }
    }

    fn vec2(v: [f32; 2]) -> QueryResult {
        let mut values = [0.0f32; 16];
        values[..2].copy_from_slice(&v);
        QueryResult { values, count: 2 }
    }
}

/// Appends `bytes` to `storage` and returns a slice into it with the
/// caller-chosen lifetime `'a` -- deliberately **not** tied to `storage`'s
/// own (much shorter) borrow. Used by [`Context::submit_with`]/
/// [`Context::apply_call_buffer`] to keep a `CALL`ed buffer's fetched bytes
/// alive long enough for a draw command inside it to borrow from, exactly
/// as an ordinary ring-inline draw borrows from `ring` itself.
///
/// # Why a called buffer's bytes must be owned in the first place
///
/// `ring` (the top-level ring) is a slice the *caller* already holds for
/// `'a` before calling `submit`/`submit_with` at all -- `board.rs`'s
/// `doorbell` reads it into a local `Vec<u8>` up front. A called buffer's
/// bytes are different: which buffer to fetch is only known once decoding
/// reaches the `CALL` command itself, so there is no `'a`-lifetime slice to
/// hand back from `fetch_call_buffer` -- especially for a guest-space
/// (`CAP_GUESTMEM`) buffer, which has no backing slice in this process at
/// all until `DeviceHost::dma_read` copies it into a fresh buffer.
///
/// # Why that forces this function to be `unsafe`
///
/// [`RenderOp::Draw`]'s inline vertices/array descriptors need to borrow
/// `&'a [u8]`, the same lifetime the whole submission's `out` uses -- so a
/// called buffer's owned bytes need to outlive `apply_call_buffer`'s own
/// stack frame, all the way out to wherever the caller eventually drains
/// `out`. Safe Rust cannot express "push an element now, hand out a
/// long-lived borrow of it, then push again" against a single
/// `&mut Vec<Vec<u8>>`: proving two non-overlapping calls to this function
/// don't alias requires knowing that each already-pushed `Vec<u8>` element
/// is never touched again, which is exactly the invariant documented below
/// but not something the borrow checker can see through `Vec`'s ordinary
/// API.
///
/// # Safety
///
/// The caller must treat `storage` as **append-only**: nothing may remove,
/// replace, reorder, or otherwise mutate an element already pushed while
/// any slice this function previously returned from it is still alive.
/// Given that, the returned slice stays valid regardless of further
/// pushes: each element is its own independent heap allocation (a
/// `Vec<u8>`), and growing the *outer* `Vec<Vec<u8>>` can move the
/// `Vec<u8>` header (pointer/len/cap) around but never the bytes it points
/// to. `Context::submit_with`/`apply_call_buffer` uphold this themselves
/// (they only ever `Vec::push` onto `call_storage`, never truncate or
/// index-assign into it); the caller across a whole `submit_with` call is
/// additionally required to keep `call_storage` itself alive at least as
/// long as `'a` -- in practice, alongside `out`, until every `RenderOp` in
/// it has been consumed (see [`RenderOp`]'s doc comment, which already
/// asks the same of `ring`).
fn stash_call_buffer<'a>(storage: &mut Vec<Vec<u8>>, bytes: Vec<u8>) -> &'a [u8] {
    storage.push(bytes);
    let last = storage.last().expect("just pushed");
    // SAFETY: see this function's doc comment.
    unsafe { std::slice::from_raw_parts(last.as_ptr(), last.len()) }
}

fn rect_as_f32(r: state::Rect) -> [f32; 4] {
    [r.x as f32, r.y as f32, r.w as f32, r.h as f32]
}

// `QUERY`'s accepted `what` enumerants (`docs/internals/c3d.md`'s
// "Queries" section). Not a `gl_enum!` in `state.rs` because resolving one
// needs live read access to several unrelated pieces of `State`
// (whichever matrix stack, the raster rect, current-vertex state) rather
// than a single typed setter -- there is no single state field a `From`
// impl could target, so the match lives here instead.
const GL_CURRENT_COLOR: u32 = 0x0B00;
const GL_CURRENT_TEXTURE_COORDS: u32 = 0x0B03;
const GL_VIEWPORT: u32 = 0x0BA2;
const GL_MODELVIEW_MATRIX: u32 = 0x0BA6;
const GL_PROJECTION_MATRIX: u32 = 0x0BA7;
const GL_TEXTURE_MATRIX: u32 = 0x0BA8;
const GL_DEPTH_RANGE: u32 = 0x0B70;
const GL_SCISSOR_BOX: u32 = 0x0C10;

/// `TEX_PARAM`'s `pname` argument. Standard GL 1.1 enumerants; not a
/// `gl_enum!` in `state.rs` because -- unlike every `gl_enum!` there --
/// this pname doesn't select a value of one fixed type, it selects *which
/// field* of [`state::TexParam`] the (separately typed) value belongs to.
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_WRAP_S: u32 = 0x2802;
const GL_TEXTURE_WRAP_T: u32 = 0x2803;

/// `TEX_ENV`'s GL-defined `pname` for the env mode (`0x2200`); the
/// device's own `TEXCOORD_SPACE` pname is [`proto::TEX_ENV_TEXCOORD_SPACE`].
/// See [`GL_TEXTURE_MAG_FILTER`]'s note on why this lives here rather than
/// as a `gl_enum!`.
const GL_TEXTURE_ENV_MODE: u32 = 0x2200;

// ---------------------------------------------------------------------
// The context
// ---------------------------------------------------------------------

/// One C3D context: its [`state::State`] plus the register-shaped values
/// `docs/internals/c3d.md`'s "Context register pages" defines that this
/// milestone owns -- the ring pointers, `FENCE_COMPLETED`, and the
/// protocol-error latch. `IRQ_STATUS`/`IRQ_ENABLE`, `CTX_CONTROL`/
/// `CTX_STATUS` themselves and `FENCE_IRQ_TARGET` are the board layer's
/// concern (register decode, out of this milestone's scope); this type
/// carries exactly the state a board layer needs to *compute* those from,
/// plus [`Context::submit`], the doorbell itself.
///
/// `error_code`/`error_offset` are stored as the plain register values
/// (`0` = no error) rather than as `Option<proto::ErrorCode>`: unlike
/// [`state::State`], [`proto::ErrorCode`] carries no `Serialize`/
/// `Deserialize` impl (it is wire-level, owned by `proto.rs`, which this
/// milestone does not modify), and the register-shaped form is what a
/// save state and a board layer both actually want anyway.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Context {
    pub state: State,

    /// `RING_BASE`. Not read by [`Context::submit`] itself -- the caller
    /// has already resolved `RING_BASE`/`RING_SIZE` into the `ring` slice
    /// it passes -- but carried here for a board layer's register reads
    /// and for save-state fidelity.
    pub ring_base: u32,
    /// `RING_SIZE` (bits 30:0 plus the `CAP_GUESTMEM` space bit in bit 31,
    /// exactly as the register packs them). See `ring_base`'s note.
    pub ring_size: u32,
    /// `RING_HEAD`: bytes before it have been consumed.
    pub ring_head: u32,
    /// `RING_TAIL`: the guest's last doorbell value.
    pub ring_tail: u32,

    /// `FENCE_COMPLETED`. Advanced only by [`Context::complete_fence`],
    /// never by `submit` itself -- see [`RenderOp::Fence`]'s doc comment.
    pub fence_completed: u32,

    /// `ERROR_CODE`: `0` = none, else a [`proto::ErrorCode`] discriminant.
    pub error_code: u32,
    /// `ERROR_OFFSET`: the ring offset of the command that raised
    /// `error_code`.
    pub error_offset: u32,
    /// `CTX_STATUS.HALTED`.
    pub halted: bool,
}

impl Context {
    pub fn new(limits: state::Limits) -> Context {
        Context {
            state: State::new(limits),
            ring_base: 0,
            ring_size: 0,
            ring_head: 0,
            ring_tail: 0,
            fence_completed: 0,
            error_code: 0,
            error_offset: 0,
            halted: false,
        }
    }

    /// The doorbell: `RING_TAIL` has just been written `new_tail`. Decodes
    /// everything from `RING_HEAD` up to `new_tail` out of `ring` (the
    /// context's whole ring, already resolved by the caller exactly as
    /// [`ring::RingCursor::new`] expects), applies each command to
    /// [`Context::state`], appends the renderer-visible operations to
    /// `out` in order, and advances `RING_HEAD`.
    ///
    /// `RING_TAIL` is recorded unconditionally -- it is an ordinary
    /// register write and the spec never makes it conditional on decode
    /// state -- but if the context is already halted, nothing is decoded:
    /// per the spec, a halted context "keeps answering registers; only its
    /// ring stops", and decoding resumes only after [`Context::error_ack`].
    pub fn submit<'a>(
        &mut self,
        ring: &'a [u8],
        new_tail: u32,
        config: &DeviceConfig,
        out: &mut Vec<RenderOp<'a>>,
    ) {
        // No `CALL` support: a `CALL` encountered here always fails to
        // fetch (skip-class `E_BAD_REF`, latched at the `CALL`'s own ring
        // offset -- see `submit_with`), which is what every caller that
        // predates `CALL` (every test in this module) already expects to
        // never happen. `Vec::new()` allocates nothing until pushed to, and
        // nothing is ever pushed to it since the fetch closure always
        // returns `None`, so this path costs the same as before `CALL`
        // existed.
        self.submit_with(ring, new_tail, config, out, &mut Vec::new(), &mut |_| None);
    }

    /// [`Context::submit`], plus `CALL` (`0x0005`) support: `fetch_call_buffer`
    /// is asked for a called buffer's bytes given its already-validated
    /// [`Ref`] (`ring.rs`'s `validate_ref` already ran on it as part of
    /// ordinary decoding); returning `None` is reported the same as any
    /// other unresolvable ref, `E_BAD_REF` at the `CALL`'s own ring offset.
    /// `board.rs`'s `doorbell` is the real caller, implementing this with
    /// exactly the aperture-range/`DeviceHost::dma_read` fetch its own
    /// guest-ring precedent already uses -- reused here for one fetch code
    /// path instead of two.
    ///
    /// `call_storage` is where the fetched bytes are kept alive: a called
    /// buffer's bytes must be owned (see `stash_call_buffer`'s doc comment
    /// for the full reasoning and the `unsafe` this implies), and the
    /// caller must keep `call_storage` alive at least as long as `out` --
    /// in practice, until every `RenderOp` in `out` has actually been
    /// consumed, exactly the same lifetime discipline `ring` itself already
    /// requires (see [`RenderOp`]'s doc comment).
    ///
    /// **Nesting is bounded to one level by construction, not by a
    /// counter**: a `CALL` decoded from the *top-level ring* is handled
    /// right here and fetches+recurses into [`Context::apply_call_buffer`]
    /// exactly once; a `CALL` decoded *from inside* `apply_call_buffer`
    /// (i.e. already one level deep) is handled entirely within that
    /// method, which never calls back into this one or into itself. There
    /// is no code path that can go two levels deep, so there is no depth
    /// counter to get wrong and no state to restore on the way back out.
    pub fn submit_with<'a>(
        &mut self,
        ring: &'a [u8],
        new_tail: u32,
        config: &DeviceConfig,
        out: &mut Vec<RenderOp<'a>>,
        call_storage: &mut Vec<Vec<u8>>,
        fetch_call_buffer: &mut dyn FnMut(Ref) -> Option<Vec<u8>>,
    ) {
        self.ring_tail = new_tail;
        if self.halted {
            return;
        }

        let mut cursor = RingCursor::new(ring, self.ring_head, new_tail, *config);
        loop {
            let start = cursor.head();
            match cursor.step() {
                Some(ring::Step::Command(Command::Call { data })) => {
                    match fetch_call_buffer(data) {
                        Some(bytes) => {
                            let bytes = stash_call_buffer(call_storage, bytes);
                            self.apply_call_buffer(bytes, config, start, out);
                            if self.halted {
                                // A halting error inside the buffer
                                // abandons the call and halts the context;
                                // the outer cursor has already advanced
                                // past the CALL command itself (it decoded
                                // structurally fine), so breaking here and
                                // letting `self.ring_head = cursor.head()`
                                // run below is exactly the spec's "after
                                // ERROR_ACK, decoding resumes in the ring
                                // after the CALL" -- no offset bookkeeping
                                // needed.
                                break;
                            }
                        }
                        None => self.latch_protocol_error(ErrorCode::BadRef, start),
                    }
                }
                Some(ring::Step::Command(cmd)) => self.apply(cmd, start, config, out),
                Some(ring::Step::Error(err)) => self.latch_protocol_error(err.code, err.offset),
                None => break,
            }
        }
        self.ring_head = cursor.head();
        self.halted = self.halted || cursor.is_halted();
    }

    /// Decodes and applies a `CALL`ed buffer's commands in place of the
    /// `CALL` itself, per `c3d-cmd-control`'s `CALL` prose: framed exactly
    /// like the ring but linear -- no wraparound, no `RING_HEAD`/
    /// `RING_TAIL` of its own ([`RingCursor::new_linear`]) -- and every
    /// error, **including a nested `CALL`** (illegal: `E_BAD_OPCODE`, skip,
    /// decoding of this buffer continues), latching `call_start` (the
    /// *outer* `CALL` command's own ring offset) rather than a position
    /// inside the buffer, which has none of its own to report. `FENCE`
    /// needs no special handling here -- it reaches the same `apply` as
    /// everywhere else, which is what "behaves identically" means.
    fn apply_call_buffer<'a>(
        &mut self,
        bytes: &'a [u8],
        config: &DeviceConfig,
        call_start: u32,
        out: &mut Vec<RenderOp<'a>>,
    ) {
        let mut cursor = RingCursor::new_linear(bytes, *config);
        loop {
            match cursor.step() {
                // Nesting is illegal: `ring.rs` itself raises `E_BAD_OPCODE`
                // for a `CALL` decoded from a `new_linear` cursor (ahead of
                // even looking at its ref), so `Command::Call` is never
                // produced here -- it always arrives via the `Step::Error`
                // arm below instead, which already latches `call_start` and
                // (being skip-class) loops on to the next command.
                Some(ring::Step::Command(cmd)) => self.apply(cmd, call_start, config, out),
                Some(ring::Step::Error(err)) => {
                    self.latch_protocol_error(err.code, call_start);
                    if err.halt {
                        self.halted = true;
                        return;
                    }
                }
                None => return,
            }
        }
    }

    /// The renderer calls this once every op up to and including fence
    /// `id` has actually taken effect. `submit`/`apply` never call it.
    pub fn complete_fence(&mut self, id: u32) {
        self.fence_completed = id;
    }

    /// `ERROR_ACK`: clears the protocol-error latch and un-halts the
    /// context. Decoding resumes at `RING_HEAD` on the next `submit` --
    /// `RING_HEAD` was left at the offending command by the halting
    /// `submit` call, so nothing further is needed here.
    pub fn error_ack(&mut self) {
        self.error_code = 0;
        self.error_offset = 0;
        self.halted = false;
    }

    /// `CTX_RESET_STATE` (ring opcode `0x0004`): GL state only, not
    /// objects, not the ring. Also invoked internally by `apply` when the
    /// opcode itself is decoded.
    pub fn reset_state(&mut self) {
        self.state.reset_state();
    }

    /// `CTX_CONTROL.RESET` (the register bit): GL state plus every texture
    /// and surface object for this context, and **both error latches**.
    /// The ring pointers and `FENCE_COMPLETED` are deliberately left
    /// alone, which is what the spec's "without touching the ring
    /// pointers" asks for; the protocol latch is not a ring pointer, and
    /// a context that stayed halted across an explicit reset -- or that
    /// inherited a halt when freed and reallocated -- would be stuck with
    /// `ERROR_ACK` as its only way out. See `docs/internals/c3d.md`'s
    /// `CTX_CONTROL` entry.
    pub fn reset_all(&mut self) {
        self.state.reset_all();
        self.error_ack();
    }

    fn latch_protocol_error(&mut self, code: ErrorCode, offset: u32) {
        if self.error_code == 0 {
            self.error_code = code as u32;
            self.error_offset = offset;
        }
    }

    /// Applies one already-decoded command: mutates `state`, and pushes
    /// zero or one [`RenderOp`] (most commands are pure state and push
    /// none). `start` is the command's ring offset, needed only for a
    /// [`state::State`]-detected protocol error (`ring.rs`'s own errors
    /// already carry their offset in [`ring::ProtoError`]).
    fn apply<'a>(&mut self, cmd: Command<'a>, start: u32, config: &DeviceConfig, out: &mut Vec<RenderOp<'a>>) {
        match cmd {
            Command::Nop | Command::Flush | Command::Finish => {}
            Command::Fence { id } => out.push(RenderOp::Fence { id }),
            Command::CtxResetState => self.state.reset_state(),
            // Intercepted in `submit_with`/`apply_call_buffer` before `apply`
            // is ever called for it (handling it means fetching more bytes
            // and running a nested decode pass, which `apply` has no cursor
            // access to do) -- see this type's doc comment. Never reached.
            Command::Call { .. } => {
                debug_assert!(false, "Command::Call must be intercepted before apply");
            }

            Command::SurfaceDefine {
                id,
                width,
                height,
                stride_bytes,
                format,
                flags,
                address,
            } => {
                let Some(fmt) = self.decode_gl(state::SurfaceFormat::from_wire(format)) else {
                    return;
                };
                let backing = if flags & proto::SURFACE_DEFINE_FLAG_GUEST_ADDR != 0 {
                    state::Backing::Guest(address)
                } else {
                    state::Backing::Aperture(address)
                };
                self.state.surface_define(
                    id,
                    state::Surface {
                        width,
                        height,
                        stride_bytes,
                        format: fmt,
                        backing,
                    },
                );
            }
            Command::SurfaceDestroy { id } => self.state.surface_destroy(id),
            Command::SetDrawSurface { id } => self.state.set_draw_surface(id),
            Command::SurfaceUpload { x, y, w, h } => {
                match self.state.check_draw_surface_rect(x as i32, y as i32, w, h) {
                    Ok(()) => out.push(RenderOp::SurfaceUpload {
                        x,
                        y,
                        w,
                        h,
                        state: self.state.clone(),
                    }),
                    Err(code) => self.latch_protocol_error(code, start),
                }
            }
            Command::SurfaceReadback { x, y, w, h } => {
                match self.state.check_draw_surface_rect(x as i32, y as i32, w, h) {
                    Ok(()) => out.push(RenderOp::SurfaceReadback {
                        x,
                        y,
                        w,
                        h,
                        state: self.state.clone(),
                    }),
                    Err(code) => self.latch_protocol_error(code, start),
                }
            }
            Command::Clear { mask } => match self.state.require_draw_surface() {
                Ok(_) => out.push(RenderOp::Clear {
                    mask,
                    state: self.state.clone(),
                }),
                Err(code) => self.latch_protocol_error(code, start),
            },

            Command::Enable { cap } => {
                let Some(cap) = self.decode_gl(state::Capability::from_gl(cap)) else {
                    return;
                };
                if cap == state::Capability::StencilTest && !config.programmable {
                    self.state.raise_gl_error(GlError::InvalidEnum);
                    return;
                }
                self.state.enable(cap);
            }
            Command::Disable { cap } => {
                let Some(cap) = self.decode_gl(state::Capability::from_gl(cap)) else {
                    return;
                };
                if cap == state::Capability::StencilTest && !config.programmable {
                    self.state.raise_gl_error(GlError::InvalidEnum);
                    return;
                }
                self.state.disable(cap);
            }
            Command::BlendFunc { sfactor, dfactor } => {
                let s = self.decode_gl(state::BlendFactor::from_gl(sfactor));
                let d = self.decode_gl(state::BlendFactor::from_gl(dfactor));
                let (Some(s), Some(d)) = (s, d) else { return };
                self.state.set_blend_func(s, d);
            }
            Command::DepthFunc { func } => {
                let Some(f) = self.decode_gl(state::DepthFunc::from_gl(func)) else {
                    return;
                };
                self.state.set_depth_func(f);
            }
            Command::DepthMask { flag } => self.state.set_depth_mask(flag != 0),
            Command::DepthRange { near, far } => self.state.set_depth_range(near, far),
            Command::AlphaFunc { func, reference } => {
                let Some(f) = self.decode_gl(state::AlphaFunc::from_gl(func)) else {
                    return;
                };
                self.state.set_alpha_func(f, reference);
            }
            Command::CullFace { mode } => {
                let Some(m) = self.decode_gl(state::Face::from_gl(mode)) else {
                    return;
                };
                self.state.set_cull_face(m);
            }
            Command::FrontFace { mode } => {
                let Some(m) = self.decode_gl(state::FrontFace::from_gl(mode)) else {
                    return;
                };
                self.state.set_front_face(m);
            }
            Command::ShadeModel { mode } => {
                let Some(m) = self.decode_gl(state::ShadeModel::from_gl(mode)) else {
                    return;
                };
                self.state.set_shade_model(m);
            }
            Command::ColorMask { r, g, b, a } => {
                self.state.set_color_mask(r != 0, g != 0, b != 0, a != 0)
            }
            Command::Scissor { x, y, w, h } => {
                // Deliberately not rectangle-checked: GL clamps SCISSOR.
                self.state.set_scissor(x as i32, y as i32, w, h)
            }
            Command::Viewport { x, y, w, h } => {
                // Deliberately not rectangle-checked: GL clamps VIEWPORT.
                self.state.set_viewport(x as i32, y as i32, w, h)
            }
            Command::PolygonOffset { factor, units } => {
                self.state.set_polygon_offset(factor, units)
            }
            Command::ClearColor { r, g, b, a } => self.state.set_clear_color(r, g, b, a),
            Command::ClearDepth { depth } => self.state.set_clear_depth(depth),
            Command::FogMode { mode } => {
                let Some(m) = self.decode_gl(state::FogMode::from_gl(mode)) else {
                    return;
                };
                self.state.set_fog_mode(m);
            }
            Command::FogParams {
                density,
                start: fog_start,
                end,
            } => self.state.set_fog_params(density, fog_start, end),
            Command::FogColor { r, g, b, a } => self.state.set_fog_color(r, g, b, a),
            Command::Hint { .. } => {} // Accepted and ignored, per the spec.
            Command::LineWidth { width } => self.state.set_line_width(width),
            Command::PointSize { size } => self.state.set_point_size(size),
            Command::PolygonMode { face, mode } => {
                let face = self.decode_gl(state::Face::from_gl(face));
                let mode = self.decode_gl(state::PolygonMode::from_gl(mode));
                let (Some(face), Some(mode)) = (face, mode) else {
                    return;
                };
                self.state.set_polygon_mode(face, mode);
            }
            Command::BlendEquation { mode } => {
                let Some(m) = self.decode_gl(state::BlendEquation::from_gl(mode)) else {
                    return;
                };
                self.state.set_blend_equation(m);
            }
            Command::BlendFuncSeparate {
                src_rgb,
                dst_rgb,
                src_a,
                dst_a,
            } => {
                let src_rgb = self.decode_gl(state::BlendFactor::from_gl(src_rgb));
                let dst_rgb = self.decode_gl(state::BlendFactor::from_gl(dst_rgb));
                let src_a = self.decode_gl(state::BlendFactor::from_gl(src_a));
                let dst_a = self.decode_gl(state::BlendFactor::from_gl(dst_a));
                let (Some(src_rgb), Some(dst_rgb), Some(src_a), Some(dst_a)) =
                    (src_rgb, dst_rgb, src_a, dst_a)
                else {
                    return;
                };
                self.state
                    .set_blend_func_separate(src_rgb, dst_rgb, src_a, dst_a);
            }
            Command::StencilFunc {
                func,
                reference,
                mask,
            } => {
                let Some(func) = self.decode_gl(state::StencilCompareFunc::from_gl(func)) else {
                    return;
                };
                self.state.set_stencil_func(func, reference, mask);
            }
            Command::StencilOp { fail, zfail, zpass } => {
                let fail = self.decode_gl(state::StencilOp::from_gl(fail));
                let zfail = self.decode_gl(state::StencilOp::from_gl(zfail));
                let zpass = self.decode_gl(state::StencilOp::from_gl(zpass));
                let (Some(fail), Some(zfail), Some(zpass)) = (fail, zfail, zpass) else {
                    return;
                };
                self.state.set_stencil_op(fail, zfail, zpass);
            }
            Command::StencilMask { mask } => self.state.set_stencil_mask(mask),
            Command::ClearStencil { value } => self.state.set_clear_stencil(value),

            Command::MatrixMode { mode } => {
                let Some(m) = self.decode_gl(state::MatrixMode::from_gl(mode)) else {
                    return;
                };
                self.state.set_matrix_mode(m);
            }
            Command::LoadMatrix { m } => self.state.load_matrix(m),
            Command::LoadIdentity => self.state.load_identity(),
            Command::MultMatrix { m } => self.state.mult_matrix(m),
            // The GL error, if any, is already raised inside `push_matrix`/
            // `pop_matrix` themselves (see this module's doc comment).
            Command::PushMatrix => {
                let _ = self.state.push_matrix();
            }
            Command::PopMatrix => {
                let _ = self.state.pop_matrix();
            }
            Command::Translate { x, y, z } => self.state.translate(x, y, z),
            Command::Rotate { angle_deg, x, y, z } => self.state.rotate(angle_deg, x, y, z),
            Command::Scale { x, y, z } => self.state.scale(x, y, z),
            Command::Frustum { l, r, b, t, n, f } => self.state.frustum(l, r, b, t, n, f),
            Command::Ortho { l, r, b, t, n, f } => self.state.ortho(l, r, b, t, n, f),

            Command::Light { light, pname, v } => {
                let light = self.decode_gl(state::LightId::from_gl(light));
                let pname = self.decode_gl(state::LightParam::from_gl(pname));
                let (Some(light), Some(pname)) = (light, pname) else {
                    return;
                };
                self.state.set_light(light, pname, v);
            }
            Command::LightModel { pname, v } => {
                let Some(pname) = self.decode_gl(state::LightModelParam::from_gl(pname)) else {
                    return;
                };
                self.state.set_light_model(pname, v);
            }
            Command::Material { face, pname, v } => {
                let face = self.decode_gl(state::Face::from_gl(face));
                let pname = self.decode_gl(state::MaterialParam::from_gl(pname));
                let (Some(face), Some(pname)) = (face, pname) else {
                    return;
                };
                self.state.set_material(face, pname, v);
            }
            Command::ColorMaterial { face, mode } => {
                let face = self.decode_gl(state::Face::from_gl(face));
                let mode = self.decode_gl(state::ColorMaterialMode::from_gl(mode));
                let (Some(face), Some(mode)) = (face, mode) else {
                    return;
                };
                self.state.set_color_material(face, mode);
            }
            Command::ClipPlane { plane, eq } => {
                let Some(plane) = self.decode_gl(state::ClipPlaneId::from_gl(plane)) else {
                    return;
                };
                self.state.set_clip_plane(plane, eq);
            }
            Command::TexGen { unit, coord, mode } => {
                let coord = self.decode_gl(state::TexCoord::from_gl(coord));
                let mode = self.decode_gl(state::TexGenMode::from_gl(mode));
                let (Some(coord), Some(mode)) = (coord, mode) else {
                    return;
                };
                self.state.set_texgen_mode(unit, coord, mode);
            }
            Command::TexGenPlane {
                unit,
                coord,
                plane,
                eq,
            } => {
                let coord = self.decode_gl(state::TexCoord::from_gl(coord));
                let plane = self.decode_gl(state::TexGenPlaneKind::from_gl(plane));
                let (Some(coord), Some(plane)) = (coord, plane) else {
                    return;
                };
                self.state.set_texgen_plane(unit, coord, plane, eq);
            }

            Command::TexCreate { id } => self.state.tex_create(id),
            Command::TexDestroy { id } => self.state.tex_destroy(id),
            Command::TexBind { unit, id } => self.state.tex_bind(unit, id),
            Command::TexImage {
                id,
                level,
                format,
                width,
                height,
                row_bytes,
                data,
            } => {
                let Some(fmt) = self.decode_gl(state::TexFormat::from_wire(format)) else {
                    return;
                };
                if let Err(code) = self.state.check_texture_kind(id, state::TextureKind::Flat) {
                    self.latch_protocol_error(code, start);
                    return;
                }
                if self
                    .state
                    .define_tex_level(id, level, fmt, width, height)
                    .is_ok()
                {
                    out.push(RenderOp::TexImage {
                        id,
                        level,
                        format: fmt,
                        width,
                        height,
                        row_bytes,
                        data,
                    });
                }
            }
            Command::TexSubImage {
                id,
                level,
                x,
                y,
                width,
                height,
                format,
                row_bytes,
                data,
            } => {
                let Some(fmt) = self.decode_gl(state::TexFormat::from_wire(format)) else {
                    return;
                };
                if let Err(code) = self.state.check_texture_kind(id, state::TextureKind::Flat) {
                    self.latch_protocol_error(code, start);
                    return;
                }
                let matches_existing_level = self
                    .state
                    .texture(id)
                    .and_then(|t| t.levels.get(&(0, level)))
                    .is_some_and(|existing| existing.format == fmt);
                if matches_existing_level {
                    out.push(RenderOp::TexSubImage {
                        id,
                        level,
                        x,
                        y,
                        width,
                        height,
                        format: fmt,
                        row_bytes,
                        data,
                    });
                } else {
                    self.state.raise_gl_error(GlError::InvalidOperation);
                }
            }
            Command::TexParam { id, pname, value } => {
                let Some(param) = self.decode_tex_param(pname, value) else {
                    return;
                };
                let _ = self.state.set_tex_param(id, param);
            }
            Command::TexEnv { unit, pname, value } => {
                let Some(param) = self.decode_tex_env(pname, value) else {
                    return;
                };
                self.state.set_tex_env(unit, param);
            }
            Command::TexEnvColor { unit, r, g, b, a } => {
                self.state.set_tex_env_color(unit, r, g, b, a)
            }
            Command::ActiveUnit { unit } => self.state.set_active_unit(unit),
            Command::TexPalette { id, entries, data } => {
                if self.state.set_tex_palette(id, entries).is_ok() {
                    out.push(RenderOp::TexPalette { id, entries, data });
                }
            }
            Command::TexCopyImage {
                id,
                level,
                format,
                x,
                y,
                width,
                height,
            } => {
                let Some(fmt) = self.decode_gl(state::TexFormat::from_wire(format)) else {
                    return;
                };
                if let Err(code) = self.state.check_texture_kind(id, state::TextureKind::Flat) {
                    self.latch_protocol_error(code, start);
                    return;
                }
                // The source rectangle comes off the draw surface, or
                // (`CAP_PROGRAMMABLE`, a non-default read framebuffer
                // bound) that framebuffer's `READ_BUFFER`-selected
                // attachment instead -- see `check_read_source_rect`'s
                // doc comment -- checked before the level is defined, so
                // a rejected copy leaves the texture object untouched.
                if let Err(code) = self
                    .state
                    .check_read_source_rect(x as i32, y as i32, width, height)
                {
                    self.latch_protocol_error(code, start);
                    return;
                }
                if self
                    .state
                    .define_tex_level(id, level, fmt, width, height)
                    .is_ok()
                {
                    out.push(RenderOp::TexCopyImage {
                        id,
                        level,
                        format: fmt,
                        x,
                        y,
                        width,
                        height,
                        state: self.state.clone(),
                    });
                }
            }
            Command::TexCopySubImage {
                id,
                level,
                xoff,
                yoff,
                x,
                y,
                width,
                height,
            } => {
                if let Err(code) = self.state.check_texture_kind(id, state::TextureKind::Flat) {
                    self.latch_protocol_error(code, start);
                    return;
                }
                // Source rectangle off the draw surface or the bound read
                // framebuffer, as above.
                if let Err(code) = self
                    .state
                    .check_read_source_rect(x as i32, y as i32, width, height)
                {
                    self.latch_protocol_error(code, start);
                    return;
                }
                let level_exists = self
                    .state
                    .texture(id)
                    .is_some_and(|t| t.levels.contains_key(&(0, level)));
                if level_exists {
                    out.push(RenderOp::TexCopySubImage {
                        id,
                        level,
                        xoff,
                        yoff,
                        x,
                        y,
                        width,
                        height,
                        state: self.state.clone(),
                    });
                } else {
                    self.state.raise_gl_error(GlError::InvalidOperation);
                }
            }
            // `id` must name an existing texture object (`E_BAD_ID`
            // already ruled out by `ring.rs`'s `check_id`; this is the
            // GL-error-class "exists at all" check every other texture
            // op above makes the same way). Level 0 itself being
            // undefined is deliberately *not* checked here -- the spec's
            // "generated levels are undefined but this is not an error"
            // -- `render.rs`'s `op_tex_generate_mips` handles that case
            // as a harmless no-op.
            Command::TexGenerateMips { id } => {
                if self.state.texture(id).is_some() {
                    out.push(RenderOp::TexGenerateMips { id });
                } else {
                    self.state.raise_gl_error(GlError::InvalidOperation);
                }
            }
            Command::TexCreateCube { id } => self.state.tex_create_cube(id),
            Command::TexImageFace {
                id,
                face,
                level,
                format,
                width,
                height,
                row_bytes,
                data,
            } => {
                let Some(fmt) = self.decode_gl(state::TexFormat::from_wire(format)) else {
                    return;
                };
                if let Err(code) = self.state.check_texture_kind(id, state::TextureKind::Cube) {
                    self.latch_protocol_error(code, start);
                    return;
                }
                if self
                    .state
                    .define_tex_level_face(id, face, level, fmt, width, height)
                    .is_ok()
                {
                    out.push(RenderOp::TexImageFace {
                        id,
                        face,
                        level,
                        format: fmt,
                        width,
                        height,
                        row_bytes,
                        data,
                    });
                }
            }
            Command::TexSubImageFace {
                id,
                face,
                level,
                x,
                y,
                width,
                height,
                format,
                row_bytes,
                data,
            } => {
                let Some(fmt) = self.decode_gl(state::TexFormat::from_wire(format)) else {
                    return;
                };
                if let Err(code) = self.state.check_texture_kind(id, state::TextureKind::Cube) {
                    self.latch_protocol_error(code, start);
                    return;
                }
                let matches_existing_level = self
                    .state
                    .texture(id)
                    .and_then(|t| t.levels.get(&(face, level)))
                    .is_some_and(|existing| existing.format == fmt);
                if matches_existing_level {
                    out.push(RenderOp::TexSubImageFace {
                        id,
                        face,
                        level,
                        x,
                        y,
                        width,
                        height,
                        format: fmt,
                        row_bytes,
                        data,
                    });
                } else {
                    self.state.raise_gl_error(GlError::InvalidOperation);
                }
            }
            Command::TexCopyImageFace {
                id,
                face,
                level,
                format,
                x,
                y,
                width,
                height,
            } => {
                let Some(fmt) = self.decode_gl(state::TexFormat::from_wire(format)) else {
                    return;
                };
                if let Err(code) = self.state.check_texture_kind(id, state::TextureKind::Cube) {
                    self.latch_protocol_error(code, start);
                    return;
                }
                if let Err(code) = self
                    .state
                    .check_read_source_rect(x as i32, y as i32, width, height)
                {
                    self.latch_protocol_error(code, start);
                    return;
                }
                if self
                    .state
                    .define_tex_level_face(id, face, level, fmt, width, height)
                    .is_ok()
                {
                    out.push(RenderOp::TexCopyImageFace {
                        id,
                        face,
                        level,
                        format: fmt,
                        x,
                        y,
                        width,
                        height,
                        state: self.state.clone(),
                    });
                }
            }
            Command::TexCopySubImageFace {
                id,
                face,
                level,
                xoff,
                yoff,
                x,
                y,
                width,
                height,
            } => {
                if let Err(code) = self.state.check_texture_kind(id, state::TextureKind::Cube) {
                    self.latch_protocol_error(code, start);
                    return;
                }
                if let Err(code) = self
                    .state
                    .check_read_source_rect(x as i32, y as i32, width, height)
                {
                    self.latch_protocol_error(code, start);
                    return;
                }
                let level_exists = self
                    .state
                    .texture(id)
                    .is_some_and(|t| t.levels.contains_key(&(face, level)));
                if level_exists {
                    out.push(RenderOp::TexCopySubImageFace {
                        id,
                        face,
                        level,
                        xoff,
                        yoff,
                        x,
                        y,
                        width,
                        height,
                        state: self.state.clone(),
                    });
                } else {
                    self.state.raise_gl_error(GlError::InvalidOperation);
                }
            }

            Command::CurrentColor { r, g, b, a } => self.state.set_current_color(r, g, b, a),
            Command::CurrentNormal { x, y, z } => self.state.set_current_normal(x, y, z),
            Command::CurrentTexCoord { unit, s, t } => self.state.set_current_texcoord(unit, s, t),
            Command::CurrentFogCoord { f } => self.state.set_current_fogcoord(f),

            Command::DrawInline {
                prim,
                format,
                count,
                vertices,
            } => self.draw(
                false,
                prim,
                format,
                count,
                DrawVertices::Inline(vertices),
                start,
                out,
            ),
            Command::DrawInlineWin {
                prim,
                format,
                count,
                vertices,
            } => self.draw(
                true,
                prim,
                format,
                count,
                DrawVertices::Inline(vertices),
                start,
                out,
            ),
            Command::DrawArrays {
                prim,
                format,
                count,
                descriptors,
            } => self.draw(
                false,
                prim,
                format,
                count,
                DrawVertices::Arrays(descriptors),
                start,
                out,
            ),
            Command::DrawArraysWin {
                prim,
                format,
                count,
                descriptors,
            } => self.draw(
                true,
                prim,
                format,
                count,
                DrawVertices::Arrays(descriptors),
                start,
                out,
            ),
            Command::DrawElements {
                prim,
                format,
                count,
                index_type,
                min_index,
                max_index,
                index_ref,
                descriptors,
            } => self.draw(
                false,
                prim,
                format,
                count,
                DrawVertices::Elements {
                    index_type,
                    min_index,
                    max_index,
                    index_ref,
                    descriptors,
                },
                start,
                out,
            ),
            Command::DrawElementsWin {
                prim,
                format,
                count,
                index_type,
                min_index,
                max_index,
                index_ref,
                descriptors,
            } => self.draw(
                true,
                prim,
                format,
                count,
                DrawVertices::Elements {
                    index_type,
                    min_index,
                    max_index,
                    index_ref,
                    descriptors,
                },
                start,
                out,
            ),

            Command::Query { what, dest } => match self.resolve_query(what) {
                Some(result) => out.push(RenderOp::Query { dest, result }),
                None => self.state.raise_gl_error(GlError::InvalidEnum),
            },
            Command::ReadPixels {
                x,
                y,
                w,
                h,
                format,
                row_bytes,
                flags,
                dest,
            } => {
                let Some(fmt) = self.decode_gl(state::SurfaceFormat::from_wire(format)) else {
                    return;
                };
                match self.state.check_read_source_rect(x as i32, y as i32, w, h) {
                    Ok(()) => out.push(RenderOp::ReadPixels {
                        x,
                        y,
                        w,
                        h,
                        format: fmt,
                        row_bytes,
                        flags,
                        dest,
                        state: self.state.clone(),
                    }),
                    Err(code) => self.latch_protocol_error(code, start),
                }
            }

            // -- CAP_PROGRAMMABLE: shaders and programs --------------
            Command::ShaderCreate { id, kind } => {
                if let Some(kind) = self.decode_gl(state::ShaderKind::from_gl(kind)) {
                    self.state.shader_create(id, kind);
                }
            }
            Command::ShaderDestroy { id } => self.state.shader_destroy(id),
            Command::ShaderSource { id, source } => {
                out.push(RenderOp::ShaderSource { id, source });
            }
            Command::ShaderCompile { id } => {
                if let Some(kind) = self.state.shader(id).map(|s| s.kind) {
                    out.push(RenderOp::ShaderCompile { id, kind });
                }
            }
            Command::ProgramCreate { id } => self.state.program_create(id),
            Command::ProgramDestroy { id } => self.state.program_destroy(id),
            Command::ProgramAttach { program, shader } => {
                let _ = self.state.program_attach(program, shader);
            }
            Command::ProgramDetach { program, shader } => {
                self.state.program_detach(program, shader)
            }
            Command::ProgramBindAttribLocation {
                program,
                index,
                name,
            } => out.push(RenderOp::ProgramBindAttribLocation {
                program,
                index,
                name,
            }),
            Command::ProgramLink { program } => {
                if let Some(prog) = self.state.program(program) {
                    out.push(RenderOp::ProgramLink {
                        program,
                        vertex: prog.attached_vertex,
                        fragment: prog.attached_fragment,
                    });
                }
            }
            Command::ProgramValidate { program } => {
                if self.state.program(program).is_some() {
                    out.push(RenderOp::ProgramValidate { program });
                }
            }
            Command::UseProgram { program } => self.state.use_program(program),

            // -- CAP_PROGRAMMABLE: programmable queries (0x0Dxx) -----
            Command::ShaderStatus { shader, dest } => {
                if self.state.shader(shader).is_some() {
                    out.push(RenderOp::ShaderStatus { shader, dest });
                }
            }
            Command::ProgramStatus { program, dest } => {
                if self.state.program(program).is_some() {
                    out.push(RenderOp::ProgramStatus { program, dest });
                }
            }

            // -- CAP_PROGRAMMABLE: buffers and vertex attributes -----
            Command::BufferCreate { id } => self.state.buffer_create(id),
            Command::BufferDestroy { id } => self.state.buffer_destroy(id),
            Command::BufferData {
                id,
                size,
                usage,
                data,
            } => {
                if self.state.check_buffer_exists(id).is_ok() {
                    out.push(RenderOp::BufferData {
                        id,
                        size,
                        usage,
                        data,
                    });
                }
            }
            Command::BufferSubData {
                id,
                offset,
                size,
                data,
            } => {
                if self.state.check_buffer_exists(id).is_ok() {
                    out.push(RenderOp::BufferSubData {
                        id,
                        offset,
                        size,
                        data,
                    });
                }
            }
            Command::VertexAttribEnable { index, flag } => {
                self.state.vertex_attrib_enable(index, flag != 0)
            }
            Command::VertexAttribPointer {
                index,
                size,
                type_,
                normalized,
                stride_bytes,
                offset,
                buffer,
            } => {
                if let Some(type_) = self.decode_gl(state::VertexAttribType::from_gl(type_)) {
                    self.state.vertex_attrib_pointer(
                        index,
                        state::VertexAttribPointer {
                            buffer,
                            size,
                            type_,
                            normalized: normalized != 0,
                            stride_bytes,
                            offset,
                        },
                    );
                }
            }
            Command::VertexAttribConstant { index, x, y, z, w } => {
                self.state.vertex_attrib_constant(index, [x, y, z, w])
            }

            // -- CAP_PROGRAMMABLE: framebuffers and renderbuffers ----
            Command::FramebufferCreate { id } => self.state.framebuffer_create(id),
            Command::FramebufferDestroy { id } => self.state.framebuffer_destroy(id),
            Command::BindFramebuffer { target, id } => {
                if let Some(target) = self.decode_gl(state::FramebufferTarget::from_gl(target)) {
                    self.state.bind_framebuffer(target, id);
                }
            }
            Command::RenderbufferCreate { id } => self.state.renderbuffer_create(id),
            Command::RenderbufferDestroy { id } => self.state.renderbuffer_destroy(id),
            Command::RenderbufferStorageColor {
                id,
                width,
                height,
                format,
            } => {
                let Some(format) = self.decode_gl(state::SurfaceFormat::from_wire(format)) else {
                    return;
                };
                self.state.renderbuffer_storage_color(id, width, height, format);
            }
            Command::RenderbufferStorageDepthStencil { id, width, height } => {
                self.state.renderbuffer_storage_depth_stencil(id, width, height);
            }
            Command::FramebufferAttachRenderbuffer {
                target,
                attachment,
                renderbuffer,
            } => {
                let Some(target) = self.decode_gl(state::FramebufferTarget::from_gl(target)) else {
                    return;
                };
                let Some(attachment) = self.decode_gl(state::FramebufferAttachmentPoint::from_gl(attachment, self.state.limits().max_color_attachments))
                else {
                    return;
                };
                if let Err(code) = self
                    .state
                    .framebuffer_attach_renderbuffer(target, attachment, renderbuffer)
                {
                    self.latch_protocol_error(code, start);
                }
            }
            Command::FramebufferAttachTexture {
                target,
                attachment,
                texture,
                level,
            } => {
                let Some(target) = self.decode_gl(state::FramebufferTarget::from_gl(target)) else {
                    return;
                };
                let Some(attachment) = self.decode_gl(state::FramebufferAttachmentPoint::from_gl(attachment, self.state.limits().max_color_attachments))
                else {
                    return;
                };
                if let Err(code) = self
                    .state
                    .framebuffer_attach_texture(target, attachment, texture, level)
                {
                    self.latch_protocol_error(code, start);
                }
            }
            Command::FramebufferStatus { target, dest } => {
                let Some(target) = self.decode_gl(state::FramebufferTarget::from_gl(target)) else {
                    return;
                };
                // Resolved immediately, like `Query` -- completeness is
                // pure state, so there is no reason to make the renderer
                // re-derive it later (and, before this would have been
                // just as exposed to the same stale-state hazard as
                // every other op below).
                let status = self.state.framebuffer_status(target);
                out.push(RenderOp::FramebufferStatus { dest, status });
            }
            Command::FramebufferAttachTextureFace {
                target,
                attachment,
                texture,
                face,
                level,
            } => {
                let Some(target) = self.decode_gl(state::FramebufferTarget::from_gl(target)) else {
                    return;
                };
                let Some(attachment) = self.decode_gl(state::FramebufferAttachmentPoint::from_gl(attachment, self.state.limits().max_color_attachments))
                else {
                    return;
                };
                if let Err(code) = self
                    .state
                    .framebuffer_attach_texture_face(target, attachment, texture, face, level)
                {
                    self.latch_protocol_error(code, start);
                }
            }
            Command::DrawBuffers { count, bufs } => {
                let values: Vec<u32> = (0..count as usize)
                    .map(|i| {
                        u32::from_be_bytes([
                            bufs[i * 4],
                            bufs[i * 4 + 1],
                            bufs[i * 4 + 2],
                            bufs[i * 4 + 3],
                        ])
                    })
                    .collect();
                if let Err(code) = self.state.draw_buffers(&values) {
                    self.latch_protocol_error(code, start);
                }
            }
            Command::ReadBuffer { buf } => {
                if let Err(code) = self.state.read_buffer(buf) {
                    self.latch_protocol_error(code, start);
                }
            }

            // -- CAP_PROGRAMMABLE: programmable draw -----------------
            Command::DrawArraysProg { prim, first, count } => {
                let Some(prim) = self.decode_gl(state::PrimitiveType::from_gl(prim)) else {
                    return;
                };
                match self.state.require_current_program() {
                    Ok(_) => out.push(RenderOp::DrawArraysProg {
                        prim,
                        first,
                        count,
                        state: self.state.clone(),
                    }),
                    Err(code) => self.latch_protocol_error(code, start),
                }
            }
            Command::DrawElementsProg {
                prim,
                count,
                index_type,
                index_ref,
            } => {
                let Some(prim) = self.decode_gl(state::PrimitiveType::from_gl(prim)) else {
                    return;
                };
                match self.state.require_current_program() {
                    Ok(_) => out.push(RenderOp::DrawElementsProg {
                        prim,
                        count,
                        index_type,
                        index_ref,
                        state: self.state.clone(),
                    }),
                    Err(code) => self.latch_protocol_error(code, start),
                }
            }

            // -- CAP_PROGRAMMABLE: uniforms ---------------------------
            Command::Uniform {
                program,
                location,
                type_,
                count,
                values,
            } => out.push(RenderOp::Uniform {
                program,
                location,
                type_,
                count,
                values,
            }),
            Command::UniformSampler {
                program,
                location,
                unit,
            } => out.push(RenderOp::UniformSampler {
                program,
                location,
                unit,
            }),
            Command::ProgramInterface { program, dest } => {
                if self.state.program(program).is_some() {
                    out.push(RenderOp::ProgramInterface { program, dest });
                }
            }
        }
    }

    /// The shared tail of the six draw opcodes: validate `prim` (a GL
    /// enumerant, so a bad value is a GL error like any other) and the
    /// draw surface (`E_NO_SURFACE`; draws carry no rectangle to check),
    /// then emit the op.
    #[allow(clippy::too_many_arguments)]
    fn draw<'a>(
        &mut self,
        window_space: bool,
        prim: u32,
        format: proto::VertexFormat,
        count: u32,
        vertices: DrawVertices<'a>,
        start: u32,
        out: &mut Vec<RenderOp<'a>>,
    ) {
        let Some(prim) = self.decode_gl(state::PrimitiveType::from_gl(prim)) else {
            return;
        };
        match self.state.require_draw_surface() {
            Ok(_) => out.push(RenderOp::Draw {
                window_space,
                prim,
                format,
                count,
                vertices,
                state: self.state.clone(),
            }),
            Err(code) => self.latch_protocol_error(code, start),
        }
    }

    /// `TEX_PARAM`'s `pname`/`value` pair, decoded into a
    /// [`state::TexParam`]. `None` (an unrecognised `pname`, or a `value`
    /// that fails the selected field's `from_gl`) has already raised
    /// `GL_INVALID_ENUM` by the time this returns.
    fn decode_tex_param(&mut self, pname: u32, value: u32) -> Option<state::TexParam> {
        match pname {
            GL_TEXTURE_MIN_FILTER => Some(state::TexParam::MinFilter(
                self.decode_gl(state::TexFilter::from_gl(value))?,
            )),
            GL_TEXTURE_MAG_FILTER => Some(state::TexParam::MagFilter(
                self.decode_gl(state::TexFilter::from_gl(value))?,
            )),
            GL_TEXTURE_WRAP_S => Some(state::TexParam::WrapS(
                self.decode_gl(state::TexWrap::from_gl(value))?,
            )),
            GL_TEXTURE_WRAP_T => Some(state::TexParam::WrapT(
                self.decode_gl(state::TexWrap::from_gl(value))?,
            )),
            _ => {
                self.state.raise_gl_error(GlError::InvalidEnum);
                None
            }
        }
    }

    /// `TEX_ENV`'s `pname`/`value` pair. See [`Self::decode_tex_param`].
    fn decode_tex_env(&mut self, pname: u32, value: u32) -> Option<state::TexEnvParam> {
        match pname {
            GL_TEXTURE_ENV_MODE => Some(state::TexEnvParam::Mode(
                self.decode_gl(state::TexEnvMode::from_gl(value))?,
            )),
            proto::TEX_ENV_TEXCOORD_SPACE => Some(state::TexEnvParam::CoordSpace(
                self.decode_gl(state::TexCoordSpace::from_wire(value))?,
            )),
            _ => {
                self.state.raise_gl_error(GlError::InvalidEnum);
                None
            }
        }
    }

    /// `QUERY`'s `what`, resolved against live state *now* -- see
    /// [`RenderOp::Query`]'s doc comment on why this cannot be deferred to
    /// the renderer. `None` means `what` is not one of the spec's
    /// accepted enumerants; the caller raises `GL_INVALID_ENUM`.
    fn resolve_query(&self, what: u32) -> Option<QueryResult> {
        Some(match what {
            GL_MODELVIEW_MATRIX => QueryResult::matrix(
                self.state
                    .top_matrix(state::MatrixMode::Modelview)
                    .unwrap_or_else(state::Mat4::identity),
            ),
            GL_PROJECTION_MATRIX => QueryResult::matrix(
                self.state
                    .top_matrix(state::MatrixMode::Projection)
                    .unwrap_or_else(state::Mat4::identity),
            ),
            GL_TEXTURE_MATRIX => QueryResult::matrix(
                self.state
                    .top_matrix(state::MatrixMode::Texture)
                    .unwrap_or_else(state::Mat4::identity),
            ),
            GL_VIEWPORT => QueryResult::vec4(rect_as_f32(self.state.raster.viewport)),
            GL_SCISSOR_BOX => QueryResult::vec4(rect_as_f32(self.state.raster.scissor)),
            GL_CURRENT_COLOR => QueryResult::vec4(self.state.current.color),
            GL_CURRENT_TEXTURE_COORDS => {
                let (s, t) = self
                    .state
                    .current
                    .texcoord
                    .get(self.state.active_unit() as usize)
                    .copied()
                    .unwrap_or((0.0, 0.0));
                QueryResult::vec2([s, t])
            }
            GL_DEPTH_RANGE => {
                let (near, far) = self.state.raster.depth_range;
                QueryResult::vec2([near, far])
            }
            _ => return None,
        })
    }

    /// Turns a `from_gl`/`from_wire` decode into `Option`, raising
    /// `GL_INVALID_ENUM` on `None` -- the "caller" `state.rs`'s own doc
    /// comment describes (see this module's doc comment).
    fn decode_gl<T>(&mut self, decoded: Option<T>) -> Option<T> {
        if decoded.is_none() {
            self.state.raise_gl_error(GlError::InvalidEnum);
        }
        decoded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c3d::proto::*;

    fn config() -> DeviceConfig {
        DeviceConfig::default()
    }

    fn context() -> Context {
        Context::new(state::Limits::default())
    }

    /// Builds one command's ring bytes: `opcode`, then `payload_words`
    /// big-endian `u32`s, with `length` computed automatically.
    fn cmd(opcode: u16, payload_words: &[u32]) -> Vec<u8> {
        let length = 1 + payload_words.len() as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(((opcode as u32) << 16) | length).to_be_bytes());
        for w in payload_words {
            bytes.extend_from_slice(&w.to_be_bytes());
        }
        bytes
    }

    fn ring_of(commands: &[Vec<u8>]) -> Vec<u8> {
        let mut ring = vec![0u8; 4096];
        let mut offset = 0usize;
        for c in commands {
            ring[offset..offset + c.len()].copy_from_slice(c);
            offset += c.len();
        }
        ring
    }

    fn f32w(v: f32) -> u32 {
        v.to_bits()
    }

    fn define_surface(id: u32, w: u32, h: u32) -> Vec<u8> {
        // SURFACE_DEFINE: id, width, height, stride_bytes, format, flags,
        // address. format 5 = A8R8G8B8 (4 bytes/pixel); aperture-backed at
        // offset 0, well inside the 32 MiB default aperture.
        cmd(OP_SURFACE_DEFINE, &[id, w, h, w * 4, 5, 0, 0])
    }

    // -- The doorbell and basic routing ---------------------------------

    #[test]
    fn a_nop_decodes_with_no_render_op_and_no_error() {
        let mut ctx = context();
        let ring = ring_of(&[cmd(OP_NOP, &[])]);
        let mut out = Vec::new();
        ctx.submit(&ring, 4, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.error_code, 0);
        assert_eq!(ctx.ring_head, 4);
    }

    #[test]
    fn a_fence_command_emits_a_fence_op_but_does_not_advance_fence_completed() {
        let mut ctx = context();
        let ring = ring_of(&[cmd(OP_FENCE, &[7])]);
        let mut out = Vec::new();
        ctx.submit(&ring, 8, &config(), &mut out);
        assert_eq!(out, vec![RenderOp::Fence { id: 7 }]);
        assert_eq!(ctx.fence_completed, 0);
        ctx.complete_fence(7);
        assert_eq!(ctx.fence_completed, 7);
    }

    #[test]
    fn state_actually_mutates_from_a_raster_state_command() {
        let mut ctx = context();
        // DEPTH_MASK 0.
        let ring = ring_of(&[cmd(OP_DEPTH_MASK, &[0])]);
        let mut out = Vec::new();
        ctx.submit(&ring, 8, &config(), &mut out);
        assert!(out.is_empty());
        assert!(!ctx.state.raster.depth_mask);
    }

    #[test]
    fn a_draw_after_set_draw_surface_emits_exactly_one_draw_op_with_the_vertex_bytes() {
        let mut ctx = context();
        let define = define_surface(1, 4, 4);
        let set = cmd(OP_SET_DRAW_SURFACE, &[1]);
        // DRAW_INLINE_WIN: prim=TRIANGLES(4), format=0 (POS only, 4
        // words), count=1, then one vertex (x,y,z,rhw).
        let vx = [f32w(1.0), f32w(2.0), f32w(0.5), f32w(1.0)];
        let draw = cmd(OP_DRAW_INLINE_WIN, &[4, 0, 1, vx[0], vx[1], vx[2], vx[3]]);
        let written = (define.len() + set.len() + draw.len()) as u32;
        let ring = ring_of(&[define, set, draw.clone()]);
        let mut out = Vec::new();
        ctx.submit(&ring, written, &config(), &mut out);

        assert_eq!(out.len(), 1);
        match &out[0] {
            RenderOp::Draw {
                window_space,
                prim,
                count,
                vertices: DrawVertices::Inline(bytes),
                ..
            } => {
                assert!(window_space);
                assert_eq!(*prim, state::PrimitiveType::Triangles);
                assert_eq!(*count, 1);
                let mut expected = Vec::new();
                for w in vx {
                    expected.extend_from_slice(&w.to_be_bytes());
                }
                assert_eq!(*bytes, expected.as_slice());
            }
            other => panic!("expected a Draw op, got {other:?}"),
        }
    }

    // -- E_NO_SURFACE / E_BAD_RECT ---------------------------------------

    #[test]
    fn a_draw_with_no_bound_surface_is_e_no_surface_and_emits_no_draw_op() {
        let mut ctx = context();
        let draw = cmd(OP_DRAW_INLINE_WIN, &[4, 0, 0]); // count 0, no vertices
        let ring = ring_of(std::slice::from_ref(&draw));
        let mut out = Vec::new();
        ctx.submit(&ring, draw.len() as u32, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.error_code, ErrorCode::NoSurface as u32);
        assert!(!ctx.halted); // E_NO_SURFACE never halts
    }

    #[test]
    fn a_draw_is_e_no_surface_again_after_its_surface_is_destroyed() {
        let mut ctx = context();
        let define = define_surface(1, 4, 4);
        let set = cmd(OP_SET_DRAW_SURFACE, &[1]);
        let destroy = cmd(OP_SURFACE_DESTROY, &[1]);
        let draw = cmd(OP_DRAW_INLINE_WIN, &[4, 0, 0]);
        let ring = ring_of(&[define.clone(), set.clone(), destroy.clone(), draw.clone()]);
        let mut out = Vec::new();
        let total = (define.len() + set.len() + destroy.len() + draw.len()) as u32;
        ctx.submit(&ring, total, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.error_code, ErrorCode::NoSurface as u32);
    }

    #[test]
    fn a_readback_partly_outside_its_surface_is_e_bad_rect() {
        let mut ctx = context();
        let define = define_surface(1, 4, 4);
        let set = cmd(OP_SET_DRAW_SURFACE, &[1]);
        // Rectangle (2,2,4,4) overflows a 4x4 surface.
        let readback = cmd(OP_SURFACE_READBACK, &[2, 2, 4, 4]);
        let ring = ring_of(&[define.clone(), set.clone(), readback.clone()]);
        let mut out = Vec::new();
        let total = (define.len() + set.len() + readback.len()) as u32;
        ctx.submit(&ring, total, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.error_code, ErrorCode::BadRect as u32);
        assert!(!ctx.halted);
    }

    /// `TEX_COPY_IMAGE`'s source rectangle must be checked against the
    /// bound *read* framebuffer's attachment, not always the draw
    /// surface, once `CAP_PROGRAMMABLE` has a non-default one bound: a
    /// rect that fits the (larger) draw surface but not the (smaller)
    /// attachment must still be rejected.
    #[test]
    fn tex_copy_image_against_a_read_framebuffer_checks_the_attachments_own_size_not_the_draw_surfaces() {
        let mut ctx = context();
        const FRAMEBUFFER: u32 = 0x8D40; // GL_FRAMEBUFFER
        const COLOR_ATTACHMENT0: u32 = 0x8CE0;
        const RGBA8: u32 = 0;
        let commands = vec![
            define_surface(1, 8, 8),
            cmd(OP_SET_DRAW_SURFACE, &[1]),
            cmd(proto::OP_FRAMEBUFFER_CREATE, &[1]),
            cmd(proto::OP_BIND_FRAMEBUFFER, &[FRAMEBUFFER, 1]),
            cmd(proto::OP_RENDERBUFFER_CREATE, &[1]),
            cmd(proto::OP_RENDERBUFFER_STORAGE_COLOR, &[1, 4, 4, 5]), // 4x4, A8R8G8B8
            cmd(proto::OP_FRAMEBUFFER_ATTACH_RENDERBUFFER, &[FRAMEBUFFER, COLOR_ATTACHMENT0, 1]),
            // (0,0,8,8) fits the 8x8 draw surface but not the 4x4
            // attachment the bound read framebuffer actually sources
            // from -- must be E_BAD_RECT.
            cmd(proto::OP_TEX_COPY_IMAGE, &[1, 0, RGBA8, 0, 0, 8, 8]),
        ];
        let ring = ring_of(&commands);
        let total = commands.iter().map(Vec::len).sum::<usize>() as u32;
        let mut out = Vec::new();
        ctx.submit(&ring, total, &config(), &mut out);
        assert!(out.is_empty(), "expected no RenderOp, got {out:?}");
        assert_eq!(ctx.error_code, ErrorCode::BadRect as u32);
        assert!(!ctx.halted);
    }

    /// The reverse of the above: a rectangle too large for the (small)
    /// draw surface but valid against the (larger) bound read
    /// framebuffer attachment must succeed -- the old, always-draw-
    /// surface check would have wrongly rejected this.
    #[test]
    fn tex_copy_image_against_a_read_framebuffer_accepts_a_rect_too_big_for_the_draw_surface() {
        let mut ctx = context();
        const FRAMEBUFFER: u32 = 0x8D40;
        const COLOR_ATTACHMENT0: u32 = 0x8CE0;
        const RGBA8: u32 = 0;
        let commands = vec![
            define_surface(1, 2, 2),
            cmd(OP_SET_DRAW_SURFACE, &[1]),
            cmd(proto::OP_FRAMEBUFFER_CREATE, &[1]),
            cmd(proto::OP_BIND_FRAMEBUFFER, &[FRAMEBUFFER, 1]),
            cmd(proto::OP_RENDERBUFFER_CREATE, &[1]),
            cmd(proto::OP_RENDERBUFFER_STORAGE_COLOR, &[1, 8, 8, 5]), // 8x8, A8R8G8B8
            cmd(proto::OP_FRAMEBUFFER_ATTACH_RENDERBUFFER, &[FRAMEBUFFER, COLOR_ATTACHMENT0, 1]),
            cmd(proto::OP_TEX_CREATE, &[1]),
            // (0,0,8,8) overflows the 2x2 draw surface but fits the 8x8
            // attachment the bound read framebuffer actually sources
            // from -- must succeed.
            cmd(proto::OP_TEX_COPY_IMAGE, &[1, 0, RGBA8, 0, 0, 8, 8]),
        ];
        let ring = ring_of(&commands);
        let total = commands.iter().map(Vec::len).sum::<usize>() as u32;
        let mut out = Vec::new();
        ctx.submit(&ring, total, &config(), &mut out);
        assert_eq!(ctx.error_code, 0, "unexpected protocol error");
        assert_eq!(out.len(), 1, "expected exactly one RenderOp, got {out:?}");
        match &out[0] {
            RenderOp::TexCopyImage { x, y, width, height, .. } => {
                assert_eq!((*x, *y, *width, *height), (0, 0, 8, 8));
            }
            other => panic!("expected a TexCopyImage op, got {other:?}"),
        }
    }

    // -- GL errors ---------------------------------------------------

    #[test]
    fn an_unknown_enable_capability_raises_gl_invalid_enum_and_is_ignored() {
        let mut ctx = context();
        let ring = ring_of(&[cmd(OP_ENABLE, &[0xDEAD])]);
        let mut out = Vec::new();
        ctx.submit(&ring, 8, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.state.gl_error(), GlError::InvalidEnum.to_gl());
        // The protocol latch is untouched by a GL error.
        assert_eq!(ctx.error_code, 0);
    }

    #[test]
    fn shader_create_with_a_bad_kind_raises_gl_invalid_enum_and_creates_nothing() {
        let mut ctx = context();
        let ring = ring_of(&[cmd(OP_SHADER_CREATE, &[1, 0xDEAD])]);
        let mut out = Vec::new();
        ctx.submit(&ring, 12, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.state.gl_error(), GlError::InvalidEnum.to_gl());
        assert!(ctx.state.shader(1).is_none());
    }

    #[test]
    fn program_attach_of_a_second_vertex_shader_is_a_pure_gl_error_with_no_render_op() {
        let mut ctx = context();
        let commands = [
            cmd(OP_SHADER_CREATE, &[1, 0x8B31]), // GL_VERTEX_SHADER
            cmd(OP_SHADER_CREATE, &[2, 0x8B31]),
            cmd(OP_PROGRAM_CREATE, &[1]),
            cmd(OP_PROGRAM_ATTACH, &[1, 1]),
            cmd(OP_PROGRAM_ATTACH, &[1, 2]),
        ];
        let total: u32 = commands.iter().map(|c| c.len() as u32).sum();
        let ring = ring_of(&commands);
        let mut out = Vec::new();
        ctx.submit(&ring, total, &config(), &mut out);
        assert!(out.is_empty(), "program lifecycle ops carry no refs and push no RenderOp");
        assert_eq!(ctx.state.gl_error(), GlError::InvalidOperation.to_gl());
        assert_eq!(ctx.state.program(1).unwrap().attached_vertex, Some(1));
    }

    #[test]
    fn shader_source_and_compile_push_render_ops_carrying_the_resolved_kind() {
        let mut ctx = context();
        let commands = [
            cmd(OP_SHADER_CREATE, &[1, 0x8B31]), // GL_VERTEX_SHADER
            cmd(OP_SHADER_SOURCE, &[1, 0x0010_0000, 4]),
            cmd(OP_SHADER_COMPILE, &[1]),
        ];
        let total: u32 = commands.iter().map(|c| c.len() as u32).sum();
        let ring = ring_of(&commands);
        let mut out = Vec::new();
        ctx.submit(&ring, total, &config(), &mut out);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], RenderOp::ShaderSource { id: 1, .. }));
        assert!(matches!(
            out[1],
            RenderOp::ShaderCompile {
                id: 1,
                kind: state::ShaderKind::Vertex
            }
        ));
    }

    #[test]
    fn buffer_data_on_a_never_created_id_is_gl_invalid_operation_with_no_render_op() {
        let mut ctx = context();
        let ring = ring_of(&[cmd(OP_BUFFER_DATA, &[1, 256, 0x88E4, 0x0010_0000, 0])]);
        let mut out = Vec::new();
        ctx.submit(&ring, 24, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.state.gl_error(), GlError::InvalidOperation.to_gl());
    }

    #[test]
    fn buffer_data_on_an_existing_buffer_pushes_a_render_op() {
        let mut ctx = context();
        let commands = [
            cmd(OP_BUFFER_CREATE, &[1]),
            cmd(OP_BUFFER_DATA, &[1, 256, 0x88E4, 0x0010_0000, 0]),
        ];
        let total: u32 = commands.iter().map(|c| c.len() as u32).sum();
        let ring = ring_of(&commands);
        let mut out = Vec::new();
        ctx.submit(&ring, total, &config(), &mut out);
        assert!(matches!(
            out.as_slice(),
            [RenderOp::BufferData { id: 1, size: 256, .. }]
        ));
    }

    #[test]
    fn vertex_attrib_pointer_with_a_bad_type_raises_gl_invalid_enum_and_records_nothing() {
        let mut ctx = context();
        let ring = ring_of(&[cmd(
            OP_VERTEX_ATTRIB_POINTER,
            &[0, 4, 0xDEAD, 0, 0, 0, 1],
        )]);
        let mut out = Vec::new();
        ctx.submit(&ring, 32, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.state.gl_error(), GlError::InvalidEnum.to_gl());
        assert!(ctx.state.vertex_attrib(0).unwrap().pointer.is_none());
    }

    #[test]
    fn enable_stencil_test_without_programmable_raises_gl_invalid_enum() {
        let mut ctx = context();
        let mut cfg = config();
        cfg.programmable = false;
        let ring = ring_of(&[cmd(OP_ENABLE, &[0x0B90])]); // GL_STENCIL_TEST
        let mut out = Vec::new();
        ctx.submit(&ring, 8, &cfg, &mut out);
        assert_eq!(ctx.state.gl_error(), GlError::InvalidEnum.to_gl());
        assert!(!ctx.state.is_enabled(state::Capability::StencilTest));
    }

    #[test]
    fn enable_stencil_test_with_programmable_actually_enables_it() {
        let mut ctx = context();
        let ring = ring_of(&[cmd(OP_ENABLE, &[0x0B90])]); // GL_STENCIL_TEST
        let mut out = Vec::new();
        ctx.submit(&ring, 8, &config(), &mut out);
        assert_eq!(ctx.state.gl_error(), state::GL_NO_ERROR);
        assert!(ctx.state.is_enabled(state::Capability::StencilTest));
    }

    #[test]
    fn stencil_func_op_mask_and_clear_stencil_mutate_raster_state() {
        let mut ctx = context();
        let commands = [
            cmd(OP_STENCIL_FUNC, &[0x0207, 2, 0xAA]), // GL_ALWAYS
            cmd(OP_STENCIL_OP, &[0x1E01, 0x1E01, 0x1E02]), // REPLACE, REPLACE, INCR
            cmd(OP_STENCIL_MASK, &[0x0F]),
            cmd(OP_CLEAR_STENCIL, &[7]),
        ];
        let total: u32 = commands.iter().map(|c| c.len() as u32).sum();
        let ring = ring_of(&commands);
        let mut out = Vec::new();
        ctx.submit(&ring, total, &config(), &mut out);
        assert_eq!(ctx.state.gl_error(), state::GL_NO_ERROR);
        assert_eq!(ctx.state.raster.stencil_func, state::StencilCompareFunc::Always);
        assert_eq!(ctx.state.raster.stencil_ref, 2);
        assert_eq!(ctx.state.raster.stencil_read_mask, 0xAA);
        assert_eq!(ctx.state.raster.stencil_fail_op, state::StencilOp::Replace);
        assert_eq!(ctx.state.raster.stencil_zfail_op, state::StencilOp::Replace);
        assert_eq!(ctx.state.raster.stencil_zpass_op, state::StencilOp::Incr);
        assert_eq!(ctx.state.raster.stencil_write_mask, 0x0F);
        assert_eq!(ctx.state.raster.clear_stencil, 7);
    }

    #[test]
    fn framebuffer_attach_renderbuffer_to_the_default_framebuffer_is_e_bad_id() {
        let mut ctx = context();
        let commands = [
            cmd(OP_RENDERBUFFER_CREATE, &[1]),
            cmd(OP_RENDERBUFFER_STORAGE_COLOR, &[1, 64, 64, 5]),
            cmd(OP_FRAMEBUFFER_ATTACH_RENDERBUFFER, &[0x8D40, 0x8CE0, 1]),
        ];
        let total: u32 = commands.iter().map(|c| c.len() as u32).sum();
        let ring = ring_of(&commands);
        let mut out = Vec::new();
        ctx.submit(&ring, total, &config(), &mut out);
        assert_eq!(ctx.error_code, ErrorCode::BadId as u32);
    }

    #[test]
    fn framebuffer_attach_renderbuffer_with_a_storage_kind_mismatch_is_e_bad_arg() {
        let mut ctx = context();
        let commands = [
            cmd(OP_FRAMEBUFFER_CREATE, &[1]),
            cmd(OP_BIND_FRAMEBUFFER, &[0x8D40, 1]),
            cmd(OP_RENDERBUFFER_CREATE, &[1]),
            cmd(OP_RENDERBUFFER_STORAGE_DEPTH_STENCIL, &[1, 64, 64]),
            cmd(OP_FRAMEBUFFER_ATTACH_RENDERBUFFER, &[0x8D40, 0x8CE0, 1]),
        ];
        let total: u32 = commands.iter().map(|c| c.len() as u32).sum();
        let ring = ring_of(&commands);
        let mut out = Vec::new();
        ctx.submit(&ring, total, &config(), &mut out);
        assert_eq!(ctx.error_code, ErrorCode::BadArg as u32);
    }

    #[test]
    fn framebuffer_status_pushes_a_render_op_once_target_decodes() {
        let mut ctx = context();
        let commands = [
            cmd(OP_FRAMEBUFFER_CREATE, &[1]),
            cmd(OP_BIND_FRAMEBUFFER, &[0x8D40, 1]),
            cmd(OP_FRAMEBUFFER_STATUS, &[0x8D40, 0, 0]),
        ];
        let total: u32 = commands.iter().map(|c| c.len() as u32).sum();
        let ring = ring_of(&commands);
        let mut out = Vec::new();
        ctx.submit(&ring, total, &config(), &mut out);
        assert_eq!(ctx.error_code, 0);
        assert_eq!(
            out,
            vec![RenderOp::FramebufferStatus {
                dest: Ref::decode(0, 0),
                // The newly created framebuffer has no attachments yet.
                status: state::GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT,
            }]
        );
    }

    #[test]
    fn bind_framebuffer_with_an_unrecognised_target_raises_gl_invalid_enum() {
        let mut ctx = context();
        let ring = ring_of(&[cmd(OP_BIND_FRAMEBUFFER, &[0xDEAD, 0])]);
        let mut out = Vec::new();
        ctx.submit(&ring, 12, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.state.gl_error(), GlError::InvalidEnum.to_gl());
    }

    #[test]
    fn draw_arrays_prog_with_no_current_program_is_e_no_program() {
        let mut ctx = context();
        let ring = ring_of(&[cmd(OP_DRAW_ARRAYS_PROG, &[4, 0, 3])]);
        let mut out = Vec::new();
        ctx.submit(&ring, 16, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.error_code, ErrorCode::NoProgram as u32);
    }

    #[test]
    fn draw_arrays_prog_with_a_current_program_pushes_a_render_op() {
        let mut ctx = context();
        let commands = [
            cmd(OP_PROGRAM_CREATE, &[1]),
            cmd(OP_USE_PROGRAM, &[1]),
            cmd(OP_DRAW_ARRAYS_PROG, &[4, 0, 3]),
        ];
        let total: u32 = commands.iter().map(|c| c.len() as u32).sum();
        let ring = ring_of(&commands);
        let mut out = Vec::new();
        ctx.submit(&ring, total, &config(), &mut out);
        assert!(matches!(
            out.as_slice(),
            [RenderOp::DrawArraysProg { first: 0, count: 3, .. }]
        ));
    }

    #[test]
    fn uniform_and_uniform_sampler_push_render_ops_unconditionally() {
        // No `state` gate at all -- see RenderOp::Uniform's doc comment
        // on why validity is entirely render.rs's concern here.
        let mut ctx = context();
        let commands = [
            cmd(OP_UNIFORM, &[1, 0, 0x1406, 1, 0]), // FLOAT, value 0.0
            cmd(OP_UNIFORM_SAMPLER, &[1, 0, 2]),
        ];
        let total: u32 = commands.iter().map(|c| c.len() as u32).sum();
        let ring = ring_of(&commands);
        let mut out = Vec::new();
        ctx.submit(&ring, total, &config(), &mut out);
        assert!(matches!(out[0], RenderOp::Uniform { program: 1, location: 0, .. }));
        assert!(matches!(
            out[1],
            RenderOp::UniformSampler { program: 1, location: 0, unit: 2 }
        ));
    }

    #[test]
    fn program_interface_is_gated_on_the_program_existing_in_state() {
        let mut ctx = context();
        let ring = ring_of(&[cmd(OP_PROGRAM_INTERFACE, &[1, 0x0010_0000, 256])]);
        let mut out = Vec::new();
        ctx.submit(&ring, 16, &config(), &mut out);
        assert!(out.is_empty(), "program 1 was never created");

        let mut ctx = context();
        let commands = [
            cmd(OP_PROGRAM_CREATE, &[1]),
            cmd(OP_PROGRAM_INTERFACE, &[1, 0x0010_0000, 256]),
        ];
        let total: u32 = commands.iter().map(|c| c.len() as u32).sum();
        let ring = ring_of(&commands);
        let mut out = Vec::new();
        ctx.submit(&ring, total, &config(), &mut out);
        assert!(matches!(out.as_slice(), [RenderOp::ProgramInterface { program: 1, .. }]));
    }

    #[test]
    fn matrix_stack_overflow_leaves_the_stack_unchanged_and_raises_stack_overflow() {
        let mut ctx = context();
        let mut pushes = Vec::new();
        // MAX_MATRIX_DEPTH_MV default is 32; push past it.
        for _ in 0..40 {
            pushes.push(cmd(OP_PUSH_MATRIX, &[]));
        }
        let ring = ring_of(&pushes);
        let mut out = Vec::new();
        let total: u32 = pushes.iter().map(|c| c.len() as u32).sum();
        ctx.submit(&ring, total, &config(), &mut out);
        assert_eq!(ctx.state.gl_error(), GlError::StackOverflow.to_gl());
        assert_eq!(ctx.state.limits().modelview_depth, 32);
    }

    #[test]
    fn a_gl_error_does_not_clear_or_overwrite_an_earlier_protocol_error() {
        let mut ctx = context();
        // First: E_NO_SURFACE from a draw with nothing bound.
        let draw = cmd(OP_DRAW_INLINE_WIN, &[4, 0, 0]);
        // Then: a GL error from an unknown ENABLE cap.
        let enable = cmd(OP_ENABLE, &[0xDEAD]);
        let ring = ring_of(&[draw.clone(), enable.clone()]);
        let mut out = Vec::new();
        let total = (draw.len() + enable.len()) as u32;
        ctx.submit(&ring, total, &config(), &mut out);
        assert_eq!(ctx.error_code, ErrorCode::NoSurface as u32);
        assert_eq!(ctx.state.gl_error(), GlError::InvalidEnum.to_gl());
    }

    #[test]
    fn first_error_wins_on_the_protocol_latch() {
        let mut ctx = context();
        // Two draws with nothing bound: only the first's offset sticks.
        let draw1 = cmd(OP_DRAW_INLINE_WIN, &[4, 0, 0]);
        let draw2 = cmd(OP_DRAW_INLINE_WIN, &[4, 0, 0]);
        let ring = ring_of(&[draw1.clone(), draw2.clone()]);
        let mut out = Vec::new();
        let total = (draw1.len() + draw2.len()) as u32;
        ctx.submit(&ring, total, &config(), &mut out);
        assert_eq!(ctx.error_code, ErrorCode::NoSurface as u32);
        assert_eq!(ctx.error_offset, 0);
    }

    #[test]
    fn first_error_wins_on_the_gl_latch() {
        let mut ctx = context();
        let bad1 = cmd(OP_ENABLE, &[0xDEAD]);
        let bad2 = cmd(OP_DEPTH_FUNC, &[0xDEAD]);
        let ring = ring_of(&[bad1.clone(), bad2.clone()]);
        let mut out = Vec::new();
        let total = (bad1.len() + bad2.len()) as u32;
        ctx.submit(&ring, total, &config(), &mut out);
        // Both are GL_INVALID_ENUM here, so this pins first-error-wins by
        // observing the latch never clears/changes across the second
        // command rather than by distinguishing codes.
        assert_eq!(ctx.state.gl_error(), GlError::InvalidEnum.to_gl());
        ctx.state.ack_gl_error();
        assert_eq!(ctx.state.gl_error(), state::GL_NO_ERROR);
    }

    // -- Halting and resuming ---------------------------------------

    #[test]
    fn a_halted_context_ignores_further_submissions_until_error_ack_then_resumes() {
        let mut ctx = context();
        // length 0 is E_BAD_LENGTH, which halts.
        let bad = vec![0u8, 0, 0, 0]; // opcode 0 (NOP), length 0
        let good = cmd(OP_NOP, &[]);
        let mut ring = vec![0u8; 4096];
        ring[0..4].copy_from_slice(&bad);
        ring[4..4 + good.len()].copy_from_slice(&good);

        let mut out = Vec::new();
        ctx.submit(&ring, 8, &config(), &mut out);
        assert!(ctx.halted);
        assert_eq!(ctx.error_code, ErrorCode::BadLength as u32);
        assert_eq!(ctx.ring_head, 0); // left at the offending command

        // A further submission (even a larger tail) decodes nothing.
        let mut out2 = Vec::new();
        ctx.submit(&ring, 8, &config(), &mut out2);
        assert!(out2.is_empty());
        assert!(ctx.halted);

        ctx.error_ack();
        assert!(!ctx.halted);
        assert_eq!(ctx.error_code, 0);

        // Resuming decodes from RING_HEAD (0) again: the same bad word is
        // still there and halts again immediately, proving decode resumed
        // at the right offset rather than skipping ahead.
        let mut out3 = Vec::new();
        ctx.submit(&ring, 8, &config(), &mut out3);
        assert!(ctx.halted);
        assert_eq!(ctx.error_offset, 0);
    }

    // -- Textures ------------------------------------------------------

    #[test]
    fn tex_image_on_an_uncreated_id_is_gl_invalid_operation_and_emits_no_op() {
        let mut ctx = context();
        // TEX_IMAGE: id, level, format, width, height, row_bytes, ref(2).
        let tex_image = cmd(OP_TEX_IMAGE, &[1, 0, 0, 4, 4, 16, 0, 16]);
        let ring = ring_of(std::slice::from_ref(&tex_image));
        let mut out = Vec::new();
        ctx.submit(&ring, tex_image.len() as u32, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.state.gl_error(), GlError::InvalidOperation.to_gl());
    }

    #[test]
    fn tex_image_on_a_created_texture_emits_one_tex_image_op_and_records_the_level() {
        let mut ctx = context();
        let create = cmd(OP_TEX_CREATE, &[1]);
        let tex_image = cmd(OP_TEX_IMAGE, &[1, 0, 0, 4, 4, 16, 0, 16]);
        let ring = ring_of(&[create.clone(), tex_image.clone()]);
        let mut out = Vec::new();
        let total = (create.len() + tex_image.len()) as u32;
        ctx.submit(&ring, total, &config(), &mut out);
        assert_eq!(out.len(), 1);
        assert!(matches!(
            out[0],
            RenderOp::TexImage {
                id: 1,
                level: 0,
                ..
            }
        ));
        let tex = ctx.state.texture(1).unwrap();
        assert_eq!(tex.levels.get(&(0, 0)).unwrap().width, 4);
    }

    // -- Reset ---------------------------------------------------------

    #[test]
    fn ctx_reset_state_opcode_resets_gl_state_via_the_ring() {
        let mut ctx = context();
        let depth_mask_off = cmd(OP_DEPTH_MASK, &[0]);
        let reset = cmd(OP_CTX_RESET_STATE, &[]);
        let ring = ring_of(&[depth_mask_off.clone(), reset.clone()]);
        let mut out = Vec::new();
        let total = (depth_mask_off.len() + reset.len()) as u32;
        ctx.submit(&ring, total, &config(), &mut out);
        assert!(ctx.state.raster.depth_mask); // back to the GL default
    }

    #[test]
    fn reset_all_clears_objects_that_reset_state_leaves_alone() {
        let mut ctx = context();
        ctx.state.tex_create(1);
        ctx.reset_state();
        assert!(ctx.state.texture(1).is_some());
        ctx.reset_all();
        assert!(ctx.state.texture(1).is_none());
    }

    // -- Copy-to-texture reads the draw surface ------------------------

    #[test]
    fn copy_to_texture_with_no_draw_surface_raises_e_no_surface() {
        // TEX_COPY_IMAGE sources its pixels from the draw surface, so with
        // none bound it is E_NO_SURFACE exactly as a readback would be.
        let mut ctx = context();
        let mut out = Vec::new();
        let ring = ring_of(&[
            cmd(OP_TEX_CREATE, &[1]),
            // id, level, format(0 = RGBA8), x, y, width, height
            cmd(OP_TEX_COPY_IMAGE, &[1, 0, 0, 0, 0, 4, 4]),
        ]);
        ctx.submit(&ring, 8 + 32, &config(), &mut out);
        assert_eq!(ctx.error_code, ErrorCode::NoSurface as u32);
        assert!(!ctx.halted, "E_NO_SURFACE skips, it does not halt");
        assert!(out.is_empty(), "no render op for a rejected copy");
    }

    #[test]
    fn copy_to_texture_outside_the_draw_surface_raises_e_bad_rect() {
        let mut ctx = context();
        let mut out = Vec::new();
        let ring = ring_of(&[
            define_surface(1, 8, 8),
            cmd(OP_SET_DRAW_SURFACE, &[1]),
            cmd(OP_TEX_CREATE, &[1]),
            // A 16x16 source rectangle out of an 8x8 surface.
            cmd(OP_TEX_COPY_IMAGE, &[1, 0, 0, 0, 0, 16, 16]),
        ]);
        let tail = (4 + 7 * 4) + (4 + 4) + (4 + 4) + (4 + 7 * 4);
        ctx.submit(&ring, tail as u32, &config(), &mut out);
        assert_eq!(ctx.error_code, ErrorCode::BadRect as u32);
        assert!(out.is_empty(), "no render op for a rejected copy");
    }

    #[test]
    fn a_rejected_copy_leaves_the_texture_level_undefined() {
        // The rect check runs before the level is defined, so a rejected
        // copy must not have half-created a level on the texture object.
        let mut ctx = context();
        let mut out = Vec::new();
        let ring = ring_of(&[
            cmd(OP_TEX_CREATE, &[1]),
            cmd(OP_TEX_COPY_IMAGE, &[1, 0, 0, 0, 0, 4, 4]),
        ]);
        ctx.submit(&ring, 8 + 32, &config(), &mut out);
        assert!(ctx.state.texture(1).is_some(), "the object still exists");
        assert!(
            ctx.state.texture(1).is_some_and(|t| t.levels.is_empty()),
            "but the rejected copy defined no level"
        );
    }

    // -- CTX_CONTROL.RESET clears both latches -------------------------

    #[test]
    fn ctx_control_reset_clears_the_protocol_latch_and_unhalts() {
        // A halted context whose only escape was ERROR_ACK would stay
        // stuck across a free/realloc; the spec makes RESET clear it.
        let mut ctx = context();
        let mut out = Vec::new();
        // A length of 0 is E_BAD_LENGTH, which halts.
        let mut ring = vec![0u8; 4096];
        ring[..4].copy_from_slice(&0x0000_0000u32.to_be_bytes());
        ctx.submit(&ring, 8, &config(), &mut out);
        assert!(ctx.halted, "a framing error halts");
        assert_ne!(ctx.error_code, 0);

        ctx.reset_all();
        assert!(!ctx.halted, "RESET un-halts");
        assert_eq!(ctx.error_code, 0, "RESET clears the protocol latch");
        assert_eq!(ctx.error_offset, 0);
    }

    #[test]
    fn ctx_reset_state_leaves_a_pending_gl_error_alone() {
        // The opcode form must not discard an error the guest has not
        // read: it is a command in the stream, not a register write.
        let mut ctx = context();
        let mut out = Vec::new();
        // DEPTH_FUNC with a bogus enumerant is GL_INVALID_ENUM.
        let ring = ring_of(&[
            cmd(OP_DEPTH_FUNC, &[0xDEAD_BEEF]),
            cmd(OP_CTX_RESET_STATE, &[]),
        ]);
        ctx.submit(&ring, 8 + 4, &config(), &mut out);
        assert_ne!(
            ctx.state.gl_error(),
            state::GL_NO_ERROR,
            "CTX_RESET_STATE preserves GL_ERROR"
        );
        ctx.reset_all();
        assert_eq!(
            ctx.state.gl_error(),
            state::GL_NO_ERROR,
            "CTX_CONTROL.RESET clears it"
        );
    }

    // -- Query -----------------------------------------------------

    #[test]
    fn query_current_color_resolves_the_value_immediately() {
        let mut ctx = context();
        let set_color = cmd(
            OP_CURRENT_COLOR,
            &[f32w(0.1), f32w(0.2), f32w(0.3), f32w(0.4)],
        );
        // QUERY: what=GL_CURRENT_COLOR(0x0B00), dest ref (aperture, addr 0, len 16).
        let query = cmd(OP_QUERY, &[0x0B00, 0, 16]);
        let ring = ring_of(&[set_color.clone(), query.clone()]);
        let mut out = Vec::new();
        let total = (set_color.len() + query.len()) as u32;
        ctx.submit(&ring, total, &config(), &mut out);
        assert_eq!(out.len(), 1);
        match &out[0] {
            RenderOp::Query { result, .. } => {
                assert_eq!(result.count, 4);
                assert_eq!(&result.values[..4], &[0.1, 0.2, 0.3, 0.4]);
            }
            other => panic!("expected a Query op, got {other:?}"),
        }
    }

    #[test]
    fn query_of_an_unknown_target_raises_gl_invalid_enum() {
        let mut ctx = context();
        let query = cmd(OP_QUERY, &[0xFFFF, 0, 16]);
        let ring = ring_of(std::slice::from_ref(&query));
        let mut out = Vec::new();
        ctx.submit(&ring, query.len() as u32, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.state.gl_error(), GlError::InvalidEnum.to_gl());
    }

    // -- A mixed submission ---------------------------------------------

    #[test]
    fn a_submission_mixing_valid_commands_with_one_of_each_error_class() {
        let mut ctx = context();
        let define = define_surface(1, 4, 4);
        let set = cmd(OP_SET_DRAW_SURFACE, &[1]);
        // Valid: CLEAR with a bound surface.
        let clear = cmd(OP_CLEAR, &[CLEAR_MASK_COLOR]);
        // GL error: unknown ENABLE cap; keeps decoding.
        let bad_enable = cmd(OP_ENABLE, &[0xDEAD]);
        // Protocol error (state.rs-detected): readback rect outside the
        // surface; keeps decoding (never halts).
        let bad_readback = cmd(OP_SURFACE_READBACK, &[10, 10, 1, 1]);
        // Valid again, after both errors: another CLEAR.
        let clear2 = cmd(OP_CLEAR, &[CLEAR_MASK_DEPTH]);

        let ring = ring_of(&[
            define.clone(),
            set.clone(),
            clear.clone(),
            bad_enable.clone(),
            bad_readback.clone(),
            clear2.clone(),
        ]);
        let mut out = Vec::new();
        // Submitted in two steps, not one, so each `RenderOp::Clear`'s
        // expected `state` snapshot can be captured at the point it was
        // actually produced -- `bad_enable` raises a pending `GL_ERROR`
        // between the two `CLEAR`s, so their snapshots must legitimately
        // differ (first: no pending error; second: one), not just their
        // `mask`.
        let before_bad_enable = (define.len() + set.len() + clear.len()) as u32;
        ctx.submit(&ring, before_bad_enable, &config(), &mut out);
        let clear1_state = ctx.state.clone();
        let total = before_bad_enable + (bad_enable.len() + bad_readback.len() + clear2.len()) as u32;
        ctx.submit(&ring, total, &config(), &mut out);
        let clear2_state = ctx.state.clone();

        assert_eq!(
            out,
            vec![
                RenderOp::Clear {
                    mask: CLEAR_MASK_COLOR,
                    state: clear1_state,
                },
                RenderOp::Clear {
                    mask: CLEAR_MASK_DEPTH,
                    state: clear2_state,
                },
            ]
        );
        assert!(!ctx.halted); // neither error class halts here
        assert_eq!(ctx.error_code, ErrorCode::BadRect as u32);
        assert_eq!(ctx.state.gl_error(), GlError::InvalidEnum.to_gl());
    }

    // -- CALL -----------------------------------------------------------

    fn buffer_of(commands: &[Vec<u8>]) -> Vec<u8> {
        commands.concat()
    }

    /// A `FENCE` inside a called buffer completes normally -- it reaches
    /// the same `apply` as everywhere else -- and `RING_HEAD` (the outer
    /// ring's) advances past the `CALL` command itself.
    #[test]
    fn call_replays_a_buffer_and_its_fence_completes() {
        let mut ctx = context();
        let mut out = Vec::new();
        let mut storage = Vec::new();
        let buffer = buffer_of(&[cmd(OP_FENCE, &[9])]);
        let call_cmd = cmd(OP_CALL, &[0x1000, buffer.len() as u32]);
        let call_len = call_cmd.len() as u32;
        let ring = ring_of(&[call_cmd]);
        let mut fetch = |r: Ref| -> Option<Vec<u8>> {
            if r.address == 0x1000 {
                Some(buffer.clone())
            } else {
                None
            }
        };
        ctx.submit_with(
            &ring,
            call_len,
            &config(),
            &mut out,
            &mut storage,
            &mut fetch,
        );
        assert_eq!(out, vec![RenderOp::Fence { id: 9 }]);
        assert_eq!(ctx.ring_head, call_len, "RING_HEAD past the CALL");
        assert_eq!(ctx.error_code, 0);
        assert!(!ctx.halted);
    }

    /// Nesting is illegal: a `CALL` decoded while already decoding a
    /// called buffer is `E_BAD_OPCODE`, skipped -- decoding of the
    /// (outer) called buffer continues past it, proven here by a marker
    /// `FENCE` right after the illegal nested `CALL` still executing.
    #[test]
    fn call_nesting_is_illegal_and_decoding_of_the_buffer_continues_past_it() {
        let mut ctx = context();
        let mut out = Vec::new();
        let mut storage = Vec::new();
        let nested_call = cmd(OP_CALL, &[0x2000, 4]);
        let marker = cmd(OP_FENCE, &[55]);
        let buffer = [nested_call, marker].concat();
        let call_cmd = cmd(OP_CALL, &[0x1000, buffer.len() as u32]);
        let call_len = call_cmd.len() as u32;
        let ring = ring_of(&[call_cmd]);
        let mut nested_fetch_invoked = false;
        let mut fetch = |r: Ref| -> Option<Vec<u8>> {
            if r.address == 0x1000 {
                Some(buffer.clone())
            } else {
                // The illegal nested CALL must never even be fetched --
                // it's rejected as E_BAD_OPCODE before any fetch happens.
                nested_fetch_invoked = true;
                None
            }
        };
        ctx.submit_with(
            &ring,
            call_len,
            &config(),
            &mut out,
            &mut storage,
            &mut fetch,
        );
        assert!(!nested_fetch_invoked, "a nested CALL must never be fetched");
        assert_eq!(
            out,
            vec![RenderOp::Fence { id: 55 }],
            "the marker after the illegal nested CALL still executes"
        );
        assert_eq!(ctx.error_code, ErrorCode::BadOpcode as u32);
        assert_eq!(
            ctx.error_offset, 0,
            "latches the OUTER CALL's own ring offset, not a position inside the buffer"
        );
        assert!(!ctx.halted, "E_BAD_OPCODE is skip-class");
    }

    /// Nesting is `E_BAD_OPCODE` even when the nested `CALL`'s own ref is
    /// malformed (here, misaligned -- which would otherwise be `E_BAD_REF`
    /// for an ordinary top-level `CALL`): nesting is checked first, before
    /// the ref is looked at at all, so a malformed nested ref must not leak
    /// through as the wrong error code.
    #[test]
    fn call_nesting_is_e_bad_opcode_even_with_a_malformed_nested_ref() {
        let mut ctx = context();
        let mut out = Vec::new();
        let mut storage = Vec::new();
        let nested_call = cmd(OP_CALL, &[0x2001, 4]); // misaligned address
        let marker = cmd(OP_FENCE, &[55]);
        let buffer = [nested_call, marker].concat();
        let call_cmd = cmd(OP_CALL, &[0x1000, buffer.len() as u32]);
        let call_len = call_cmd.len() as u32;
        let ring = ring_of(&[call_cmd]);
        let mut fetch = |r: Ref| -> Option<Vec<u8>> {
            if r.address == 0x1000 {
                Some(buffer.clone())
            } else {
                panic!("the illegal nested CALL must never be fetched");
            }
        };
        ctx.submit_with(
            &ring,
            call_len,
            &config(),
            &mut out,
            &mut storage,
            &mut fetch,
        );
        assert_eq!(
            out,
            vec![RenderOp::Fence { id: 55 }],
            "the marker after the illegal nested CALL still executes"
        );
        assert_eq!(
            ctx.error_code,
            ErrorCode::BadOpcode as u32,
            "nesting is rejected before the malformed ref is ever validated"
        );
        assert!(!ctx.halted, "E_BAD_OPCODE is skip-class");
    }

    /// A skip-class error raised by a command *inside* a called buffer
    /// (here `SURFACE_DESTROY 0`, `E_BAD_ID`) latches the enclosing
    /// `CALL`'s own ring offset -- not a position inside the buffer, which
    /// has none of its own -- and decoding of the buffer continues past
    /// it (a marker command in the same buffer still executes).
    #[test]
    fn a_skip_class_error_inside_a_called_buffer_latches_the_calls_offset_and_continues() {
        let mut ctx = context();
        let mut out = Vec::new();
        let mut storage = Vec::new();
        let filler = cmd(OP_CTX_RESET_STATE, &[]);
        let bad = cmd(OP_SURFACE_DESTROY, &[0]); // id 0: E_BAD_ID
        let marker = cmd(OP_FENCE, &[77]);
        let buffer = [filler, bad, marker].concat();
        // A leading FENCE in the outer ring gives the CALL a non-zero
        // offset, distinguishable from any buffer-local offset (the bad
        // command's own buffer-local offset is non-zero too, at 4).
        let leading = cmd(OP_FENCE, &[1]);
        let call_cmd = cmd(OP_CALL, &[0x1000, buffer.len() as u32]);
        let call_start = leading.len() as u32;
        let total = (leading.len() + call_cmd.len()) as u32;
        let ring = ring_of(&[leading, call_cmd]);
        let mut fetch = |r: Ref| -> Option<Vec<u8>> {
            if r.address == 0x1000 {
                Some(buffer.clone())
            } else {
                None
            }
        };
        ctx.submit_with(&ring, total, &config(), &mut out, &mut storage, &mut fetch);
        assert_eq!(
            out,
            vec![RenderOp::Fence { id: 1 }, RenderOp::Fence { id: 77 }],
            "the marker after the skipped bad command still executes"
        );
        assert_eq!(ctx.error_code, ErrorCode::BadId as u32);
        assert_eq!(
            ctx.error_offset, call_start,
            "latches the CALL's own ring offset, not the buffer-local one"
        );
        assert!(!ctx.halted);
    }

    /// A framing error against the *buffer's own end* (its last command's
    /// declared length runs past it) is `E_BAD_LENGTH`, halting -- the
    /// call is abandoned entirely, `ERROR_OFFSET` latches the `CALL`'s own
    /// ring offset (not anything inside the buffer), and after the halt
    /// `RING_HEAD` (the outer ring's) is left right after the `CALL`,
    /// exactly as if it had completed -- because the outer cursor already
    /// advanced past the `CALL` itself before the buffer was ever decoded.
    #[test]
    fn a_framing_error_in_a_called_buffer_halts_and_latches_the_calls_offset() {
        let mut ctx = context();
        let mut out = Vec::new();
        let mut storage = Vec::new();
        // A leading FENCE in the outer ring gives the CALL a non-zero
        // offset, and a harmless filler command in the buffer gives the
        // truncated FENCE a non-zero *buffer-local* offset too -- so a
        // latch that used either the wrong (buffer-local) offset or offset
        // 0 by coincidence would be caught here, unlike a same-valued
        // 0/0 case.
        let leading = cmd(OP_FENCE, &[1]);
        let filler = cmd(OP_CTX_RESET_STATE, &[]);
        // Declares FENCE (length 2 words = 8 bytes) but is truncated to 4
        // bytes -- runs past the buffer's own end.
        let mut truncated_fence = cmd(OP_FENCE, &[9]);
        truncated_fence.truncate(4);
        let buffer = [filler, truncated_fence].concat();
        let call_cmd = cmd(OP_CALL, &[0x1000, buffer.len() as u32]);
        let call_start = leading.len() as u32;
        let total = (leading.len() + call_cmd.len()) as u32;
        let ring = ring_of(&[leading, call_cmd]);
        let mut fetch = |r: Ref| -> Option<Vec<u8>> {
            if r.address == 0x1000 {
                Some(buffer.clone())
            } else {
                None
            }
        };
        ctx.submit_with(&ring, total, &config(), &mut out, &mut storage, &mut fetch);
        assert_eq!(
            out,
            vec![RenderOp::Fence { id: 1 }],
            "only the outer ring's leading FENCE took effect"
        );
        assert!(ctx.halted, "E_BAD_LENGTH halts");
        assert_eq!(ctx.error_code, ErrorCode::BadLength as u32);
        assert_eq!(
            ctx.error_offset, call_start,
            "latches the CALL's own ring offset, not inside the buffer"
        );
        assert_eq!(
            ctx.ring_head, total,
            "resumes in the ring right after the CALL, as if it had completed"
        );
    }

    /// A `CALL` to a guest-space (`CAP_GUESTMEM`) buffer works exactly
    /// like an aperture one -- the fetch closure just sees `RefSpace::Guest`.
    #[test]
    fn call_to_a_guest_space_buffer_works_under_cap_guestmem() {
        let mut ctx = context();
        let mut out = Vec::new();
        let mut storage = Vec::new();
        let buffer = buffer_of(&[cmd(OP_FENCE, &[42])]);
        let ref_len_word = 0x8000_0000 | buffer.len() as u32; // space 1 (guest)
        let call_cmd = cmd(OP_CALL, &[0x2000_0000, ref_len_word]);
        let call_len = call_cmd.len() as u32;
        let ring = ring_of(&[call_cmd]);
        let mut seen_space = None;
        let mut fetch = |r: Ref| -> Option<Vec<u8>> {
            seen_space = Some(r.space);
            if r.address == 0x2000_0000 {
                Some(buffer.clone())
            } else {
                None
            }
        };
        ctx.submit_with(
            &ring,
            call_len,
            &config(),
            &mut out,
            &mut storage,
            &mut fetch,
        );
        assert_eq!(seen_space, Some(RefSpace::Guest));
        assert_eq!(out, vec![RenderOp::Fence { id: 42 }]);
        assert_eq!(ctx.error_code, 0);
    }

    /// A `CALL` to a space-1 buffer with `CAP_GUESTMEM` masked off must go
    /// through the ordinary `validate_ref` path exactly like any other
    /// ref -- `E_BAD_REF` at the `CALL`'s own ref, raised by `ring.rs`
    /// before `dispatch.rs` ever sees `Command::Call`, so the fetch
    /// closure must never even run.
    #[test]
    fn call_to_a_guest_space_buffer_is_e_bad_ref_without_cap_guestmem() {
        let mut ctx = context();
        let mut out = Vec::new();
        let mut storage = Vec::new();
        let mut c = config();
        c.guestmem = false;
        let ref_len_word = 0x8000_0000 | 16u32;
        let call_cmd = cmd(OP_CALL, &[0x2000_0000, ref_len_word]);
        let call_len = call_cmd.len() as u32;
        let ring = ring_of(&[call_cmd]);
        let mut fetch_invoked = false;
        let mut fetch = |_r: Ref| -> Option<Vec<u8>> {
            fetch_invoked = true;
            None
        };
        ctx.submit_with(&ring, call_len, &c, &mut out, &mut storage, &mut fetch);
        assert!(
            !fetch_invoked,
            "validate_ref must reject this before dispatch ever sees Command::Call"
        );
        assert!(out.is_empty());
        assert_eq!(ctx.error_code, ErrorCode::BadRef as u32);
    }

    /// The no-`CALL` path (`Context::submit`) is unaffected: a `CALL`
    /// command reaching it (no caller wired up `submit_with`) always
    /// fails to fetch, which is skip-class `E_BAD_REF` at the `CALL`'s own
    /// offset -- proving `submit`'s convenience wrapper still behaves
    /// sanely even if a ring somehow contained a `CALL`, without needing
    /// any of `submit_with`'s extra parameters.
    #[test]
    fn plain_submit_treats_an_unfetchable_call_as_e_bad_ref() {
        let mut ctx = context();
        let mut out = Vec::new();
        let call_cmd = cmd(OP_CALL, &[0x1000, 16]);
        let call_len = call_cmd.len() as u32;
        let ring = ring_of(&[call_cmd]);
        ctx.submit(&ring, call_len, &config(), &mut out);
        assert!(out.is_empty());
        assert_eq!(ctx.error_code, ErrorCode::BadRef as u32);
        assert_eq!(ctx.error_offset, 0);
        assert!(!ctx.halted);
    }
}
