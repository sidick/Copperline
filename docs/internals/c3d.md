# The C3D 3D accelerator board

C3D is a fixed-function 3D accelerator for the Zorro bus, described in
OpenGL 1.x terms. **The protocol -- the autoconfig identity, register
file, command stream and semantics -- is specified outside this
repository**, as a standalone document under a permissive documentation
licence, so that another emulator or a hardware implementation can build
the board without reading Copperline's source and without inheriting
Copperline's licence. This chapter is Copperline's *implementation* of
that specification, and nothing here changes it; where the two disagree,
the specification wins and this chapter is the bug.

Copperline appears in the specification's identity registry as
manufacturer `0x1448`, product **9** (Zorro III) or **10** (Zorro II) --
the next free product numbers under the ID the project already uses (see
[](../zorro)'s
[manufacturer ID table](../zorro.md#the-copperline-manufacturer-id)).
Guest software finds a board by walking that registry rather than
hard-coding one pair, so Copperline is one row among however many come
later, not the definition.

The pure-logic half of the board -- the ring decoder, the OpenGL state
machine, the dispatch layer that drives one into the other, and the
conformance-trace container (`src/c3d/`) -- is deliberately **not**
behind the `c3d` cargo feature. It depends on nothing else in the
emulator and on no GPU API, so it builds and tests everywhere the core
does, including wasm32, and the conformance logic is never silently
excluded from CI. Only the renderer needs the feature.

## Implementation notes

These describe the Copperline host board (`src/c3d/`, `[c3d]`, cargo
feature `c3d`, **off by default**). They are updated as the milestones
land; the specification is revised separately and first. Where an
earlier draft of this section described a shape the code ended up not
taking, the note says so rather than quietly disappearing -- a design
decision proven wrong by building it is exactly the kind of thing this
project's spec-first loop exists to surface.

- **Feature and configuration.** The board is behind the `c3d` cargo
  feature and a `[c3d]` table (`src/c3d/board.rs`,
  `docs/guide/configuration.md`'s `[c3d]` section): `enabled` fits the
  board, and `mask_caps` is the spec's conformance switch (forces named
  `CAPS0` bits clear). `size`, `contexts`, `timing` and `trace` are not
  yet configuration knobs -- the board fits one fixed 32 MiB Zorro III
  window with 4 contexts. `enabled` is also a System-tab row in the GUI
  launcher (`src/video/launcher/fields.rs`'s `F::C3d`, a `c3d`-build-only
  toggle passed through unconditionally otherwise, the same pattern
  `F::Mhi` uses); `mask_caps` is config-file/CLI only, being a
  conformance-testing knob rather than something normal use ever
  touches. A build without the feature still parses and ignores the
  table, the same contract every other optional board's config keeps.
  Browser and `--no-default-features` builds carry none of the board or
  the renderer; the pure-logic modules stay in them regardless (see this
  chapter's opening paragraphs).
- **GPU.** The board owns a headless `wgpu` device of its own (created
  lazily, on the first doorbell that needs it, so `[c3d] enabled = true`
  alone never requires a GPU); it never shares the window's `pixels`
  device, so it works identically in headless captures (which have no
  window) and under a software adapter in CI. `wgpu` is already in the
  default build's dependency graph via `pixels`, so enabling the feature
  adds no new crate to the lock file. A missing adapter is not a hard
  failure: the board still autoconfigs and answers every register, and a
  doorbell's rendering work is silently dropped (logged once) rather
  than panicking or refusing to boot.
- **Threading: no worker, on purpose, for now.** A `RING_TAIL` write is
  handled synchronously, start to finish, inside the register write
  itself: the ring is decoded, every command is applied to the GL state
  machine, and the resulting renderer operations are executed against
  `wgpu` with `pollster`'s blocking wait, all before the write returns.
  There is no render worker thread and `tick` does nothing. An earlier
  draft of this section described a worker thread draining completions
  on `tick`, the same shape `copperhf`'s worker takes (see
  [](copperhf.md), "The determinism model"); M2 does not need it, since
  running synchronously satisfies
  [Determinism and timing](#c3d-determinism) trivially (nothing is
  asynchronous, so there is nothing for the guest to observe early) and
  is materially simpler to get right first. Its cost is real and
  specific: because Copperline's emulated machine runs on the main
  thread inside the event loop, a slow doorbell stalls that loop, not a
  background thread, so a heavy M3 workload may need to move this to a
  worker after all. That is a decision for when a real workload argues
  for it, not one this milestone had to make.
- **Surfaces.** Aperture-backed surfaces are the M2 path, and the
  aperture is a **plain buffer the board owns directly**, not a second,
  RAM-backed chained autoconfig identity as an earlier draft of this
  section proposed. The reason is addressing, not convenience: the
  specification's `APERTURE_OFFSET` register is a promise that the
  aperture sits at a fixed offset from the *same* base the guest found
  with `FindConfigDev()`, and a chained identity gets its own, unrelated
  base that autoconfig placement does not keep contiguous with anything.
  Owning the buffer keeps that promise exactly, at the cost of not
  reusing the existing board-RAM accessor path. `CAP_SURFACE_GUESTADDR`
  is not yet set -- every surface is aperture-backed in this milestone;
  reaching into another board's VRAM needs the `DeviceHost` accessor for
  RTG boards' linear apertures (the same capability-hook pattern as the
  audio rings) scheduled for M3, and reaching fast RAM through the
  existing DMA decode is untried but expected to need no further
  plumbing when it is picked up. Because doorbell execution is
  synchronous (above), there is no benefit yet to the
  [readback contract](#c3d-readback)'s lazy-materialisation allowance --
  a readback's `Memory::write` lands in the aperture buffer before the
  doorbell returns, which is also "by the fence" and "on first read",
  so the distinction is dormant until an asynchronous render path gives
  it something to do.
- **Capability honesty.** `CAPS0` reports `CAP_IRQ`, `CAP_REF_SYNC`,
  `CAP_SURFACE_GUESTADDR` (since the cross-board DMA hook),
  `CAP_GUESTMEM` (since `ApertureMemory` resolves space-1 refs and a
  guest-memory ring through `DeviceHost`, completing the spec's stated
  Copperline capability set: bits 0, 1, 2, 3, 4, 6), and -- since the
  transform-tier MVP landed in the renderer -- `CAP_TRANSFORM` and
  `CAP_MULTITEXTURE`; the ring decoder is derived from the same constant
  (`board.rs`'s `device_config`), so the register and the decoder's idea
  of what is legal can never drift apart. The transform-tier limit
  registers (`MAX_LIGHTS`, `MAX_CLIP_PLANES`, the three matrix depths,
  `MAX_TEXTURE_UNITS`) report exactly the values each context's
  `state::Limits::default()` enforces. Lighting and `CLIP_PLANE` state is
  decoded and tracked per the spec but not yet applied by the renderer
  (both are unused by the first client -- see "Implementation order"
  below); a guest that enables them gets unlit, unclipped rendering, not
  an error.
- **`CAP_GUESTMEM`.** A space-1 ref names ordinary guest memory (chip/
  slow/motherboard/accelerator RAM, or a RAM-backed Zorro board), reached
  through the same `DeviceHost::dma_read`/`dma_write` decode the A2091
  and CDTV bus masters use. Reads happen synchronously inside the
  doorbell (`dma_read` takes `&self`, so it never conflicts with the
  aperture/`pending_dma` borrows already live there) -- which is also
  exactly what `CAP_REF_SYNC` promises, trivially true here since the
  whole doorbell is synchronous, but worth stating since `CAP_REF_SYNC`
  is "meaningful only with `CAP_GUESTMEM`" and now it is. Writes
  deliberately stay on the existing one-tick-deferred `pending_dma` path
  the `CAP_SURFACE_GUESTADDR` readback machinery already established,
  rather than a new synchronous `dma_write` call: `dma_write`/
  `dma_write_byte` only resolve ordinary RAM and *RAM-backed* Zorro
  windows, never a *device-backed* board's window, so only the bus's
  `drain_cross_board_dma` pass can correctly land a write regardless of
  what the guest address actually names. A `QUERY` result written to
  plain guest RAM gets the same one-tick deferral a `CAP_SURFACE_GUESTADDR`
  readback already gets. **`RING_HEAD`/`CTX_STATUS.IDLE_RING` are not
  part of that deferral**: `Context::submit` advances `RING_HEAD`
  synchronously every doorbell regardless of any write still pending,
  matching the spec's own wording for the register -- "commands before
  it have been *consumed* ..., not necessarily executed". The only
  architected completion signal is `FENCE_COMPLETED`, and that is the
  one actually deferred alongside the write; a guest inferring
  completion from `RING_HEAD`/`IDLE_RING` instead is reading a meaning
  the spec never assigned it. `RING_BASE` can itself name guest memory
  too (`RING_SIZE` bit 31): the doorbell fetches the ring bytes with
  `dma_read` instead of `read_aperture_range` when the bit is set and
  `CAP_GUESTMEM` is not masked off; a masked device returns before
  building a ring slice at all (nothing decoded, `RING_HEAD` does not
  advance) rather than raising a protocol error, since `RING_BASE`/
  `RING_SIZE` are ordinary registers with no error latch of their own --
  and this is a dedicated early return, not a fallback into the
  aperture-read path, because `ring_base` is a guest address here and
  reading the aperture at that offset would decode unrelated bytes as
  commands instead of correctly decoding nothing. **Known gap:** an
  unreachable space-1 address is not rejected with `E_BAD_REF` --
  `ring.rs`'s `validate_ref` explicitly punts guest-memory reachability
  checking to "a higher layer" that does not exist yet, so `dma_read`
  simply returns the bus's open-bus filler for a hole in the guest
  address map, and the device never notices. As of spec draft 0.13 this
  is a *should-where-detectable*, not a requirement (a hardware bus
  master has no reachability oracle -- it sees open bus, exactly what
  this board mimics today), so Copperline is conformant as-is; an
  emulator *can* detect, though, and implementing the check remains
  planned as the quality-of-implementation debugging aid the spec's
  draft-history entry describes.
- **Implementation order.** The first intended client is an OpenGL 1.x
  subset of the Quake-engine shape: immediate mode, matrices, textures
  with texenv, blend/alpha/depth/fog/scissor, vertex arrays with
  `DRAW_ELEMENTS` and compiled vertex arrays, two-unit multitexture,
  `READ_PIXELS`, polygon mode, blend equation and separate blend
  functions, copy-to-texture, texgen (sphere map only, S and T
  independently enabled); and -- confirmed from its function
  table -- **no GL lighting, materials or clip planes** (not stubbed:
  absent as symbols). Those are therefore specified and
  conformance-traced but implemented last; `DRAW_ELEMENTS` with
  multitexture and `COLOR_PACKED` are implemented first. Indexed
  textures are applied at upload by the first client, so `TEX_PALETTE`
  is a later optimisation, not a dependency. M2 itself implements only
  the window-space rasteriser path (`DRAW_INLINE_WIN`), `CLEAR` and
  `SURFACE_READBACK` -- enough for the M2 protocol test program
  (`guest/c3d/test/c3dtest.c`) and nothing more; M1's renderer already
  covers considerably more of the baseline tier (textures, blending,
  every primitive's triangulation) than M2's board exercises yet.
- **Two-unit multitexture.** `src/c3d/render.rs` implements true
  single-pass two-unit multitexture: unit 0 and unit 1 are each
  independently enabled/bound/`TEX_ENV`'d (`TEXTURE_2D` and `TEX_ENV`
  are per-unit with no rule coupling the units, matching GL 1.1's fixed-
  function pipeline). When both units are active, unit 1's texel
  cascades onto unit 0's result per unit 1's own `TEX_ENV` mode; when
  only one unit is active (either one), that unit alone textures the
  draw from its own texture and its own texcoord. `TEX_ENV_COLOR` is not
  yet consulted by either unit's `BLEND` mode (an existing unit-0
  approximation, carried through unchanged to unit 1). `CAPS0`
  advertises `CAP_MULTITEXTURE` (see "Capability honesty" above), so
  this is guest-reachable.
- **GL-space `DRAW_ARRAYS`/`DRAW_ELEMENTS`.** `src/c3d/render.rs`
  combines the window-space array/indexed-array vertex-fetch machinery
  with the GL-space (`CAP_TRANSFORM`) position-transform path: array and
  element (indexed) draws now go through matrix transform, `VIEWPORT` and
  `DEPTH_RANGE` exactly like `DRAW_INLINE` (GL-space) does, not just the
  window-space rasteriser. Implemented as a parse-wrapper/shared-tail
  split mirroring the window-space refactor: `op_draw_inline_gl` (parses
  `DRAW_INLINE`'s bytes), `op_draw_arrays_gl` and `op_draw_elements_gl`
  (fetch from array/index memory) all hand a `Vec<GlVertex>` to the
  shared `op_draw_gl` tail, which is `op_draw_inline_gl`'s pre-refactor
  body moved verbatim -- no transform-math changes. `NORMAL` is a legal
  format bit for a GL-space array draw (unlike the window-space fetch,
  which `ring.rs` rejects outright before this module ever sees it), but
  `GlVertex` still carries no normal field this milestone (no lighting is
  implemented yet) -- its descriptor is decoded (so later descriptors in
  the same command land at the right slot) but never dereferenced,
  exactly like `TEXCOORD2`-`3`. A follow-up milestone adding lighting
  would extend `GlVertex` and its fetch functions to actually read it.
  Position component-count validation (`2..=4`, no `glVertex1f`) and the
  `min_index`/`max_index`-are-never-consulted reasoning (including the
  spec's `0xFFFF_FFFF` self-scan sentinel) and the
  `MAX_ARRAY_DRAW_VERTICES` allocation-size cap are shared with the
  window-space array fetch, unchanged: the DoS/allocation-abort reasoning
  and the position-component-count rule are both space-independent.
  `CAPS0` advertises `CAP_TRANSFORM` (see "Capability honesty" above),
  so every GL-space draw path is guest-reachable.
- **Texgen.** `TEXGEN`/`TEXGEN_PLANE` are implemented per unit, per
  coordinate (`S`/`T` independently enabled, matching the first client's
  own usage -- see "Implementation order" above), for `OBJECT_LINEAR`,
  `EYE_LINEAR` and `SPHERE_MAP`, computed CPU-side in the shared
  `op_draw_gl` tail alongside the existing modelview/projection
  transform and folded straight into `GpuVertex`'s existing
  `texcoord0`/`texcoord1` fields -- no new WGSL, pipeline key, or bind
  group. `OBJECT_LINEAR`/`EYE_LINEAR` are the same `dot(pos, plane)`
  formula fed different position/plane pairs (`EYE_LINEAR`'s plane is
  already transformed by the inverse modelview at `TEXGEN_PLANE` command
  time, per `state.rs`'s `set_texgen_plane`, so the render side does no
  further transform); `SPHERE_MAP` is GL 1.1's normative reflection
  formula from the eye-space vertex position and the eye-space normal.
  The eye-space normal needs `NORMAL`, which `parse_vertices_raw` used to
  walk-and-discard for every draw kind; it now extracts the value (into
  `GlVertex::normal`; `WinVertex` still never copies it, matching
  `NORMAL`'s "illegal in a window-space draw" rule) and transforms it by
  the modelview's inverse-transpose (`Mat4::transform_normal3`), not the
  modelview itself -- the two only coincide when the modelview's linear
  part is orthogonal (pure rotation), and diverge under scale. Because
  texgen lives in the shared `op_draw_gl` tail, it applies to every
  GL-space draw shape (`DRAW_INLINE`, `DRAW_ARRAYS`, `DRAW_ELEMENTS`)
  -- but the array/element vertex fetch decodes the `NORMAL`
  descriptor without reading its data yet (see the bullet above), so an
  array-supplied per-vertex normal falls back to the current-state
  normal there; `SPHERE_MAP`/`EYE_LINEAR` with true per-vertex array
  normals is the documented follow-up alongside lighting.
- **Texture matrix.** The per-unit `GL_TEXTURE` matrix (the third
  matrix mode, its own stack per unit) is applied to each unit's
  texture coordinates in GL-space draws, *after* texgen per GL 1.1's
  pipeline order, CPU-side like the rest of the transform path. The
  transformed `q` is divided out per vertex, not per fragment (the wire
  format has no per-texcoord projective `q` in version 1 -- the spec's
  one-`rhw` rule -- so a projective texture matrix gets the same
  per-vertex approximation period fixed-function hardware gave it; the
  common translate/scale/rotate matrices keep `q == 1` and are exact).
  The default texture-stack depth is 16 (`MAX_MATRIX_DEPTH_TEX`), above
  the spec's minimum of 2, so a guest library carrying the first
  client's 10-deep `GL_TEXTURE` stack can map its pushes 1:1 instead of
  flattening client-side.
- **`CALL`.** `CALL` (`0x0005`, Control group, draft 0.14) replays a
  linear command buffer -- the aperture or, under `CAP_GUESTMEM`, guest
  memory -- in place of the `CALL` command itself: baseline tier, no
  capability bit, since it is purely a decode-side addressing mode. The
  buffer is decoded with a second [`RingCursor`](../../src/c3d/ring.rs)
  built by `RingCursor::new_linear`, a dedicated constructor rather than
  a `head`/`tail` pair fed to the ordinary one: the ordinary constructor's
  `E_RING_OVERRUN` check exists only to tell "full" from "empty" on a
  *circular* ring, and applying that same "one word short of full"
  arithmetic to a buffer whose `tail` always equals its own length would
  spuriously halt on every buffer a guest fills exactly to the end.
  `new_linear` also disables the ordinary cursor's "a command landing
  exactly on the physical end wraps `head` back to `0`" behaviour (correct
  only for a real ring, where `tail` wraps independently) -- without that,
  a called buffer entirely consumed by its last command would wrap back to
  offset `0` and silently re-decode itself instead of stopping. Nesting
  (a `CALL` decoded while already decoding a called buffer) is bounded to
  one level **by construction**: `Context::apply_call_buffer` never calls
  itself or `Context::submit_with`, so there is no code path that can go
  two levels deep, and no depth counter to get wrong. Every error raised
  while decoding a called buffer -- including a nested `CALL`, which is
  `E_BAD_OPCODE` -- latches the *outer* `CALL`'s own ring offset in
  `ERROR_OFFSET`, per the spec's "a command in a called buffer has no ring
  offset to report"; a halting error abandons the call and halts the
  context, and since the outer ring cursor has already advanced past the
  `CALL` command by the time its buffer is decoded (`CALL` itself decoded
  structurally fine), `RING_HEAD` is left exactly where the spec wants it
  after `ERROR_ACK`: right after the `CALL`, as if it had completed.
  `FENCE` needs no special-casing inside a called buffer -- it reaches the
  same `Context::apply` as everywhere else. The buffer-fetch itself is an
  injected closure (`Context::submit_with`'s `fetch_call_buffer`), reusing
  `board.rs`'s doorbell's existing aperture-range/`DeviceHost::dma_read`
  fetch (the same two primitives it already used to fetch a guest-memory
  ring, one level down) for both address spaces through one code path; a
  fetch returning bytes stores them in a caller-owned `Vec<Vec<u8>>`
  (`call_storage`, living exactly as long as the submission's `RenderOp`s
  do) via `stash_call_buffer`, a small, carefully-documented `unsafe`
  helper -- a called buffer's bytes must be owned (there is no borrowable
  backing for a guest-space fetch, and an aperture-backed one cannot alias
  the board's own `&mut` aperture the way the top-level `ring` slice,
  captured once up front, already does), yet a `DRAW_INLINE`/
  `DRAW_ARRAYS`/`DRAW_ELEMENTS` inside the buffer still needs to borrow a
  `'a`-lifetime slice of it, and safe Rust cannot prove that pushing a
  second buffer into a `Vec<Vec<u8>>` doesn't invalidate a slice already
  handed out from the first (it factually doesn't -- each `Vec<u8>`
  element is its own stable heap allocation -- but the borrow checker
  cannot see that fact through `Vec`'s ordinary API). The lifetime
  exemption from `CAP_REF_SYNC` a called buffer (and every ref inside it)
  gets is purely documentation/behavioural: Copperline's doorbell is
  already fully synchronous, so it satisfies the baseline lifetime rule
  trivially and needs no eager-capture mechanism to build. The no-`CALL`
  path (`Context::submit`) costs nothing extra: it forwards to
  `submit_with` with a throwaway empty `Vec::new()` (no allocation until
  pushed to) and a closure that always returns `None`, never invoking
  `stash_call_buffer`, so every one of this milestone's pre-existing tests
  keeps calling `submit` unchanged. Two edge cases `RingCursor::step`
  handles explicitly for a `new_linear` cursor, since a called buffer's
  length comes straight from a `CALL`'s ref length word with only
  alignment/aperture-bound checking (`validate_ref`), not a 4-byte
  minimum: a buffer shorter than one command header (1-3 bytes) is
  `E_BAD_LENGTH`, not an out-of-bounds read of the header word that isn't
  there; and nesting is rejected as `E_BAD_OPCODE` by `ring.rs` itself,
  ahead of even decoding the nested `CALL`'s ref, so a nested `CALL` with
  a malformed ref reports `E_BAD_OPCODE` rather than `E_BAD_REF`.
- **`POLYGON_MODE`.** State tracking (`state::set_polygon_mode`) predates
  this note; `render.rs`'s pipeline construction was the missing half.
  `wgpu::PolygonMode::Line`/`Point` are gated behind the optional wgpu
  features `POLYGON_MODE_LINE`/`POLYGON_MODE_POINT`, requested only as
  the intersection of what this module ever wants and what the adapter
  actually advertises (`Renderer::new`, never blindly requested) --
  notably Metal has no native polygon-mode point at all, only fill/line,
  so requesting it unconditionally would fail device creation outright on
  that backend. A guest asking for `GL_POINT` where the adapter lacks
  that feature falls back to `GL_LINE` (still strictly more faithful than
  silently filling); asking for `GL_LINE` where even that is unsupported
  falls all the way back to `Fill`. Like `CULL_FACE`, a decoded
  `POLYGON_MODE` only ever reaches the pipeline for `Triangles` topology
  -- a POINTS/LINES-family draw is already just points or lines, nothing
  to "fill" or "outline". Unlike `cull_mode`/`front_face`, which discard
  a whole face, wgpu has no per-face polygon mode: `glPolygonMode` can
  set front and back independently and both can be visible at once with
  culling disabled, but a pipeline can only rasterise one mode. This is
  approximated as a single pipeline-wide mode taken from the *front*
  face (`PipelineKey::polygon_mode`, `Renderer::pipeline_key_from_state`)
  -- exact whenever front and back agree, which is the overwhelmingly
  common case (`glPolygonMode(GL_FRONT_AND_BACK, ...)` is what almost
  every real client calls), and a documented approximation when a guest
  genuinely sets them differently.
- **Fence state across page reallocation (draft 0.15).** `FENCE_COMPLETED`/
  `FENCE_IRQ_TARGET` must return to `0` when `CTX_CONTROL.ALLOC` transitions
  from clear to set, and must NOT be touched by `CTX_CONTROL.RESET` (bit
  2) -- a live owner's mid-life reset must not strand a wait on an
  already-emitted fence ID. `board.rs`'s `write_context` already satisfied
  both halves incidentally before this draft existed: the freeing branch
  (`ALLOC` 1->0) replaces the whole `ContextSlot` with its `Default`,
  which already zeroes both registers well before any later realloc --
  behaviourally equivalent to zeroing on the 0->1 transition itself, since
  a context can only reach `ALLOC=1` a second time by first passing back
  through `ALLOC=0`; and `RESET`'s handler (`Context::reset_all`) only
  ever touches GL state and the error latches, never `fence_completed` or
  the board-level `fence_irq_target` field. No code change was needed --
  two tests (`fence_state_does_not_survive_a_free_and_realloc`,
  `ctx_control_reset_leaves_fence_state_untouched`) pin this down as an
  explicit, spec-cited invariant instead of leaving it an incidental
  side effect of the freeing branch's unrelated full wipe.
- **`READ_PIXELS` format independence (draft 0.17).** `format` names the
  client layout of the *destination* and is independent of
  `SURFFMT_SUPPORTED`, which governs render targets (`SURFACE_DEFINE`)
  only -- every device converts a readback to any non-reserved format
  regardless of what it can render to. `ring.rs`'s decode previously
  gated `READ_PIXELS`'s `format` through the same `check_surffmt` render-
  target check `SURFACE_DEFINE` uses (special-casing only `DEPTH`=255),
  so a guest requesting a readback in a format the device happened not
  to advertise as a render target incorrectly got `E_UNSUPPORTED_FORMAT`
  -- concretely reachable, since Copperline's own `B8G8R8A8` bit was
  itself missing from the default `SURFFMT_SUPPORTED` bitmask until
  `d494119e`, meaning a `READ_PIXELS` in that format would have failed
  even after `B8G8R8A8` became a legal render target. Fixed with a
  dedicated `check_read_pixels_format` that accepts every non-reserved
  format (`1..=9`, or `255`=`DEPTH`) unconditionally and never consults
  `surffmt_supported` -- `CLUT8` (32) and the other reserved ranges stay
  `E_UNSUPPORTED_FORMAT` regardless of that register's value, since the
  exemption is from the capability register, not from the format table
  itself. The draft's second clarification (`render.rs`'s row loop
  writing exactly the rectangle's byte length and leaving inter-row
  padding untouched) was already correct -- documented as deliberate in
  `op_read_pixels`'s own doc comment rather than left to read as
  incidental.
- **Snapshots.** The board serialises in the `ZORR` chunk like every
  other board: GL state, texture images and surface definitions as
  plain data; `wgpu` objects are rebuilt on restore (the renderer field
  itself is `#[serde(skip)]` and recreated lazily on the next doorbell
  that needs it). No GPU handle is ever in a state.
- **Capture and debugger panel.** Not built yet. `[c3d] trace = path`
  and a debugger panel are M6 items in the proposal's own milestone
  table; nothing in M2 depends on either.
