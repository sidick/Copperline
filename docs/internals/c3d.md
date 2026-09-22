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
  feature and a `[c3d] enabled = true` table (`src/c3d/board.rs`,
  `docs/guide/configuration.md`'s `[c3d]` section). As of M2 `enabled` is
  the only key; `size`, `contexts`, `caps_mask`, `timing` and `trace` are
  not yet configuration knobs -- the board fits one fixed 32 MiB Zorro
  III window with 4 contexts. A build without the feature still parses
  and ignores the table, the same contract every other optional board's
  config keeps. Browser and `--no-default-features` builds carry none of
  the board or the renderer; the pure-logic modules stay in them
  regardless (see this chapter's opening paragraphs).
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
- **Capability honesty.** `CAPS0` reports only `CAP_IRQ` and
  `CAP_REF_SYNC` -- `CAP_TRANSFORM`, `CAP_MULTITEXTURE`,
  `CAP_GUESTMEM` and `CAP_SURFACE_GUESTADDR` are all clear, and the ring
  decoder is configured to match: a transform-tier opcode is `E_BAD_OPCODE`
  before it would ever reach a renderer that cannot execute it, rather
  than the register advertising a capability M2 does not deliver.
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
  approximation, carried through unchanged to unit 1). This is renderer-
  side only -- `CAPS0` still does not advertise `CAP_MULTITEXTURE` (see
  "Capability honesty" above), so no real guest can reach it yet.
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
  `CAPS0` still does not advertise `CAP_TRANSFORM` (see "Capability
  honesty" above), so no real guest can reach any GL-space draw path yet.
- **Snapshots.** The board serialises in the `ZORR` chunk like every
  other board: GL state, texture images and surface definitions as
  plain data; `wgpu` objects are rebuilt on restore (the renderer field
  itself is `#[serde(skip)]` and recreated lazily on the next doorbell
  that needs it). No GPU handle is ever in a state.
- **Capture and debugger panel.** Not built yet. `[c3d] trace = path`
  and a debugger panel are M6 items in the proposal's own milestone
  table; nothing in M2 depends on either.
