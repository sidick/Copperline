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
land; the specification is revised separately and first.

- **Feature and configuration.** The board is behind an opt-in cargo
  feature and a `[c3d] enabled = true` table with `size`, `contexts`,
  `caps_mask`, `timing` (`fast`/`period`), `adapter`
  (`auto`/`hardware`/`software`) and `trace` (a capture path). A build
  without the feature parses and ignores the table. Browser and
  `--no-default-features` builds carry none of it.
- **GPU.** The board owns a headless `wgpu` device of its own; it never
  shares the window's `pixels` device, so it works identically in
  headless captures (which have no window) and under a software adapter
  in CI. `wgpu` is already in the default build's dependency graph.
- **Threads.** Ring decode, validation, capture of by-reference data
  (`CAP_REF_SYNC`) and GL state updates happen on the emulation thread
  inside the `RING_TAIL` write and the board's `tick`; a render worker
  drives `wgpu`. Copperline's emulated machine runs on the main thread,
  so the "stall until the host has finished" rule in
  [Determinism and timing](#c3d-determinism) stalls the event loop; the
  `fast` preset's latency is chosen so that is rare. This is the same
  model as `copperhf`'s worker (see [](copperhf.md), "The determinism
  model").
- **Surfaces.** Aperture-backed surfaces are the M2 path. The data
  aperture is a RAM-backed chained autoconfig identity, so the host
  writes readbacks through the ordinary board-RAM accessor.
  `CAP_SURFACE_GUESTADDR` into fast RAM works through the shared DMA
  decode today; into another board's VRAM it needs a `DeviceHost`
  accessor for RTG boards' linear apertures (the same capability-hook
  pattern as the audio rings), scheduled for M3. Readbacks into
  aperture-backed surfaces are materialised lazily, per the
  [readback contract](#c3d-readback): the aperture is a board window,
  so both guest reads and guest writes of a deferred region route
  through the board and can trigger materialisation. Guest-address
  surfaces (fast RAM, RTG VRAM) are reached by the CPU's fast path,
  which the board cannot observe, so those readbacks are eager, by the
  fence.
- **Implementation order.** The first client is an OpenGL 1.x subset of
  the Quake-engine shape: immediate mode, matrices, textures with
  texenv, blend/alpha/depth/fog/scissor, vertex arrays with
  `DRAW_ELEMENTS` and compiled vertex arrays, two-unit multitexture,
  `READ_PIXELS`, polygon mode, blend equation and separate blend
  functions, copy-to-texture, texgen (sphere map only, S and T
  independently enabled); and -- confirmed from its function
  table -- **no GL lighting, materials or clip planes** (not stubbed:
  absent as symbols). Those are therefore specified and
  conformance-traced but implemented last; `DRAW_ELEMENTS` with
  multitexture and `COLOR_PACKED` are implemented first. Indexed
  textures are applied at upload by the first client, so `TEX_PALETTE`
  is a later optimisation, not a dependency.
- **Snapshots.** The board serialises in the `ZORR` chunk like every
  other board: GL state, texture images and surface definitions as
  plain data; `wgpu` objects are rebuilt on restore. No GPU handle is
  ever in a state.
- **Capture.** `[c3d] trace = path` records the command stream and
  referenced data in the conformance trace format; the debugger's C3D
  panel shows per-context state, ring pointers, counters and the
  capability mask, and can toggle wireframe, texturing and draw
  coalescing.
