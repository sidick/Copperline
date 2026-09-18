# The C3D 3D accelerator board: device specification

**Draft 0.10.** This chapter specifies the register file, command stream,
and semantics of C3D, a virtual fixed-function 3D accelerator board
implemented in Copperline (`src/c3d/`, `[c3d]`) and targeted by a guest
`minigl.library`. It is written to be implementable by another emulator
or by hardware without reference to Copperline's source, in the same
spirit as [](mhi.md); see [Porting to another implementation](#c3d-porting).
Protocol changes must follow the [versioning rules](#c3d-versioning).

The board is described in OpenGL 1.x terms because that is the vocabulary
its first client speaks, but it is **not** MiniGL-shaped: no MiniGL
context, lock mode, structure layout or `mgl*` name appears anywhere in
this protocol, and any such notion belongs in the guest library that
translates it into device operations. See
[The API/device split](#c3d-api-device-split). GL enumerant values used
on the wire are those of the public Khronos OpenGL registry
(`GL_TRIANGLES` = `0x0004`, and so on); the registry, and the OpenGL
1.1-1.3 specifications, are the only GL references this document relies
on.

## Design principle

**The 68k does the least possible work per triangle; the device does
everything else.** Every choice below is measured against the cost of a
GL call on the guest CPU: the command stream is written straight into
memory with no per-command register access, immediate-mode geometry is
batched into one command per `glBegin`/`glEnd`, doorbells are rung by
writing one register, bulk data may be passed by reference so the guest
never copies it, and matrix, lighting, clipping and rasterisation all
run in the device. A device that lacks the transform tier (see
[Tiers and capabilities](#c3d-tiers)) still gets the same command
stream; the guest library then supplies window-space vertices itself.

(c3d-zorro-identity)=
## Zorro identity

**The autoconfig identity below belongs to the board, not to any one
implementation of it, and every conforming implementation presents the
same one.** This is not a convention but a requirement, and the reason is
guest binary compatibility: an Amiga program and its `minigl.library`
find the board with `FindConfigDev()` on a manufacturer and product pair
compiled into them. If a second emulator, or an FPGA card, presented its
own identity instead, the *same guest binaries* would fail on it, and the
portability this specification exists for would be lost. The precedent is
ordinary: every emulator that models Village Tronic's Picasso II presents
Village Tronic's identity, because that is what the guest driver looks
for.

- Manufacturer **5192** / `0x1448`, registered to dec0de Consulting, who
  publish this specification and **grant its use, for these product
  numbers only, to any implementation that conforms to it** -- emulator
  or hardware, whoever writes it. An implementation that deviates from
  this specification in a guest-visible way must *not* present this
  identity; presenting it asserts conformance, and a guest that finds it
  is entitled to assume everything in this chapter.
- An implementer who would rather use **their own registered
  manufacturer ID** -- which is the more orthodox reading of what an
  autoconfig manufacturer ID means -- may do so, and adds a row to the
  [identity registry](#c3d-identity-registry) below. That keeps guest
  software working, at the cost of needing a guest-library update before
  the new board is found; using the identity above needs none. Both are
  supported on purpose, because the trade-off is the implementer's to
  make, not this specification's.
- Product **9** for the Zorro III profile, **10** for the Zorro II
  profile. Distinct products let a guest probe for the wider bus first
  and fall back, without having to inspect `er_Type` -- the same shape
  Copperline's emulated ZZ9000 and Graffity boards use for their own two
  profiles.
- The autoconfig **serial number is implementation-defined** and is the
  one field that identifies *who* implemented the board. An
  implementation may encode its own name and version there. Guest
  software must never key on it, or on any behaviour derived from it,
  beyond display and diagnostics; everything a guest needs to decide
  functionally is in `VERSION`, `CAPS0` and the limits registers.
- No autoboot ROM, and not in the Exec free-memory list (`ERTF_MEMLIST`
  clear): the board's window belongs to whichever library claims it.
- A guest **should confirm the board before writing to it** by reading
  `ID` (offset `0x000`) and checking the `"C3D "` magic, then `VERSION`
  for a major it knows. `FindConfigDev()` finds a board by identity;
  the magic proves the window is this register file before anything
  writes a doorbell into it.
(c3d-identity-registry)=
### Identity registry

Guest software finds the board by walking this table, not by hard-coding
one pair. A guest library carries the table, tries `FindConfigDev()` for
each row in order, and confirms the first match by reading `ID` and
`VERSION` (below) before it writes anything. **This table is the
coordination point**: an implementation that uses its own identity adds a
row here by proposing a change to this specification, and guest libraries
pick it up at their next release.

| Manufacturer | Product | Bus | Implementation |
|---|---|---|---|
| `0x1448` (5192) | 9 | Zorro III | The specification's own identity -- any conforming implementation, including Copperline |
| `0x1448` (5192) | 10 | Zorro II | The same, on the 16-bit bus |

A guest must treat every row as equally valid and must not prefer one
implementation's behaviour over another's: everything it needs to decide
functionally is in `VERSION`, `CAPS0` and the limits registers, never in
the identity it was found by. A guest that finds several boards may use
any of them, and should prefer the first that reports the capabilities it
wants.

- Two bus profiles share one register file and one command set:

  | Profile | Product | Bus | Window | Notes |
  |---|---|---|---|---|
  | **Z3** | 9 | Zorro III | one window, default 32 MiB, any power of two from 4 MiB to 256 MiB | Copperline ships this profile. Needs a 32-bit CPU (68020 or later, not 68EC020). |
  | **Z2** | 10 | Zorro II | one window, 4 MiB or 8 MiB | For hardware implementations on the 16-bit bus. The data aperture is correspondingly small; texture data streams through the ring rather than living in the aperture. |

  A guest library must read `APERTURE_OFFSET`/`APERTURE_SIZE` rather than
  assume either layout.
- **Window layout** (offsets from the configured base), identical in both
  profiles:

  | Offset | Size | Contents |
  |---|---|---|
  | `0x0000_0000` | 64 KiB | [Global registers](#c3d-global-registers) |
  | `0x0001_0000` | `MAX_CONTEXTS` &times; 256 bytes | [Context register pages](#c3d-context-registers) |
  | `APERTURE_OFFSET` (`0x0010_0000`) | `APERTURE_SIZE` (rest of window) | [Data aperture](#c3d-aperture): rings, bulk data, aperture-backed surfaces |

  Offsets in the register area that are not listed are **reserved**: they
  read as `0x0000_0000` and discard writes, in every protocol version.

(c3d-conventions)=
## Conventions

- All registers and all command-stream words are **32-bit, big-endian**.
  Floating-point values are IEEE-754 single precision, big-endian. The
  68k never byte-swaps. GL `double` parameters (`glOrtho`, `glFrustum`,
  `glClearDepth`, `glDepthRange`, `glRotated`, `glClipPlane`...) are
  narrowed to single precision by the guest library.
- Reserved register bits read as zero and must be written as zero.
  Reserved command fields must be zero; a non-zero reserved field is a
  [protocol error](#c3d-errors), not ignored, so that future extension
  stays honest.
- `RO` = read only (writes discarded), `WO` = write only (reads return
  `0`), `RW` = both, `W1C` = write 1 to clear. No register read has a side
  effect except where a register is explicitly described as a
  read-to-consume value.
- The device never hangs and never stalls the bus on bad input. Every
  malformed input is either skipped with a latched error or halts the
  offending context until acknowledged; the register file always answers.
- "Guest address" means a physical address in the emulated Amiga's
  address space. Under the Z2 profile only the low 24 bits are driven.

(c3d-access-size)=
## Access size and alignment

Registers are 32-bit at longword-aligned offsets. **Longword access
(`move.l`) is the primary and recommended access size**, and is what the
guest library uses exclusively for registers.

- On the **Z3 profile** a longword access is one bus cycle and is atomic.
- On the **Z2 profile** the bus is 16 bits wide and a `move.l` arrives as
  two word cycles, high word first at the lower address (68k order). A
  Z2 implementation must make the pair atomic from the guest's point of
  view: a register **write** takes effect when its low word (offset+2)
  is written, the high word having been latched; a register **read**
  latches the whole value when its high word (offset+0) is read, and
  the following low-word read returns the latched low half. A guest
  library must therefore always access a register as one `move.l` and
  never as two independent `move.w`s in the other order.
- **Word and byte accesses** to registers are honoured on the Z3 profile
  as partial accesses to the big-endian longword (a byte write changes
  that byte only) and are the two-cycle mechanism above on Z2. They
  are not recommended and the guest library never issues them.
- Accesses to the **data aperture** are plain memory accesses of any
  size and alignment the bus allows; the aperture has no side effects.
- Reads of a `WO` register return `0`; writes to an `RO` register are
  discarded. Neither is an error.

(c3d-global-registers)=
## Global registers

| Offset | Name | Access | Reset | Purpose |
|---|---|---|---|---|
| `0x000` | `ID` | RO | `0x4333_4420` | Magic `"C3D "`; identifies a C3D device |
| `0x004` | `VERSION` | RO | `0x0000_000A` | `major << 16 \| minor`; see [Versioning](#c3d-versioning) |
| `0x008` | `CAPS0` | RO | board-fixed | [Capability bits](#c3d-tiers) |
| `0x00C` | `CAPS1` | RO | `0` | Reserved for future capability bits |
| `0x010` | `STATUS` | RO | `0x0000_0001` | Bit 0 `READY`; bit 1 `RESETTING`; bit 2 `FATAL` (an implementation-internal failure; every context is halted; only `CONTROL.RESET` recovers) |
| `0x014` | `CONTROL` | RW | `0x0000_0001` | Bit 0 `ENABLE` (clear: every context stops decoding and no interrupt is asserted; ring pointers are preserved); bit 1 `RESET` (write 1: full device reset -- every context freed, every object destroyed, all memory untouched; self-clearing, `STATUS.RESETTING` set meanwhile) |
| `0x018` | `IRQ_STATUS` | W1C | `0` | Bit `n` (0..15): context `n` has a pending fence interrupt; bit `16+n`: context `n` has latched a protocol error. Writing 1 clears the bit; the line deasserts when `IRQ_STATUS & IRQ_ENABLE` is zero |
| `0x01C` | `IRQ_ENABLE` | RW | `0` | Mask with the same layout as `IRQ_STATUS` |
| `0x020` | `APERTURE_OFFSET` | RO | `0x0010_0000` | Window-relative start of the data aperture |
| `0x024` | `APERTURE_SIZE` | RO | board-fixed | Bytes in the data aperture |
| `0x028` | `MAX_CONTEXTS` | RO | board-fixed, &ge; 1 | Number of context pages (Copperline: 4; at most 16) |
| `0x02C` | `MAX_RING_SIZE` | RO | board-fixed | Largest `RING_SIZE` accepted, bytes, a power of two |
| `0x040` | `MAX_TEXTURE_SIZE` | RO | board-fixed | Largest texture edge, a power of two (&ge; 256) |
| `0x044` | `MAX_TEXTURE_UNITS` | RO | board-fixed, &ge; 1 | Texture units (&gt; 1 implies `CAP_MULTITEXTURE`) |
| `0x048` | `MAX_TEXTURES` | RO | board-fixed | Highest texture object ID, per context |
| `0x04C` | `MAX_SURFACES` | RO | board-fixed | Highest surface ID, per context |
| `0x050` | `MAX_LIGHTS` | RO | board-fixed | Lights (&ge; 8 when `CAP_TRANSFORM`; `0` otherwise) |
| `0x054` | `MAX_CLIP_PLANES` | RO | board-fixed | User clip planes (&ge; 6 when `CAP_TRANSFORM`; `0` otherwise) |
| `0x058` | `MAX_MATRIX_DEPTH_MV` | RO | board-fixed | Modelview stack depth (&ge; 32 when `CAP_TRANSFORM`) |
| `0x05C` | `MAX_MATRIX_DEPTH_PROJ` | RO | board-fixed | Projection stack depth (&ge; 2) |
| `0x060` | `MAX_MATRIX_DEPTH_TEX` | RO | board-fixed | Texture stack depth (&ge; 2) |
| `0x064` | `TEXFMT_SUPPORTED` | RO | board-fixed | Bitmap of [texture formats](#c3d-texture-formats): bit `n` set means format `n` is accepted |
| `0x068` | `SURFFMT_SUPPORTED_LO` | RO | board-fixed | Bitmap of [surface formats](#c3d-surface-formats) `0`-`31` |
| `0x06C` | `SURFFMT_SUPPORTED_HI` | RO | board-fixed | Bitmap of surface formats `32`-`63` |
| `0x070` | `MAX_SURFACE_WIDTH` | RO | board-fixed | Largest surface width, pixels (&ge; 1024) |
| `0x074` | `MAX_SURFACE_HEIGHT` | RO | board-fixed | Largest surface height (&ge; 768) |

Limits are always queried, never assumed. A guest library that needs
more than a limit allows reports failure to its client rather than
guessing.

(c3d-tiers)=
## Tiers and capabilities

`CAPS0` (`0x008`) is a bitmask. **The baseline -- no bits set -- is a
complete, conformant device**: a rasteriser accepting window-space
vertices, with all bulk data passed through the aperture, fences by
polling, one texture unit, and surfaces in the aperture. Every bit is an
optional extension. The guest library must run correctly against a
baseline-only device.

| Bit | Name | Meaning |
|---|---|---|
| 0 | `CAP_GUESTMEM` | [References](#c3d-refs) may name guest addresses (space `1`), for rings, bulk data and query results. The device reads and writes guest memory itself. |
| 1 | `CAP_IRQ` | Fence and error interrupts are available on INT2. Polling `FENCE_COMPLETED` is always valid regardless. |
| 2 | `CAP_TRANSFORM` | The **transform tier**: the [matrix](#c3d-cmd-matrix), [lighting](#c3d-cmd-lighting), viewport/depth-range and clip-plane commands, and the GL-space draw commands. Without it those opcodes are unknown, `MAX_LIGHTS`/`MAX_CLIP_PLANES` read `0`, and only the window-space draw commands exist. |
| 3 | `CAP_MULTITEXTURE` | `MAX_TEXTURE_UNITS` &gt; 1; `ACTIVE_UNIT` and per-unit `TEX_BIND`/`TEX_ENV`/`CURRENT_TEXCOORD` accept units above `0`. |
| 4 | `CAP_SURFACE_GUESTADDR` | Surfaces may be backed by an arbitrary guest address range (another board's VRAM, fast RAM) instead of the aperture. The device writes readbacks there itself. |
| 5 | -- | Reserved (was a present-layer overlay in an earlier draft; withdrawn -- see [Presentation](#c3d-presentation)). Reads as `0`. |
| 6 | `CAP_REF_SYNC` | Every by-reference operand is captured by the device **before the `RING_TAIL` write that submits it returns**. The guest may reuse or free referenced memory as soon as that write retires. Without this bit the [baseline lifetime rule](#c3d-ref-lifetime) applies. Meaningful only with `CAP_GUESTMEM`. |
| 7-31 | -- | Reserved, read as `0` |

Copperline sets bits 0, 1, 2, 3, 4 and 6, and provides a configuration
switch that masks any subset off so the baseline and each intermediate
combination are exercised against the same guest library.

Tier summary:

| | Baseline (rasteriser) | `CAP_TRANSFORM` |
|---|---|---|
| Vertex position | window space `(x, y, z, rhw)` | object space `(x, y, z, w)`; transformed, lit, clipped by the device |
| Draw opcodes | `DRAW_INLINE_WIN`, `DRAW_ARRAYS_WIN`, `DRAW_ELEMENTS_WIN` | those plus `DRAW_INLINE`, `DRAW_ARRAYS`, `DRAW_ELEMENTS` |
| Matrix stacks, lighting, materials, clip planes, viewport, depth range | absent | present |
| Fog distance | vertex `FOGCOORD`, else derived from vertex `rhw` | vertex `FOGCOORD`, else eye-space distance per GL |
| Everything else (textures, texenv, blend, depth, alpha test, scissor, colour mask, polygon offset, fog, surfaces, fences, errors) | identical | identical |

The rasteriser tier is what period hardware did and what a hardware
implementation is expected to build. It is also the interface a
post-transform client such as a Warp3D-style front end uses. A guest
library serving a GL client on a baseline device performs transform and
lighting itself and submits window-space vertices.

(c3d-context-registers)=
## Context register pages

One context = one command ring = one independent GL state machine.
Context `n` (`0 <= n < MAX_CONTEXTS`) has a 256-byte register page at
`0x0001_0000 + n * 0x100`:

| Offset | Name | Access | Reset | Purpose |
|---|---|---|---|---|
| `0x00` | `CTX_CONTROL` | RW | `0` | Bit 0 `ALLOC` (set: page in use; clear: context freed, all its objects destroyed, ring stopped); bit 1 `ENABLE` (decode the ring); bit 2 `RESET` (write 1: return this context's GL state and objects to their initial values, and clear `GL_ERROR`, `ERROR_CODE`/`ERROR_OFFSET` and `CTX_STATUS.HALTED`, without touching the ring pointers; self-clearing). Unlike `CTX_RESET_STATE`, this **does** clear both error latches -- it is the guest asking for a clean slate, and a context that stayed halted across it, or that inherited a halt when freed and reallocated, would be stuck |
| `0x04` | `CTX_STATUS` | RO | `0` | Bit 0 `BUSY` (commands submitted and not yet complete); bit 1 `HALTED` (a framing error halted decoding; see [Errors](#c3d-errors)); bit 2 `IDLE_RING` (`RING_HEAD == RING_TAIL`) |
| `0x08` | `RING_BASE` | RW | `0` | Ring start: an aperture offset, or a guest address if bit 31 of `RING_SIZE` is set (`CAP_GUESTMEM`). Must be longword aligned. |
| `0x0C` | `RING_SIZE` | RW | `0` | Bits 30:0: ring bytes, a power of two, 4 KiB..`MAX_RING_SIZE`; bit 31: `RING_BASE` is a guest address. Written only while `ENABLE` is clear; writing it also resets `RING_HEAD` and `RING_TAIL` to `0`. |
| `0x10` | `RING_TAIL` | RW | `0` | Guest write offset, bytes, longword aligned, `< RING_SIZE`. **Writing this register is the doorbell**: the device may consume up to the new tail. |
| `0x14` | `RING_HEAD` | RO | `0` | Device read offset, bytes. Commands before it have been *consumed* (decoded and, if by-reference and `CAP_REF_SYNC`, captured), not necessarily executed. |
| `0x18` | `FENCE_COMPLETED` | RO | `0` | ID of the most recent [fence](#c3d-fences) whose preceding commands have all taken effect |
| `0x1C` | `ERROR_CODE` | RO | `0` | First latched [protocol error](#c3d-errors) since the last `ERROR_ACK`; `0` = none |
| `0x20` | `ERROR_OFFSET` | RO | `0` | Ring offset of the command that raised `ERROR_CODE` |
| `0x24` | `ERROR_ACK` | WO | -- | Any write clears `ERROR_CODE`/`ERROR_OFFSET`, clears `CTX_STATUS.HALTED`, and resumes decoding at `RING_HEAD` |
| `0x28` | `GL_ERROR` | RO | `0` | First pending [GL error](#c3d-gl-errors) (a GL enumerant, e.g. `0x0500` `GL_INVALID_ENUM`); `0` = `GL_NO_ERROR` |
| `0x2C` | `GL_ERROR_ACK` | WO | -- | Any write clears `GL_ERROR` (the `glGetError` read-and-clear, split into a read and a write so no register read has a side effect) |
| `0x30` | `FENCE_IRQ_TARGET` | RW | `0` | When `CAP_IRQ`: a fence interrupt is raised when `FENCE_COMPLETED` reaches or passes this value (unsigned compare). `0` disables. |

Allocation is a guest-side concern: the guest library arbitrates pages
among its own openers (a semaphore in its library base) and the device
only provides `MAX_CONTEXTS` independent pages. A context that is freed
(`ALLOC` cleared) or reset by `CONTROL.RESET` releases every texture,
surface and ring the device holds for it. Per-context rings mean the
submit path needs no cross-task locking.

(c3d-aperture)=
## The data aperture

`APERTURE_SIZE` bytes of ordinary board memory at `APERTURE_OFFSET`. The
guest reads and writes it like RAM. It holds command rings, by-reference
bulk data (texture images, vertex arrays), query results, and
aperture-backed surfaces. Allocation within it is entirely the guest
library's business; the device only ever sees offsets handed to it in
registers and commands.

The aperture is the **baseline** location for everything; `CAP_GUESTMEM`
and `CAP_SURFACE_GUESTADDR` let the same things live in guest memory
instead. A guest library must be able to operate with the aperture alone.

(c3d-command-stream)=
## Command stream

### Framing

A ring is a circular byte buffer of `RING_SIZE` bytes. Commands are
sequences of 32-bit words:

```
word 0:  opcode[31:16] | length[15:0]     (length in words, including word 0; >= 1)
word 1.. payload
```

- The guest writes commands starting at `RING_TAIL`, then writes the new
  `RING_TAIL`. The device decodes from `RING_HEAD` up to `RING_TAIL`.
- **A command never wraps the end of the ring.** When the remaining space
  before the end is too small, the guest pads it with a single `NOP`
  whose `length` covers the remainder (any length &ge; 1 is legal for
  `NOP`) and continues at offset `0`.
- The guest must not advance `RING_TAIL` past `RING_HEAD - 4` (the ring
  is full when one word short of empty); a device is entitled to treat a
  tail that overtakes the head as a `E_RING_OVERRUN` framing error.
- `length` of `0`, or a command extending past `RING_TAIL`, is a framing
  error and halts the context.
- The maximum command is `0xFFFF` words. Anything larger (a big texture
  image) is passed by [reference](#c3d-refs).

(c3d-refs)=
### References

A **ref** names a byte range outside the command itself:

```
word A: address      -- aperture offset (space 0) or guest address (space 1)
word B: space[31] | length[30:0]
```

Space `1` requires `CAP_GUESTMEM`; on a device without it, space `1` is
`E_BAD_REF`. A ref must lie entirely within the aperture (space `0`) or
within guest memory the implementation can reach (space `1`); otherwise
`E_BAD_REF`. Refs are longword aligned unless a command says otherwise.

(c3d-ref-lifetime)=
**Lifetime.** Baseline rule: memory named by a ref must stay valid and
unchanged until the first `FENCE` submitted after the referencing
command has completed. With `CAP_REF_SYNC`, the device has captured the
data before the submitting `RING_TAIL` write returns, and the guest may
reuse the memory immediately. A guest library that needs to honour a
client's "free on return" contract on a device without `CAP_REF_SYNC`
copies the data inline into the ring (always legal, and for small
payloads cheaper anyway) or into aperture space it manages.

### Opcode map

Opcodes are grouped by the high byte. Unknown opcodes are `E_BAD_OPCODE`
and are skipped (their `length` is trusted so decoding can continue).

| Range | Group | Tier |
|---|---|---|
| `0x00xx` | [Control](#c3d-cmd-control) | baseline |
| `0x01xx` | [Surfaces](#c3d-cmd-surfaces) | baseline |
| `0x02xx` | [Raster state](#c3d-cmd-raster) | baseline (viewport/depth range: transform) |
| `0x03xx` | [Matrix](#c3d-cmd-matrix) | `CAP_TRANSFORM` |
| `0x04xx` | [Lighting, clipping and texgen](#c3d-cmd-lighting) | `CAP_TRANSFORM` |
| `0x05xx` | [Textures](#c3d-cmd-textures) | baseline |
| `0x06xx` | [Current vertex state](#c3d-cmd-current) | baseline |
| `0x07xx` | [Draw](#c3d-cmd-draw) | baseline (`*_WIN`) / `CAP_TRANSFORM` (GL space) |
| `0x08xx` | [Queries](#c3d-cmd-queries) | baseline |
| `0x09xx`-`0xFFxx` | -- | Reserved |

(c3d-cmd-control)=
### Control (`0x00xx`)

| Opcode | Name | Payload | Effect |
|---|---|---|---|
| `0x0000` | `NOP` | any | Ignored. `length` may be anything &ge; 1 (used for end-of-ring padding). |
| `0x0001` | `FENCE` | `id` | When every preceding command in this ring has taken effect (including readbacks and query results landing in memory), `FENCE_COMPLETED` becomes `id` and, if enabled, the context's fence interrupt is raised. `id` must be greater (unsigned) than the previous fence's; `0` is never a valid fence ID. |
| `0x0002` | `FLUSH` | -- | Hint that the guest will wait; the device should not defer work. No guest-visible effect. |
| `0x0003` | `FINISH` | -- | Equivalent to `FLUSH`; provided so a trace reads like the GL calls that produced it. The guest waits by fencing. |
| `0x0004` | `CTX_RESET_STATE` | -- | Return this context's GL state (not its objects, not its ring) to initial values. **`GL_ERROR` and the protocol-error latch are left untouched**: this is a command in the stream, and silently discarding an error the guest has not yet read would lose it. |

(c3d-fences)=
**Fences** are the only synchronisation primitive. `glFinish` is
`FENCE` + wait; a buffer swap is `SURFACE_READBACK` + `FENCE` + wait; a
synchronous query is `QUERY` + `FENCE` + wait. Waiting means polling
`FENCE_COMPLETED` or, with `CAP_IRQ`, setting `FENCE_IRQ_TARGET` and
sleeping on the INT2 server. Fence IDs are per context and need only be
monotonic; the guest library typically uses a counter.

(c3d-cmd-surfaces)=
### Surfaces (`0x01xx`)

A **surface** is a colour render target: an ID, a size, a stride, a
[pixel format](#c3d-surface-formats), and a backing location. The device
keeps a depth buffer per surface internally, sized to the surface; it is
never guest-visible except through `READ_PIXELS` with the depth format.
IDs are guest-allocated, `1..MAX_SURFACES`; `0` means "none".

| Opcode | Name | Payload | Effect |
|---|---|---|---|
| `0x0100` | `SURFACE_DEFINE` | `id, width, height, stride_bytes, format, flags, address` | Create or **redefine** surface `id`. `flags` bit 0: `address` is a guest address (`CAP_SURFACE_GUESTADDR`), else an aperture offset. Redefining an existing surface keeps its depth buffer if the size is unchanged and otherwise recreates it. `stride_bytes` must be at least the row's byte length (`width` &times; the format's bytes per pixel) or `E_BAD_ARG`. An **aperture-backed** surface must lie wholly inside the aperture -- its last row begins at `address + (height - 1) &times; stride_bytes` and runs for the row's byte length -- or `E_BAD_REF`; a guest-address surface is not range-checked, exactly as a space-`1` reference is not. The colour contents of a freshly defined surface are **undefined** until a `CLEAR` or `SURFACE_UPLOAD`. |
| `0x0101` | `SURFACE_DESTROY` | `id` | Release it. If it is the draw surface, the draw surface becomes `0` and draws are `E_NO_SURFACE`. |
| `0x0102` | `SET_DRAW_SURFACE` | `id` | Subsequent draws, clears and readbacks target `id`. |
| `0x0103` | `SURFACE_UPLOAD` | `x, y, w, h` | Read the rectangle from the surface's backing memory into the render target (the guest has drawn 2D into the bitmap and wants 3D composited over it). |
| `0x0104` | `SURFACE_READBACK` | `x, y, w, h` | Make the rendered contents of the rectangle observable in the backing memory, converted to the surface format. See the [readback contract](#c3d-readback). |
| `0x0105` | `CLEAR` | `mask` | `mask` bit 0: colour (to `CLEAR_COLOR`); bit 1: depth (to `CLEAR_DEPTH`). Honours the scissor and colour mask. |

Rectangles are in surface pixels, origin top-left, and must lie within
the surface (`E_BAD_RECT` otherwise). All of these except
`SURFACE_DEFINE`/`SURFACE_DESTROY` act on the current draw surface.

(c3d-readback)=
**Readback contract.** A `SURFACE_READBACK` is **logically complete when
its covering `FENCE` completes**: from that moment the rectangle's
backing memory holds the rendered pixels, any agent -- the guest CPU,
another board, a capture -- that reads it sees them, and any *write* to
it by the guest takes precedence over the readback (a guest drawing 2D
over the 3D after the fence must never have it overwritten later).
Until the covering fence completes, the guest must not write the
rectangle (its write may be overwritten). The device never writes
outside the rectangle.

An implementation is free to keep the frame on its own side and
materialise it into the backing memory lazily, provided nothing can
observe the difference -- which means it must intercept **both reads
and writes** of the deferred region: a read materialises first; a write
materialises first and then lands on top. An implementation that cannot
observe writes to a region (memory the guest CPU reaches directly) must
materialise eagerly, by the fence. A hardware implementation simply
copies at readback time.

(c3d-presentation)=
**Presentation is not a device concept.** The guest library keeps doing
whatever its window system does -- Intuition ScreenBuffers page flipping,
blitting an off-screen bitmap into a window -- and simply points
`SET_DRAW_SURFACE` at whichever bitmap is now the back buffer. Two
bitmaps are two surfaces. Because a P96 bitmap's address is stable only
while it is locked, the guest's per-frame sequence is: lock the bitmap,
compare its address with the surface's definition, re-emit
`SURFACE_DEFINE` if it moved, render, `SURFACE_READBACK`, `FENCE`,
wait, unlock. A device never draws directly onto a display; an earlier
draft's present-layer overlay capability was withdrawn because it would
make the frame unobservable in memory, breaking 2D-over-3D composition,
guest-side pixel reads and every memory-based capture.

**Client locks and 2D composition.** Period 3D APIs expose a hardware
*lock*: while the client holds it, nothing else may touch the
framebuffer, and a client draws 2D (a HUD, console text) into the
bitmap only between an unlock and the next lock, in whichever of the
API's lock modes actually guarantees a release. **A lock is not a device
concept, and the device never blocks anyone's access to backing
memory** -- it is ordinary memory throughout. The only hazard is
staleness, and the guest library removes it with two commands: a
`SURFACE_READBACK` + `FENCE` at every point its API promises the pixels
are visible (an unlock; the end of a per-batch auto-lock; a
time-sliced lock's expiry; a swap), and a `SURFACE_UPLOAD` at every
point the client may have drawn 2D since (a lock; the start of the next
frame), of the dirtied rectangle where the API lets the library know it
and of the whole surface where it does not. A mode that unlocks after
every primitive batch therefore costs a readback per batch -- which is
exactly the case the lazy materialisation above exists for: on an
implementation that defers, a readback nobody reads costs nothing.

(c3d-surface-formats)=
### Surface formats

The device's own enum. The guest library maps its window system's pixel
format identifiers onto these; this specification does not reference any
window system's constants. `SURFFMT_SUPPORTED_*` reports which a device
implements; a conforming device implements at least `R5G6B5` and
`A8R8G8B8`.

| Value | Name | Layout (bytes in memory order) |
|---|---|---|
| `1` | `R5G6B5` | 16-bit big-endian `RRRRRGGG GGGBBBBB` |
| `2` | `R5G6B5_LE` | the same, little-endian (byte-swapped) |
| `3` | `R5G5B5` | 16-bit big-endian `xRRRRRGG GGGBBBBB` |
| `4` | `R5G5B5_LE` | the same, little-endian |
| `5` | `A8R8G8B8` | `A R G B` |
| `6` | `B8G8R8A8` | `B G R A` |
| `7` | `R8G8B8A8` | `R G B A` |
| `8` | `R8G8B8` | `R G B`, 3 bytes per pixel |
| `9` | `B8G8R8` | `B G R` |
| `10`-`31` | -- | Reserved chunky formats |
| `32` | `CLUT8` | Reserved: 8-bit indexed, palette via a future `SURFACE_PALETTE` command. Not in version 1. |
| `33`-`47` | -- | Reserved |
| `48`-`63` | -- | Reserved for **planar** formats (Amiga bitplanes: plane count, interleave, modulo, palette). Not in version 1; reserved so a chipset-screen target can be added without renumbering. |
| `255` | `DEPTH` | Only for `READ_PIXELS`: 32-bit unsigned depth, `0` = near, `0xFFFF_FFFF` = far |

Alpha in a readback of an alpha-carrying format is the rendered alpha.
Formats without alpha discard it. Conversion from the device's internal
precision rounds to nearest; a device may dither to 16-bit formats and
says so in its documentation.

(c3d-cmd-raster)=
### Raster state (`0x02xx`)

All payload words are `u32` unless marked `f32`. Enumerants are GL's.

| Opcode | Name | Payload | Notes |
|---|---|---|---|
| `0x0200` | `ENABLE` | `cap` | `cap` is a GL enumerant from the accepted set below |
| `0x0201` | `DISABLE` | `cap` | |
| `0x0202` | `BLEND_FUNC` | `sfactor, dfactor` | The GL 1.1 factor set: `ZERO`, `ONE`, `SRC_COLOR`, `ONE_MINUS_SRC_COLOR`, `DST_COLOR`, `ONE_MINUS_DST_COLOR`, `SRC_ALPHA`, `ONE_MINUS_SRC_ALPHA`, `DST_ALPHA`, `ONE_MINUS_DST_ALPHA`, `SRC_ALPHA_SATURATE` (source only) |
| `0x0203` | `DEPTH_FUNC` | `func` | `NEVER`..`ALWAYS` |
| `0x0204` | `DEPTH_MASK` | `flag` | `0`/`1` |
| `0x0205` | `DEPTH_RANGE` | `near f32, far f32` | **Transform tier.** Window-space vertices already carry final depth. |
| `0x0206` | `ALPHA_FUNC` | `func, ref f32` | |
| `0x0207` | `CULL_FACE` | `mode` | `FRONT`, `BACK`, `FRONT_AND_BACK` |
| `0x0208` | `FRONT_FACE` | `mode` | `CW`, `CCW`. Winding is evaluated in window space with y down; the guest library accounts for its window system's origin. |
| `0x0209` | `SHADE_MODEL` | `mode` | `FLAT`, `SMOOTH`. Flat shading takes the **last** vertex of each primitive (the GL 1.x provoking vertex); see [Draw](#c3d-cmd-draw) for quads and polygons. |
| `0x020A` | `COLOR_MASK` | `r, g, b, a` | each `0`/`1` |
| `0x020B` | `SCISSOR` | `x, y, w, h` | Surface pixels, origin top-left |
| `0x020C` | `VIEWPORT` | `x, y, w, h` | **Transform tier.** |
| `0x020D` | `POLYGON_OFFSET` | `factor f32, units f32` | Applied when `POLYGON_OFFSET_FILL` is enabled. `units` scales the smallest resolvable depth difference of the device's depth buffer, as GL specifies. |
| `0x020E` | `CLEAR_COLOR` | `r f32, g f32, b f32, a f32` | |
| `0x020F` | `CLEAR_DEPTH` | `depth f32` | |
| `0x0210` | `FOG_MODE` | `mode` | `LINEAR`, `EXP`, `EXP2` |
| `0x0211` | `FOG_PARAMS` | `density f32, start f32, end f32` | |
| `0x0212` | `FOG_COLOR` | `r f32, g f32, b f32, a f32` | |
| `0x0213` | `HINT` | `target, mode` | Accepted and ignored. |
| `0x0214` | `LINE_WIDTH` | `width f32` | A device must draw at least a one-pixel line; wider lines may be drawn at one pixel. |
| `0x0215` | `POINT_SIZE` | `size f32` | Likewise. |
| `0x0216` | `LOGIC_OP` | `opcode` | Reserved; `E_BAD_OPCODE` in version 1. GL 1.x logic ops are not part of this device. |
| `0x0217` | `POLYGON_MODE` | `face, mode` | `FILL`, `LINE`, `POINT`. Baseline: the device converts the polygon's edges to lines (or its vertices to points) before rasterisation, so this is primitive conversion, not a rasteriser feature. `face` is honoured as GL specifies. |
| `0x0218` | `BLEND_EQUATION` | `mode` | `FUNC_ADD`, `FUNC_SUBTRACT`, `FUNC_REVERSE_SUBTRACT`, `MIN`, `MAX`. Baseline. |
| `0x0219` | `BLEND_FUNC_SEPARATE` | `src_rgb, dst_rgb, src_a, dst_a` | Baseline. `BLEND_FUNC` sets both pairs alike. |

Accepted `ENABLE`/`DISABLE` caps: `ALPHA_TEST`, `BLEND`, `CULL_FACE`,
`DEPTH_TEST`, `DITHER`, `FOG`, `POLYGON_OFFSET_FILL`, `SCISSOR_TEST`,
`TEXTURE_2D` (per active unit), and -- transform tier only --
`LIGHTING`, `LIGHT0`..`LIGHT7`, `COLOR_MATERIAL`, `NORMALIZE`,
`RESCALE_NORMAL`, `CLIP_PLANE0`..`CLIP_PLANE5`, `TEXTURE_GEN_S`,
`TEXTURE_GEN_T` (per active unit). Any other cap is a
`GL_INVALID_ENUM` [GL error](#c3d-gl-errors), not a protocol error.
There is no stencil buffer in version 1; `STENCIL_TEST` is
`GL_INVALID_ENUM`.

Initial state is GL's: everything disabled except `DITHER`; blend
`ONE, ZERO`; depth func `LESS`, mask `1`, range `0..1`; alpha func
`ALWAYS, 0`; cull `BACK`, front `CCW`; shade `SMOOTH`; colour mask all
`1`; scissor and viewport the full draw surface at
`SET_DRAW_SURFACE` time; clear colour `0,0,0,0`, clear depth `1`; fog
`EXP`, density `1`, start `0`, end `1`, colour `0,0,0,0`; blend
equation `FUNC_ADD`, and `BLEND_FUNC_SEPARATE`'s two pairs equal to the
blend func's; polygon mode `FILL` for both faces; line width and point
size `1`.

(c3d-cmd-matrix)=
### Matrix (`0x03xx`) -- `CAP_TRANSFORM`

Matrices are 16 `f32` in GL's column-major order. Three stacks:
modelview, projection, texture (one per texture unit, selected by the
active unit).

| Opcode | Name | Payload |
|---|---|---|
| `0x0300` | `MATRIX_MODE` | `mode` (`MODELVIEW`, `PROJECTION`, `TEXTURE`) |
| `0x0301` | `LOAD_MATRIX` | `m[16] f32` |
| `0x0302` | `LOAD_IDENTITY` | -- |
| `0x0303` | `MULT_MATRIX` | `m[16] f32` |
| `0x0304` | `PUSH_MATRIX` | -- (`GL_STACK_OVERFLOW` past the reported depth) |
| `0x0305` | `POP_MATRIX` | -- (`GL_STACK_UNDERFLOW` at depth 1) |
| `0x0306` | `TRANSLATE` | `x, y, z` (`f32`) |
| `0x0307` | `ROTATE` | `angle_deg, x, y, z` (`f32`) |
| `0x0308` | `SCALE` | `x, y, z` (`f32`) |
| `0x0309` | `FRUSTUM` | `l, r, b, t, n, f` (`f32`) |
| `0x030A` | `ORTHO` | `l, r, b, t, n, f` (`f32`) |

Matrix stacks are **device-authoritative**: the guest streams
operations and reads a matrix back with a [query](#c3d-cmd-queries).
Vertex transformation, lighting, and user clipping follow the OpenGL
1.1 specification's fixed-function pipeline exactly, including the
`[-1, 1]` clip-space depth range mapped through `DEPTH_RANGE`.

(c3d-cmd-lighting)=
### Lighting, clipping and texgen (`0x04xx`) -- `CAP_TRANSFORM`

| Opcode | Name | Payload |
|---|---|---|
| `0x0400` | `LIGHT` | `light, pname, v[4] f32` -- `light` is `LIGHT0`..; `pname` one of `AMBIENT`, `DIFFUSE`, `SPECULAR`, `POSITION`, `SPOT_DIRECTION`, `SPOT_EXPONENT`, `SPOT_CUTOFF`, `CONSTANT_ATTENUATION`, `LINEAR_ATTENUATION`, `QUADRATIC_ATTENUATION` (scalars in `v[0]`, unused entries zero) |
| `0x0401` | `LIGHT_MODEL` | `pname, v[4] f32` -- `LIGHT_MODEL_AMBIENT`, `LIGHT_MODEL_LOCAL_VIEWER`, `LIGHT_MODEL_TWO_SIDE` |
| `0x0402` | `MATERIAL` | `face, pname, v[4] f32` -- `AMBIENT`, `DIFFUSE`, `SPECULAR`, `EMISSION`, `SHININESS`, `AMBIENT_AND_DIFFUSE` |
| `0x0403` | `COLOR_MATERIAL` | `face, mode` |
| `0x0404` | `CLIP_PLANE` | `plane, eq[4] f32` -- transformed by the inverse modelview at the time of the command, per GL |
| `0x0405` | `TEXGEN` | `unit, coord, mode` -- `coord` is `S` or `T`; `mode` is `OBJECT_LINEAR`, `EYE_LINEAR` or `SPHERE_MAP`. Takes effect when `TEXTURE_GEN_S`/`TEXTURE_GEN_T` is enabled for the unit. |
| `0x0406` | `TEXGEN_PLANE` | `unit, coord, plane, eq[4] f32` -- `plane` is `OBJECT_PLANE` or `EYE_PLANE`; an eye plane is transformed by the inverse modelview at the time of the command, per GL |

`POSITION` and `SPOT_DIRECTION` are transformed by the current modelview
at the time of the command, per GL.

(c3d-cmd-textures)=
### Textures (`0x05xx`)

Texture object IDs are guest-allocated, `1..MAX_TEXTURES`, per context;
`0` is the null texture (texturing disabled for the unit). A 2D target
only in version 1.

| Opcode | Name | Payload | Notes |
|---|---|---|---|
| `0x0500` | `TEX_CREATE` | `id` | Creates an empty object with default parameters. Creating an existing ID resets its parameters and discards its images, but **leaves any unit bindings to it intact** -- the object's identity persists, and only `TEX_DESTROY` unbinds. |
| `0x0501` | `TEX_DESTROY` | `id` | Unbinds it from every unit. |
| `0x0502` | `TEX_BIND` | `unit, id` | |
| `0x0503` | `TEX_IMAGE` | `id, level, format, width, height, row_bytes, ref` | Defines mip `level` (`0..`) of `id` from the pixels at `ref`, `height` rows of `row_bytes` bytes (so rows may carry GL's unpack padding without repacking). `width`/`height` powers of two, &le; `MAX_TEXTURE_SIZE`. Level `n` must be half of level `n-1` in each dimension (min 1). The image is copied at consume time per the [lifetime rule](#c3d-ref-lifetime). |
| `0x0504` | `TEX_SUBIMAGE` | `id, level, x, y, width, height, format, row_bytes, ref` | Replaces a rectangle of an existing level. `format` must match the level's. |
| `0x0505` | `TEX_PARAM` | `id, pname, value` | `TEXTURE_MIN_FILTER`, `TEXTURE_MAG_FILTER` (`NEAREST`, `LINEAR`, and the four mipmap modes for min), `TEXTURE_WRAP_S`, `TEXTURE_WRAP_T` (`REPEAT`, `CLAMP`, `CLAMP_TO_EDGE`). The two wrap axes are **independent and both mandatory**: a device must honour `REPEAT` on one axis and `CLAMP` on the other. There is no asymmetric-wrap capability to probe. |
| `0x0506` | `TEX_ENV` | `unit, pname, value` | `TEXTURE_ENV_MODE`: `MODULATE`, `REPLACE`, `DECAL`, `BLEND`, `ADD`. Also the device's own pname `TEXCOORD_SPACE` (`0x0001_0000`): value `0` `NORMALISED` (default; `s, t` in `0..1` span the texture, per GL) or `1` `TEXEL` (`s, t` in texels of the bound texture's level 0, as post-transform clients such as Warp3D supply). Per unit; applies to every texcoord source (inline, array, current). Neither costs the guest anything per vertex -- the scale is applied by the device. |
| `0x0507` | `TEX_ENV_COLOR` | `unit, r f32, g f32, b f32, a f32` | |
| `0x0508` | `ACTIVE_UNIT` | `unit` | Selects the unit for `TEXTURE_2D` enable, `CURRENT_TEXCOORD` default, and the texture matrix stack. `> 0` needs `CAP_MULTITEXTURE`. |
| `0x0509` | `TEX_PALETTE` | `id, entries, ref` | Binds a palette of `entries` (`<= 256`) `RGBA8` colours at `ref` to texture `id`; with a palette bound, an `I8` image is interpreted as **indices** into it, sampled after lookup. `entries` of `0` unbinds. **Optional**: a device reports it with `TEXFMT_SUPPORTED` bit 9 (`I8_INDEXED`); without it `TEX_PALETTE` is `E_BAD_OPCODE`. The first client applies its colour table at upload (`GL_EXT_color_table`: indexed input expanded to RGB/RGBA on the guest, nothing indexed ever stored), so it loses nothing on a device without the bit; the opcode exists so a device may take that per-texel expansion off the guest CPU at texture-load time. |
| `0x050A` | `TEX_COPY_IMAGE` | `id, level, format, x, y, width, height` | Defines mip `level` of `id` from the draw surface's rectangle (`glCopyTexImage2D`), converted to the texture `format`. Baseline; a render-to-texture path with no guest copy. |
| `0x050B` | `TEX_COPY_SUBIMAGE` | `id, level, xoff, yoff, x, y, width, height` | Replaces a rectangle of an existing level from the draw surface (`glCopyTexSubImage2D`). Baseline. |

`CLAMP` is implemented as `CLAMP_TO_EDGE` (no border colour); this is
the behaviour period software expects and a device must not introduce a
border. Sampling a level that has not been defined yields undefined
colour but is not an error. A texture with an incomplete mipmap chain
and a mipmapping min filter samples level 0 only.

(c3d-texture-formats)=
#### Texture formats

`TEXFMT_SUPPORTED` bit `n` reports format `n`. A conforming device
accepts at least `RGBA8`, `RGB8`, `RGB565`, `RGBA4444`, `RGBA5551`, `L8`,
`LA8` and `A8`. Pixels are packed big-endian in memory order. Sixteen-bit
formats are first-class: a device may store them natively and is not
required to expand to 8 bits per channel.

| Value | Name | Bytes/pixel | Layout |
|---|---|---|---|
| `0` | `RGBA8` | 4 | `R G B A` |
| `1` | `RGB8` | 3 | `R G B` |
| `2` | `RGB565` | 2 | `RRRRRGGG GGGBBBBB` |
| `3` | `RGBA4444` | 2 | `RRRRGGGG BBBBAAAA` |
| `4` | `RGBA5551` | 2 | `RRRRRGGG GGBBBBBA` |
| `5` | `L8` | 1 | luminance (`R=G=B=L`, `A=1`) |
| `6` | `LA8` | 2 | `L A` |
| `7` | `A8` | 1 | alpha (`R=G=B=0`) |
| `8` | `I8` | 1 | intensity (`R=G=B=A=I`); **indexed** into the bound palette when one is bound with `TEX_PALETTE` |
| `9` | `I8_INDEXED` | -- | Not an image format: `TEXFMT_SUPPORTED` bit 9 reports that `TEX_PALETTE` is implemented |
| `10`-`31` | -- | Reserved |

(c3d-cmd-current)=
### Current vertex state (`0x06xx`)

Used for any vertex component a draw command's format omits.

| Opcode | Name | Payload |
|---|---|---|
| `0x0600` | `CURRENT_COLOR` | `r, g, b, a` (`f32`) |
| `0x0601` | `CURRENT_NORMAL` | `x, y, z` (`f32`) -- transform tier |
| `0x0602` | `CURRENT_TEXCOORD` | `unit, s, t` (`f32`) |
| `0x0603` | `CURRENT_FOGCOORD` | `f` (`f32`) |

Initial: colour `1,1,1,1`; normal `0,0,1`; texcoords `0,0`; fog `0`.

(c3d-cmd-draw)=
### Draw (`0x07xx`)

#### Vertex format

A draw command carries a `format` word saying which components each
vertex has, in this fixed order:

| Bit | Component | Words | Contents |
|---|---|---|---|
| -- | `POS` | 4, 3 or 2 | always present: `x, y, z, w` (`f32`); `w` is `rhw` in window space (see below). The word count is set by `POS_COUNT`; omitted components are implied `z = 0`, `w`/`rhw = 1.0`. |
| 0 | `COLOR` | 4 | `r, g, b, a` (`f32`, `0..1`) |
| 1 | `NORMAL` | 3 | `nx, ny, nz` (`f32`) -- transform tier. **Illegal in a window-space draw** (`E_BAD_ARG`) on any device: such a vertex has already been transformed *and* lit, so a normal cannot mean anything there. |
| 2 | `TEXCOORD0` | 2 | `s, t` (`f32`) |
| 3 | `TEXCOORD1` | 2 | unit 1 (`CAP_MULTITEXTURE`) |
| 4 | `TEXCOORD2` | 2 | unit 2 |
| 5 | `TEXCOORD3` | 2 | unit 3 |
| 6 | `FOGCOORD` | 1 | `f` (`f32`) |
| 7 | `COLOR_PACKED` | 1 | `COLOR` as one `0xRRGGBBAA` word (`glColor4ub`'s bytes, unnormalised; the device scales to `0..1`). Mutually exclusive with bit 0. |
| 8-9 | `POS_COUNT` | -- | `0`: `POS` is 4 words; `1`: 3 words (`glVertex3f`, the common case); `2`: 2 words (`glVertex2f`, 2D overlays and HUDs); `3`: reserved. |
| 10 | `TEXCOORD_STQ` | -- | Reserved: 4-component texcoords with an explicit projective `q`. Not in version 1; see the one-field `rhw` rule below. A transform-tier device ignores a client's `q` in version 1 and the guest library submits `s, t` only. |
| 11-31 | -- | -- | Reserved, must be zero |

An omitted component takes the [current](#c3d-cmd-current) value.

**Position semantics** differ by opcode:

- **GL-space** (`DRAW_INLINE`, `DRAW_ARRAYS`, `DRAW_ELEMENTS`; transform
  tier): `(x, y, z, w)` is object-space, transformed by the modelview
  and projection matrices, lit, clipped, and mapped through the
  viewport and depth range as GL 1.1 specifies.
- **Window-space** (`DRAW_INLINE_WIN`, `DRAW_ARRAYS_WIN`,
  `DRAW_ELEMENTS_WIN`; baseline): `x, y` are surface pixels (origin
  top-left, pixel centres at `.5`), `z` is depth in `0..1`, and the
  fourth component is **`rhw` = `1 / w_clip`**, the reciprocal of the
  clip-space `w`, used for perspective-correct interpolation (`1.0` for
  affine). It is the *reciprocal* deliberately: a guest that has
  transformed a vertex already computed `1/w` for its own viewport
  divide and simply stores it, a post-transform API such as Warp3D
  hands over its reciprocal-`w` field unchanged, and the device never
  divides per vertex. (A shim that passes `w` instead of `1/w` produces
  correct geometry with swimming textures -- the field is named `rhw`
  so that cannot happen silently.) Texture coordinates and colours are
  supplied *undivided*; the device interpolates `attr * rhw` and
  recovers `attr` per fragment. **One `rhw` governs every attribute**:
  there is no separate projective `q` per texcoord in version 1, which
  matches what period software assumes (a client that overloads the
  slot with `q` gets the picture it got on period hardware); an
  explicit `q` is reserved as the `TEXCOORD_STQ` format bit. Fog
  distance is `FOGCOORD` if present, else derived from `rhw` (a device
  may compute `1/rhw` or index a table on `rhw`; the conformance
  tolerance covers the difference). `z` is single precision: a front
  end carrying double-precision depth (Warp3D's `W3D_Double` z) narrows
  it at the boundary. No clipping is performed other than the scissor
  and the surface bounds; a vertex outside the surface is legal and the
  primitive is clipped by the rasteriser.
- **Texture coordinates** are normalised (`0..1` spans the texture) by
  default in both tiers, or texel-space per unit under
  `TEX_ENV(TEXCOORD_SPACE, TEXEL)`. A device's interpolator must handle
  texel-space coordinates up to `MAX_TEXTURE_SIZE` with `REPEAT` wrap
  without loss; this pins the fixed-point range a hardware
  implementation needs.

#### Primitive types

`prim` is GL's: `POINTS` (`0`), `LINES` (`1`), `LINE_LOOP` (`2`),
`LINE_STRIP` (`3`), `TRIANGLES` (`4`), `TRIANGLE_STRIP` (`5`),
`TRIANGLE_FAN` (`6`), `QUADS` (`7`), `QUAD_STRIP` (`8`), `POLYGON`
(`9`). The device converts quads and polygons to triangles as follows,
and this order is normative so that flat shading and culling match:

- `QUADS` `(v0 v1 v2 v3)` &rarr; `(v0 v1 v2)`, `(v0 v2 v3)`; the flat
  colour of both is `v3`'s.
- `QUAD_STRIP` `(v0 v1 v2 v3 ...)` &rarr; `(v0 v1 v3)`, `(v0 v3 v2)` per
  quad; flat colour from the quad's last vertex (`v3`).
- `POLYGON` `(v0 .. vn)` &rarr; fan from `v0`; flat colour from `vn`.
- Strips and fans take GL's provoking vertex (the last of each
  generated triangle); `TRIANGLES` the third of each.

A count that does not complete the last primitive drops the incomplete
one, per GL.

#### Commands

| Opcode | Name | Payload | Tier |
|---|---|---|---|
| `0x0700` | `DRAW_INLINE` | `prim, format, count, vertices...` -- `count` vertices interleaved per `format` | `CAP_TRANSFORM` |
| `0x0701` | `DRAW_INLINE_WIN` | same layout, window-space | baseline |
| `0x0702` | `DRAW_ARRAYS` | `prim, format, count, arrays...` -- for each set `format` bit, in bit order, one **array descriptor**: `type, stride_bytes, ref`. The position array is first, always. | `CAP_TRANSFORM` |
| `0x0703` | `DRAW_ARRAYS_WIN` | same, window-space | baseline |
| `0x0704` | `DRAW_ELEMENTS` | `prim, format, count, index_type, min_index, max_index, index_ref, arrays...` -- `count` indices at `index_ref` of `index_type` (`UNSIGNED_BYTE`, `UNSIGNED_SHORT`, `UNSIGNED_INT`); the array descriptors cover `min_index..=max_index`. If `min_index`/`max_index` are both `0xFFFF_FFFF` the device scans the indices itself (`CAP_GUESTMEM` devices must support this). | `CAP_TRANSFORM` |
| `0x0705` | `DRAW_ELEMENTS_WIN` | same layout, window-space | baseline |

`DRAW_ELEMENTS_WIN` is in the **baseline** tier on purpose: for a
rasteriser it is only an addressing mode -- an indirection on vertex
fetch -- and post-transform clients use it constantly, not for general
indexed geometry but to name a run's vertices non-contiguously (a
clipped triangle fan whose pivot is a shared vertex elsewhere in the
buffer). Without it such a client must copy the pivot to make the run
contiguous, per fan, on the guest CPU.

Array descriptors reference memory; they never copy it. The same arrays
may be referenced by **any number of draws** within the
[lifetime rule](#c3d-ref-lifetime), so a client's compiled-vertex-array
lock (`glLockArraysEXT`) maps to "submit once, draw many": under
`CAP_GUESTMEM` nothing moves at all, and under the baseline the guest
copies the locked arrays into the aperture once per lock rather than
once per draw.

Array `type` is a GL enumerant: `FLOAT`, `SHORT`, `UNSIGNED_BYTE`,
`INT`. Fixed-point types are converted per GL's array rules
(`UNSIGNED_BYTE` colours normalised to `0..1`; `SHORT` positions and
texcoords as integers). Component counts are those of the vertex-format
table (positions may be `2`, `3` or `4` components -- the descriptor's
`type` word carries the count in bits `31:28`, `0` meaning the table's
default). A `stride_bytes` of `0` means tightly packed.

`DRAW_INLINE*` is the batching target for immediate mode: a
`glBegin`..`glEnd` becomes one command with the vertices written straight
into ring space. A `count` of `0` is legal and draws nothing.

(c3d-cmd-queries)=
### Queries (`0x08xx`)

Results are written to a ref (the destination) and are guaranteed
present when the next `FENCE` completes.

| Opcode | Name | Payload | Result |
|---|---|---|---|
| `0x0800` | `QUERY` | `what, dest_ref` | `what`: a GL enumerant from the set below; the result is the values `glGet*` would return, as `f32`s (matrices in column-major order) |
| `0x0801` | `READ_PIXELS` | `x, y, w, h, format, row_bytes, flags, dest_ref` | The draw surface's rectangle in the given [surface format](#c3d-surface-formats) (`DEPTH` allowed), `h` rows of `row_bytes`. `flags` bit 0 `ROWS_BOTTOM_UP`: the first row written is the rectangle's *bottom* row, so a client whose image origin is bottom-left (`glReadPixels`) gets its layout without a guest-side row reversal. `row_bytes` also carries the client's pack alignment. |

Matrix stacks being device-authoritative, `QUERY` is how a client reads
them back; the first client fetches only the modelview and projection
matrices and does so rarely, which is what makes device-side stacks the
right trade (each matrix operation costs the guest a few ring words
instead of a 4&times;4 multiply).

`QUERY` accepts: `MODELVIEW_MATRIX`, `PROJECTION_MATRIX`,
`TEXTURE_MATRIX` (transform tier), `VIEWPORT`, `SCISSOR_BOX`,
`CURRENT_COLOR`, `CURRENT_TEXTURE_COORDS`, `DEPTH_RANGE`. Everything
else a client can `glGet` is guest-library state and never reaches the
device.

(c3d-errors)=
## Errors

Two classes, kept apart because they have different consumers.

### Protocol errors

Raised by the decoder for malformed or impossible input. The context
latches the first one in `ERROR_CODE`/`ERROR_OFFSET`, sets its
`IRQ_STATUS` error bit, and then either **skips** the command or
**halts** (`CTX_STATUS.HALTED`, decoding stops at the offending command
until `ERROR_ACK`). Subsequent errors before the ACK are counted
nowhere and lost; the guest is expected to ACK promptly.

| Code | Name | Action | Cause |
|---|---|---|---|
| `1` | `E_BAD_LENGTH` | halt | `length` `0`, or the command extends past `RING_TAIL` |
| `2` | `E_RING_OVERRUN` | halt | `RING_TAIL` passed `RING_HEAD` |
| `3` | `E_BAD_OPCODE` | skip | Unknown opcode, or an opcode of a tier the device lacks |
| `4` | `E_BAD_ARG` | skip | A payload word outside its documented range, a `length` inconsistent with `count`/`format`, or a non-zero reserved field |
| `5` | `E_BAD_REF` | skip | A ref outside the aperture, outside reachable guest memory, misaligned, or in space `1` without `CAP_GUESTMEM` |
| `6` | `E_BAD_ID` | skip | A texture or surface ID of `0` or above its limit |
| `7` | `E_NO_SURFACE` | skip | A draw, clear, readback, upload, `READ_PIXELS`, or copy-to-texture (`TEX_COPY_IMAGE`/`TEX_COPY_SUBIMAGE`) with no draw surface |
| `8` | `E_BAD_RECT` | skip | A rectangle outside its surface. Applies to `SURFACE_UPLOAD`, `SURFACE_READBACK`, `READ_PIXELS` and the copy-to-texture commands' source rectangle; **not** to `SCISSOR` or `VIEWPORT`, which GL clamps rather than rejecting |
| `9` | `E_UNSUPPORTED_FORMAT` | skip | A surface or texture format the device does not report |
| `10` | `E_LIMIT` | skip | Beyond a reported limit (texture size, unit, light, plane) |

A halted context keeps answering registers; only its ring stops.

(c3d-gl-errors)=
### GL errors

GL-semantic errors (`GL_INVALID_ENUM`, `GL_INVALID_VALUE`,
`GL_INVALID_OPERATION`, `GL_STACK_OVERFLOW`, `GL_STACK_UNDERFLOW`,
`GL_OUT_OF_MEMORY`) are tracked per context in `GL_ERROR` with GL's
first-error-wins semantics, and the command is otherwise ignored. The
device is **required** to detect: bad enumerants in any command that
takes one; matrix stack overflow/underflow; `TEXTURE_2D` sampling of an
unbound unit is *not* an error. The guest library may validate cheap
cases itself to avoid a round trip, and answers `glGetError` from its
own pending error first, then `GL_ERROR` + `GL_ERROR_ACK`.

(c3d-determinism)=
## Determinism and timing

**Guest-visible completion is a pure function of the command stream and
never of host speed.** Two runs of the same stream on the same
implementation observe `FENCE_COMPLETED`, `RING_HEAD`, `CTX_STATUS` and
interrupts at the same guest-visible moments. An emulator achieves this
by assigning each command a cost in emulated time and making a fence
visible at `submit_time + accumulated_cost`; if the host has not
actually finished the work when the guest is about to observe the fence
(a register read, an interrupt delivery), **the emulator stalls the
emulated machine** until it has. The guest never sees a late fence; a
slow host slows emulation, it does not change behaviour.

The cost table is **informative, not normative**: implementations may
use any, including a fixed small latency per submit (Copperline's
default, `fast`) or per-vertex/per-pixel costs plausible for late-1990s
hardware (`period`). A guest library must never assume one -- it polls
or takes the interrupt. On real hardware, time is real time.

Pixel output is **not** bit-identical across implementations. The
[conformance suite](#c3d-conformance) compares with a stated tolerance.

**Floating-point results are guaranteed reproducible per
implementation and per platform, not across platforms.** The device
computes in IEEE-754 single or double precision, and a transcendental
an implementation needs -- `ROTATE`'s sine and cosine, a fog table --
comes from whatever maths library that build uses. Those differ in the
last unit in the last place between platform maths libraries, so a
value a guest can read back with `QUERY` (a matrix a chain of `ROTATE`s
built) may differ in its lowest bits between two hosts running the
same implementation. Within one build on one platform the result is
reproducible run to run, which is what a save state, a scripted
capture and a regression comparison rely on; a conformance golden is
compared with the suite's tolerance, which covers it. An
implementation that wants cross-platform bit-exactness must supply its
own transcendentals rather than call the platform's, and this
specification does not require it. (Copperline states the same
guarantee for its MPEG audio decoder, and for the same reason; see
[](mhi.md)'s Copperline implementation notes.)

(c3d-api-device-split)=
## The API/device split

This protocol is deliberately innocent of any client API's own
vocabulary. For its first client, a `minigl.library`:

| Concern | Lives in |
|---|---|
| Context objects, screens, windows, page flipping, bitmap locking, lock modes, `glGetString` identity strings, extension strings | Guest library only |
| Texture and surface **ID allocation**, client vertex-array pointers and enables, the current `glBegin` batch, cheap error validation, cached limits | Guest library (state kept guest-side only where it avoids a round trip) |
| Every GL state variable the device consumes; matrix stacks; lighting; texture objects and their images; the draw surface | Device (this protocol) |
| Transform and lighting on a device without `CAP_TRANSFORM` | Guest library, submitting window-space vertices |
| `glGet*`/`glIsEnabled` answers other than the matrices (enables, blend factors, shade model, polygon mode, unpack state) | Guest library, from its own shadow of the state it streamed |
| `GL_MAX_TEXTURE_SIZE`, `GL_MAX_TEXTURE_UNITS` | Guest library, read once from the `MAX_*` registers -- never hard-coded |
| `GL_RENDERER` | Guest library: names this device and its `VERSION`; `GL_VENDOR`/`GL_VERSION` are whatever the client API expects |
| `GL_EXTENSIONS` | Guest library, derived from `CAPS0` (multitexture from `CAP_MULTITEXTURE`; compiled vertex arrays always) |
| Colour tables applied at upload (`GL_EXT_color_table`), `glPixelStore` skip/row-length | Guest library: expands indices before `TEX_IMAGE`, or points `ref`/`row_bytes` past the skipped pixels and rows with no copy |
| `gluPerspective`/`gluLookAt`-style helpers, rotate variants with precomputed trigonometry | Guest library: emitted as `FRUSTUM`/`MULT_MATRIX` |
| Any client-API version, ABI or backend-identity query | Guest library only; the device has `VERSION` and `CAPS0` and nothing else |

Keeping the client API's constants out of the wire format is what lets
this specification describe a device that any fixed-function 3D client
can drive, and is what makes the porting section below possible without
porting client glue.

(c3d-versioning)=
## Versioning

`VERSION` (`0x004`) is `major << 16 | minor`; this draft is `0.10` and the
first released protocol will be `1.0`. A **major** bump is incompatible:
a guest library refuses a major it does not know. A **minor** bump is
additive: new opcodes, new capability bits, new limits registers at
reserved offsets, new enum values. New opcodes are gated by a
capability bit or by a minimum minor version, and a guest library
degrades gracefully across minors. Register offsets, widths, access
rules, bit meanings and documented semantics never change within a
major.

### Draft history

- **0.10** -- an identity registry added, so an implementer may use
  their own registered manufacturer ID rather than the shared one. Guest
  software walks the registry instead of hard-coding a single pair, which
  is how a driver in this ecosystem normally supports a family of boards.
  The trade-off between the two routes -- found with no guest change,
  versus an orthodox manufacturer ID -- is left to the implementer.
- **0.9** -- the autoconfig identity stated as belonging to the board
  rather than to Copperline, with an explicit grant of its use to any
  conforming implementation: guest binaries find the board by
  manufacturer and product, so a second implementation presenting its own
  identity would not run them. The serial number becomes the
  implementation identifier, and the Zorro II profile takes its own
  product number.
- **0.8** -- the trace container finalised against its first
  implementation and its runner, replacing the sketch: exact section
  layouts, the big-endian `u32` rule, prefix-then-payload shape,
  skip-unknown-tags, and an address *space* on `BLOB`, which the sketch
  omitted entirely and without which a runner must guess which memory a
  captured blob belongs in.
- **0.7** -- gaps the dispatch layer exposed: `E_NO_SURFACE` and
  `E_BAD_RECT` extended to `READ_PIXELS` and the copy-to-texture
  commands, which read the draw surface but were never listed (they were
  added in 0.3 without updating the error table); the effect of both
  resets on `GL_ERROR` and the protocol-error latch stated.
- **0.6** -- validation rules the first implementation showed were stated
  loosely or not at all: `NORMAL` is `E_BAD_ARG` in a window-space draw;
  `stride_bytes` below the row's byte length is `E_BAD_ARG`; an
  aperture-backed surface whose extent leaves the aperture is `E_BAD_REF`.
- **0.5** -- after the first implementation of the ring decoder and state
  machine: floating-point reproducibility stated as per-platform, not
  cross-platform; `BLEND_EQUATION`, `BLEND_FUNC_SEPARATE`, polygon mode,
  line width and point size given initial values (the initial-state
  paragraph predated them); `TEX_CREATE`'s effect on existing unit
  bindings stated.
- **0.4** -- after the first client's lock-mode, swap, multitexture and
  pixel-read semantics were checked: readback contract tightened (logically
  complete at the fence; later guest writes take precedence; deferral must
  intercept writes as well as reads, else materialise eagerly); "client
  locks and 2D composition" mapping added; `READ_PIXELS` gains a `flags`
  word with `ROWS_BOTTOM_UP`.
- **0.3** -- after the first client's exported function table was
  checked: `TEX_PALETTE` demoted to optional (the client expands colour
  tables at upload and never stores indexed textures); `POLYGON_MODE`,
  `BLEND_EQUATION`, `BLEND_FUNC_SEPARATE`, `TEX_COPY_IMAGE`,
  `TEX_COPY_SUBIMAGE` added to the baseline; `TEXGEN`/`TEXGEN_PLANE` and
  `TEXTURE_GEN_S/T` added to the transform tier; `POS3` generalised to a
  `POS_COUNT` field (4/3/2 words); device-authoritative matrix stacks
  confirmed as the decision; lighting confirmed absent from the first
  client; API/device split table extended.
- **0.2** -- after the first check of the draft against a post-transform
  client's conventions: window-space fourth component is `rhw` (the
  reciprocal), one `rhw` governs all attributes with `TEXCOORD_STQ`
  reserved; `DRAW_ELEMENTS_WIN` added to the baseline tier; per-unit
  `TEXCOORD_SPACE` (normalised/texel); wrap axes stated independent and
  mandatory; `COLOR_PACKED` and `POS3` vertex-format bits added;
  `TEX_PALETTE` specified and required in 1.0 pending confirmation;
  `z` narrowing stated.
- **0.1** -- first draft, written before any contact with a client
  implementation.

(c3d-conformance)=
## Conformance

A conformance suite ships with this specification: command-stream
**traces** in the capture format, golden images, and a per-pixel and
per-image tolerance. A native runner (trace in, images out, compare)
lets an implementer test a backend with no Amiga, no guest library and
no ROM.

**Trace container** (version 1). A file is a sequence of sections back to
back, with no padding, no alignment and no trailer; a reader stops
cleanly when the bytes run out. Each section is:

```text
tag[4] length[4] payload[length]
```

`length` is the payload's byte count as a big-endian `u32`, capping one
section at 4 GiB. Every fixed field below is likewise a big-endian
`u32`, matching the wire format's own convention rather than introducing
a second. Each section is a fixed-size prefix followed by one run of
variable-length bytes, so a reader can locate every field at a fixed
offset and slice the remainder without scanning.

| Tag | Fixed prefix | Trailing bytes | Repeats |
|---|---|---|---|
| `C3DT` | `container_version, protocol_version, capability_mask` | none | **first section, exactly once** |
| `APER` | `offset` | aperture image | any |
| `RING` | `context, ring_tail` | ring bytes | any |
| `BLOB` | `space, address` | referenced bytes | any |
| `GOLD` | `context, fence_id, surface_id` | PNG | any |

- `C3DT`'s `capability_mask` is the `CAPS0` bits the trace exercises; a
  runner refuses to replay it against a device lacking any of them.
- `RING` is one submission: a runner writes the bytes at the ring's
  current tail and then sets `RING_TAIL`, in that order, reproducing the
  guest's two-step sequence. Replayed in file order.
- `BLOB` is the bytes a reference named at capture time, materialised
  before the `RING` section that uses them. `space` is `0` for the data
  aperture and `1` for guest memory, matching a
  [reference](#c3d-refs)'s own space field -- a trace captured from a
  `CAP_GUESTMEM` device carries both, and replaying one into the wrong
  space corrupts the replay silently.
- `GOLD` is the expected contents of `surface_id` once `context`'s
  `FENCE_COMPLETED` reaches `fence_id`. Compared with the suite's
  tolerance, never exactly.
- **An unrecognised tag is skipped, not rejected**, so a later revision
  may add a section kind without breaking an older runner -- the same
  forward-compatibility rule unknown opcodes follow. Ordering beyond
  "`C3DT` first" is a runner's concern, not the container's.

Coverage targets for 1.0: each primitive type, flat and smooth, each
texenv mode, the common blend factor pairs, alpha funcs, depth funcs,
range and polygon offset, fog modes, lighting (directional, positional,
spot, colour-material, two-sided), clip planes, scissor, colour mask,
each texture format, filter and wrap mode, mipmaps and subimage, each
surface format, upload/readback rectangles, every protocol error path,
and every capability masked off in turn -- the baseline-only suite is
what a rasteriser-tier implementation runs.

(c3d-porting)=
## Porting to another implementation

Everything above is expressed in terms of the autoconfigured window's
offsets, the Amiga address space, and GL 1.1 semantics; nothing
references Copperline's internal types or its savestate format. An
emulator or hardware design wanting to run the same guest library needs
to:

1. Autoconfig a board at manufacturer `0x1448`, product `9` (Zorro III)
   or `10` (Zorro II), with no autoboot ROM, and report its layout
   through `APERTURE_OFFSET`/`APERTURE_SIZE`/`MAX_CONTEXTS`. Use that
   identity if you want existing guest software to find your board with
   no changes -- the grant in [Zorro identity](#c3d-zorro-identity)
   exists so that you can, and your own name goes in the serial number.
   If you would rather use your own registered manufacturer ID, add a row
   to the [identity registry](#c3d-identity-registry) instead, and expect
   to wait for a guest-library release before your board is found.
2. Implement the global and context register files, honouring the
   [access rules](#c3d-access-size) for its bus width.
3. Implement the ring decoder and the baseline command set over its own
   rasteriser; set `CAP_TRANSFORM` only if it implements GL 1.1
   fixed-function transform and lighting.
4. Choose its capabilities honestly. In particular, set `CAP_GUESTMEM`
   and `CAP_SURFACE_GUESTADDR` only if it can read and write guest
   memory from the device -- by whatever mechanism it has (a DMA
   engine, a direct memory-array access; unconstrained here) -- and
   `CAP_REF_SYNC` only if it can capture referenced data inside the
   `RING_TAIL` write.
5. Make fences guest-visible on a schedule that is a function of the
   command stream, per [Determinism and timing](#c3d-determinism).
6. Pass the baseline conformance traces, and the traces for each
   capability it claims.

The rasteriser tier is deliberately the size of a late-1990s 3D chip:
triangle setup, perspective-correct interpolation, one or more texture
units with the GL 1.1 texenv modes, depth, blend, alpha test and fog.

## Copperline implementation notes

This section describes the Copperline host board (`src/c3d/`, `[c3d]`,
cargo feature `c3d`, **off by default**); nothing here changes the
protocol above. As of draft 0.1 it describes the planned shape; it is
updated as the milestones land.

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
