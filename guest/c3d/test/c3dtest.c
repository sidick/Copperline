/* SPDX-License-Identifier: GPL-3.0-or-later
 *
 * c3dtest: the M2 protocol test program for the C3D board. Speaks the
 * wire protocol directly -- FindConfigDev, register peek/poke, a hand
 * assembled command stream -- and touches nothing MiniGL-shaped, per the
 * device specification's section 11.1 ("protocol test programs") and
 * this project's clean-room rules: this file is written from the
 * specification alone and exists specifically so the board's behaviour
 * can be checked before any guest library exists at all.
 *
 * Probes the board, allocates context 0, defines a 4x4 A8R8G8B8 surface
 * in the data aperture, clears it to opaque red, reads it back, and
 * checks the actual pixel bytes -- the same sequence and the same
 * expected bytes as src/c3d/board.rs's own
 * `a_full_command_stream_is_decoded_and_a_readback_produces_the_cleared_pixels`
 * test, run for real through a 68k CPU and real MMIO instead of Rust
 * calling the board's ZorroDevice methods directly. Every assertion line
 * starts with "C3DTEST:" and is otherwise plain text -- grep-friendly for
 * the integration test, matching guest/mhi/test/mhitest.c's own
 * convention, which this file follows closely (dos.library Write() for
 * output, no stdio, a hand written hex formatter since there is no
 * hardware 32-bit divide on this target and no C runtime linked in to
 * supply a software one -- see mhitest.c's own comment on udiv10 for why
 * that trap matters here too, even though this file only ever needs hex).
 *
 * Because the board's doorbell (RING_TAIL write) executes synchronously
 * end to end in Copperline (src/c3d/board.rs's own module doc comment),
 * every register this test reads immediately after ringing the doorbell
 * is already final by the time the `move.l` that wrote RING_TAIL
 * returns. The bounded poll loop below is still the spec-correct thing
 * to write, since another implementation's FENCE_COMPLETED may not be
 * available quite that promptly, and a test written against "it happens
 * to be synchronous today" would silently stop proving anything portable.
 */

#include <exec/execbase.h>
#include <exec/types.h>
#include <libraries/configvars.h>
#include <libraries/expansion.h>

#include <dos/dos.h>
#include <dos/dosextens.h>

#define __NOLIBBASE__
#define EXEC_BASE_NAME _sysbase
#define DOS_BASE_NAME _dosbase
#define EXPANSION_BASE_NAME _expbase
#include <inline/dos.h>
#include <inline/exec.h>
#include <inline/expansion.h>

/* -- Board identity and register/opcode layout (docs/internals/c3d.md's
 * specification -- see the published spec repository for the source of
 * truth this file was written from). Kept local rather than shared with
 * src/c3d/proto.rs on purpose: a guest program has no access to the host
 * crate, and the specification is the contract both sides implement
 * independently. */

#define C3D_MANUFACTURER 0x1448
#define C3D_PRODUCT_Z3   9

#define REG_ID          0x000
#define REG_VERSION     0x004
#define REG_CAPS0       0x008
#define REG_APERTURE_OFFSET 0x020
#define ID_MAGIC        0x43334420UL /* "C3D " */

#define CTX_PAGE_BASE   0x00010000UL
#define CTX_CONTROL     0x00
#define CTX_RING_BASE   0x08
#define CTX_RING_SIZE   0x0C
#define CTX_RING_TAIL   0x10
#define CTX_FENCE_COMPLETED 0x18
#define CTX_ERROR_CODE  0x1C

#define CTX_CONTROL_ALLOC  (1UL << 0)
#define CTX_CONTROL_ENABLE (1UL << 1)

#define OP_SURFACE_DEFINE   0x0100UL
#define OP_SET_DRAW_SURFACE 0x0102UL
#define OP_CLEAR_COLOR      0x020EUL
#define OP_CLEAR            0x0105UL
#define OP_SURFACE_READBACK 0x0104UL
#define OP_FENCE             0x0001UL

#define CLEAR_MASK_COLOR (1UL << 0)

#define SURFACE_FMT_A8R8G8B8 5UL

#define RING_BASE_APERTURE_REL 0UL
#define RING_SIZE               0x1000UL
#define SURFACE_APERTURE_REL    RING_SIZE
#define SURFACE_W 4UL
#define SURFACE_H 4UL
#define SURFACE_STRIDE (SURFACE_W * 4UL)

#define POLL_LIMIT 1000000UL

static struct ExecBase *sysbase(void)
{
    struct ExecBase *base;
    __asm("move.l 4.w,%0" : "=r"(base));
    return base;
}

static LONG strlen_local(const char *s)
{
    LONG n = 0;
    while (s[n] != '\0') {
        n++;
    }
    return n;
}

static void put(struct Library *_dosbase, const char *msg)
{
    Write(Output(), (APTR)msg, strlen_local(msg));
}

/* Hex only -- no division needed, unlike mhitest.c's decimal formatter
 * (see this file's header comment). `buf` must be at least 9 bytes. */
static char *format_hex32(ULONG value, char *buf)
{
    static const char digits[] = "0123456789ABCDEF";
    int i;
    for (i = 7; i >= 0; i--) {
        buf[i] = digits[value & 0xF];
        value >>= 4;
    }
    buf[8] = '\0';
    return buf;
}

static void put_result(struct Library *_dosbase, const char *check, BOOL ok)
{
    put(_dosbase, "C3DTEST: ");
    put(_dosbase, ok ? "PASS " : "FAIL ");
    put(_dosbase, check);
    put(_dosbase, "\n");
}

static void put_hex_kv(struct Library *_dosbase, const char *name, ULONG value)
{
    char buf[9];
    put(_dosbase, "C3DTEST: INFO ");
    put(_dosbase, name);
    put(_dosbase, "=0x");
    put(_dosbase, format_hex32(value, buf));
    put(_dosbase, "\n");
}

static ULONG rd32(volatile UBYTE *board, ULONG off)
{
    return *(volatile ULONG *)(board + off);
}

static void wr32(volatile UBYTE *board, ULONG off, ULONG value)
{
    *(volatile ULONG *)(board + off) = value;
}

/* Appends one command's header word (opcode<<16 | length_in_words) at
 * `*off`, advancing it by 4. */
static void put_cmd_header(volatile UBYTE *board, ULONG *off, ULONG opcode, ULONG length_words)
{
    wr32(board, *off, (opcode << 16) | length_words);
    *off += 4;
}

static void put_word(volatile UBYTE *board, ULONG *off, ULONG value)
{
    wr32(board, *off, value);
    *off += 4;
}

/* IEEE-754 single precision bits for small non-negative integer and
 * simple fractional constants this test needs (0.0 and 1.0 only) --
 * avoids linking any floating-point support at all. */
#define F32_ZERO 0x00000000UL
#define F32_ONE  0x3F800000UL

LONG entry(char *cmdline __asm("a0"), long cmdlen __asm("d0"))
{
    (void)cmdline;
    (void)cmdlen;

    struct ExecBase *_sysbase = sysbase();
    struct Library *_dosbase = OpenLibrary((STRPTR) "dos.library", 34);
    if (_dosbase == NULL) {
        return 20;
    }

    BOOL all_ok = TRUE;
    struct Library *_expbase = OpenLibrary((STRPTR) "expansion.library", 0);
    if (_expbase == NULL) {
        put_result(_dosbase, "open expansion.library", FALSE);
        put(_dosbase, "C3DTEST: SUMMARY FAIL\n");
        CloseLibrary(_dosbase);
        return 20;
    }

    struct ConfigDev *cd = FindConfigDev(NULL, C3D_MANUFACTURER, C3D_PRODUCT_Z3);
    if (cd == NULL || cd->cd_BoardAddr == NULL) {
        put_result(_dosbase, "find C3D board", FALSE);
        put(_dosbase, "C3DTEST: SUMMARY FAIL\n");
        CloseLibrary(_expbase);
        CloseLibrary(_dosbase);
        return 20;
    }
    put_result(_dosbase, "find C3D board", TRUE);

    volatile UBYTE *board = (volatile UBYTE *)cd->cd_BoardAddr;

    ULONG id = rd32(board, REG_ID);
    put_hex_kv(_dosbase, "id", id);
    BOOL id_ok = (id == ID_MAGIC);
    put_result(_dosbase, "id magic", id_ok);
    all_ok = all_ok && id_ok;

    put_hex_kv(_dosbase, "version", rd32(board, REG_VERSION));
    put_hex_kv(_dosbase, "caps0", rd32(board, REG_CAPS0));

    ULONG aperture_offset = rd32(board, REG_APERTURE_OFFSET);
    put_hex_kv(_dosbase, "aperture_offset", aperture_offset);

    /* Allocate context 0's ring at aperture-relative 0 and build the
     * command stream directly into the aperture (spec: every address in
     * the stream, and RING_BASE itself, is aperture-relative -- 0 is the
     * aperture's own first byte, not the window's). */
    volatile UBYTE *aperture = board + aperture_offset;
    ULONG off = RING_BASE_APERTURE_REL;

    put_cmd_header(aperture, &off, OP_SURFACE_DEFINE, 8);
    put_word(aperture, &off, 1);                    /* id */
    put_word(aperture, &off, SURFACE_W);
    put_word(aperture, &off, SURFACE_H);
    put_word(aperture, &off, SURFACE_STRIDE);
    put_word(aperture, &off, SURFACE_FMT_A8R8G8B8);
    put_word(aperture, &off, 0);                    /* flags: aperture-backed */
    put_word(aperture, &off, SURFACE_APERTURE_REL);  /* address */

    put_cmd_header(aperture, &off, OP_SET_DRAW_SURFACE, 2);
    put_word(aperture, &off, 1);

    put_cmd_header(aperture, &off, OP_CLEAR_COLOR, 5);
    put_word(aperture, &off, F32_ONE);   /* r */
    put_word(aperture, &off, F32_ZERO);  /* g */
    put_word(aperture, &off, F32_ZERO);  /* b */
    put_word(aperture, &off, F32_ONE);   /* a */

    put_cmd_header(aperture, &off, OP_CLEAR, 2);
    put_word(aperture, &off, CLEAR_MASK_COLOR);

    put_cmd_header(aperture, &off, OP_SURFACE_READBACK, 5);
    put_word(aperture, &off, 0); /* x */
    put_word(aperture, &off, 0); /* y */
    put_word(aperture, &off, SURFACE_W);
    put_word(aperture, &off, SURFACE_H);

    put_cmd_header(aperture, &off, OP_FENCE, 2);
    put_word(aperture, &off, 1); /* fence id */

    ULONG stream_bytes = off - RING_BASE_APERTURE_REL;

    ULONG ctx = CTX_PAGE_BASE;
    wr32(board, ctx + CTX_RING_BASE, RING_BASE_APERTURE_REL);
    wr32(board, ctx + CTX_RING_SIZE, RING_SIZE);
    wr32(board, ctx + CTX_CONTROL, CTX_CONTROL_ALLOC | CTX_CONTROL_ENABLE);

    /* The doorbell. */
    wr32(board, ctx + CTX_RING_TAIL, stream_bytes);

    ULONG error_code = rd32(board, ctx + CTX_ERROR_CODE);
    put_hex_kv(_dosbase, "error_code", error_code);
    BOOL no_error = (error_code == 0);
    put_result(_dosbase, "no protocol error", no_error);
    all_ok = all_ok && no_error;

    ULONG waited;
    ULONG fence_completed = 0;
    for (waited = 0; waited < POLL_LIMIT; waited++) {
        fence_completed = rd32(board, ctx + CTX_FENCE_COMPLETED);
        if (fence_completed >= 1) {
            break;
        }
    }
    BOOL fence_ok = (fence_completed >= 1);
    put_result(_dosbase, "fence completed", fence_ok);
    all_ok = all_ok && fence_ok;

    /* A8R8G8B8 is "A R G B" bytes per the specification's surface-format
     * table: opaque red is 0xFF 0xFF 0x00 0x00, every one of the 16
     * pixels of a 4x4 surface with no stride padding. */
    BOOL pixels_ok = TRUE;
    ULONG row, col;
    for (row = 0; row < SURFACE_H; row++) {
        for (col = 0; col < SURFACE_W; col++) {
            ULONG px = SURFACE_APERTURE_REL + row * SURFACE_STRIDE + col * 4;
            UBYTE a = aperture[px + 0];
            UBYTE r = aperture[px + 1];
            UBYTE g = aperture[px + 2];
            UBYTE b = aperture[px + 3];
            if (a != 0xFF || r != 0xFF || g != 0x00 || b != 0x00) {
                pixels_ok = FALSE;
            }
        }
    }
    put_result(_dosbase, "readback pixels are opaque red", pixels_ok);
    all_ok = all_ok && pixels_ok;

    put(_dosbase, all_ok ? "C3DTEST: SUMMARY PASS\n" : "C3DTEST: SUMMARY FAIL\n");

    CloseLibrary(_expbase);
    CloseLibrary(_dosbase);
    return all_ok ? 0 : 20;
}
