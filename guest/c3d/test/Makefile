# Builds c3dtest, the M2 protocol test program for the C3D board (see
# c3dtest.c's own header comment), with the dockerized m68k-amigaos
# cross-GCC (see ../../toolchain.mk). Same shape as guest/mhi/test/Makefile:
# a normal relocatable hunk executable (LoadSeg relocates it), -nostartfiles
# with its own start.s as the real entry point.

include ../../toolchain.mk

# -noixemul, not -nostdlib: c3dtest.c avoids ULONG division entirely (its
# hex formatter needs only shifts and masks, see the file's own comment),
# so unlike mhitest.c it does not strictly need libc.a's software divide --
# kept anyway for consistency with every other guest/*/test/Makefile in
# this tree and because it costs nothing measurable at this program's size.
CFLAGS  = -Os -m68000 -fomit-frame-pointer -Wall -Wextra \
          -noixemul -nostartfiles

c3dtest: start.s c3dtest.c
	$(CC) $(CFLAGS) start.s c3dtest.c -o $@

clean:
	rm -f c3dtest

.PHONY: clean
