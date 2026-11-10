// SPDX-License-Identifier: GPL-3.0-or-later

//! Cross-board bus-master DMA, exercised through two C3D boards. The
//! mechanism under test is generic -- `ZorroDevice::take_pending_dma_writes`
//! and `Bus::drain_cross_board_dma` -- but the only board in-tree that
//! produces such writes is C3D, so the test needs the `c3d` feature and
//! lives here rather than in [`super::tests`], which every other bus test
//! shares.
use super::tests::empty_bus;
use super::*;

#[test]
fn c3d_surface_guestaddr_readback_lands_bytes_in_a_second_boards_window() {
    // Two C3D boards on the same chain: board A (slot 0) does a
    // CAP_SURFACE_GUESTADDR readback whose surface address points into
    // board B's (slot 1) own data aperture, exercising the whole
    // cross-board DMA path -- ZorroDevice::take_pending_dma_writes,
    // Bus::drain_cross_board_dma, and the two-stage deferred-fence
    // bookkeeping -- with no RTG board needed: any device-backed board's
    // window works as the target, since the mechanism under test is
    // generic (see PendingDmaWrite's own doc comment).
    use crate::c3d::board::{C3dBoard, CONTEXT_PAGE_BASE};
    use crate::zorro::BoardSpec;
    use crate::zorro_device::{BoardDevice, DeviceHost, ZorroDevice};

    const BASE_A: u32 = 0x1000_0000;
    const BASE_B: u32 = 0x2000_0000;
    const WINDOW: usize = 0x0020_0000; // 2 MiB: smallest legal Zorro III
                                       // size that still leaves room past
                                       // the 1 MiB register/context area.
    const APERTURE_OFFSET: u32 = 0x0010_0000; // proto::APERTURE_OFFSET_DEFAULT

    let mut bus = empty_bus();
    bus.mem
        .zorro
        .add_board_configured_at(BoardSpec::c3d(0, WINDOW), BASE_A)
        .unwrap();
    bus.devices
        .push(BoardDevice::C3d(Box::new(C3dBoard::new(WINDOW as u32))));
    bus.mem
        .zorro
        .add_board_configured_at(BoardSpec::c3d(1, WINDOW), BASE_B)
        .unwrap();
    bus.devices
        .push(BoardDevice::C3d(Box::new(C3dBoard::new(WINDOW as u32))));

    // The guest address of a 4x4 A8R8G8B8 surface living at
    // aperture-relative offset 0x2000 inside board B.
    const SURFACE_APERTURE_REL_B: u32 = 0x2000;
    let surface_guest_addr = BASE_B + APERTURE_OFFSET + SURFACE_APERTURE_REL_B;

    // Build the command stream in board A's own aperture (offset 0 in
    // A's aperture, i.e. window address BASE_A + APERTURE_OFFSET).
    let opcode_len = |op: u16, words: u32| ((op as u32) << 16) | words;
    let mut cmds = Vec::<u32>::new();
    cmds.push(opcode_len(crate::c3d::proto::OP_SURFACE_DEFINE, 8));
    cmds.extend([
        1,                  // id
        4,                  // width
        4,                  // height
        16,                 // stride_bytes (4 px * 4 bytes)
        5,                  // format: A8R8G8B8
        1,                  // flags: SURFACE_DEFINE_FLAG_GUEST_ADDR
        surface_guest_addr, // address
    ]);
    cmds.push(opcode_len(crate::c3d::proto::OP_SET_DRAW_SURFACE, 2));
    cmds.push(1);
    cmds.push(opcode_len(crate::c3d::proto::OP_CLEAR_COLOR, 5));
    cmds.extend([0, 1.0f32.to_bits(), 0, 1.0f32.to_bits()]); // g=1 (green)
    cmds.push(opcode_len(crate::c3d::proto::OP_CLEAR, 2));
    cmds.push(crate::c3d::proto::CLEAR_MASK_COLOR);
    cmds.push(opcode_len(crate::c3d::proto::OP_SURFACE_READBACK, 5));
    cmds.extend([0, 0, 4, 4]);
    cmds.push(opcode_len(crate::c3d::proto::OP_FENCE, 2));
    cmds.push(1);
    let bytes: Vec<u8> = cmds.iter().flat_map(|w| w.to_be_bytes()).collect();

    let write_a = |bus: &mut Bus, off: u32, size: usize, value: u32| {
        let mut host = DeviceHost::for_slot(&mut bus.mem, 0);
        ZorroDevice::write(&mut bus.devices[0], off, size, value, &mut host);
    };
    let read_a = |bus: &mut Bus, off: u32, size: usize| -> u32 {
        let mut host = DeviceHost::for_slot(&mut bus.mem, 0);
        ZorroDevice::read(&mut bus.devices[0], off, size, &mut host)
    };

    for (i, b) in bytes.iter().enumerate() {
        write_a(&mut bus, APERTURE_OFFSET + i as u32, 1, *b as u32);
    }
    write_a(
        &mut bus,
        CONTEXT_PAGE_BASE + 0x08, /* RING_BASE */
        4,
        0,
    );
    write_a(
        &mut bus,
        CONTEXT_PAGE_BASE + 0x0C, /* RING_SIZE */
        4,
        0x1000,
    );
    write_a(
        &mut bus,
        CONTEXT_PAGE_BASE, /* CTX_CONTROL */
        4,
        0b11, /* ALLOC | ENABLE */
    );
    // The doorbell.
    write_a(
        &mut bus,
        CONTEXT_PAGE_BASE + 0x10, /* RING_TAIL */
        4,
        bytes.len() as u32,
    );

    assert_eq!(
        read_a(&mut bus, CONTEXT_PAGE_BASE + 0x1C /* ERROR_CODE */, 4),
        0,
        "no protocol error decoding a well-formed stream"
    );

    let read_b_pixel = |bus: &mut Bus| -> u32 {
        let mut host = DeviceHost::for_slot(&mut bus.mem, 1);
        ZorroDevice::read(
            &mut bus.devices[1],
            APERTURE_OFFSET + SURFACE_APERTURE_REL_B,
            4,
            &mut host,
        )
    };

    // Right after the doorbell: nothing has reached board B yet (the bus
    // has not drained anything), and the fence has not completed.
    assert_eq!(
        read_b_pixel(&mut bus),
        0,
        "board B's window must be untouched before any drain"
    );
    assert_eq!(
        read_a(
            &mut bus,
            CONTEXT_PAGE_BASE + 0x18, /* FENCE_COMPLETED */
            4
        ),
        0,
        "the fence cannot be complete before the cross-board write has landed"
    );

    // First drain: the bytes land in board B, but per the two-stage
    // design the fence only becomes "ready", not complete yet.
    bus.advance_devices(1);
    assert_eq!(
        read_b_pixel(&mut bus),
        0xFF00_FF00,
        "opaque green (A R G B) must have landed in board B's window after one drain"
    );
    assert_eq!(
        read_a(&mut bus, CONTEXT_PAGE_BASE + 0x18, 4),
        0,
        "the fence must not complete on the same drain that applied the write"
    );

    // Second drain, with nothing new queued: now the fence completes.
    bus.advance_devices(1);
    assert_eq!(
        read_a(&mut bus, CONTEXT_PAGE_BASE + 0x18, 4),
        1,
        "the fence completes on the drain after the one that applied the write"
    );
}
