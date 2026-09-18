// SPDX-License-Identifier: GPL-3.0-or-later

//! `copperline-c3d-trace`: the C3D conformance-trace runner
//! (`docs/internals/c3d.md`, "Conformance"). Replays a capture-format
//! trace (`src/c3d/trace.rs`) through the pure decode/dispatch layers
//! (`src/c3d/{ring,dispatch,state}.rs`) and the wgpu renderer
//! (`src/c3d/render.rs`), with no Amiga, no guest library and no ROM.
//!
//! ```text
//! copperline-c3d-trace info <trace>
//! copperline-c3d-trace run <trace> [--out <dir>]
//! copperline-c3d-trace compare <trace>
//! copperline-c3d-trace record <trace> --out <trace-with-goldens>
//! copperline-c3d-trace selftest
//! ```
//!
//! `run` replays and writes each fence's draw surface to a PNG; `compare`
//! does the same and checks each rendered frame against the trace's
//! `GOLD` sections within a stated tolerance (pixel output is never
//! bit-identical across implementations, per the spec's "Determinism and
//! timing" section, so this is always tolerance-based); `record` replays
//! and writes the rendered frames back out as `GOLD` sections, which is
//! how the three M1 traces below get their goldens the first time;
//! `info` inspects a trace's header and section inventory without
//! rendering (and so without a GPU adapter) at all; `selftest` builds the
//! three M1 traces in code -- clear, triangle, textured quad -- renders
//! them, and sanity-checks the result, so the milestone is verifiable
//! with no committed binary trace or golden at all.
//!
//! No argument-parsing dependency: the CLI is hand-rolled below.

use copperline::c3d::dispatch::{Context, RenderOp};
use copperline::c3d::proto;
use copperline::c3d::render::{MemLoc, Memory, Renderer};
use copperline::c3d::ring::DeviceConfig;
use copperline::c3d::state::{self, Limits};
use copperline::c3d::trace::{Section, TraceReader, TraceWriter};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------
// CLI entry point
// ---------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let status = match run_cli(&args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("copperline-c3d-trace: {e}");
            1
        }
    };
    std::process::exit(status);
}

fn run_cli(args: &[String]) -> Result<i32, String> {
    let Some(cmd) = args.first() else {
        print_usage();
        return Ok(1);
    };
    match cmd.as_str() {
        "info" => {
            let path = args.get(1).ok_or("info: missing <trace>")?;
            cmd_info(Path::new(path))
        }
        "run" => {
            let path = args.get(1).ok_or("run: missing <trace>")?;
            let out_dir = parse_out_flag(&args[2..]).unwrap_or_else(|| PathBuf::from("."));
            cmd_run(Path::new(path), &out_dir)
        }
        "compare" => {
            let path = args.get(1).ok_or("compare: missing <trace>")?;
            cmd_compare(Path::new(path))
        }
        "record" => {
            let path = args.get(1).ok_or("record: missing <trace>")?;
            let out = parse_out_flag(&args[2..]).ok_or("record: missing --out <path>")?;
            cmd_record(Path::new(path), &out)
        }
        "selftest" => cmd_selftest(),
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(0)
        }
        other => Err(format!("unknown subcommand '{other}' (see --help)")),
    }
}

fn parse_out_flag(rest: &[String]) -> Option<PathBuf> {
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        if a == "--out" {
            return it.next().map(PathBuf::from);
        }
    }
    None
}

fn print_usage() {
    eprintln!(
        "usage:\n\
         \x20 copperline-c3d-trace info <trace>\n\
         \x20 copperline-c3d-trace run <trace> [--out <dir>]\n\
         \x20 copperline-c3d-trace compare <trace>\n\
         \x20 copperline-c3d-trace record <trace> --out <trace-with-goldens>\n\
         \x20 copperline-c3d-trace selftest"
    );
}

// ---------------------------------------------------------------------
// `info`: header and section inventory, no rendering
// ---------------------------------------------------------------------

fn cmd_info(path: &Path) -> Result<i32, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut counts: HashMap<&'static str, (u32, u64)> = HashMap::new();
    let mut header = None;
    for section in TraceReader::new(&bytes) {
        let section = section.map_err(|e| format!("{}: {e}", path.display()))?;
        let (tag, len) = match &section {
            Section::Header(h) => {
                header = Some(*h);
                ("C3DT", 12u64)
            }
            Section::Aperture(a) => ("APER", a.data.len() as u64),
            Section::Ring(r) => ("RING", r.bytes.len() as u64),
            Section::Blob(b) => ("BLOB", b.data.len() as u64),
            Section::Gold(g) => ("GOLD", g.png.len() as u64),
            Section::Unknown { payload, .. } => ("?", payload.len() as u64),
        };
        let e = counts.entry(tag).or_insert((0, 0));
        e.0 += 1;
        e.1 += len;
    }
    let Some(h) = header else {
        return Err("trace has no C3DT header".into());
    };
    println!("trace: {}", path.display());
    println!(
        "  container_version={} protocol_version=0x{:08x} capability_mask={}",
        h.container_version,
        h.protocol_version,
        describe_caps(h.capability_mask)
    );
    for tag in ["APER", "RING", "BLOB", "GOLD"] {
        if let Some((n, bytes)) = counts.get(tag) {
            println!("  {tag}: {n} section(s), {bytes} payload byte(s)");
        }
    }
    Ok(0)
}

fn describe_caps(mask: u32) -> String {
    let bits: &[(u32, &str)] = &[
        (proto::CAP_GUESTMEM, "GUESTMEM"),
        (proto::CAP_IRQ, "IRQ"),
        (proto::CAP_TRANSFORM, "TRANSFORM"),
        (proto::CAP_MULTITEXTURE, "MULTITEXTURE"),
        (proto::CAP_SURFACE_GUESTADDR, "SURFACE_GUESTADDR"),
        (proto::CAP_REF_SYNC, "REF_SYNC"),
    ];
    let names: Vec<&str> = bits
        .iter()
        .filter(|(bit, _)| mask & bit != 0)
        .map(|(_, name)| *name)
        .collect();
    if names.is_empty() {
        format!("0x{mask:08x} (baseline)")
    } else {
        format!("0x{mask:08x} ({})", names.join("|"))
    }
}

// ---------------------------------------------------------------------
// Replay: shared by `run`, `compare`, `record` and `selftest`
// ---------------------------------------------------------------------

/// A rendered frame captured at a `FENCE` during replay.
struct CapturedFrame {
    context: u32,
    fence_id: u32,
    surface_id: u32,
    width: u32,
    height: u32,
    /// Straight RGBA8, top-left origin.
    rgba: Vec<u8>,
}

/// Flat guest-visible memory backing both address spaces a [`Ref`] can
/// name. Sized generously and grown on write rather than fixed up front:
/// a trace runner's traces are captured artefacts of modest size (per
/// `trace.rs`'s own module doc comment on what this container is for),
/// not a bound this module needs to defend adversarially.
struct FlatMemory {
    aperture: Vec<u8>,
    guest: Vec<u8>,
}

impl FlatMemory {
    fn new() -> Self {
        FlatMemory {
            aperture: Vec::new(),
            guest: Vec::new(),
        }
    }

    fn buf_mut(&mut self, is_guest: bool) -> &mut Vec<u8> {
        if is_guest {
            &mut self.guest
        } else {
            &mut self.aperture
        }
    }

    fn buf(&self, is_guest: bool) -> &Vec<u8> {
        if is_guest {
            &self.guest
        } else {
            &self.aperture
        }
    }

    fn put(&mut self, is_guest: bool, address: u32, data: &[u8]) {
        let buf = self.buf_mut(is_guest);
        let end = address as usize + data.len();
        if buf.len() < end {
            buf.resize(end, 0);
        }
        buf[address as usize..end].copy_from_slice(data);
    }
}

impl Memory for FlatMemory {
    fn read(&self, loc: MemLoc, len: usize) -> Option<&[u8]> {
        let (buf, addr) = match loc {
            MemLoc::Aperture(a) => (self.buf(false), a),
            MemLoc::Guest(a) => (self.buf(true), a),
        };
        buf.get(addr as usize..addr as usize + len)
    }

    fn write(&mut self, loc: MemLoc, data: &[u8]) -> bool {
        match loc {
            MemLoc::Aperture(a) => self.put(false, a, data),
            MemLoc::Guest(a) => self.put(true, a, data),
        }
        true
    }
}

/// One context's replay-time bookkeeping: its [`Context`], its ring
/// buffer, and how far the guest has written into it. The ring buffer is
/// sized generously and never wraps for these traces (each is a handful
/// of small submissions); a trace exercising real wraparound is out of
/// this milestone's scope.
struct ReplayContext {
    ctx: Context,
    ring: Vec<u8>,
}

const RING_BUF_SIZE: usize = 1 << 20; // 1 MiB: comfortably larger than any M1 trace's ring usage.

/// Replays every section of `bytes` against a fresh set of contexts and a
/// fresh [`FlatMemory`], executing every [`RenderOp`] through `renderer`
/// as it goes and capturing a frame at each `FENCE`. Returns the captured
/// frames in trace order, plus any renderer errors encountered (recorded,
/// not fatal -- a conformance run wants to see every failure).
fn replay(
    bytes: &[u8],
    renderer: &mut Renderer,
) -> Result<(Vec<CapturedFrame>, Vec<String>), String> {
    let mut contexts: HashMap<u32, ReplayContext> = HashMap::new();
    let mut mem = FlatMemory::new();
    let mut frames = Vec::new();
    let mut errors = Vec::new();
    let config = DeviceConfig::default();

    for section in TraceReader::new(bytes) {
        let section = section.map_err(|e| e.to_string())?;
        match section {
            Section::Header(_) => {}
            Section::Aperture(a) => mem.put(false, a.offset, a.data),
            Section::Blob(b) => {
                // `space` is the container's own flag: 0 the data
                // aperture, 1 guest memory, matching a command-stream
                // reference. A trace captured from a `CAP_GUESTMEM`
                // device carries blobs in both, and replaying one into
                // the wrong space would corrupt the replay silently, so
                // the flag is honoured rather than assumed.
                mem.put(b.space == 1, b.address, b.data);
            }
            Section::Gold(_) => {}
            Section::Unknown { .. } => {}
            Section::Ring(r) => {
                let entry = contexts.entry(r.context).or_insert_with(|| ReplayContext {
                    ctx: Context::new(Limits::default()),
                    ring: vec![0u8; RING_BUF_SIZE],
                });
                // The bytes this RING section carries are exactly what the
                // guest appended before writing `ring_tail`; place them at
                // the ring's previous tail, matching the doorbell sequence
                // `trace.rs`'s module doc comment describes.
                let prev_tail = entry.ctx.ring_tail as usize;
                let end = prev_tail + r.bytes.len();
                if end > entry.ring.len() {
                    return Err(format!(
                        "context {} RING section overflows the {}-byte replay ring buffer",
                        r.context, RING_BUF_SIZE
                    ));
                }
                entry.ring[prev_tail..end].copy_from_slice(r.bytes);

                let mut ops: Vec<RenderOp> = Vec::new();
                entry
                    .ctx
                    .submit(&entry.ring, r.ring_tail, &config, &mut ops);

                if entry.ctx.error_code != 0 {
                    errors.push(format!(
                        "context {} protocol error {} at ring offset {}",
                        r.context, entry.ctx.error_code, entry.ctx.error_offset
                    ));
                }
                let gl_err = entry.ctx.state.gl_error();
                if gl_err != state::GL_NO_ERROR {
                    errors.push(format!(
                        "context {} GL error 0x{gl_err:04x} pending",
                        r.context
                    ));
                }

                let render_errors = renderer.execute(&ops, &entry.ctx.state, &mut mem);
                for e in render_errors {
                    errors.push(format!("context {}: {e}", r.context));
                }

                for op in &ops {
                    if let RenderOp::Fence { id } = op {
                        entry.ctx.complete_fence(*id);
                        let surface_id = entry.ctx.state.draw_surface();
                        match renderer.read_surface_rgba8(surface_id) {
                            Ok((width, height, rgba)) => frames.push(CapturedFrame {
                                context: r.context,
                                fence_id: *id,
                                surface_id,
                                width,
                                height,
                                rgba,
                            }),
                            Err(e) => errors.push(format!(
                                "context {} fence {id}: could not read back surface {surface_id}: {e}",
                                r.context
                            )),
                        }
                    }
                }
            }
        }
    }

    Ok((frames, errors))
}

fn write_frame_png(dir: &Path, frame: &CapturedFrame) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("creating {}: {e}", dir.display()))?;
    let path = dir.join(format!(
        "ctx{}_fence{}_surf{}.png",
        frame.context, frame.fence_id, frame.surface_id
    ));
    encode_png(&path, frame.width, frame.height, &frame.rgba)?;
    Ok(path)
}

fn encode_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    let file =
        std::fs::File::create(path).map_err(|e| format!("creating {}: {e}", path.display()))?;
    let writer = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut w = encoder
        .write_header()
        .map_err(|e| format!("writing PNG header to {}: {e}", path.display()))?;
    w.write_image_data(rgba)
        .map_err(|e| format!("writing PNG data to {}: {e}", path.display()))?;
    Ok(())
}

fn decode_png(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("decoding PNG: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("decoding PNG frame: {e}"))?;
    buf.truncate(info.buffer_size());
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => buf
            .chunks(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        other => return Err(format!("unsupported golden PNG colour type {other:?}")),
    };
    Ok((info.width, info.height, rgba))
}

// ---------------------------------------------------------------------
// `run`
// ---------------------------------------------------------------------

fn cmd_run(path: &Path, out_dir: &Path) -> Result<i32, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut renderer = new_renderer()?;
    let (frames, errors) = replay(&bytes, &mut renderer)?;
    for e in &errors {
        eprintln!("warning: {e}");
    }
    for frame in &frames {
        let out = write_frame_png(out_dir, frame)?;
        println!("wrote {}", out.display());
    }
    if frames.is_empty() {
        eprintln!("warning: the trace produced no fences, so no frames were captured");
    }
    Ok(if errors.is_empty() { 0 } else { 1 })
}

// ---------------------------------------------------------------------
// `compare`
// ---------------------------------------------------------------------

/// Per-pixel tolerance (max absolute difference per channel, 0..255) and
/// per-image tolerance (fraction of pixels allowed to exceed it) a
/// `compare` run accepts. Pixel output is never bit-identical across
/// implementations (`docs/internals/c3d.md`'s "Determinism and timing"),
/// so this comparison is deliberately never exact.
const PIXEL_TOLERANCE: u8 = 12;
const IMAGE_TOLERANCE_FRACTION: f64 = 0.01;

struct CompareResult {
    context: u32,
    fence_id: u32,
    surface_id: u32,
    differing_pixels: usize,
    total_pixels: usize,
    max_diff: u8,
    passed: bool,
}

fn compare_frame(gold: (u32, u32, &[u8]), actual: &CapturedFrame) -> Result<CompareResult, String> {
    let (gw, gh, gpix) = gold;
    if gw != actual.width || gh != actual.height {
        return Err(format!(
            "size mismatch: golden {gw}x{gh}, rendered {}x{}",
            actual.width, actual.height
        ));
    }
    let mut differing = 0usize;
    let mut max_diff = 0u8;
    let total = (gw * gh) as usize;
    for i in 0..total {
        let g = &gpix[i * 4..i * 4 + 4];
        let a = &actual.rgba[i * 4..i * 4 + 4];
        let mut pixel_max = 0u8;
        for c in 0..4 {
            let d = g[c].abs_diff(a[c]);
            pixel_max = pixel_max.max(d);
        }
        max_diff = max_diff.max(pixel_max);
        if pixel_max > PIXEL_TOLERANCE {
            differing += 1;
        }
    }
    let fraction = differing as f64 / total.max(1) as f64;
    Ok(CompareResult {
        context: actual.context,
        fence_id: actual.fence_id,
        surface_id: actual.surface_id,
        differing_pixels: differing,
        total_pixels: total,
        max_diff,
        passed: fraction <= IMAGE_TOLERANCE_FRACTION,
    })
}

fn cmd_compare(path: &Path) -> Result<i32, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut golds = Vec::new();
    for section in TraceReader::new(&bytes) {
        if let Section::Gold(g) = section.map_err(|e| e.to_string())? {
            golds.push((g.context, g.fence_id, g.surface_id, g.png.to_vec()));
        }
    }
    if golds.is_empty() {
        return Err("trace has no GOLD sections to compare against".into());
    }

    let mut renderer = new_renderer()?;
    let (frames, errors) = replay(&bytes, &mut renderer)?;
    for e in &errors {
        eprintln!("warning: {e}");
    }

    let mut all_passed = true;
    for (context, fence_id, surface_id, png_bytes) in &golds {
        let Some(actual) = frames.iter().find(|f| {
            f.context == *context && f.fence_id == *fence_id && f.surface_id == *surface_id
        }) else {
            println!(
                "FAIL ctx{context} fence{fence_id} surf{surface_id}: no rendered frame at this fence"
            );
            all_passed = false;
            continue;
        };
        let (gw, gh, gpix) = decode_png(png_bytes)?;
        match compare_frame((gw, gh, &gpix), actual) {
            Ok(r) => {
                let status = if r.passed { "PASS" } else { "FAIL" };
                println!(
                    "{status} ctx{} fence{} surf{}: {}/{} pixel(s) differ beyond tolerance {} (max channel diff {})",
                    r.context, r.fence_id, r.surface_id, r.differing_pixels, r.total_pixels, PIXEL_TOLERANCE, r.max_diff
                );
                all_passed &= r.passed;
            }
            Err(e) => {
                println!("FAIL ctx{context} fence{fence_id} surf{surface_id}: {e}");
                all_passed = false;
            }
        }
    }

    Ok(if all_passed && errors.is_empty() {
        0
    } else {
        1
    })
}

// ---------------------------------------------------------------------
// `record`
// ---------------------------------------------------------------------

fn cmd_record(path: &Path, out_path: &Path) -> Result<i32, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut renderer = new_renderer()?;
    let (frames, errors) = replay(&bytes, &mut renderer)?;
    for e in &errors {
        eprintln!("warning: {e}");
    }

    let out_file = std::fs::File::create(out_path)
        .map_err(|e| format!("creating {}: {e}", out_path.display()))?;
    let mut writer = TraceWriter::new(std::io::BufWriter::new(out_file));

    for section in TraceReader::new(&bytes) {
        let section = section.map_err(|e| e.to_string())?;
        match section {
            Section::Header(h) => writer
                .header(h.protocol_version, h.capability_mask)
                .map_err(|e| e.to_string())?,
            Section::Aperture(a) => writer
                .aperture(a.offset, a.data)
                .map_err(|e| e.to_string())?,
            Section::Ring(r) => writer
                .ring(r.context, r.ring_tail, r.bytes)
                .map_err(|e| e.to_string())?,
            Section::Blob(b) => writer
                .blob(b.space, b.address, b.data)
                .map_err(|e| e.to_string())?,
            // Regenerated below instead of carried over: `record`'s whole
            // job is to replace whatever goldens the trace had (if any)
            // with freshly rendered ones.
            Section::Gold(_) => {}
            Section::Unknown { .. } => {}
        }
    }

    for frame in &frames {
        let mut png_bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png_bytes, frame.width, frame.height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut w = encoder.write_header().map_err(|e| e.to_string())?;
            w.write_image_data(&frame.rgba).map_err(|e| e.to_string())?;
        }
        writer
            .gold(frame.context, frame.fence_id, frame.surface_id, &png_bytes)
            .map_err(|e| e.to_string())?;
    }
    writer.into_inner().flush().map_err(|e| e.to_string())?;

    println!(
        "wrote {} with {} golden frame(s)",
        out_path.display(),
        frames.len()
    );
    Ok(if errors.is_empty() { 0 } else { 1 })
}

// ---------------------------------------------------------------------
// Renderer setup
// ---------------------------------------------------------------------

fn new_renderer() -> Result<Renderer, String> {
    let renderer = Renderer::new().map_err(|e| format!("could not create a renderer: {e}"))?;
    eprintln!("using adapter: {}", renderer.describe());
    Ok(renderer)
}

// ---------------------------------------------------------------------
// `selftest`: builds the three M1 traces in code and sanity-checks them
// ---------------------------------------------------------------------

fn cmd_selftest() -> Result<i32, String> {
    let mut renderer = new_renderer()?;
    let mut ok = true;

    ok &= selftest_one(
        "clear",
        &sample_traces::clear_trace(),
        &mut renderer,
        |frames, _| {
            let f = frames.first().ok_or("no frame captured")?;
            check_uniform_color(f, [0, 128, 255, 255])
        },
    )?;

    ok &= selftest_one(
        "triangle",
        &sample_traces::triangle_trace(),
        &mut renderer,
        |frames, _| {
            let f = frames.first().ok_or("no frame captured")?;
            check_triangle(f)
        },
    )?;

    ok &= selftest_one(
        "textured quad",
        &sample_traces::textured_quad_trace(),
        &mut renderer,
        |frames, _| {
            let f = frames.first().ok_or("no frame captured")?;
            check_textured_quad(f)
        },
    )?;

    if ok {
        println!("selftest: all traces passed");
        Ok(0)
    } else {
        println!("selftest: FAILED");
        Ok(1)
    }
}

fn selftest_one(
    name: &str,
    trace_bytes: &[u8],
    renderer: &mut Renderer,
    check: impl FnOnce(&[CapturedFrame], &mut Renderer) -> Result<(), String>,
) -> Result<bool, String> {
    let (frames, errors) = replay(trace_bytes, renderer)?;
    if !errors.is_empty() {
        println!("FAIL {name}: renderer errors: {errors:?}");
        return Ok(false);
    }
    match check(&frames, renderer) {
        Ok(()) => {
            println!("PASS {name}");
            Ok(true)
        }
        Err(e) => {
            println!("FAIL {name}: {e}");
            Ok(false)
        }
    }
}

fn check_uniform_color(frame: &CapturedFrame, expect: [u8; 4]) -> Result<(), String> {
    for (i, px) in frame.rgba.chunks(4).enumerate() {
        let diff: u8 = px
            .iter()
            .zip(expect.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(0);
        if diff > PIXEL_TOLERANCE {
            return Err(format!(
                "pixel {i} is {px:?}, expected close to {expect:?} (tolerance {PIXEL_TOLERANCE})"
            ));
        }
    }
    Ok(())
}

/// Checks the triangle trace's frame: roughly the expected screen
/// fraction is covered by the vertex colour, and the surface centre (well
/// inside the triangle sample_traces::triangle_trace draws) is that
/// colour.
fn check_triangle(frame: &CapturedFrame) -> Result<(), String> {
    let expect = [255u8, 0, 0, 255]; // the trace's vertex colour: opaque red.
    let bg = [0u8, 0, 0, 255]; // the trace's clear colour: opaque black.
                               // The trace's triangle is (1,1),(7,1),(7,7): its interior is exactly
                               // where y <= x (the half of the (1,1)-(7,7) diagonal containing
                               // (7,1)); (6,2) is well inside that with margin from every edge.
    let cx = 6u32.min(frame.width.saturating_sub(1));
    let cy = 2u32.min(frame.height.saturating_sub(1));
    let idx = ((cy * frame.width + cx) * 4) as usize;
    let center = &frame.rgba[idx..idx + 4];
    let diff: u8 = center
        .iter()
        .zip(expect.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(255);
    if diff > PIXEL_TOLERANCE {
        return Err(format!(
            "triangle interior at ({cx},{cy}) is {center:?}, expected close to {expect:?}"
        ));
    }

    let mut covered = 0usize;
    let total = (frame.width * frame.height) as usize;
    for px in frame.rgba.chunks(4) {
        let d: u8 = px
            .iter()
            .zip(expect.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap_or(255);
        if d <= PIXEL_TOLERANCE {
            covered += 1;
        }
    }
    let fraction = covered as f64 / total as f64;
    // A right triangle spanning most of an 8x8 surface covers roughly
    // half of it; a loose band avoids over-fitting to exact rasteriser
    // edge behaviour.
    if !(0.2..=0.8).contains(&fraction) {
        let _ = bg;
        return Err(format!(
            "triangle covers {:.1}% of the surface, expected roughly 20-80%",
            fraction * 100.0
        ));
    }
    Ok(())
}

/// Checks the textured quad trace's frame reproduces the 2x2 checkerboard
/// texture `sample_traces::textured_quad_trace` uploads (red at texel
/// (0,0) and (1,1), green at (0,1) and (1,0); `MODULATE` with a white
/// vertex colour reproduces it unchanged): the quad's screen top-left
/// (texcoord (0,0)) should be red and its top-right (texcoord (1,0))
/// should be green.
fn check_textured_quad(frame: &CapturedFrame) -> Result<(), String> {
    let sample = |x: u32, y: u32| -> [u8; 4] {
        let idx = ((y * frame.width + x) * 4) as usize;
        [
            frame.rgba[idx],
            frame.rgba[idx + 1],
            frame.rgba[idx + 2],
            frame.rgba[idx + 3],
        ]
    };
    let top_left = sample(1, 1);
    let top_right = sample(frame.width - 2, 1);
    let close = |a: [u8; 4], b: [u8; 4]| {
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.abs_diff(*y) <= PIXEL_TOLERANCE)
    };
    if !close(top_left, [255, 0, 0, 255]) {
        return Err(format!(
            "top-left texel is {top_left:?}, expected close to red"
        ));
    }
    if !close(top_right, [0, 255, 0, 255]) {
        return Err(format!(
            "top-right texel is {top_right:?}, expected close to green"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// The three M1 sample traces, built in code (not committed binary
// blobs): clear, triangle, textured quad. Each is one context, one RING
// submission, one FENCE.
// ---------------------------------------------------------------------

mod sample_traces {
    use copperline::c3d::proto::*;
    use copperline::c3d::trace::TraceWriter;

    fn f32w(v: f32) -> u32 {
        v.to_bits()
    }

    /// One command's ring bytes: `opcode`, then `payload_words`
    /// big-endian `u32`s, with `length` computed automatically. Mirrors
    /// `dispatch.rs`'s own private test helper of the same shape -- that
    /// one is `#[cfg(test)]`-only and not exported, so this binary (which
    /// needs to build real command streams, not just test fixtures)
    /// carries its own copy rather than depending on test-only code.
    fn cmd(opcode: u16, payload_words: &[u32]) -> Vec<u8> {
        let length = 1 + payload_words.len() as u32;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(((opcode as u32) << 16) | length).to_be_bytes());
        for w in payload_words {
            bytes.extend_from_slice(&w.to_be_bytes());
        }
        bytes
    }

    const SURFACE_ID: u32 = 1;
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 8;

    // GL enumerants this module needs on the wire (see `state.rs` for the
    // canonical list; duplicated here as plain constants because this
    // binary builds raw command streams, not typed `state.rs` values).
    const GL_TEXTURE_2D: u32 = 0x0DE1;
    const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
    const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
    const GL_TEXTURE_WRAP_S: u32 = 0x2802;
    const GL_TEXTURE_WRAP_T: u32 = 0x2803;
    const GL_NEAREST: u32 = 0x2600;
    const GL_CLAMP_TO_EDGE: u32 = 0x812F;
    const GL_TRIANGLES: u32 = 0x0004;
    const SURFACE_FORMAT_A8R8G8B8: u32 = 5;
    const TEX_FORMAT_RGBA8: u32 = 0;

    fn define_and_bind_surface() -> Vec<u8> {
        let mut b = Vec::new();
        // SURFACE_DEFINE: id, width, height, stride_bytes, format, flags, address.
        b.extend(cmd(
            OP_SURFACE_DEFINE,
            &[
                SURFACE_ID,
                WIDTH,
                HEIGHT,
                WIDTH * 4,
                SURFACE_FORMAT_A8R8G8B8,
                0,
                0,
            ],
        ));
        b.extend(cmd(OP_SET_DRAW_SURFACE, &[SURFACE_ID]));
        b
    }

    /// A `CLEAR_COLOR` of opaque `(0, 0.5, 1, 1)` (roughly `[0, 128, 255,
    /// 255]` once quantised), then `CLEAR` (colour + depth), then `FENCE`.
    /// Exercises the pipeline's simplest path end to end: ring decode,
    /// dispatch, a `RenderOp::Clear`, a surface readback.
    pub fn clear_trace() -> Vec<u8> {
        let mut ring = Vec::new();
        ring.extend(define_and_bind_surface());
        ring.extend(cmd(
            OP_CLEAR_COLOR,
            &[f32w(0.0), f32w(0.5), f32w(1.0), f32w(1.0)],
        ));
        ring.extend(cmd(OP_CLEAR, &[CLEAR_MASK_COLOR | CLEAR_MASK_DEPTH]));
        ring.extend(cmd(OP_FENCE, &[1]));

        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        w.header(PROTOCOL_VERSION, 0).unwrap();
        w.ring(0, ring.len() as u32, &ring).unwrap();
        out
    }

    /// Clears black, then draws one opaque-red window-space triangle
    /// (`DRAW_INLINE_WIN`, `POS_COUNT`=1, `COLOR`) covering roughly the
    /// surface's lower-right half, then `FENCE`. Exercises vertex
    /// parsing, triangulation/provoking-vertex handling and rasterisation
    /// end to end.
    pub fn triangle_trace() -> Vec<u8> {
        let mut ring = Vec::new();
        ring.extend(define_and_bind_surface());
        ring.extend(cmd(
            OP_CLEAR_COLOR,
            &[f32w(0.0), f32w(0.0), f32w(0.0), f32w(1.0)],
        ));
        ring.extend(cmd(OP_CLEAR, &[CLEAR_MASK_COLOR | CLEAR_MASK_DEPTH]));

        // format: POS_COUNT=1 (3 words: x,y,z), COLOR (4 words).
        let format = (1u32 << VertexFormat::POS_COUNT_SHIFT) | VertexFormat::COLOR;
        let vertex = |x: f32, y: f32, r: f32, g: f32, b: f32| -> Vec<u32> {
            vec![
                f32w(x),
                f32w(y),
                f32w(0.5),
                f32w(r),
                f32w(g),
                f32w(b),
                f32w(1.0),
            ]
        };
        let mut payload = vec![GL_TRIANGLES, format, 3];
        payload.extend(vertex(1.0, 1.0, 1.0, 0.0, 0.0));
        payload.extend(vertex(7.0, 1.0, 1.0, 0.0, 0.0));
        payload.extend(vertex(7.0, 7.0, 1.0, 0.0, 0.0));
        ring.extend(cmd(OP_DRAW_INLINE_WIN, &payload));
        ring.extend(cmd(OP_FENCE, &[1]));

        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        w.header(PROTOCOL_VERSION, 0).unwrap();
        w.ring(0, ring.len() as u32, &ring).unwrap();
        out
    }

    /// Uploads a 2x2 `RGBA8` checkerboard texture (red/green diagonal),
    /// binds it to unit 0 with `NEAREST`/`CLAMP_TO_EDGE`, and draws one
    /// window-space quad covering the whole surface with `TEXCOORD0`
    /// spanning `0..1` and a white vertex colour (so `MODULATE`, the
    /// default texenv, reproduces the texture unchanged). Exercises
    /// `TEX_IMAGE`, sampling, wrap/filter mapping and the `TEXCOORD0`
    /// vertex path end to end.
    pub fn textured_quad_trace() -> Vec<u8> {
        const TEX_ID: u32 = 1;
        const TEX_REF_ADDR: u32 = 0x0010_0000; // an aperture offset, arbitrary but ref-aligned.

        // 2x2 RGBA8 checkerboard: (0,0)=red, (1,0)=green, (0,1)=green, (1,1)=red.
        let texel = |r: u8, g: u8, b: u8| -> [u8; 4] { [r, g, b, 255] };
        let mut texels = Vec::new();
        texels.extend_from_slice(&texel(255, 0, 0));
        texels.extend_from_slice(&texel(0, 255, 0));
        texels.extend_from_slice(&texel(0, 255, 0));
        texels.extend_from_slice(&texel(255, 0, 0));

        let mut ring = Vec::new();
        ring.extend(define_and_bind_surface());
        ring.extend(cmd(
            OP_CLEAR_COLOR,
            &[f32w(0.0), f32w(0.0), f32w(0.0), f32w(1.0)],
        ));
        ring.extend(cmd(OP_CLEAR, &[CLEAR_MASK_COLOR | CLEAR_MASK_DEPTH]));

        ring.extend(cmd(OP_TEX_CREATE, &[TEX_ID]));
        ring.extend(cmd(
            OP_TEX_IMAGE,
            &[
                TEX_ID,
                0, // level
                TEX_FORMAT_RGBA8,
                2, // width
                2, // height
                8, // row_bytes (2 texels * 4 bytes/texel)
                TEX_REF_ADDR,
                16, // ref length: row_bytes * height
            ],
        ));
        ring.extend(cmd(
            OP_TEX_PARAM,
            &[TEX_ID, GL_TEXTURE_MIN_FILTER, GL_NEAREST],
        ));
        ring.extend(cmd(
            OP_TEX_PARAM,
            &[TEX_ID, GL_TEXTURE_MAG_FILTER, GL_NEAREST],
        ));
        ring.extend(cmd(
            OP_TEX_PARAM,
            &[TEX_ID, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE],
        ));
        ring.extend(cmd(
            OP_TEX_PARAM,
            &[TEX_ID, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE],
        ));
        ring.extend(cmd(OP_TEX_BIND, &[0, TEX_ID]));
        ring.extend(cmd(OP_ENABLE, &[GL_TEXTURE_2D]));

        // format: POS_COUNT=2 (2 words: x,y), COLOR, TEXCOORD0.
        let format =
            (2u32 << VertexFormat::POS_COUNT_SHIFT) | VertexFormat::COLOR | VertexFormat::TEXCOORD0;
        let vertex = |x: f32, y: f32, s: f32, t: f32| -> Vec<u32> {
            vec![
                f32w(x),
                f32w(y),
                f32w(1.0),
                f32w(1.0),
                f32w(1.0),
                f32w(1.0), // white vertex colour
                f32w(s),
                f32w(t),
            ]
        };
        // Two triangles covering the whole 8x8 surface, texcoords 0..1.
        let mut payload = vec![GL_TRIANGLES, format, 6];
        payload.extend(vertex(0.0, 0.0, 0.0, 0.0));
        payload.extend(vertex(8.0, 0.0, 1.0, 0.0));
        payload.extend(vertex(8.0, 8.0, 1.0, 1.0));
        payload.extend(vertex(0.0, 0.0, 0.0, 0.0));
        payload.extend(vertex(8.0, 8.0, 1.0, 1.0));
        payload.extend(vertex(0.0, 8.0, 0.0, 1.0));
        ring.extend(cmd(OP_DRAW_INLINE_WIN, &payload));
        ring.extend(cmd(OP_FENCE, &[1]));

        let mut out = Vec::new();
        let mut w = TraceWriter::new(&mut out);
        w.header(PROTOCOL_VERSION, CAP_GUESTMEM).unwrap();
        // BLOB: the texel bytes at TEX_REF_ADDR, materialised before the
        // RING section that references them, per trace.rs's contract.
        // Space 0: the data aperture. The M1 traces use aperture refs
        // only, so nothing here needs CAP_GUESTMEM.
        w.blob(0, TEX_REF_ADDR, &texels).unwrap();
        w.ring(0, ring.len() as u32, &ring).unwrap();
        out
    }
}
