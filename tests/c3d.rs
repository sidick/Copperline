//! C3D virtual 3D accelerator board: M2 integration and headless
//! verification.
//!
//! `c3d_m2_board_probe_clear_and_readback` needs no local assets: it
//! boots the bundled AROS ROM with `[c3d] enabled = true` and a
//! `[[filesys]]` mount built from the committed
//! `guest/c3d/test/c3dtest` protocol test binary (see
//! `guest/c3d/test/c3dtest.c` -- a committed artifact, referenced
//! directly by path, exactly like `tests/mhi.rs` does for
//! `guest/mhi/test/mhitest`), and asserts every `C3DTEST: ...` line the
//! guest probe writes back to the host (via a Shell output redirect on
//! the mounted host directory, the same mechanism `tests/mhi.rs`'s own
//! M1 test uses) is a PASS -- proving `FindConfigDev`, the register
//! file, the doorbell, and a real `CLEAR` + `SURFACE_READBACK` landing
//! the right pixel bytes in the aperture, all against the real board
//! through a real 68k CPU rather than through Rust calling
//! `src/c3d/board.rs`'s `ZorroDevice` methods directly (which
//! `board.rs`'s own unit tests already cover).
//!
//! This is the M2 milestone's guest-visible proof: the proposal's
//! section 11.1 protocol test program, run headless, with no MiniGL and
//! no assets.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

static EMULATOR_TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock_emulator_tests() -> std::sync::MutexGuard<'static, ()> {
    EMULATOR_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn run_copperline(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_copperline"))
        .current_dir(repo_root())
        .env("RUST_LOG", "copperline=warn,copperline::emulator=info")
        .env("COPPERLINE_AROS_DIR", repo_root().join("assets/aros"))
        .args(args)
        .output()
        .expect("run emulator")
}

fn scratch_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("copperline-c3d-test-{name}-{}", std::process::id()))
}

/// Every `C3DTEST: ...` check line as `(kind, rest)`, mirroring
/// `tests/mhi.rs`'s `parse_probe_lines`.
fn parse_probe_lines(output: &str, prefix: &str) -> Vec<(String, String)> {
    output
        .lines()
        .filter_map(|line| line.strip_prefix(prefix))
        .filter_map(|rest| {
            let mut parts = rest.splitn(2, ' ');
            let kind = parts.next()?.to_string();
            let tail = parts.next().unwrap_or("").to_string();
            Some((kind, tail))
        })
        .collect()
}

/// Stage a boot volume: `S/Startup-Sequence` runs `c3dtest` redirected to
/// `c3dtest.out`, then drops a `done` marker.
fn stage_m2_mount(mount: &Path, probe_src: &Path) {
    let _ = std::fs::remove_dir_all(mount);
    std::fs::create_dir_all(mount.join("S")).expect("create S/");
    std::fs::copy(probe_src, mount.join("c3dtest")).expect("stage probe binary");
    std::fs::write(
        mount.join("S").join("Startup-Sequence"),
        "FailAt 21\nSYS:c3dtest >SYS:c3dtest.out\nEcho >\"SYS:done\" \"done\"\n",
    )
    .expect("write Startup-Sequence");
}

fn write_m2_config(cfg_path: &Path, mount: &Path) {
    // The C3D board is Zorro III (docs/zorro.md's product 9), which needs
    // a 32-bit-bus CPU -- the default no-[machine] profile is an A500
    // (68000), which cannot address it at all (confirmed the hard way:
    // without this, register reads come back as whatever garbage sits at
    // the aliased 24-bit address rather than a clean rejection). A4000
    // is the same profile tests/graffity.rs and tests/picasso2.rs already
    // use for their own Zorro III boards.
    std::fs::write(
        cfg_path,
        format!(
            "rom = \"<bundled-aros>\"\n\n\
             [machine]\n\
             profile = \"A4000\"\n\n\
             [c3d]\n\
             enabled = true\n\n\
             [[filesys]]\n\
             path = '{}'\n\
             volume = \"C3DBOOT\"\n\
             bootpri = 6\n",
            mount.display()
        ),
    )
    .expect("write test config");
}

/// See this file's own doc comment.
#[test]
#[ignore = "runs the emulator"]
fn c3d_m2_board_probe_clear_and_readback() {
    let _guard = lock_emulator_tests();
    let mount = scratch_dir("m2-mount");
    stage_m2_mount(&mount, &repo_root().join("guest/c3d/test/c3dtest"));

    let cfg_path = scratch_dir("m2-cfg").with_extension("toml");
    write_m2_config(&cfg_path, &mount);

    let shot = scratch_dir("m2-shot").with_extension("png");
    let out = run_copperline(&[
        "--config",
        cfg_path.to_str().unwrap(),
        "--noaudio",
        "--screenshot-after",
        "40",
        shot.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "emulator run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("c3d: 3D accelerator board"),
        "expected the C3D board-attach log line; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(
        mount.join("done").is_file(),
        "Startup-Sequence never reached its completion marker under {}",
        mount.display()
    );

    let raw = std::fs::read_to_string(mount.join("c3dtest.out")).unwrap_or_else(|e| {
        panic!(
            "reading c3dtest.out under {}: {e} (mount contents: {:?})",
            mount.display(),
            std::fs::read_dir(&mount).map(|d| d
                .filter_map(|e| e.ok().map(|e| e.file_name()))
                .collect::<Vec<_>>())
        )
    });
    let lines = parse_probe_lines(&raw, "C3DTEST: ");
    assert!(
        !lines.is_empty(),
        "no C3DTEST: lines captured; raw output:\n{raw}"
    );

    let fails: Vec<&(String, String)> = lines.iter().filter(|(k, _)| k == "FAIL").collect();
    assert!(
        fails.is_empty(),
        "c3dtest reported failing checks: {fails:?}\nfull output:\n{raw}"
    );
    let (last_kind, last_rest) = lines.last().unwrap();
    assert_eq!(
        (last_kind.as_str(), last_rest.as_str()),
        ("SUMMARY", "PASS"),
        "c3dtest did not end with a PASS summary; full output:\n{raw}"
    );
}
