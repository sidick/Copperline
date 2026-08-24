// SPDX-License-Identifier: GPL-3.0-or-later

//! Android entry point for the full Copperline emulator.
//!
//! Stage 2 of the Android port (see the Android port plan): drives
//! `copperline`'s real `App`/window loop, the same machine-build path
//! `src/main.rs` uses for an ordinary run with no CLI flags. What's
//! Android-specific is small and isolated:
//!
//! - the event loop, built around the `AndroidApp` GameActivity handle via
//!   [`copperline::video::window::App::run_android`] instead of the
//!   desktop `App::run`;
//! - the Kickstart path, since [`copperline::romsearch::find_bundled_aros`]
//!   only searches desktop-shaped locations (exe-relative, `assets/aros`
//!   relative to the CWD) that don't exist on Android. The same bundled
//!   AROS ROM instead ships as an APK asset (see the Gradle project's
//!   `copyArosAssets` task) and is extracted into the app's internal data
//!   directory on first run, once, by [`extract_bundled_aros`] -- the
//!   default a fresh install boots. A real Kickstart the user drops into
//!   `external/kickstart/` (see [`find_user_kickstart`]) overrides it, IDed
//!   by content the same way the desktop asset-gated tests are, never by
//!   file name. WHDLoad games remain unaddressed (still needs the wider
//!   SAF storage case WP5's own detail section describes).
//!
//! Everything else -- config defaults, machine build, window/render/input,
//! and audio (`CpalSink`, falling back to silence if it fails to open) --
//! is exactly `copperline`'s own code, unmodified.

use std::io::Read as _;
use std::path::Path;

use android_activity::AndroidApp;
use anyhow::{anyhow, Context, Result};
use copperline::audio::{AudioSink, CpalSink, NullSink};
use copperline::config::{self, Config, ConfigOverrides};
use copperline::emulator;
use copperline::video::window::App;

/// Files the Gradle project's `copyArosAssets` task bundles under
/// `assets/aros/` in the APK, matching `assets/aros/` at the repo root.
const AROS_ASSETS: &[&str] = &["aros-amiga-m68k-rom.bin", "aros-amiga-m68k-ext.bin"];

/// Where a user's own Kickstart goes, under the app's external-files
/// directory (the same one WP5 mounts as `ANDROID:`) -- a fixed,
/// discoverable subfolder rather than the mount's root, so it reads as
/// "put your ROM here" to someone browsing over USB or a file manager,
/// distinct from WHDLoad games or anything else they park in `ANDROID:`.
const KICKSTART_DIR: &str = "kickstart";

/// A user-supplied Kickstart found in [`KICKSTART_DIR`]: the primary image,
/// and a second one (if exactly one other file in the folder also
/// identifies) to use as the extended ROM -- the CDTV/CD32 second flash
/// bank `cfg.extended_rom_path` exists for. Identification is by content
/// ([`copperline::romdb::describe_file`]), never by file name, so it does
/// not matter what a dumper or a collection manager happened to call the
/// file -- the same convention the desktop asset-gated tests use.
struct UserKickstart {
    rom_path: std::path::PathBuf,
    extended_rom_path: Option<std::path::PathBuf>,
}

/// Look for a user's own Kickstart in `external/kickstart/`. Creates the
/// folder if it does not exist yet (so it is there to find, empty, the
/// first time someone goes looking for where to put a ROM) and returns
/// `None` -- keeping the bundled AROS default -- when it holds nothing
/// [`copperline::romdb::describe_file`] recognises. An Amiga-Forever
/// encrypted image is named in the log but not selected: decoding it needs
/// a `rom.key` this path does not look for.
fn find_user_kickstart(external: &Path) -> Option<UserKickstart> {
    let dir = external.join(KICKSTART_DIR);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::warn!("kickstart: could not prepare {} ({e:#})", dir.display());
        return None;
    }
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) => {
            log::warn!("kickstart: could not read {} ({e:#})", dir.display());
            return None;
        }
    };
    let mut candidates: Vec<std::path::PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    // Sorted so which file becomes primary vs. extended is deterministic
    // (and stable across runs) when a folder holds more than one.
    candidates.sort();

    let mut recognized = Vec::new();
    for path in candidates {
        match copperline::romdb::describe_file(&path) {
            Some(copperline::romdb::Identified::Known(entry)) => {
                log::info!(
                    "kickstart: {} identified as {}",
                    path.display(),
                    entry.label
                );
                recognized.push(path);
            }
            Some(copperline::romdb::Identified::Encrypted) => {
                log::info!(
                    "kickstart: {} is an Amiga Forever encrypted image; \
                     decoding needs a rom.key this path does not supply, skipping",
                    path.display()
                );
            }
            None => {
                log::info!(
                    "kickstart: {} is not a recognised Kickstart image, skipping",
                    path.display()
                );
            }
        }
    }

    let mut recognized = recognized.into_iter();
    let rom_path = recognized.next()?;
    let extended_rom_path = recognized.next();
    if let Some(extra) = recognized.next() {
        log::warn!(
            "kickstart: {} also identified, but only two ROMs are used; ignoring it and any others",
            extra.display()
        );
    }
    Some(UserKickstart {
        rom_path,
        extended_rom_path,
    })
}

/// Copy the bundled AROS ROM out of the APK's assets and into `internal`
/// (the app's private data directory), skipping any file already there --
/// this runs on every launch, so it needs to be cheap once installed.
fn extract_bundled_aros(android_app: &AndroidApp, internal: &Path) -> Result<()> {
    let dir = internal.join("aros");
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let mgr = android_app.asset_manager();
    for name in AROS_ASSETS {
        let dest = dir.join(name);
        if dest.is_file() {
            continue;
        }
        let asset_path = format!("aros/{name}");
        let cpath =
            std::ffi::CString::new(asset_path.as_str()).expect("asset path has no interior NUL");
        let mut asset = mgr
            .open(&cpath)
            .ok_or_else(|| anyhow!("asset {asset_path} not found in the APK"))?;
        let mut buf = Vec::new();
        asset
            .read_to_end(&mut buf)
            .with_context(|| format!("reading asset {asset_path}"))?;
        std::fs::write(&dest, &buf).with_context(|| format!("writing {}", dest.display()))?;
    }
    Ok(())
}

#[unsafe(no_mangle)]
fn android_main(android_app: AndroidApp) {
    copperline_android_host::init_logging();
    log::info!(
        "copperline-android: starting, copperline {}",
        env!("CARGO_PKG_VERSION")
    );
    if let Err(e) = run(android_app) {
        log::error!("copperline-android: {e:#}");
    }
}

fn run(android_app: AndroidApp) -> Result<()> {
    let internal = android_app
        .internal_data_path()
        .ok_or_else(|| anyhow!("no internal data path (Context.getFilesDir())"))?;
    extract_bundled_aros(&android_app, &internal).context("extracting the bundled AROS ROM")?;

    let raw = Config::load_raw(None, &ConfigOverrides::default())?;
    copperline::paths::adopt(raw.paths());
    let mut cfg = Config::try_from(raw.clone())?;

    // Stand-in for config::resolve_bundled_rom, which only looks in
    // desktop-shaped locations; see the module doc. A user's own Kickstart
    // in external/kickstart/, if one identifies, overrides this -- checked
    // after the external directory is known (just below).
    cfg.rom_path = internal.join("aros/aros-amiga-m68k-rom.bin");
    cfg.extended_rom_path = Some(internal.join("aros/aros-amiga-m68k-ext.bin"));

    // WP5 (storage): a real Kickstart the user dropped in external/
    // kickstart/ boots a real machine instead of the bundled AROS ROM --
    // real Workbench, real WHDLoad-compatible ROM behaviour, whatever the
    // user's own dump supports that AROS does not. Falls back to AROS
    // (already set above) when the folder holds nothing recognised, so
    // the app always has something to boot.
    if let Some(external) = android_app.external_data_path() {
        match find_user_kickstart(&external) {
            Some(kickstart) => {
                log::info!(
                    "kickstart: booting the user-supplied ROM at {}",
                    kickstart.rom_path.display()
                );
                cfg.rom_path = kickstart.rom_path;
                cfg.extended_rom_path = kickstart.extended_rom_path;
            }
            None => log::info!(
                "kickstart: no recognised ROM in external/{KICKSTART_DIR}/; booting the bundled AROS ROM"
            ),
        }
    }

    // WP5 (storage): no config file exists to have written a `[[filesys]]`
    // entry (there is no desktop-shaped `./copperline.toml` on Android), so
    // without this the guest would have no host storage at all. Auto-mount
    // the app's external-files directory -- `Context.getExternalFilesDir`,
    // real POSIX paths under `/sdcard/Android/data/<pkg>/files`, readable
    // from a connected computer or a file manager with "show Android/data"
    // enabled -- needing no runtime permission or SAF picker, unlike an
    // arbitrary user-chosen folder. That wider case (browsing to any
    // directory the user wants, e.g. an existing WHDLoad library) needs a
    // Kotlin `GameActivity` subclass to receive the picker's result plus a
    // URI/fd-based `filesys.rs` backend for the resulting `content://`
    // handle, since SAF hands back a document tree, not a path this
    // directory mount can use -- not attempted here.
    if cfg.filesys.is_empty() {
        if let Some(dir) = android_app.external_data_path() {
            match std::fs::create_dir_all(&dir) {
                Ok(()) => {
                    log::info!("storage: mounting {} as HOSTFS0:", dir.display());
                    cfg.filesys.push(copperline::filesys::MountSpec {
                        path: dir,
                        volume: "ANDROID".to_string(),
                        boot_pri: -128,
                        readonly: false,
                    });
                }
                Err(e) => log::warn!("storage: could not prepare {} ({e:#})", dir.display()),
            }
        } else {
            log::warn!("storage: no external-files directory available; guest has no HOSTFS");
        }
    }

    // WP7: on a panel with pixels to spare, default to the CRT look --
    // there's no settings screen yet to have overridden `[display]
    // shader` away from its (always, today) default of None, so this is
    // the one place that default is actually decided for this run.
    // Below the threshold a phone-class panel can't spare the pixels to
    // scanline structure without just looking dim, so it keeps None.
    if cfg.shader == config::ShaderMode::None {
        let screen_px = android_app.config().screen_width_dp().zip(
            android_app
                .config()
                .screen_height_dp()
                .zip(android_app.config().density()),
        );
        if let Some((width_dp, (height_dp, density))) = screen_px {
            let scale = f64::from(density) / 160.0;
            let px = (f64::from(width_dp) * scale) * (f64::from(height_dp) * scale);
            if px > 1920.0 * 1080.0 {
                log::info!("display: {px:.0} panel pixels, defaulting to the CRT shader");
                cfg.shader = config::ShaderMode::Crt;
            }
        }
    }

    // Fall back to silence rather than aborting the whole run: a device
    // with no usable audio output shouldn't stop the machine from booting
    // and displaying, any more than desktop's --noaudio does.
    let (audio, audio_output_enabled): (Box<dyn AudioSink>, bool) = match CpalSink::new(false, None)
    {
        Ok(sink) => (Box::new(sink), true),
        Err(e) => {
            log::warn!("audio: cpal init failed ({e:#}); continuing without sound");
            (Box::new(NullSink), false)
        }
    };

    // The panel refresh rate to request (WP7): the config's video standard
    // at boot, not necessarily what a state loaded below ends up running --
    // close enough for a display-mode hint, and simpler than threading a
    // live accessor through Emulator/Bus for it.
    let refresh_hz = match cfg.video_standard {
        copperline::chipset::agnus::VideoStandard::Pal => 50.0,
        copperline::chipset::agnus::VideoStandard::Ntsc => 60.0,
    };

    // A state saved by App::suspended the last time this process was
    // backgrounded: if Android killed the process rather than just
    // suspending it (the common case under memory pressure), this cold
    // start resumes it instead of rebooting AROS from scratch.
    let suspend_state_path = internal.join("suspend.clstate");
    let resuming = suspend_state_path.is_file();
    let mut emu =
        emulator::build_machine(&cfg, audio, true, resuming).context("building the machine")?;
    if resuming {
        match emu.load_state(&suspend_state_path) {
            Ok(outcome) => log::info!("resumed from suspend state: {}", outcome.summary),
            Err(e) => log::warn!("suspend state failed to load ({e:#}); starting fresh"),
        }
    }

    let mut app = App::new(
        emu,
        cfg.emulation.power_on,
        Vec::new(), // screenshot_after
        Vec::new(), // save_state_after
        None,       // frame_dump
        Vec::new(), // press_after
        Vec::new(), // click_after
        Vec::new(), // joy_after
        Vec::new(), // mouse_after
        Vec::new(), // mouse_to_after
        Vec::new(), // pot_after
        Vec::new(), // disk_insert_after
        Vec::new(), // cd_insert_after
        None,       // record_input
        None,       // run_warp_target
        cfg.floppy_playlists.clone(),
        [false, false, false, false], // disk_write_protected
        config::resolve_overscan(cfg.overscan),
        cfg.tv_centre,
        config::resolve_deinterlace(cfg.deinterlace),
        config::resolve_phosphor(cfg.phosphor),
        config::resolve_shader(cfg.shader.clone()),
        config::resolve_shader_strength(cfg.shader_strength),
        config::resolve_bezel(cfg.bezel),
        config::resolve_bezel_stickers(cfg.bezel_stickers.clone()),
        config::resolve_perf_overlay(cfg.perf_overlay),
        config::resolve_tint(cfg.tint),
        cfg.full_screen,
        !cfg.status_bar,
        cfg.emulation.warp_speed,
        cfg.joystick_input_mode,
        cfg.mouse_sensitivity,
        cfg.mouse_capture,
        config::about_machine_lines(&cfg),
        raw,
        audio_output_enabled,
        copperline::sampler::SamplerRequest::from_config(&cfg.parallel),
    );
    app.set_suspend_save_path(suspend_state_path);
    // A cheap clone (Arc-backed): App needs its own handle to request a
    // matching refresh rate each time it builds a window/surface, and
    // run_android below needs to keep consuming the original.
    app.set_android_frame_rate_hint(android_app.clone(), refresh_hz);

    // WP8: keep the pacer thread on the SoC's fastest core and at the
    // scheduler's front of the queue, rather than leaving a handheld's
    // big.LITTLE scheduler free to migrate/deprioritise it under load --
    // there is no desktop-style "don't be antisocial to other apps"
    // tradeoff here, this process owns the foreground.
    copperline::priority::pin_to_fastest_core();
    copperline::priority::elevate_pacer_thread();

    app.run_android(android_app)
}
