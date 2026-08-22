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
//!   relative to the CWD) that don't exist on Android. For now this reads
//!   from the app's internal data directory, pushed there by hand
//!   (`adb push ... /data/data/<app>/files/aros/`) -- shipping the ROM as
//!   an APK asset and extracting it on first run is storage work (WP5),
//!   not part of this stage.
//!
//! Everything else -- config defaults, machine build, window/render/input
//! -- is exactly `copperline`'s own code, unmodified.

use android_activity::AndroidApp;
use anyhow::{anyhow, Context, Result};
use copperline::audio::NullSink;
use copperline::config::{self, Config, ConfigOverrides};
use copperline::emulator;
use copperline::video::window::App;

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

    let raw = Config::load_raw(None, &ConfigOverrides::default())?;
    copperline::paths::adopt(raw.paths());
    let mut cfg = Config::try_from(raw.clone())?;

    // Stand-in for config::resolve_bundled_rom, which only looks in
    // desktop-shaped locations; see the module doc.
    cfg.rom_path = internal.join("aros/aros-amiga-m68k-rom.bin");
    cfg.extended_rom_path = Some(internal.join("aros/aros-amiga-m68k-ext.bin"));
    if !cfg.rom_path.is_file() {
        return Err(anyhow!(
            "no Kickstart at {} -- push the bundled AROS ROM there first \
             (adb push assets/aros/aros-amiga-m68k-*.bin <internal-data>/aros/)",
            cfg.rom_path.display()
        ));
    }

    let emu = emulator::build_machine(&cfg, Box::new(NullSink), true, false)
        .context("building the machine")?;

    let app = App::new(
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
        false, // audio_output_enabled -- NullSink above; cpal on Android is WP8
        copperline::sampler::SamplerRequest::from_config(&cfg.parallel),
    );

    app.run_android(android_app)
}
