# Android (in progress)

Copperline's Android port is under active development. This chapter
tracks what exists today; see the Android port plan tracked alongside
this work for the full work-package sequence.

## What exists today

Two standalone crates, in the same non-workspace shape as
`crates/copperline-web` and `crates/copperline-player`:

- `crates/copperline-android-host` -- the reusable host layer (logcat
  logging today; storage, input and display policy land here as their own
  work packages). A plain library, not a native lib itself, and not
  buildable standalone on Android -- see its `Cargo.toml`.
- `crates/copperline-android` -- the actual app: a `cdylib` with the
  `android_main` GameActivity entry point, depending on the host layer and
  on `copperline` itself (`frontend`, `cpu-jit`). `android_main` extracts
  the bundled AROS ROM (see below), builds the machine the same way
  `src/main.rs` does for an ordinary run with no CLI flags
  (`Config::load_raw(None, ...)`, `emulator::build_machine`), then drives
  `copperline`'s real `App`/window loop via [`App::run_android`], winit's
  Android-specific counterpart to the desktop `App::run`.

...plus a minimal Gradle project at `crates/copperline-android/android/`
that packages the two into a real, installable APK.

**Verified end-to-end on a headless AVD, fully self-contained: `adb
install` + launch, with no manual setup, boots the bundled AROS ROM and
renders it live**, with Copperline's usual status bar chrome, showing
"Waiting for bootable media" (correct -- no `df0` is configured yet).
Confirmed via `logcat`: the ROM extracted from the APK's assets on first
launch, window/surface creation, the Zorro identification board
autoconfiguring, chipset register writes -- the real machine, not a stub.
Audio is real too: `CpalSink` (the same sink desktop uses) opens
successfully on the AVD's default output device (48 kHz, resampled from
Paula's 44.1 kHz mix rate, no underruns observed), falling back to
silence rather than aborting the run if it fails to open -- untested on a
physical device, and unverified whether sound is actually audible versus
merely opened without error (the AVD was run headless).

**Backgrounding is handled** (WP4): `ApplicationHandler::suspended` (a
winit callback essentially every other platform never fires) drops the
window/surface, which for free stops `about_to_wait` from stepping the
machine -- backgrounding already means "pause", with no Android-specific
code needed for that half. If `App::set_suspend_save_path` was called
(only `copperline-android` calls it), `suspended` also saves a state
there first, and `android_main` loads it back on a cold start if the file
exists -- so a process Android kills outright while backgrounded (the
common case under memory pressure, not just occlusion) resumes instead of
rebooting AROS from scratch. Verified on the AVD: backgrounding (HOME)
saves a state and the surface tears down cleanly; bringing the app back
without a kill rebuilds the surface and keeps running (screen was
confirmed broken -- solid black, un-recoverable -- before the
`suspended` fix, for the record); `adb shell am kill` followed by
relaunch loads the saved state and logs "resumed from suspend state: ...".
Resize/orientation handling (the rest of WP4) already works, unmodified:
the manifest's `android:configChanges` keeps the Activity alive across a
rotation, and the same `WindowEvent::Resized` / `resync_surface_size`
path desktop's resizable window already exercises handles the new native
window size winit reports -- no Android-specific code needed. Verified on
the AVD (`adb shell settings put system user_rotation 1`, then `0` to
rotate back): both directions resize and recentre correctly, live, with
no black-screen glitch and no crash.

**Display policy (WP7).** Named scaling modes already existed
generically (`[display] scaling`, `App::resumed`'s `plan_present_scaling`):
`Smooth` (aspect-fit, the default) and `Integer` (pixel-perfect, falling
back to smooth when the surface is too small for even 1x) -- both were
already exercised, correctly, by the WP4 rotation test above, so nothing
Android-specific was needed there. `App::apply_android_frame_rate_hint`
is new: `resumed` asks the compositor (`ANativeWindow_setFrameRate`,
`FrameRateCompatibility::FixedSource`) to match the emulated machine's own
PAL/NTSC rate (50/60 Hz) rather than whatever the panel defaults to, on
panels that support switching. Verified on the AVD: the call succeeds and
`dumpsys display` shows the vote registered
(`frameRateOverride {uid=..., frameRateHz=...}`), but the AVD's virtual
display only advertises one 60 Hz mode, so there's nothing to actually
switch to here -- unverified whether a real variable-refresh-rate panel
follows the request. `android_main` also defaults `[display] shader` to
`Crt` above a 1920x1080-pixel panel (there's no settings screen yet to
have chosen something else, so this is the one place the choice is
actually made for a given run today) -- computed from `AndroidApp::
config()`'s `screen_width_dp`/`screen_height_dp`/`density` before the
window exists, since the shader mode has to be decided before `App::new`.
Below that threshold the desktop default (`None`) stands: a phone-class
panel can't spare the pixels to scanline structure without just looking
dim. Verified on the AVD (a 1080x2400 panel, above the threshold): the
log confirms the computed pixel count and the CRT look (tube curvature,
vignette, phosphor/dot-mask texture) is visibly active in a screenshot.

**Performance and power (WP8).** `crates/copperline-android`'s
`android_main` calls two new `src/priority.rs` functions before starting
the event loop:

- `pin_to_fastest_core` reads every online CPU's `cpuinfo_max_freq` and
  `sched_setaffinity`s the calling (pacer) thread to whichever core(s)
  report the highest value -- on a big.LITTLE/big.MID.little handheld SoC
  this keeps the hot thread off an efficiency core the scheduler would
  otherwise be free to migrate it onto under load. Best effort, like the
  rest of the module: logs and continues unpinned if `cpufreq` isn't
  readable or the syscall fails. On the AVD's uniform-frequency virtual
  CPUs this pins to all four (`[0, 1, 2, 3]`) -- a degenerate but correct
  case; a real big.LITTLE SoC would narrow to just the big cores.
- `elevate_pacer_thread` (already existed; desktop calls it only under
  `[emulation] realtime_priority`) is now called unconditionally on
  Android -- there's no desktop-style "don't be antisocial to other apps"
  tradeoff when this process owns the foreground.

New also: `priority::android_thermal_throttling`, a direct binding to the
stable NDK `<android/thermal.h>` API (`AThermal_acquireManager` /
`AThermal_getCurrentThermalStatus` / `AThermal_releaseManager`, linked via
`#[link(name = "android")]` -- no JNI, no `ndk`/`ndk-sys` wrapper needed,
they don't have one). `App::check_android_thermal`, called from
`about_to_wait` and throttled to once every 5 seconds, surfaces
`THERMAL_STATUS_SEVERE` and worse as an OSD warning (once per throttling
episode, not repeated every interval) rather than letting the machine
silently drop below real time with no explanation on screen. Verified on
the AVD end-to-end, including the parts that don't have a real sensor to
trigger them: `adb shell cmd thermalservice override-status 3` forces the
status the same way real thermal stress would, and the OSD
("Device is thermal throttling -- performance may drop") appeared,
confirmed by screenshot, within one 5-second check cycle of the override
taking effect; `cmd thermalservice reset` before uninstalling put the AVD
back to normal so the override doesn't leak into other testing.

## WP9 (full mode): analysis-only, empirically unverifiable here

WP9 (hardware keyboard, pointer capture, DeX) has no code changes from
this pass -- what follows is what reading winit's Android backend
(`platform_impl/android/keycodes.rs`, `mod.rs`) turned up, and why it
couldn't be confirmed on the AVD.

**Hardware keyboard mapping already covers what the proposal asks for, on
paper.** Standard letter/number/function keys map through winit's
existing `to_physical_key`/`to_logical`, the same generic path desktop
uses -- nothing Android-specific needed. So does everything WP9 names
specifically: `Keycode::MetaLeft/MetaRight` (a real keyboard's
Windows/Command-equivalent key) map to `KeyCode::SuperLeft/SuperRight`,
which `src/keymap.rs` already binds to the Amiga keys; `Keycode::Numpad*`
map to `KeyCode::Numpad*`, matching the numeric keypad row-for-row. Help
has no host-keyboard equivalent by design on *any* platform, Android
included (see `src/video/window/kbdpanel.rs`'s doc comment) -- it's
reached through the existing on-screen keyboard panel instead, which
isn't Android-specific either.

**None of this could be empirically confirmed.** `adb shell input
keyevent` (the only synthetic input tool available without real
hardware) doesn't reach the app as a genuine hardware KeyEvent would:
`dumpsys window` confirms the app has focus, but `logcat` shows every
synthetic key intercepted by GameActivity's bundled GameTextInput layer
(`gti.InputConnection.onKey`, `source=0x0`) rather than reaching the
native input queue -- for a D-pad key, a plain letter, or anything else
tried. This is a testing-environment limitation (no real Bluetooth/USB
keyboard or gamepad attached to the AVD), not a confirmed defect; the
mapping analysis above is sound, but unverified the way the gamepad
D-pad's *behavior* was never actually verified either (WP6's docs
say "confirmed by reading winit's source", not confirmed on a device --
same caveat applies here for the same reason).

**Pointer capture is blocked on the identical root cause as WP6's analog
gamepad input**, not a separate gap: `ActiveEventLoop::set_cursor_grab`
is a hard `Err(NotSupported)` on Android in winit (no capture API is
wired up at all), and even without that, a real mouse's relative motion
would arrive as a `MotionEvent` and get misrouted through winit's
touch-only handling exactly like a joystick axis does -- winit's Android
`MotionEvent` handling doesn't discriminate by `source()` at all. A JNI
`View.requestPointerCapture()` shim (as the proposal describes) would
still hit this same wall on the Rust side.

**New finding, not previously known:** `WindowEvent::Touch` has no
handler anywhere in `src/video/window.rs` -- tapping the screen (tried at
several plausible status-bar-button coordinates, and confirmed via
`dumpsys window` that the tap coordinates were computed correctly) does
nothing at all. This isn't a WP9 gap, though: the proposal's own "Out of
scope" section excludes "any touch-first UI beyond the on-screen keyboard
overlay" for the whole port, on the premise that handheld mode is
gamepad/D-pad-navigated (`src/video/window/app_nav.rs`'s existing
controller-walkable UI, which reads `KeyCode::ArrowUp/Down/Left/Right` --
the same keys Android's D-pad already maps to per the keyboard analysis
above, and *also* unverified for the same GameTextInput-interception
reason). Worth knowing regardless, since a real user reaching for the
screen first is a natural instinct this doesn't reward.

**Not attempted:** runtime handheld/full mode switching (`Configuration.
keyboard`/`InputDevice` sources) has no second UI layout to switch into
-- this session built one UI, not the proposal's two presentation modes
-- so there's nothing to detect *for* yet. DeX/multi-window needs a
DeX-capable device or a resizable-emulator configuration neither of which
exist here.

## Building and running

```sh
# 1. Build the Rust side straight into the Gradle project's jniLibs dir.
cargo ndk -t arm64-v8a --platform 35 \
  -o crates/copperline-android/android/app/src/main/jniLibs \
  build --release --manifest-path crates/copperline-android/Cargo.toml

# 2. Build and install the APK (copyArosAssets, a Gradle task, bundles
#    the ROM from the repo's assets/aros/ automatically -- see below).
cd crates/copperline-android/android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n org.copperlinehq.copperline/com.google.androidgamesdk.GameActivity
```

Needs an NDK (r28c or newer; `cargo-ndk` finds one via `ANDROID_NDK_HOME`
or `$ANDROID_HOME/ndk`) and, for the Gradle step, a JDK 17+. The Gradle
wrapper (`./gradlew`) downloads its own Gradle 9.5.0 and the Android
Gradle Plugin on first run; no separate Gradle install is needed.

### Kickstart path: the one bundled asset, for now

`copperline::romsearch::find_bundled_aros` only searches desktop-shaped
locations (exe-relative, `assets/aros` relative to the CWD), which don't
exist for an Android app. Instead, the Gradle project's `copyArosAssets`
task copies the same `assets/aros/` files from the repo root into the
APK's own `assets/` (Android's normal read-only per-APK asset store), and
`android_main`'s `extract_bundled_aros` copies them out into the app's
internal data directory (`Context.getFilesDir()`, writable, where a real
file path exists) on first launch, then points `cfg.rom_path`/
`extended_rom_path` there directly -- bypassing `find_bundled_aros`
entirely rather than teaching it Android's layout. A user's own Kickstart
or WHDLoad games, from arbitrary host-chosen storage, is still WP5; this
is only the one asset every install already needs, done now because a
manual `adb push` step for every fresh install was worse than doing it
properly.

### Non-obvious Gradle/manifest pieces

In case they need touching again:

- `GameActivity`'s real class name is `com.google.androidgamesdk.
  GameActivity` -- the `androidx.games:games-activity` artifact ships it
  under the legacy AGDK package, not `androidx.games.activity`.
- `GameActivity` extends `AppCompatActivity`, which `games-activity`'s POM
  does not declare as a dependency, and which refuses to start under
  anything but a `Theme.AppCompat` descendant. Both `androidx.appcompat:
  appcompat` and an AppCompat-derived theme are required or the app
  crashes on launch (a `ClassNotFoundException` if appcompat is missing
  entirely -- the unresolved superclass reads as "class not found" rather
  than a clearer error -- or an `IllegalStateException` about the theme
  once appcompat is present but the theme isn't).
- AGP 9's classic `sourceSets { ... }` DSL rejects a `Provider<Directory>`
  outright ("You cannot add Provider instances to the Android SourceSet
  API"), which is what `layout.buildDirectory.dir(...)` returns. Resolve
  it to a plain `File` first (`.get().asFile`) before passing it to
  `assets.srcDir(...)` (see `copyArosAssets` in `app/build.gradle.kts`).

### The android-activity flavor choice

`android-activity` (which both `android-activity` itself and, through it,
`winit` depend on) needs a `game-activity` or `native-activity` feature
selected somewhere in the build graph, or the crate refuses to compile at
all (`compile_error!`). Neither the root `copperline` crate nor
`copperline-android-host` selects one -- that choice belongs to whichever
app crate actually ships, so the host layer stays usable by a future
`copperline-player-android` that might choose differently. Only
`crates/copperline-android`'s `Cargo.toml` selects `game-activity`, on
both its direct `android-activity` dependency and (separately, since it's
winit's own copy of the same choice) `winit`'s `android-game-activity`
feature. A consequence: `copperline-android-host` cannot be built or
type-checked standalone on `target_os = "android"` -- only as
`copperline-android`'s path dependency.

## Gamepads (WP6): blocked on winit, not just unimplemented

`src/gamepad.rs`'s `android_backend` stub (see below) still reports no
pads connected. That's not just "not written yet" -- winit 0.30's Android
backend cannot deliver analog gamepad input at all, confirmed by reading
its source (`platform_impl/android/mod.rs`):

- `InputEvent::MotionEvent` handling only recognises touch phases
  (`Down`/`Up`/`Move`/`Cancel`) and doesn't check the event's `source()`.
  A joystick's `SOURCE_JOYSTICK` axis motion arrives as the same
  `MotionAction::Move` a touchscreen sends, so it's read through the same
  path as a touch point (`pointer.x()`/`.y()`, meant for screen
  coordinates) -- not exposed as axis data anywhere.
- There's no `DeviceEvent` axis passthrough either; `listen_device_events`
  is a stub that does nothing.
- `AndroidApp`'s native input queue is a single consumable stream, and
  `App::run_android`'s `EventLoop::builder().with_android_app(...)` hands
  it to winit exclusively -- there's no way for `crates/copperline-android`
  to also poll it independently the way `gilrs` polls raw OS APIs
  untouched by winit on desktop.

Buttons are a different story: Android `KeyEvent`s (`BUTTON_A/B/X/Y`,
`DPAD_*`, `L1`/`R1`, ...) go through winit's ordinary `KeyboardInput`
window event, `device_id` intact -- `to_physical_key` maps recognised
gamepad buttons to `PhysicalKey::Unidentified(NativeKeyCode::Android(code))`
and, notably, DPAD already maps to `KeyCode::ArrowUp/Down/Left/Right`, the
same physical keys `src/video/window.rs`'s existing keyboard-joystick
fallback (the desktop numpad stand-in) already reads. A button-only
gamepad path (D-pad + fire, no analog stick, no gilrs-style calibration)
is buildable on top of that existing mechanism without touching winit;
full WP6 (analog input, SDL-style calibration, verification against real
device pads) needs either a winit fix upstream or bypassing its event
loop for input, neither of which is done here. Caveat added once this
became clear during WP9's testing: this button/D-pad mapping is source-
verified, not device-verified -- `adb shell input keyevent`'s synthetic
events don't reach a GameActivity app's native input queue (see WP9's
section below), so there's currently no way to confirm it here short of
a real gamepad.

## Root crate feature support

Every root `Cargo.toml` feature's Android support is recorded as a comment
next to that feature, checked with `cargo check --target
aarch64-linux-android`. `rfd` (file dialogs), `arboard` (clipboard) and
`gilrs` (gamepads) have no Android backend and are target-gated out; the
`frontend` feature's call sites go through `src/host/` (file dialogs,
clipboard) or `src/gamepad.rs`'s `android_backend` module (gamepads)
instead, each backed by a stub -- see "Gamepads (WP6)" above for why that
stub isn't just an unimplemented placeholder.
