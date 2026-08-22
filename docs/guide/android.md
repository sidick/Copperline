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

## Root crate feature support

Every root `Cargo.toml` feature's Android support is recorded as a comment
next to that feature, checked with `cargo check --target
aarch64-linux-android`. `rfd` (file dialogs), `arboard` (clipboard) and
`gilrs` (gamepads) have no Android backend and are target-gated out; the
`frontend` feature's call sites go through `src/host/` (file dialogs,
clipboard) or `src/gamepad.rs`'s `android_backend` module (gamepads)
instead, each backed by a stub until its own work package (storage for
files, WP6 for gamepads) lands the real thing.
