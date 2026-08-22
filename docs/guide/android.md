# Android (in progress)

Copperline's Android port is under active development. This chapter
tracks what exists today; see the Android port plan tracked alongside
this work for the full work-package sequence.

## What exists today

Two standalone crates, in the same non-workspace shape as
`crates/copperline-web` and `crates/copperline-player`:

- `crates/copperline-android-host` -- the reusable host layer (logcat
  logging today; activity lifecycle, surface handling, storage, input and
  display policy land here as their own work packages). A plain library,
  not a native lib itself, and not buildable standalone on Android -- see
  its `Cargo.toml`.
- `crates/copperline-android` -- the actual app: a `cdylib` with the
  `android_main` GameActivity entry point, depending on the host layer and
  on `copperline` itself (`frontend`, `cpu-jit`). `android_main` builds
  the machine the same way `src/main.rs` does for an ordinary run with no
  CLI flags (`Config::load_raw(None, ...)`, `emulator::build_machine`),
  then drives `copperline`'s real `App`/window loop via
  [`App::run_android`], winit's Android-specific counterpart to the
  desktop `App::run`.

...plus a minimal Gradle project at `crates/copperline-android/android/`
that packages the two into a real, installable APK.

**Verified end-to-end on a headless AVD: the emulator boots the bundled
AROS ROM and renders it live**, with Copperline's usual status bar chrome,
showing "Waiting for bootable media" (correct -- no `df0` is configured
yet; the Kickstart path is a temporary stand-in, see below). Confirmed via
`logcat`: window/surface creation, the Zorro identification board
autoconfiguring, chipset register writes -- the real machine, not a stub.

## Building and running

```sh
# 1. Build the Rust side straight into the Gradle project's jniLibs dir.
cargo ndk -t arm64-v8a --platform 35 \
  -o crates/copperline-android/android/app/src/main/jniLibs \
  build --release --manifest-path crates/copperline-android/Cargo.toml

# 2. Build and install the APK.
cd crates/copperline-android/android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk

# 3. Stage a Kickstart in the app's internal storage (temporary -- see
#    below) and launch.
adb push assets/aros/aros-amiga-m68k-rom.bin /data/local/tmp/aros-rom.bin
adb push assets/aros/aros-amiga-m68k-ext.bin /data/local/tmp/aros-ext.bin
adb shell run-as org.copperlinehq.copperline mkdir -p files/aros
adb shell run-as org.copperlinehq.copperline cp /data/local/tmp/aros-rom.bin files/aros/aros-amiga-m68k-rom.bin
adb shell run-as org.copperlinehq.copperline cp /data/local/tmp/aros-ext.bin files/aros/aros-amiga-m68k-ext.bin
adb shell am start -n org.copperlinehq.copperline/com.google.androidgamesdk.GameActivity
```

Needs an NDK (r28c or newer; `cargo-ndk` finds one via `ANDROID_NDK_HOME`
or `$ANDROID_HOME/ndk`) and, for the Gradle step, a JDK 17+. The Gradle
wrapper (`./gradlew`) downloads its own Gradle 9.5.0 and the Android
Gradle Plugin on first run; no separate Gradle install is needed.

### Kickstart path: a temporary stand-in

`copperline::romsearch::find_bundled_aros` only searches desktop-shaped
locations (exe-relative, `assets/aros` relative to the CWD), which don't
exist for an Android app. `android_main` reads instead from the app's
internal data directory (`Context.getFilesDir()`, staged above by hand),
bypassing `find_bundled_aros` entirely. Shipping the ROM as an APK asset
and extracting it on first run -- so a real install needs no `adb push`
step -- is storage work (WP5), not done yet.

### Two non-obvious Gradle/manifest pieces

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
