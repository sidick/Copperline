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
  not a native lib itself.
- `crates/copperline-android` -- the actual app: a `cdylib` with the
  `android_main` GameActivity entry point, depending on the host layer and
  on `copperline` itself (`frontend`, `cpu-jit`).

...plus a minimal Gradle project at `crates/copperline-android/android/`
that packages the two into a real, installable APK. Verified end-to-end
on a headless AVD: builds, installs, launches, native code runs (confirmed
via `logcat`), and stays up rather than crashing. `android_main`
currently only sets up logging and confirms it ran -- it does not yet
drive `copperline`'s `App`/window loop, so the app currently shows a
blank black screen once launched. That's WP4 (lifecycle/surface) next.

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
adb shell am start -n org.copperlinehq.copperline/com.google.androidgamesdk.GameActivity
```

Needs an NDK (r28c or newer; `cargo-ndk` finds one via `ANDROID_NDK_HOME`
or `$ANDROID_HOME/ndk`) and, for the Gradle step, a JDK 17+. The Gradle
wrapper (`./gradlew`) downloads its own Gradle 9.5.0 and the Android
Gradle Plugin on first run; no separate Gradle install is needed.

Two non-obvious pieces the manifest and `build.gradle.kts` get right, in
case they need touching again:

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

## Root crate feature support

Every root `Cargo.toml` feature's Android support is recorded as a comment
next to that feature, checked with `cargo check --target
aarch64-linux-android`. `rfd` (file dialogs), `arboard` (clipboard) and
`gilrs` (gamepads) have no Android backend and are target-gated out; the
`frontend` feature's call sites go through `src/host/` (file dialogs,
clipboard) or `src/gamepad.rs`'s `android_backend` module (gamepads)
instead, each backed by a stub until its own work package (storage for
files, WP6 for gamepads) lands the real thing.
