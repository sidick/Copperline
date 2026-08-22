# Android (in progress)

Copperline's Android port is under active development and does not yet
produce an installable emulator. This chapter tracks what exists today;
see the Android port plan tracked alongside this work for the full
work-package sequence.

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

Both build and link for real with [`cargo-ndk`](https://github.com/bbqsrc/cargo-ndk):

```sh
cargo ndk -t arm64-v8a --platform 35 -o out build --release
```

against Android NDK r28c or newer (`ANDROID_NDK_HOME` set, or `cargo-ndk`
finds one via `$ANDROID_HOME/ndk`). `android_main` currently only sets up
logging and confirms it ran -- it does not yet drive `copperline`'s
`App`/window loop.

## What doesn't exist yet

There is no Gradle project in this repo, so there is no way to package
either crate's `.so` into a real, installable APK yet: GameActivity needs
the `androidx.games:games-activity` Java/Kotlin glue on the classpath,
which only a Gradle build (or hand-assembling a matching AAR/dex, which is
not maintainable) can provide. The load-and-run pipeline itself --
`cargo-ndk` build, package, `adb install`, launch, confirm via `logcat` --
has been verified end-to-end using a throwaway `NativeActivity` stand-in
(the framework's built-in activity class, which needs no Java glue), not
committed to the repo. Setting up the real Gradle/GameActivity packaging
is tracked as part of the Android port's WP3.

## Root crate feature support

Every root `Cargo.toml` feature's Android support is recorded as a comment
next to that feature, checked with `cargo check --target
aarch64-linux-android`. `rfd` (file dialogs), `arboard` (clipboard) and
`gilrs` (gamepads) have no Android backend and are target-gated out; the
`frontend` feature's call sites go through `src/host/` (file dialogs,
clipboard) or `src/gamepad.rs`'s `android_backend` module (gamepads)
instead, each backed by a stub until its own work package (storage for
files, WP6 for gamepads) lands the real thing.
