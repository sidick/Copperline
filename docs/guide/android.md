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

## WP9 (full mode): keyboard input confirmed, one real bug found and fixed

The earlier draft of this section said WP9's keyboard-mapping analysis
was correct but empirically unverifiable, because every `adb shell input
keyevent` was getting intercepted by GameActivity's bundled GameTextInput
layer before reaching the app. The actual cause turned out to be
simpler and specific to this AVD: **`copperline-test` was created with
`hw.keyboard=no`** (no virtual hardware keyboard device at all), so
Android had nothing to attribute a "hardware" key event to and fell back
to routing synthetic input through the IME/text-input path instead.
Setting `hw.keyboard=yes` in the AVD's `config.ini` and restarting fixes
this outright -- `adb shell input keyevent` then reaches the app as a
genuine `WindowEvent::KeyboardInput` with correct `physical_key`/
`logical_key`, confirmed directly by a temporary diagnostic log on real
key presses (`ArrowDown` for `KEYCODE_DPAD_DOWN`, `AltLeft`/`KeyE` for
Alt+E, etc.) before it was removed again. This retroactively confirms
WP6's D-pad-button claim too, which carried the same "source-verified,
not device-verified" caveat for the same reason.

**With that fixed, one real gap surfaced: winit's Android backend never
emits `WindowEvent::ModifiersChanged` at all** (there is no such event
anywhere in `platform_impl/android/mod.rs`). Every individual key press
maps correctly, but `self.modifiers` never updated, so every host
shortcut that gates on it silently never fired -- Alt+E for the menu,
mouse-capture toggle, all of them -- even though the keys making them up
arrived perfectly. Fixed with `App::track_android_modifier_key`: on
Android only, the four modifier keys' press/release are watched directly
in the ordinary `KeyboardInput` handler and fed through the same
`update_host_modifiers` every other platform's `ModifiersChanged` already
drives, so nothing downstream needed to change. Verified end-to-end on
the AVD: Alt+E now opens Copperline's full pop-up menu (Machine
Configuration, Audio/Video/Input Settings, Debugger, and the rest),
confirmed by screenshot, which it did not before this fix.

**Pointer capture is blocked on the identical root cause as WP6's analog
gamepad input**, not a separate gap: `ActiveEventLoop::set_cursor_grab`
is a hard `Err(NotSupported)` on Android in winit (no capture API wired
up at all), and even past that, a real mouse's relative motion would
arrive as a `MotionEvent` and get misrouted through winit's touch-only
handling exactly like a joystick axis does -- confirmed, not just read
from source, this time: a screen tap now visibly logs as
`WindowEvent::Touch { phase: Started/Ended, location: ... }` with the
exact coordinates tapped, so touch itself reaches the app correctly --
the `MotionEvent`-only-produces-Touch limitation is real, not a
figment of unverified source-reading. A JNI `View.requestPointerCapture()`
shim (as the proposal describes) would still hit this same wall on the
Rust side.

**`WindowEvent::Touch` has no handler anywhere in
`src/video/window.rs`** -- confirmed by the same tap-and-log test:
the event arrives correctly, Copperline's own code just does nothing
with it, so tapping the screen has no effect. Not a WP9 gap as such (the
proposal scopes touch-first UI out of the whole port, on the premise
that handheld mode is gamepad/D-pad-navigated via the existing
controller-walkable UI in `src/video/window/app_nav.rs`, which reads
`KeyCode::ArrowUp/Down/Left/Right` -- now confirmed to receive Android's
D-pad correctly, same fix as above). Worth knowing regardless, since a
real user's first instinct is often to tap the screen.

**Dead end, for the record so it isn't retried:** the Android emulator
console's `event send` command (`adb emu event send EV_KEY:...`), which
injects at the kernel/HID level and would in principle sidestep any
framework-level routing question entirely, does not deliver events on
this AVD/system-image combination at all -- `getevent -lt` on every
input device while injecting shows nothing arriving. Likely a
virtio-input vs. legacy goldfish-events incompatibility in this image;
`hw.keyboard=yes` plus ordinary `adb shell input` is the fix that
actually works, not this.

**Not attempted:** runtime handheld/full mode switching (`Configuration.
keyboard`/`InputDevice` sources) has no second UI layout to switch into
-- this session built one UI, not the proposal's two presentation modes
-- so there's nothing to detect *for* yet. DeX/multi-window needs a
DeX-capable device or a resizable-emulator configuration neither of which
exist here.

### The controller-walkable UI works end to end, D-pad-driven

With the `hw.keyboard=yes` fix and the modifier-tracking fix above,
Copperline's existing controller-walkable menu system
(`src/video/window/app_nav.rs`) was tested for real, not just read from
source, and it holds up completely -- this is the actual mechanism WP6's
proposed "handheld, gamepad-only" mode depends on, and it needed no
Android-specific code at all:

1. Alt+E opens the pop-up menu (About highlighted by default).
2. Three `KEYCODE_DPAD_UP` presses move the highlight exactly three
   items, About → Keyboard Shortcuts → Load Kickstart ROM → Save State --
   confirmed by screenshot.
3. Enter drills into the highlighted item's submenu (Save State's Quick
   Save / Quick Load / Save State... / Load State..., Quick Save
   highlighted by default) -- confirmed by screenshot.
4. Escape backs out of the submenu, a second Escape closes the menu
   entirely, back to the running machine with no glitch and no crash.

Since `KEYCODE_DPAD_*` and Android gamepad button keycodes go through
the exact same `WindowEvent::KeyboardInput` path as the hardware-keyboard
keys just confirmed above (see WP6's section), this is as close to
confirming "a gamepad can drive the whole UI" as this environment gets
without an actual gamepad -- the remaining gap is specifically the
analog stick (still blocked on winit), not the digital D-pad/button path
this test exercised.

## Storage (WP5): the app's own external-files directory, auto-mounted

Without a config file, `cfg.filesys` starts empty on every launch -- there
is no desktop-shaped `./copperline.toml` on Android for a `[[filesys]]`
entry to live in, so without this the guest would have no host storage at
all: no way to hand it a real Kickstart, a WHDLoad game, or get a save
file back out.

`android_main` (`crates/copperline-android/src/lib.rs`) auto-mounts
`AndroidApp::external_data_path()` -- `Context.getExternalFilesDir(null)`,
a real POSIX path under `/sdcard/Android/data/<pkg>/files` -- as `HOSTFS0:
-> ANDROID:` whenever `cfg.filesys` is empty, `boot_pri = -128` (mounted,
but never a boot candidate, matching the config default: an empty
directory with no `C:`/`S:`/`Devs:` is not something AROS can usefully
boot from). No runtime permission dialog and no SAF picker is needed for
this directory specifically -- it's the one location Android grants an
app without asking, visible from a connected computer or a file manager
with "show Android/data" enabled, which is enough to drop a Kickstart ROM
or a WHDLoad `.lha` in before a session and pull a save back out after.

This is deliberately the narrow half of WP5, not the whole thing. The
wider case -- browsing to any directory the user wants, e.g. an existing
WHDLoad library or Kickstart collection kept elsewhere -- needs the
Storage Access Framework: a Kotlin `GameActivity` subclass to launch
`ACTION_OPEN_DOCUMENT_TREE` and receive its `onActivityResult`, since
there's no pure-NDK path to a picker result, plus a URI/fd-based
`filesys.rs` backend, since SAF hands back a `content://` document tree
rather than a path the existing directory mount can open with `std::fs`.
Neither is attempted here.

Verified end-to-end on the AVD: `logcat` confirms both the host-side
mount (`storage: mounting /storage/emulated/0/.../files as HOSTFS0:`) and
the guest-side one (`filesys: HOSTFS0: handler started (ANDROID: ->
...)`, logged by the filesys board itself when the guest's handler
process actually starts) -- and, with `boot_pri` temporarily raised for
the test, AROS booted directly off it into a shell with no other media
inserted, proof DOS actually mounted and recognised the volume as a valid
bootable AmigaDOS device, not just that the host-side config plumbing ran.
`boot_pri` reverted to `-128` afterwards, the real, permanent value: an
auto-mounted, otherwise-empty support directory should never be handed to
the boot-device race.

### A user's own Kickstart

`android_main` also looks in `external/kickstart/` (a fixed, discoverable
subfolder of the same `ANDROID:` directory above, not its root -- so it
reads as "put your ROM here" rather than mixing with WHDLoad games or
anything else parked in general storage) for a real Kickstart to boot
instead of the bundled AROS ROM. Identification is by content
(`copperline::romdb::describe_file`), the same convention the desktop's
own asset-gated tests use (see `tests/README.md`) -- never by file name,
so it does not matter what a dumper or a collection manager happened to
call the file. The folder is created (empty) on every launch if it does
not exist yet, so there is somewhere to find by browsing over USB or a
file manager even before a ROM is ever added.

`find_user_kickstart` sorts the folder's entries for a deterministic
pick, identifies each with `describe_file`, and takes the first
recognised file as `cfg.rom_path`; a second recognised file (if any)
becomes `cfg.extended_rom_path` (the CDTV/CD32 second flash bank); a
third and beyond are logged and ignored. An Amiga Forever encrypted image
is named in the log but never selected -- decoding it needs a `rom.key`
this path does not look for. An empty or all-unrecognised folder falls
back to the bundled AROS ROM already set above it, logged either way, so
a run's `logcat` always says which ROM actually booted and why.

Verified on the AVD: `adb push`ing a real Kickstart 3.1 image into
`external/kickstart/` and relaunching logs `kickstart: .../KICK31.ROM
identified as Kickstart 3.1 (40.63) A500/A600/A2000` followed by
`kickstart: booting the user-supplied ROM at ...`, and a screenshot
confirms the real Kickstart boot banner ("AMIGA ROM Operating System and
Libraries... Copyright (C) 1985-1993 Commodore-Amiga, Inc.") in place of
AROS's own -- a real, non-AROS machine, not just the log line claiming
one. Removing the file and relaunching falls back to AROS again, logged
as `kickstart: no recognised ROM in external/kickstart/; booting the
bundled AROS ROM`.

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
loop for input, neither of which is done here. This D-pad/button mapping
is now device-confirmed, not just source-verified: see WP9's section
below for how (the AVD needed `hw.keyboard=yes`, which it didn't have
at first) -- a real analog gamepad is still needed to test anything
beyond digital buttons, since that's the actual blocked part.

### Full WP6 implementation plan

**Enumeration and hotplug can be built now, independently of the winit
blocker below.** `InputManager.getInputDeviceIds()`/`InputDeviceListener`
is a system-service query, not a read off `AndroidApp`'s native input
queue -- it doesn't compete with winit for that single consumable stream
the way motion/button *event delivery* does, so wiring device
appearance/removal into port assignment doesn't need anything below
resolved first.

**Analog axis delivery is genuinely blocked, not just missing an
Activity subclass -- and deferred rather than solved here.**
`android-activity`'s input queue enforces single-consumer access at
runtime: a second, independent call to `input_events_iter()` while
winit's own is outstanding returns `Err(InputUnavailable)`
(`android-activity` 0.6.1's `game_activity/mod.rs`, the `Weak`/`Arc`
strong-count check backing `input_receiver`). Worse, there's no "leave
it for someone else" gap to exploit even if that weren't true: winit's
Android backend (`platform_impl/android/mod.rs`, `winit` 0.30.13) fully
drains the queue every pump and marks *every* event `Handled` before
moving on, including joystick axis motion it doesn't recognise --
`input_status` starts `Handled` and is never downgraded for the motion
branch, so the axis payload is read, acknowledged, and thrown away in
the same step, not left pending. A second reader alongside winit
doesn't get "what winit left behind"; there is no window where anything
is left behind.

The only architecturally sound fix is to stop winit from being the
queue's sole consumer at all -- either fork/patch winit's Android
backend to forward joystick-class events instead of discarding them, or
replace its Android input handling entirely and reimplement
touch/key/modifier handling on our own side (the same handling WP9
already device-confirmed working: Alt+E menu, D-pad-as-arrow-keys,
modifier tracking). Either path is real work against code that
currently works correctly, for a payoff nothing currently needs.
**Decision: defer.** Ship the digital D-pad/button-only path (already
device-confirmed, see above) as WP6's v1; revisit true analog only if a
specific target title needs it. WP9's pointer-capture gap is blocked on
the identical root cause and is deferred for the same reason.

**Consequence: gamepad-driven mouse mode is blocked too, not just
analog joystick mode.** The natural design for Mouse mode -- right
stick drives the cursor proportionally -- needs exactly the axis data
that's unreachable, so it's deferred alongside analog joystick rather
than shippable in a v1 that only has digital D-pad/button input. There
is no digital substitute that isn't materially worse than a real analog
stick for pointer control (D-pad-to-cursor is coarse and slow compared
to touchscreen or physical mouse). Practically, that leaves the
Amiga-side pointer driven by the touchscreen or a real Bluetooth/USB
mouse for now, not the gamepad -- worth keeping in mind for any UI that
assumes a gamepad can always drive every port mode.

**What's still buildable for a v1 given the deferral**: Joystick mode
(D-pad + two buttons, matching `JoystickState`, `src/gamepad.rs:147`)
and CD32 pad mode (same directional input plus
`green`/`yellow`/`play`/`rwd`/`ffw`, `PadState`, `src/gamepad.rs:165`)
are both reachable today through the existing button path -- Android
`KeyEvent`s for D-pad and face/shoulder buttons already arrive via
winit's ordinary `KeyboardInput` (device-confirmed, see above and
WP9's section) -- feeding the same pipeline buttons already use:
`apply_joystick_state` (`src/video/window.rs:2546`) ->
`Bus::input.set_joystick`/`set_cd32_buttons` (`src/bus.rs:2331,2357`),
replacing `android_backend`'s stub (`src/gamepad/android_backend.rs`).
Enumeration/hotplug via `InputManager` (above) is unaffected by any of
this and can be built alongside it.

Recommended shape for the digital-only v1:

- **Source filtering**: `(device.getSources() & InputDevice.SOURCE_CLASS_JOYSTICK)
  != 0` -- a bitmask test, not `==` (the same trap
  [SDL hit](https://github.com/libsdl-org/SDL/issues/2718)) -- still
  matters for identifying gamepad-class devices even with buttons only.
- **A small logical-action enum** (`DirUp/Down/Left/Right`, `Fire1/Fire2`,
  `Cd32Red/Blue/Yellow/Green`, `Cd32ShoulderL/R`, `MenuToggle`) sits
  between raw `KeyEvent.KEYCODE_BUTTON_*`/`KEYCODE_DPAD_*` codes and
  `JoystickState`/`PadState`, driven by a per-device-class default
  table rather than hardcoded keycodes -- Android gamepads don't agree
  on button codes across vendors, and a table means a new controller's
  quirks are a data edit later, not a code change. `MouseLeft/Right`
  drop out of this enum for now along with Mouse mode itself. One
  default table covering the standard Android gamepad profile is
  enough to start.
- **`MenuToggle`** defaults to `KEYCODE_BUTTON_MODE` (the nearest thing
  to a platform convention for a guide/home-style button), with `Start`
  long-press as a fallback -- some OEM skins intercept `BUTTON_MODE` for
  a system overlay before it reaches the app, so this needs verifying
  against real hardware early rather than assumed.

**Explicitly out of scope for a first pass**: analog joystick input and
gamepad-driven Mouse mode (deferred above, not forgotten -- revisit if
a target title needs either); SDL2-style remappable controller config
or a community mapping database; force feedback/rumble
(`InputDevice.getVibrator()`/`VibratorManager` device coverage is
patchy, API 31+ only); Amiberry's Wheel Mouse, Analog joystick, and CDTV
remote port modes (low value relative to effort); "Default" isn't a
mode to replicate, it's Amiberry's autodetect UX.

**Open questions**: which of Joystick/CD32 pad mode matters first for a
real target title (affects how much of the button set needs mapping);
known controllers to validate the digital path against (Bluetooth
Xbox/PlayStation pads, USB-OTG generic pads -- device/vendor
fragmentation here is a real testing cost, not just a coding one); and
whether `KEYCODE_BUTTON_MODE` actually reaches the app on target
devices before relying on it as the default binding.

### Digital v1 implemented, source-verified only -- device confirmation
### blocked by an unrelated pre-existing crash

The digital-only plan above is implemented: `src/gamepad/android_backend.rs`
now holds a real (if minimal) synthetic single-pad queue instead of the
permanent no-op stub, fed by two new `src/video/window.rs` methods --
`track_android_gamepad_dpad` (D-pad, hooked into the existing
`PhysicalKey::Code` `KeyboardInput` arm alongside
`track_android_modifier_key`) and `handle_android_gamepad_button`
(face/shoulder buttons, a new `PhysicalKey::Unidentified` arm, since
winit has no typed `KeyCode` for them). Both call
`android_backend::push_button`, which queues a `Connected` event on
first use and then `ButtonPressed`/`ButtonReleased`, exactly the shape
`gamepad.rs`'s existing `RawGamepads::pump`/`GamepadReader::poll`
already expects from real gilrs -- no changes needed there.
`Gamepad::mapping_source()` reports `SdlMappings` once connected so
`poll()` takes its calibration-free "standard layout" path
(`MappedPadState::resolve_pad`, already written, already used by every
real desktop pad): South=fire/red, East=blue, West=green, North=yellow,
Start=play/pause, shoulders/triggers=rewind/forward, Select/Mode=host
Menu -- which also means `KEYCODE_BUTTON_MODE` opening Copperline's
menu (one of the open questions above) comes for free rather than
needing its own binding.

Verified: compiles clean on both `cargo check`/`clippy -D warnings`
(desktop) and `cargo ndk ... clippy -- -D warnings` (`aarch64-linux-android`,
via `cargo-ndk` against NDK r28.2.13676358); the existing desktop
`gamepad`/`video::window` unit test suite (23 tests) passes unchanged,
confirming nothing in the shared pipeline regressed. **Not yet
device-confirmed**: launching the APK on the `copperline-test` AVD hits
a reproducible `SIGSEGV` in the `android_main` thread immediately after
"display: N panel pixels, defaulting to the CRT shader" and before any
input ever reaches the app -- confirmed to be pre-existing and unrelated
to this work by reproducing the identical crash (same fault address) on
the unmodified stub via `git stash`. This blocks *all* Android testing
on this AVD right now, not just gamepad input, and is worth its own
investigation (a shader/Vulkan init issue on this particular AVD image
is the leading suspect, given where it happens) before anything Android
-side can be device-confirmed again.

## Root crate feature support

Every root `Cargo.toml` feature's Android support is recorded as a comment
next to that feature, checked with `cargo check --target
aarch64-linux-android`. `rfd` (file dialogs), `arboard` (clipboard) and
`gilrs` (gamepads) have no Android backend and are target-gated out; the
`frontend` feature's call sites go through `src/host/` (file dialogs,
clipboard) or `src/gamepad.rs`'s `android_backend` module (gamepads)
instead, each backed by a stub -- see "Gamepads (WP6)" above for why that
stub isn't just an unimplemented placeholder.
