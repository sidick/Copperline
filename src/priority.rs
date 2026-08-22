// SPDX-License-Identifier: GPL-3.0-or-later

//! Best-effort host thread-scheduling priority.
//!
//! An interactive emulator has two latency-critical threads: the wall-clock
//! pacer (the main thread, where jitter shows up as frame stutter and audio
//! drift) and the audio callback (a glitch there is instantly audible). On a
//! busy desktop the OS scheduler can preempt either at the wrong moment. When
//! the user opts in (`[emulation] realtime_priority = true` or the
//! `COPPERLINE_REALTIME_PRIORITY` env var), Copperline asks the OS to schedule
//! those threads above normal.
//!
//! "Real-time priority" is portable in neither API nor semantics, so this is
//! deliberately best-effort: every call logs what it did and never fails the
//! run. The whole feature is off by default, so an unprivileged or sandboxed
//! launch behaves exactly as before.
//!
//! Per platform:
//! * **macOS** -- the pacer thread joins the `USER_INTERACTIVE` QoS class
//!   (`pthread_set_qos_class_self_np`), the idiomatic way for an app thread to
//!   ask for low-latency scheduling without elevated privileges. The audio
//!   callback is deliberately left untouched: Core Audio already runs it on a
//!   real-time thread, and pinning a QoS class onto it would only *demote* it.
//! * **Windows** -- `SetThreadPriority` raises the thread to `HIGHEST` (via
//!   the [`thread_priority`] crate); no privilege required. WASAPI's callback
//!   runs on a cpal-spawned thread, so it is raised too.
//! * **Linux / other Unix** -- raising priority needs privilege (an `rtprio`
//!   rlimit, `CAP_SYS_NICE`, or root). Without it the request is declined and
//!   the thread keeps normal scheduling; that is logged once and is not fatal.

use std::sync::atomic::{AtomicBool, Ordering};

/// Resolve whether realtime-like scheduling was requested: the
/// `COPPERLINE_REALTIME_PRIORITY` env var overrides the
/// `[emulation] realtime_priority` config for one run. A value of
/// `0`/`false`/`off`/`no` forces it off; any other value (including an empty
/// string, i.e. the bare variable) forces it on.
pub fn requested(from_config: bool) -> bool {
    match crate::envcfg::var("COPPERLINE_REALTIME_PRIORITY") {
        Some(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        None => from_config,
    }
}

/// Elevate the pacer (main) thread, which runs the emulator core and the
/// wall-clock pacer. Best effort; logs the outcome. Because the pacer sleeps
/// between work chunks rather than spinning (see
/// `Emulator::sleep_until_realtime_device_time`), even the strongest
/// scheduling class it can land in still yields the CPU and cannot starve the
/// host.
pub fn elevate_pacer_thread() {
    elevate_current_thread("pacer");
}

/// Promote the calling thread -- the cpal audio callback -- the first time it
/// runs. Idempotent across callbacks via an internal latch, so the one-time
/// scheduling syscall happens on the first callback only and steady-state
/// audio output does no extra work.
pub fn promote_audio_thread_once() {
    static PROMOTED: AtomicBool = AtomicBool::new(false);
    if PROMOTED.swap(true, Ordering::Relaxed) {
        return;
    }
    #[cfg(target_os = "macos")]
    {
        // Core Audio already hands this callback a real-time thread; joining a
        // QoS class here would only lower it, so leave it as the OS set it.
        log::info!("priority: audio thread left as-is (Core Audio runs it real-time)");
    }
    #[cfg(not(target_os = "macos"))]
    {
        elevate_current_thread("audio");
    }
}

#[cfg(target_os = "macos")]
fn elevate_current_thread(label: &str) {
    // QOS_CLASS_USER_INTERACTIVE is the highest standard QoS class and the
    // conventional way for a latency-sensitive UI/media thread to request
    // low-latency scheduling without privileges. A relative priority of 0
    // keeps the thread at the class's reference priority.
    let ret = unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0)
    };
    if ret == 0 {
        log::info!("priority: {label} thread joined the USER_INTERACTIVE QoS class");
    } else {
        log::warn!(
            "priority: could not raise {label} thread QoS \
             (pthread_set_qos_class_self_np returned {ret}); continuing at normal priority"
        );
    }
}

// No host threads to schedule on wasm32; the whole feature is a no-op there
// (the thread-priority crate is excluded from the wasm32 dependency graph).
#[cfg(target_arch = "wasm32")]
fn elevate_current_thread(label: &str) {
    log::info!("priority: {label} thread left as-is (no thread scheduling on wasm)");
}

#[cfg(not(any(target_os = "macos", target_arch = "wasm32")))]
fn elevate_current_thread(label: &str) {
    use thread_priority::{set_current_thread_priority, ThreadPriority};
    // On Windows this maps to THREAD_PRIORITY_HIGHEST (no privilege needed).
    // On Linux/other Unix (Android included) it raises to the top of the
    // current scheduling policy, which requires privilege to exceed normal;
    // without it the call returns an error that we log and shrug off.
    match set_current_thread_priority(ThreadPriority::Max) {
        Ok(()) => log::info!("priority: {label} thread elevated (ThreadPriority::Max)"),
        Err(e) => log::warn!(
            "priority: could not elevate {label} thread ({e:?}); \
             continuing at normal priority"
        ),
    }
}

/// Pin the calling thread to the SoC's highest-clocked core(s). On a
/// big.LITTLE (or big.MID.little) handheld SoC, the scheduler is free to
/// migrate the pacer thread onto a slow efficiency core under load,
/// exactly when the emulator needs the fast one most; this asks for the
/// opposite. Best effort, like the rest of this module: reads each
/// online CPU's `cpuinfo_max_freq`, pins to whichever core(s) report the
/// highest value, and logs-and-continues if that information (or the
/// pin itself) isn't available rather than failing the run.
#[cfg(target_os = "android")]
pub fn pin_to_fastest_core() {
    let Some(cpus) = fastest_cpus() else {
        log::info!("priority: core pinning skipped (no readable cpufreq info)");
        return;
    };
    // SAFETY: `set` is a plain-old-data libc type, zero-initialised before
    // use; `sched_setaffinity(0, ...)` operates on the calling thread only
    // (POSIX: pid 0 means "the calling thread"), so this can't reach outside
    // the thread that called this function.
    let ret = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in &cpus {
            libc::CPU_SET(cpu, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set)
    };
    if ret == 0 {
        log::info!("priority: pacer thread pinned to core(s) {cpus:?}");
    } else {
        log::warn!(
            "priority: sched_setaffinity failed ({}); continuing unpinned",
            std::io::Error::last_os_error()
        );
    }
}

/// The CPU indices reporting the highest `cpuinfo_max_freq` among however
/// many `/sys/devices/system/cpu/cpuN/cpufreq/cpuinfo_max_freq` files this
/// host has readable (0 if none are -- an unprivileged app on some Android
/// builds can't read them, which is why this is a `Some`/`None`, not an
/// assumption that CPU 0 exists). 32 is comfortably past any phone/tablet
/// SoC's core count.
#[cfg(target_os = "android")]
fn fastest_cpus() -> Option<Vec<usize>> {
    let freqs: Vec<(usize, u64)> = (0..32)
        .filter_map(|cpu| {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq");
            std::fs::read_to_string(&path)
                .ok()?
                .trim()
                .parse()
                .ok()
                .map(|f| (cpu, f))
        })
        .collect();
    let max = freqs.iter().map(|&(_, f)| f).max()?;
    Some(
        freqs
            .into_iter()
            .filter(|&(_, f)| f == max)
            .map(|(cpu, _)| cpu)
            .collect(),
    )
}

/// Android's thermal status, coarsened to "the OS is materially throttling
/// (or about to)" versus "running fine" -- `None` if the platform API
/// isn't available (pre-Android 11, or the call itself failed), so a
/// caller can tell "don't know" from "not throttling". Best effort: reads
/// `AThermalManager`'s status through the stable NDK `<android/thermal.h>`
/// API and never fails the run.
#[cfg(target_os = "android")]
pub fn android_thermal_throttling() -> Option<bool> {
    use std::ffi::c_void;

    #[link(name = "android")]
    extern "C" {
        fn AThermal_acquireManager() -> *mut c_void;
        fn AThermal_releaseManager(manager: *mut c_void);
        fn AThermal_getCurrentThermalStatus(manager: *mut c_void) -> i32;
    }

    // SAFETY: a null manager is checked before use; the manager is released
    // exactly once, right after the one call that needs it, so there is no
    // dangling handle for anything else to touch.
    unsafe {
        let manager = AThermal_acquireManager();
        if manager.is_null() {
            return None;
        }
        let status = AThermal_getCurrentThermalStatus(manager);
        AThermal_releaseManager(manager);
        // THERMAL_STATUS_SEVERE (3) and worse: the OS is materially
        // throttling, not just running warm. status < 0 means the call
        // itself failed.
        (status >= 0).then_some(status >= 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elevating_threads_is_always_safe() {
        // Best effort and privilege-dependent: on an unprivileged host the
        // underlying syscall may decline, but the wrappers must always return
        // cleanly rather than panic. This exercises the real macOS QoS path
        // (and the thread-priority path elsewhere) on the test thread.
        elevate_pacer_thread();
        // The audio promotion latches after its first call; both the first
        // call and the latched no-op second call must be safe.
        promote_audio_thread_once();
        promote_audio_thread_once();
    }

    #[test]
    fn requested_passes_config_through_without_env_override() {
        // `requested` only consults the config value when the env override is
        // absent. The unit suite sets no COPPERLINE_* vars, so guard on that
        // (envcfg snapshots the environment once) to keep the test hermetic.
        if crate::envcfg::var("COPPERLINE_REALTIME_PRIORITY").is_none() {
            assert!(requested(true));
            assert!(!requested(false));
        }
    }
}
