//! `[c3d]`: the C3D virtual 3D accelerator board's resolved configuration.
//!
//! Kept in its own file, rather than inline in [`super::Config`]'s field
//! list and [`super::validate`]'s one big `try_from`, so the whole `[c3d]`
//! surface is one field and one call at those two shared sites. See
//! `docs/internals/c3d.md`.
use super::RawC3d;
use anyhow::{anyhow, bail, Result};
use std::path::PathBuf;

/// `[c3d]` resolved: the board and its boot-time trace-capture defaults.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct C3dConfig {
    /// `[c3d] enabled`: when true, a C3D board autoconfigs on the Zorro
    /// chain. Needs a build with the `c3d` feature (off by default).
    pub enabled: bool,
    /// `[c3d] mask_caps`: `CAPS0` bits to force clear, resolved from the
    /// raw config's capability names (see [`RawC3d::mask_caps`]). `0` when
    /// unset -- the full fitted capability set.
    pub mask_caps: u32,
    /// `[c3d] trace`: the default path a `c3d.trace_start` control call
    /// falls back to when it names none of its own. `None` means that
    /// method falls all the way back to a timestamped file in the
    /// `[paths]` traces directory.
    pub trace: Option<PathBuf>,
    /// `[c3d] trace_enabled`: start a live capture to [`Self::trace`] (or
    /// the timestamped default, if unset) immediately at boot.
    pub trace_enabled: bool,
    /// `[c3d] trace_rolling_bytes`: bound [`Self::trace_enabled`]'s
    /// boot-time capture in memory instead of streaming unbounded to
    /// disk. `None` is the original unbounded behaviour.
    pub trace_rolling_bytes: Option<u64>,
    /// `[c3d] trace_trigger_texture`: defer [`Self::trace_enabled`]'s
    /// boot-time capture until this texture id is bound, instead of
    /// starting immediately.
    pub trace_trigger_texture: Option<u32>,
}

impl C3dConfig {
    /// Resolves the raw `[c3d]` table against `defaults` (the machine
    /// profile's own `[c3d]`, as every other section resolves).
    pub(crate) fn from_raw(raw: &RawC3d, defaults: &Self) -> Result<Self> {
        Ok(Self {
            enabled: raw.enabled.unwrap_or(defaults.enabled),
            mask_caps: Self::resolve_mask_caps(raw)?,
            trace: raw.trace.as_ref().map(PathBuf::from),
            trace_enabled: raw.trace_enabled.unwrap_or(false),
            trace_rolling_bytes: match raw.trace_rolling_bytes {
                Some(0) => bail!("[c3d] trace_rolling_bytes must be at least 1"),
                other => other,
            },
            trace_trigger_texture: raw.trace_trigger_texture,
        })
    }

    /// `[c3d] mask_caps`: names -> `CAPS0` bits to clear. Validated even
    /// when the board is disabled, so a typo never hides until a
    /// conformance run flips `enabled` on.
    fn resolve_mask_caps(raw: &RawC3d) -> Result<u32> {
        let mut mask = 0u32;
        for name in raw.mask_caps.as_deref().unwrap_or(&[]) {
            mask |= match name.as_str() {
                "guestmem" => 1 << 0,
                "irq" => 1 << 1,
                "transform" => 1 << 2,
                "multitexture" => 1 << 3,
                "surface-guestaddr" => 1 << 4,
                "ref-sync" => 1 << 6,
                "shared-contexts" => crate::c3d::proto::CAP_SHARED_CONTEXTS,
                other => {
                    return Err(anyhow!(
                        "[c3d] mask_caps: unknown capability \"{other}\" (expected \
                         guestmem, irq, transform, multitexture, surface-guestaddr, \
                         ref-sync or shared-contexts)"
                    ))
                }
            };
        }
        Ok(mask)
    }
}
