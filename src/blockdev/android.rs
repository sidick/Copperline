// SPDX-License-Identifier: GPL-3.0-or-later

//! Android block devices: there are none to offer.
//!
//! Raw access to whole host media needs a privilege escalation path (see the
//! module docs in [`super`]) that Android simply does not grant an
//! unprivileged app -- there is no `authopen`-equivalent broker, and no
//! `/dev/sdX`-style node an app can be handed. A guest's storage on Android
//! comes from the Storage Access Framework instead (see `docs/guide/android.md`
//! once it lands), which is a filesystem tree, not a block device, so it goes
//! through `host-disks` rather than this module. Enumeration always succeeds
//! with an empty list -- "no such devices here" is the honest answer, not an
//! error -- and taking/lending one is unreachable because nothing is ever
//! offered to take.

use super::HostDevice;
use anyhow::{bail, Result};

pub fn list_devices() -> Result<Vec<HostDevice>> {
    Ok(Vec::new())
}

pub(super) fn taking_needs_privilege() -> bool {
    true
}

pub(super) struct Held;

pub(super) fn take_disks(_wanted: &[(HostDevice, bool)]) -> Result<Vec<(String, Held)>> {
    bail!("raw host block devices are not available on Android")
}

pub(super) fn lend(_device: &HostDevice, _write: bool, _held: &Held) -> Result<super::BlockDevice> {
    bail!("raw host block devices are not available on Android")
}
