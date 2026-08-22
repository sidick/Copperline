// SPDX-License-Identifier: GPL-3.0-or-later

//! Stand-in for the slice of gilrs's API that `gamepad.rs` uses, so the
//! module compiles on Android without gilrs (which doesn't build there;
//! see `Cargo.toml`'s `frontend` feature).
//!
//! No pad is ever connected: `next_event` never yields, `gamepads()` is
//! always empty. Real Android gamepad input (`InputDevice`/`MotionEvent`
//! feeding this same standard layout and focus map) is WP6 in the Android
//! port plan; until then a build has no gamepad, the same as a desktop
//! build with nothing plugged in.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Button {
    South,
    North,
    East,
    West,
    Start,
    Select,
    Mode,
    LeftTrigger,
    LeftTrigger2,
    RightTrigger,
    RightTrigger2,
    DPadUp,
    DPadDown,
    DPadLeft,
    DPadRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    LeftStickX,
    LeftStickY,
    DPadX,
    DPadY,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Code(u32);

impl Code {
    pub fn into_u32(self) -> u32 {
        self.0
    }
}

/// Mirrors gilrs's `ev` submodule, whose `Code` is what `gilrs::Code` really
/// re-exports -- kept so `backend::ev::Code` resolves the same way on both
/// backends.
pub mod ev {
    pub use super::Code;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GamepadId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MappingSource {
    SdlMappings,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventType {
    Connected,
    Disconnected,
    AxisChanged(Axis, f32, Code),
    ButtonChanged(Button, f32, Code),
    ButtonPressed(Button, Code),
    ButtonReleased(Button, Code),
}

pub struct Event {
    pub id: GamepadId,
    pub event: EventType,
}

/// Always the "no pad" answer; there is never a real one behind an id.
pub struct Gamepad;

impl Gamepad {
    pub fn name(&self) -> &str {
        ""
    }

    pub fn uuid(&self) -> [u8; 16] {
        [0; 16]
    }

    pub fn mapping_source(&self) -> MappingSource {
        MappingSource::None
    }

    pub fn button_code(&self, _button: Button) -> Option<Code> {
        None
    }
}

pub struct Gilrs;

impl Gilrs {
    pub fn next_event(&mut self) -> Option<Event> {
        None
    }

    pub fn gamepad(&self, _id: GamepadId) -> Gamepad {
        Gamepad
    }

    pub fn gamepads(&self) -> std::iter::Empty<(GamepadId, Gamepad)> {
        std::iter::empty()
    }
}

#[derive(Default)]
pub struct GilrsBuilder;

impl GilrsBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn add_included_mappings(self, _enabled: bool) -> Self {
        self
    }

    pub fn add_env_mappings(self, _enabled: bool) -> Self {
        self
    }

    pub fn build(self) -> Result<Gilrs, GilrsError> {
        Ok(Gilrs)
    }
}

#[derive(Debug)]
pub struct GilrsError;

impl fmt::Display for GilrsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gamepad input is not implemented on Android yet")
    }
}

impl std::error::Error for GilrsError {}
