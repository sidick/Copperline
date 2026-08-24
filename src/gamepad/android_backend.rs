// SPDX-License-Identifier: GPL-3.0-or-later

//! Stand-in for the slice of gilrs's API that `gamepad.rs` uses, so the
//! module compiles on Android without gilrs (which doesn't build there;
//! see `Cargo.toml`'s `frontend` feature).
//!
//! Real gamepad input, digital buttons only (WP6's v1 -- see
//! docs/guide/android.md's "Full WP6 implementation plan" for why analog
//! axes are deferred: winit's Android backend already exclusively drains
//! and finishes AndroidApp's input queue every pump, including joystick
//! motion it discards, so there is no gap for a second reader to recover
//! anything from). D-pad and face/shoulder button `KeyEvent`s already
//! reach `src/video/window.rs`'s ordinary `WindowEvent::KeyboardInput`
//! handling (D-pad as real winit `KeyCode::ArrowUp/Down/Left/Right`,
//! buttons as `PhysicalKey::Unidentified(NativeKeyCode::Android(code))`);
//! `push_button` below is how that handler feeds a decoded press/release
//! into this module's synthetic single-pad queue, which `Gilrs::next_event`
//! then drains the same way real gilrs's would. `gamepad.rs`'s own
//! `MappedPadState::resolve_pad` (the "standard layout": South=fire,
//! East=blue, West=green, North=yellow, Start=play, shoulders/triggers=
//! rewind/forward, Select/Mode=host Menu) does the rest unmodified --
//! this module only needs to get the right `Button` variant pushed per
//! Android key code, not reimplement any Amiga-side mapping.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Mutex;

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

/// One fixed synthetic pad, "connected" the first time an Android gamepad
/// key event arrives. There is only ever one -- Android's `InputManager`
/// enumeration (WP6, separately) is what would tell multiple devices
/// apart; this module doesn't need to yet, since `gamepad.rs`'s
/// `RawGamepads::first_gamepad` only ever drives the first one anyway.
const PAD_ID: GamepadId = GamepadId(0);

struct State {
    connected: bool,
    queue: VecDeque<Event>,
}

static STATE: Mutex<State> = Mutex::new(State {
    connected: false,
    queue: VecDeque::new(),
});

/// Feed a decoded button press/release from Android's key event stream
/// (`src/video/window.rs`'s `WindowEvent::KeyboardInput` handling) into
/// the synthetic pad. Connects the pad on its first call.
pub fn push_button(button: Button, pressed: bool) {
    let mut state = STATE.lock().unwrap();
    if !state.connected {
        state.connected = true;
        state.queue.push_back(Event {
            id: PAD_ID,
            event: EventType::Connected,
        });
    }
    let code = Code(button as u32);
    let event = if pressed {
        EventType::ButtonPressed(button, code)
    } else {
        EventType::ButtonReleased(button, code)
    };
    state.queue.push_back(Event { id: PAD_ID, event });
}

/// Always reports "recognised by the controller database" once connected,
/// so `GamepadReader::poll` takes the calibration-free "standard layout"
/// path (`gamepad.rs`'s `None if mapped => ...`) rather than demanding the
/// user run `--calibrate-gamepad` for a pad that was never plugged in in
/// the USB/Bluetooth sense gilrs's calibration flow assumes.
pub struct Gamepad {
    connected: bool,
}

impl Gamepad {
    pub fn name(&self) -> &str {
        if self.connected {
            "Android Gamepad"
        } else {
            ""
        }
    }

    pub fn uuid(&self) -> [u8; 16] {
        // Fixed and arbitrary: there is only ever one synthetic pad, so a
        // stable non-zero UUID is enough to give it its own calibration
        // slot if a user's [[input]] setup ever wants one, without
        // colliding with a real gilrs UUID's all-zero unset convention.
        *b"CopperlineAndrPd"
    }

    pub fn mapping_source(&self) -> MappingSource {
        if self.connected {
            MappingSource::SdlMappings
        } else {
            MappingSource::None
        }
    }

    pub fn button_code(&self, _button: Button) -> Option<Code> {
        // Only consulted by `collapsed_dpad_half_axis`'s SDL half-axis
        // d-pad workaround, which is meaningless here: every button push
        // already carries its own final direction, never a shared axis
        // code two buttons could collide on.
        None
    }
}

pub struct Gilrs;

impl Gilrs {
    pub fn next_event(&mut self) -> Option<Event> {
        STATE.lock().unwrap().queue.pop_front()
    }

    pub fn gamepad(&self, _id: GamepadId) -> Gamepad {
        Gamepad {
            connected: STATE.lock().unwrap().connected,
        }
    }

    pub fn gamepads(&self) -> std::vec::IntoIter<(GamepadId, Gamepad)> {
        let connected = STATE.lock().unwrap().connected;
        if connected {
            vec![(PAD_ID, Gamepad { connected: true })].into_iter()
        } else {
            Vec::new().into_iter()
        }
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
