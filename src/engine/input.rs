use super::Vec2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// A snapshot of user input for one update.
///
/// Simulations are handed one of these rather than polling the host, so a test
/// can drive them by constructing the exact input it wants.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Input {
    /// Cursor position in the simulation's own coordinate space.
    pub mouse_pos: Vec2,
    pub left_down: bool,
    pub right_down: bool,
    pub middle_down: bool,
}

impl Input {
    /// No buttons held, cursor at the origin.
    pub fn none() -> Self {
        Self::default()
    }

    /// No buttons held, cursor at `mouse_pos`.
    #[must_use]
    pub fn at(mouse_pos: Vec2) -> Self {
        Self {
            mouse_pos,
            ..Self::default()
        }
    }

    /// The same input with `button` held down.
    #[must_use]
    pub fn with_held(self, button: MouseButton) -> Self {
        match button {
            MouseButton::Left => Self {
                left_down: true,
                ..self
            },
            MouseButton::Right => Self {
                right_down: true,
                ..self
            },
            MouseButton::Middle => Self {
                middle_down: true,
                ..self
            },
        }
    }

    pub fn is_down(&self, button: MouseButton) -> bool {
        match button {
            MouseButton::Left => self.left_down,
            MouseButton::Right => self.right_down,
            MouseButton::Middle => self.middle_down,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::vec2;
    use super::*;

    #[test]
    fn default_input_has_nothing_held() {
        let input = Input::none();

        assert!(!input.is_down(MouseButton::Left));
        assert!(!input.is_down(MouseButton::Right));
        assert!(!input.is_down(MouseButton::Middle));
    }

    #[test]
    fn with_held_sets_only_that_button() {
        let input = Input::at(vec2(3., 4.)).with_held(MouseButton::Right);

        assert_eq!(vec2(3., 4.), input.mouse_pos);
        assert!(input.is_down(MouseButton::Right));
        assert!(!input.is_down(MouseButton::Left));
    }
}
