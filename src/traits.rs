use crate::engine::{Canvas, Input, Vec2};

use std::fmt::Debug;
use std::time::Duration;

/// A simulation that changes over time and can describe itself to a [`Canvas`].
///
/// [`HasSize`] supplies the extent of the simulation's own coordinate space,
/// which is what the host fits into wherever it is displayed.
pub trait Simulation: HasSize + Debug + 'static {
    /// Advances the simulation by `dt`.
    fn update(&mut self, dt: Duration, input: &Input);

    /// Describes the current state to `canvas`.
    fn draw(&mut self, canvas: &mut dyn Canvas);
}

/// An object that has a position.
pub trait HasPosition {
    fn pos(&self) -> Vec2;

    fn x(&self) -> f32 {
        self.pos().x
    }

    fn y(&self) -> f32 {
        self.pos().y
    }
}

/// An object that has a size.
pub trait HasSize {
    fn size(&self) -> Vec2;

    fn width(&self) -> f32 {
        self.size().x
    }

    fn height(&self) -> f32 {
        self.size().y
    }
}
