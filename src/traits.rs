use crate::component::Frame;
use macroquad::prelude::*;
use std::fmt::Debug;

pub trait Draw {
    fn draw(&mut self);
}

pub trait DrawWithContext {
    fn draw_with_context(&mut self, context: &mut Frame);
}

impl<T> DrawWithContext for T
where
    T: Draw,
{
    fn draw_with_context(&mut self, _context: &mut Frame) {
        self.draw();
    }
}

pub trait Update {
    fn update(&mut self) {}
}

pub trait UpdateWithContext {
    fn update_with_context(&mut self, context: &Frame);
}

impl<T> UpdateWithContext for T
where
    T: Update,
{
    fn update_with_context(&mut self, _context: &Frame) {
        self.update();
    }
}

pub trait HasSize {
    fn size(&self) -> Vec2;

    fn width(&self) -> f32 {
        self.size().x
    }

    fn height(&self) -> f32 {
        self.size().y
    }
}

pub trait HasPosition {
    fn pos(&self) -> Vec2;

    fn x(&self) -> f32 {
        self.pos().x
    }

    fn y(&self) -> f32 {
        self.pos().y
    }
}

pub trait Component: HasSize + DrawWithContext + UpdateWithContext + Debug + 'static {}

impl<T> Component for T where T: HasSize + DrawWithContext + UpdateWithContext + Debug + 'static {}
