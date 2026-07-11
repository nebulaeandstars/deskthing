use crate::component::Frame;
use macroquad::prelude::*;
use std::fmt::Debug;

pub trait Draw {
    fn draw(&self, frame: &mut Frame);
}

pub trait Update {
    fn update(&mut self, _frame: &Frame) {}
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

pub trait Component: HasSize + Draw + Update + Debug + 'static {}

impl<T> Component for T where T: HasSize + Draw + Update + Debug + 'static {}
