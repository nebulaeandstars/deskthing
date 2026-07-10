use crate::traits::*;
use crate::Draw;

use macroquad::prelude::*;
use macroquad::window;

#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub parent: Option<Frame>,
    pub frame: Frame,
    x_percent: f32,
    y_percent: f32,
    width_percent: f32,
    height_percent: f32,
}

#[allow(unused)]
impl Layout {
    pub fn new(
        parent: Option<Frame>,
        x_percent: f32,
        y_percent: f32,
        width_percent: f32,
        height_percent: f32,
    ) -> Self {
        let frame = Frame::relative(
            parent.unwrap_or_default(),
            x_percent,
            y_percent,
            width_percent,
            height_percent,
        );

        Self {
            parent,
            frame,
            x_percent,
            y_percent,
            width_percent,
            height_percent,
        }
    }

    pub fn without_parent(
        x_percent: f32,
        y_percent: f32,
        width_percent: f32,
        height_percent: f32,
    ) -> Self {
        Self::new(None, x_percent, y_percent, width_percent, height_percent)
    }

    pub fn with_parent(
        parent: Frame,
        x_percent: f32,
        y_percent: f32,
        width_percent: f32,
        height_percent: f32,
    ) -> Self {
        Self::new(
            Some(parent),
            x_percent,
            y_percent,
            width_percent,
            height_percent,
        )
    }

    pub fn refresh(&mut self) {
        self.frame = Frame::relative(
            self.parent.unwrap_or_default(),
            self.x_percent,
            self.y_percent,
            self.width_percent,
            self.height_percent,
        );
    }

    pub fn update_parent(&mut self, parent: Frame) {
        self.parent = Some(parent);
        self.refresh();
    }
}

/// A sub-window containing multiple components.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pos: Vec2,
    size: Vec2,
}

impl Frame {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            pos: Vec2::new(x, y),
            size: Vec2::new(width, height),
        }
    }

    pub fn relative(
        frame: Frame,
        x_percent: f32,
        y_percent: f32,
        width_percent: f32,
        height_percent: f32,
    ) -> Self {
        let x = frame.pos.x + x_percent * frame.width();
        let y = frame.pos.y + y_percent * frame.height();
        let width = width_percent * frame.width();
        let height = height_percent * frame.height();
        Self::new(x, y, width, height)
    }

    pub fn x(&self) -> f32 {
        self.pos.x
    }

    pub fn y(&self) -> f32 {
        self.pos.y
    }

    pub fn pos(&self) -> Vec2 {
        self.pos
    }

    pub fn width(&self) -> f32 {
        self.size.x
    }

    pub fn height(&self) -> f32 {
        self.size.y
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new(0., 0., window::screen_width(), window::screen_height())
    }
}

pub struct FrameOutline {
    layout: Layout,
    thickness: f32,
    color: Color,
}

impl FrameOutline {
    pub fn new(layout: Layout, thickness: f32, color: Color) -> Self {
        Self {
            thickness,
            color,
            layout,
        }
    }
}

impl Draw for FrameOutline {
    fn draw(&self) {
        draw_rectangle_lines(
            self.layout.frame.x(),
            self.layout.frame.y(),
            self.layout.frame.width(),
            self.layout.frame.height(),
            self.thickness,
            self.color,
        );
    }
}

impl Update for FrameOutline {
    fn update(&mut self) {
        self.layout.refresh();
    }
}
