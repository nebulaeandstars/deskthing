//! Host-side layout and presentation.
//!
//! This is where a [`Simulation`] meets the screen: it owns the render target
//! the simulation draws into, the canvas it draws through, and the arithmetic
//! that fits it into a rectangle. The simulation knows none of it.

use crate::engine::{Input, MacroquadCanvas};
use crate::traits::{HasPosition, HasSize, Simulation};

use macroquad::prelude as mq;
use macroquad::prelude::{Vec2, vec2};
use std::fmt::Debug;
use std::time::Duration;

#[derive(Debug)]
pub struct ComponentFrame {
    component: Box<dyn Simulation>,
    frame: Frame,
    canvas: MacroquadCanvas,
}

#[allow(unused)]
impl ComponentFrame {
    pub fn new<T: Simulation>(component: T, pos: Vec2, size: Vec2) -> Self {
        let frame = Frame::new(vec2(component.width(), component.height()), pos, size);
        let canvas = MacroquadCanvas::new(&frame.camera);

        Self {
            component: Box::new(component),
            frame,
            canvas,
        }
    }

    pub fn relative<T: Simulation>(
        component: T,
        parent_pos: Vec2,
        parent_size: Vec2,
        relative_frame_pos: Vec2,
        relative_frame_size: Vec2,
    ) -> Self {
        let component_size = vec2(component.width(), component.height());

        let frame = Frame::relative_to(
            parent_pos,
            parent_size,
            component_size,
            relative_frame_pos,
            relative_frame_size,
        );
        let canvas = MacroquadCanvas::new(&frame.camera);

        Self {
            component: Box::new(component),
            frame,
            canvas,
        }
    }

    pub fn relative_to_screen<T: Simulation>(
        component: T,
        relative_frame_pos: Vec2,
        relative_frame_size: Vec2,
    ) -> Self {
        Self::relative(
            component,
            vec2(0., 0.),
            vec2(mq::screen_width(), mq::screen_height()),
            relative_frame_pos,
            relative_frame_size,
        )
    }

    pub fn refit_to(
        &mut self,
        parent_pos: Vec2,
        parent_size: Vec2,
        relative_frame_pos: Vec2,
        relative_frame_size: Vec2,
    ) {
        self.frame.refit_to(
            parent_pos,
            parent_size,
            relative_frame_pos,
            relative_frame_size,
        );
    }

    pub fn refit_to_component(&mut self) {
        let current_aspect_ratio = self.width() / self.height();
        let new_aspect_ratio = self.component.width() / self.component.height();

        let mut new_size = self.size();
        if new_aspect_ratio >= current_aspect_ratio {
            new_size.y = self.width() / new_aspect_ratio;
        } else {
            new_size.x = self.height() * new_aspect_ratio;
        }

        self.frame.pos += (self.size() - new_size) / 2.;
        self.frame.size = new_size;
    }

    pub fn refit_to_screen(&mut self, relative_frame_pos: Vec2, relative_frame_size: Vec2) {
        self.frame.refit_to(
            vec2(0., 0.),
            vec2(mq::screen_width(), mq::screen_height()),
            relative_frame_pos,
            relative_frame_size,
        );
    }

    pub fn set_component<T: Simulation>(&mut self, component: T) {
        *self = Self::new(component, self.pos(), self.size());
    }

    /// Reads the host's input devices and hands the simulation a snapshot of
    /// them, in the simulation's own coordinate space.
    pub fn update(&mut self, dt: Duration) {
        let input = Input {
            mouse_pos: self.frame.relative_mouse_pos(),
            left_down: mq::is_mouse_button_down(mq::MouseButton::Left),
            right_down: mq::is_mouse_button_down(mq::MouseButton::Right),
            middle_down: mq::is_mouse_button_down(mq::MouseButton::Middle),
        };

        self.component.update(dt, &input);
    }

    pub fn draw(&mut self) {
        // Draw the simulation into its own render target...
        self.canvas.retarget(&self.frame.camera);
        mq::set_camera(&self.frame.camera);
        mq::clear_background(mq::BLANK);

        self.component.draw(&mut self.canvas);

        // ...then blit that target into our slot on screen.
        mq::set_default_camera();
        let offset = crate::OUTLINE_THICKNESS / 2.;
        mq::draw_texture_ex(
            &self.frame.camera.render_target.as_ref().unwrap().texture,
            self.x() + offset,
            self.y() + offset,
            mq::WHITE,
            mq::DrawTextureParams {
                dest_size: Some(vec2(
                    self.width() - offset * 2.,
                    self.height() - offset * 2.,
                )),
                flip_y: true,
                ..Default::default()
            },
        );
    }

    pub fn draw_outline(&self, thickness: f32, color: mq::Color) {
        mq::draw_rectangle_lines(
            self.x(),
            self.y(),
            self.width(),
            self.height(),
            thickness,
            color,
        );
    }
}

impl HasPosition for ComponentFrame {
    fn pos(&self) -> Vec2 {
        self.frame.pos()
    }
}

impl HasSize for ComponentFrame {
    fn size(&self) -> Vec2 {
        self.frame.size()
    }
}

#[derive(Debug)]
pub struct Frame {
    camera: mq::Camera2D,
    pos: Vec2,
    size: Vec2,
    component_size: Vec2,
}

impl Frame {
    pub fn new(component_size: Vec2, frame_pos: Vec2, frame_size: Vec2) -> Self {
        Self {
            camera: component_camera(component_size),
            pos: frame_pos,
            size: frame_size,
            component_size,
        }
    }

    pub fn relative_to(
        parent_pos: Vec2,
        parent_size: Vec2,
        component_size: Vec2,
        relative_frame_pos: Vec2,
        relative_frame_size: Vec2,
    ) -> Self {
        let x = parent_pos.x + relative_frame_pos.x * parent_size.x;
        let y = parent_pos.y + relative_frame_pos.y * parent_size.y;
        let width = relative_frame_size.x * parent_size.x;
        let height = relative_frame_size.y * parent_size.y;

        Self {
            camera: component_camera(component_size),
            pos: vec2(x, y),
            size: vec2(width, height),
            component_size,
        }
    }

    pub fn refit_to(
        &mut self,
        parent_pos: Vec2,
        parent_size: Vec2,
        relative_frame_pos: Vec2,
        relative_frame_size: Vec2,
    ) {
        let x = parent_pos.x + relative_frame_pos.x * parent_size.x;
        let y = parent_pos.y + relative_frame_pos.y * parent_size.y;
        let width = relative_frame_size.x * parent_size.x;
        let height = relative_frame_size.y * parent_size.y;

        self.pos = vec2(x, y);
        self.size = vec2(width, height);
    }

    pub fn relative_mouse_pos(&self) -> Vec2 {
        let mut mouse_pos = Vec2::from(mq::mouse_position()) - self.pos();
        mouse_pos.x = mouse_pos.x * self.component_size.x / self.width();
        mouse_pos.y = mouse_pos.y * self.component_size.y / self.height();
        mouse_pos
    }
}

/// A camera rendering to an offscreen target the size of the component, with
/// the component's own coordinates spanning the view.
fn component_camera(component_size: Vec2) -> mq::Camera2D {
    let render_target = mq::render_target(component_size.x as u32, component_size.y as u32);
    render_target.texture.set_filter(mq::FilterMode::Nearest);

    mq::Camera2D {
        render_target: Some(render_target),
        zoom: vec2(2.0 / component_size.x, -2.0 / component_size.y),
        target: component_size / 2.0,
        ..Default::default()
    }
}

impl HasPosition for Frame {
    fn pos(&self) -> Vec2 {
        self.pos
    }
}

impl HasSize for Frame {
    fn size(&self) -> Vec2 {
        self.size
    }
}
