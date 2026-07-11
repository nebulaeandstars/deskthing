#![allow(unused)]

use crate::traits::*;
use crate::Draw;

use macroquad::prelude::*;
use macroquad::window;

#[derive(Debug)]
pub struct Component<T> {
    inner: T,
    frame: Frame,
}

impl<T: HasSize> Component<T> {
    pub fn new(object: T, pos: Vec2, size: Vec2) -> Self {
        let frame = Frame::new(vec2(object.width(), object.height()), pos, size);
        Self {
            inner: object,
            frame,
        }
    }

    pub fn relative(
        object: T,
        parent_pos: Vec2,
        parent_size: Vec2,
        relative_frame_pos: Vec2,
        relative_frame_size: Vec2,
    ) -> Self {
        let object_size = vec2(object.width(), object.height());

        let frame = Frame::relative_to(
            parent_pos,
            parent_size,
            object_size,
            relative_frame_pos,
            relative_frame_size,
        );

        Self {
            inner: object,
            frame,
        }
    }

    pub fn relative_to_screen(
        object: T,
        relative_frame_pos: Vec2,
        relative_frame_size: Vec2,
    ) -> Self {
        Self::relative(
            object,
            vec2(0., 0.),
            vec2(screen_width(), screen_height()),
            relative_frame_pos,
            relative_frame_size,
        )
    }
}

impl<T> Component<T> {
    pub fn draw_outline(&self, thickness: f32, color: Color) {
        draw_rectangle_lines(
            self.x(),
            self.y(),
            self.width(),
            self.height(),
            thickness,
            color,
        );
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

    pub fn refit_to_screen(&mut self, relative_frame_pos: Vec2, relative_frame_size: Vec2) {
        self.frame.refit_to(
            vec2(0., 0.),
            vec2(screen_width(), screen_height()),
            relative_frame_pos,
            relative_frame_size,
        );
    }
}

impl<T: Update> Component<T> {
    pub fn update(&mut self) {
        self.inner.update(&self.frame)
    }
}

impl<T: Draw> Component<T> {
    pub fn draw(&mut self) {
        // Render the component to the camera,
        self.inner.draw(&mut self.frame);

        // then draw it.
        set_default_camera();
        let offset = crate::OUTLINE_THICKNESS / 2.;
        draw_texture_ex(
            &self.frame.camera.render_target.as_ref().unwrap().texture,
            self.x() + offset,
            self.y() + offset,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(
                    self.width() - offset * 2.,
                    self.height() - offset * 2.,
                )),
                flip_y: true,
                ..Default::default()
            },
        );
    }
}

impl<T> HasPosition for Component<T> {
    fn pos(&self) -> Vec2 {
        self.frame.pos()
    }
}

impl<T> HasSize for Component<T> {
    fn size(&self) -> Vec2 {
        self.frame.size()
    }
}

#[derive(Debug)]
pub struct Frame {
    camera: Camera2D,
    pos: Vec2,
    size: Vec2,
    component_size: Vec2,
}

impl Frame {
    pub fn new(component_size: Vec2, frame_pos: Vec2, frame_size: Vec2) -> Self {
        let render_target = render_target(component_size.x as u32, component_size.y as u32);
        render_target.texture.set_filter(FilterMode::Nearest);

        let camera = Camera2D {
            render_target: Some(render_target.clone()),
            zoom: vec2(2.0 / component_size.x, -2.0 / component_size.y),
            target: vec2(component_size.x / 2.0, component_size.y / 2.0),
            ..Default::default()
        };

        Self {
            camera,
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
        let render_target = render_target(component_size.x as u32, component_size.y as u32);
        render_target.texture.set_filter(FilterMode::Nearest);

        let camera = Camera2D {
            render_target: Some(render_target.clone()),
            zoom: vec2(2.0 / component_size.x, -2.0 / component_size.y),
            target: vec2(component_size.x / 2.0, component_size.y / 2.0),
            ..Default::default()
        };

        let x = parent_pos.x + relative_frame_pos.x * parent_size.x;
        let y = parent_pos.y + relative_frame_pos.y * parent_size.y;
        let width = relative_frame_size.x * parent_size.x;
        let height = relative_frame_size.y * parent_size.y;

        Self {
            camera,
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

    pub fn camera(&mut self) -> &mut Camera2D {
        &mut self.camera
    }

    pub fn relative_mouse_pos(&self) -> Vec2 {
        let mut mouse_pos = Vec2::from(mouse_position()) - self.pos();
        mouse_pos.x = mouse_pos.x * self.component_size.x / self.width();
        mouse_pos.y = mouse_pos.y * self.component_size.y / self.height();
        mouse_pos
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

#[derive(Clone, Copy, Debug)]
pub struct DrawFrameLayout {
    pub parent: Option<DrawFrame>,
    pub frame: DrawFrame,
    x_percent: f32,
    y_percent: f32,
    width_percent: f32,
    height_percent: f32,
}

#[allow(unused)]
impl DrawFrameLayout {
    pub fn new(
        parent: Option<DrawFrame>,
        x_percent: f32,
        y_percent: f32,
        width_percent: f32,
        height_percent: f32,
    ) -> Self {
        let frame = DrawFrame::relative(
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
        parent: DrawFrame,
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
        self.frame = DrawFrame::relative(
            self.parent.unwrap_or_default(),
            self.x_percent,
            self.y_percent,
            self.width_percent,
            self.height_percent,
        );
    }

    pub fn update_parent(&mut self, parent: DrawFrame) {
        self.parent = Some(parent);
        self.refresh();
    }
}

/// A sub-window containing multiple components.
#[derive(Clone, Copy, Debug)]
pub struct DrawFrame {
    pos: Vec2,
    size: Vec2,
}

impl DrawFrame {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            pos: Vec2::new(x, y),
            size: Vec2::new(width, height),
        }
    }

    pub fn relative(
        frame: DrawFrame,
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
}

impl HasPosition for DrawFrame {
    fn pos(&self) -> Vec2 {
        self.pos
    }
}

impl HasSize for DrawFrame {
    fn size(&self) -> Vec2 {
        self.size
    }
}

impl Default for DrawFrame {
    fn default() -> Self {
        Self::new(0., 0., window::screen_width(), window::screen_height())
    }
}
