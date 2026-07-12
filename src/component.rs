use crate::traits::*;

use macroquad::prelude::*;
use std::fmt::Debug;

#[derive(Debug)]
pub struct ComponentFrame {
    component: Box<dyn Component>,
    frame: Frame,
}

#[allow(unused)]
impl ComponentFrame {
    pub fn new<T: Component>(component: T, pos: Vec2, size: Vec2) -> Self {
        let frame = Frame::new(vec2(component.width(), component.height()), pos, size);
        Self {
            component: Box::new(component),
            frame,
        }
    }

    pub fn relative<T: Component>(
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

        Self {
            component: Box::new(component),
            frame,
        }
    }

    pub fn relative_to_screen<T: Component>(
        component: T,
        relative_frame_pos: Vec2,
        relative_frame_size: Vec2,
    ) -> Self {
        Self::relative(
            component,
            vec2(0., 0.),
            vec2(screen_width(), screen_height()),
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

    pub fn refit_to_screen(&mut self, relative_frame_pos: Vec2, relative_frame_size: Vec2) {
        self.frame.refit_to(
            vec2(0., 0.),
            vec2(screen_width(), screen_height()),
            relative_frame_pos,
            relative_frame_size,
        );
    }

    pub fn set_component<T: Component>(&mut self, component: T) {
        *self = Self::new(component, self.pos(), self.size());
    }

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

impl Update for ComponentFrame {
    fn update(&mut self) {
        self.component.update_with_context(&self.frame);
    }
}

impl Draw for ComponentFrame {
    fn draw(&mut self) {
        // Start using the component's camera,
        set_camera(&self.frame.camera);
        clear_background(BLANK);

        // render the internal drawable object to the camera,
        self.component.draw_with_context(&mut self.frame);

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

    #[allow(unused)]
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
