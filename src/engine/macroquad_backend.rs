//! The macroquad implementation of [`Canvas`].
//!
//! This is the only place that knows both vocabularies. Everything
//! macroquad-shaped stops here.

use super::{Canvas, Color, Effect, ImageBuffer, LayerId, TextureId, Vec2};
use crate::shaders::liquid_material;

use macroquad::prelude as mq;

/// Owns the GPU resources that simulations refer to by handle.
///
/// Keep one of these alive across frames: the handles a simulation stores are
/// indices into these tables, so dropping it invalidates them.
pub struct MacroquadCanvas {
    textures: Vec<mq::Texture2D>,
    layers: Vec<Layer>,
    liquid: Option<mq::Material>,
    /// `Camera2D` is not `Clone`, so the view is kept as its parts and the
    /// camera rebuilt whenever it needs to be applied.
    target: Option<mq::RenderTarget>,
    zoom: Vec2,
    view_centre: Vec2,
}

struct Layer {
    target: mq::RenderTarget,
    size: Vec2,
}

impl std::fmt::Debug for MacroquadCanvas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacroquadCanvas")
            .field("textures", &self.textures.len())
            .field("layers", &self.layers.len())
            .finish_non_exhaustive()
    }
}

impl MacroquadCanvas {
    /// Draws into `camera`'s render target. The camera is cloned, so later
    /// `set_view` calls reshape this canvas's view without disturbing the
    /// caller's copy — but the render target is shared, which is the point.
    pub fn new(camera: &mq::Camera2D) -> Self {
        Self {
            textures: Vec::new(),
            layers: Vec::new(),
            liquid: None,
            target: camera.render_target.clone(),
            zoom: camera.zoom,
            view_centre: camera.target,
        }
    }

    /// Points this canvas at a different render target, reusing the texture and
    /// layer tables so simulation-held handles stay valid across a resize.
    ///
    /// The view is left alone: the simulation sets that itself, every frame.
    pub fn retarget(&mut self, camera: &mq::Camera2D) {
        self.target = camera.render_target.clone();
    }

    /// Builds a camera for the current view, optionally overriding the target.
    fn camera(&self, target: Option<mq::RenderTarget>) -> mq::Camera2D {
        mq::Camera2D {
            render_target: target,
            zoom: self.zoom,
            target: self.view_centre,
            ..Default::default()
        }
    }

    fn apply_view(&self) {
        mq::set_camera(&self.camera(self.target.clone()));
    }

    /// Compiled on first use so a canvas that never asks for the effect never
    /// pays for the shader, and so construction stays free of GPU work.
    fn liquid_material(&mut self) -> &mq::Material {
        self.liquid.get_or_insert_with(liquid_material)
    }

    fn texture(&self, id: TextureId) -> &mq::Texture2D {
        &self.textures[id.0]
    }
}

/// Translates an engine colour into macroquad's. Public so the host's own
/// macroquad calls — window clears, outlines — can use the same palette the
/// simulations do.
pub fn to_mq_color(color: Color) -> mq::Color {
    mq::Color::new(color.r, color.g, color.b, color.a)
}

fn to_mq_image(image: &ImageBuffer) -> mq::Image {
    mq::Image {
        bytes: image.bytes().to_vec(),
        width: image.width(),
        height: image.height(),
    }
}

impl Canvas for MacroquadCanvas {
    fn set_view(&mut self, pos: Vec2, size: Vec2) {
        self.zoom = mq::vec2(2.0 / size.x, -2.0 / size.y);
        self.view_centre = pos + size / 2.0;
        self.apply_view();
    }

    fn clear(&mut self, color: Color) {
        mq::clear_background(to_mq_color(color));
    }

    fn circle(&mut self, centre: Vec2, radius: f32, color: Color) {
        mq::draw_circle(centre.x, centre.y, radius, to_mq_color(color));
    }

    fn rect(&mut self, pos: Vec2, size: Vec2, color: Color) {
        mq::draw_rectangle(pos.x, pos.y, size.x, size.y, to_mq_color(color));
    }

    fn triangle(&mut self, a: Vec2, b: Vec2, c: Vec2, color: Color) {
        mq::draw_triangle(a, b, c, to_mq_color(color));
    }

    fn create_texture(&mut self, image: &ImageBuffer) -> TextureId {
        let texture = mq::Texture2D::from_image(&to_mq_image(image));
        texture.set_filter(mq::FilterMode::Nearest);

        self.textures.push(texture);
        TextureId(self.textures.len() - 1)
    }

    fn update_texture(&mut self, texture: TextureId, image: &ImageBuffer) {
        self.texture(texture).update(&to_mq_image(image));
    }

    fn draw_texture(&mut self, texture: TextureId, pos: Vec2, size: Vec2, tint: Color) {
        if tint.is_transparent() {
            return;
        }

        mq::draw_texture_ex(
            self.texture(texture),
            pos.x,
            pos.y,
            to_mq_color(tint),
            mq::DrawTextureParams {
                flip_y: false,
                dest_size: Some(size),
                ..Default::default()
            },
        );
    }

    fn with_layer(
        &mut self,
        pos: Vec2,
        size: Vec2,
        effect: Effect,
        draw: &mut dyn FnMut(&mut dyn Canvas),
    ) {
        let layer = self.acquire_layer(size);

        // Redirect at the layer, keeping the caller's coordinates so whatever
        // `draw` emits lands where the simulation expects.
        let layer_target = Some(self.layers[layer.0].target.clone());
        mq::set_camera(&self.camera(layer_target));
        mq::clear_background(mq::BLANK);

        draw(self);

        self.apply_view();

        if effect == Effect::Liquid {
            let material = self.liquid_material().clone();
            mq::gl_use_material(&material);
        }

        mq::draw_texture_ex(
            &self.layers[layer.0].target.texture,
            pos.x,
            pos.y,
            mq::WHITE,
            mq::DrawTextureParams {
                flip_y: true,
                dest_size: Some(size),
                ..Default::default()
            },
        );

        if effect != Effect::None {
            mq::gl_use_default_material();
        }
    }
}

impl MacroquadCanvas {
    /// Reuses an existing layer of the right size, or allocates one. Layers are
    /// keyed by size alone because they are scratch space, never held across
    /// frames by a simulation.
    fn acquire_layer(&mut self, size: Vec2) -> LayerId {
        if let Some(index) = self.layers.iter().position(|layer| layer.size == size) {
            return LayerId(index);
        }

        let target = mq::render_target(size.x as u32, size.y as u32);
        target.texture.set_filter(mq::FilterMode::Nearest);

        self.layers.push(Layer { target, size });
        LayerId(self.layers.len() - 1)
    }
}
