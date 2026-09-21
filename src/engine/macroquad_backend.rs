//! The macroquad implementation of [`Canvas`].
//!
//! This is the only place that knows both vocabularies. Everything
//! macroquad-shaped stops here.

use super::{Audio, Canvas, Color, Effect, ImageBuffer, LayerId, SoundId, TextureId, Vec2};
use crate::shaders::liquid_material;

use macroquad::audio as mq_audio;
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

    /// How many sides to approximate a circle of `radius` with.
    ///
    /// `macroquad::draw_circle` is hard-coded to twenty sides regardless of
    /// size, which is a lot of triangles for a three-pixel particle. Choosing
    /// the count from the on-screen radius keeps the error under half a pixel
    /// while costing a fraction of the geometry. This is the backend's call to
    /// make — a simulation asks for a circle and should not have to think
    /// about tessellation.
    fn circle_sides(&self, radius: f32) -> u8 {
        // zoom.x is 2/view_width, so this recovers pixels per world unit.
        let target_width = self
            .target
            .as_ref()
            .map_or_else(mq::screen_width, |target| target.texture.width());
        let pixels_per_unit = (target_width * self.zoom.x / 2.0).abs();
        let radius_px = radius * pixels_per_unit;

        // Sagitta error for an n-gon is about r*pi^2/(2n^2); under half a
        // pixel means n > pi*sqrt(r).
        let ideal = std::f32::consts::PI * radius_px.max(0.).sqrt();

        (ideal.ceil() as u32).clamp(6, 40) as u8
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
        let sides = self.circle_sides(radius);
        mq::draw_poly(centre.x, centre.y, sides, radius, 0., to_mq_color(color));
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
        let existing = self.texture(texture);

        debug_assert_eq!(
            (existing.width() as u16, existing.height() as u16),
            (image.width(), image.height()),
            "update_texture requires the image to match the texture it was created with"
        );

        existing.update(&to_mq_image(image));
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Reproduces the tessellation heuristic without a graphics context: the
    /// real method needs a render target, but the arithmetic is the same.
    fn sides_for(radius_px: f32) -> u8 {
        let ideal = std::f32::consts::PI * radius_px.max(0.).sqrt();
        (ideal.ceil() as u32).clamp(6, 40) as u8
    }

    #[test]
    fn small_marks_get_far_fewer_sides_than_macroquads_fixed_twenty() {
        // A fluid particle is drawn at radius 3.
        assert!(sides_for(3.) < 20, "{} sides", sides_for(3.));
    }

    #[test]
    fn sides_never_drop_below_a_recognisable_polygon() {
        for radius in [0., 0.1, 1.] {
            assert_eq!(6, sides_for(radius));
        }
    }

    #[test]
    fn sides_grow_with_radius_and_stay_bounded() {
        assert!(sides_for(10.) > sides_for(3.));
        assert!(sides_for(1000.) <= 40);
    }

    #[test]
    fn the_half_pixel_error_bound_holds() {
        for radius in [1., 3., 10., 50., 160.] {
            let n = f32::from(sides_for(radius));
            // Exact sagitta: r * (1 - cos(pi/n)).
            let error = radius * (1. - (std::f32::consts::PI / n).cos());
            assert!(
                error < 0.5,
                "radius {radius} with {n} sides errs by {error}px"
            );
        }
    }
}

/// Owns the decoded sounds that callers refer to by handle.
///
/// Loading lives here rather than on the [`Audio`] trait: decoding is a
/// backend concern, and macroquad's loader is asynchronous, which a plain
/// trait method cannot express.
#[derive(Default)]
pub struct MacroquadAudio {
    sounds: Vec<mq_audio::Sound>,
}

impl std::fmt::Debug for MacroquadAudio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacroquadAudio")
            .field("sounds", &self.sounds.len())
            .finish()
    }
}

impl MacroquadAudio {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decodes a sound file and returns a handle to it.
    ///
    /// Ogg Vorbis and WAV; macroquad decodes nothing else, so an mp4's audio
    /// track has to be extracted first.
    pub async fn load(&mut self, path: &str) -> Result<SoundId, macroquad::Error> {
        let sound = mq_audio::load_sound(path).await?;

        self.sounds.push(sound);
        Ok(SoundId(self.sounds.len() - 1))
    }

    fn sound(&self, id: SoundId) -> &mq_audio::Sound {
        &self.sounds[id.0]
    }
}

impl Audio for MacroquadAudio {
    fn play(&mut self, sound: SoundId, looped: bool) {
        mq_audio::play_sound(
            self.sound(sound),
            mq_audio::PlaySoundParams {
                looped,
                volume: 1.0,
            },
        );
    }

    fn stop(&mut self, sound: SoundId) {
        mq_audio::stop_sound(self.sound(sound));
    }

    fn stop_all(&mut self) {
        for sound in &self.sounds {
            mq_audio::stop_sound(sound);
        }
    }

    fn set_volume(&mut self, sound: SoundId, volume: f32) {
        mq_audio::set_sound_volume(self.sound(sound), volume.clamp(0., 1.));
    }
}
