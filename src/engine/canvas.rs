use super::{Color, ImageBuffer, Vec2};

/// An opaque handle to a host-owned texture. A simulation can hold one across
/// frames but can never look inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureId(pub(crate) usize);

/// An opaque handle to a host-owned offscreen layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerId(pub(crate) usize);

/// A visual treatment a simulation can ask for by name.
///
/// Simulations request an effect; they never supply shader source, because
/// shader source is written in whatever language the backend happens to speak.
/// A backend that does not implement one falls back to drawing unfiltered, so
/// asking for an effect is always safe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Effect {
    /// Draw exactly as composed.
    #[default]
    None,
    /// Treat the layer's coverage as a density field and shade it as a liquid
    /// surface — blobby, lit, with a rim highlight. Expects the layer to have
    /// been drawn with wide, low-alpha marks.
    Liquid,
}

/// Everything a simulation is allowed to ask a renderer for.
///
/// Object-safe on purpose: simulations are held as trait objects, and the
/// recording implementation used in tests has to be interchangeable with the
/// real backend.
pub trait Canvas {
    /// Maps `size` units of the caller's coordinate space, with `pos` at the
    /// top left, onto the whole output surface. Everything drawn afterwards is
    /// in those coordinates.
    fn set_view(&mut self, pos: Vec2, size: Vec2);

    /// Clears the whole surface to `color`. The host already clears before
    /// each frame, so this is for a simulation that wants its own backdrop.
    fn clear(&mut self, color: Color);

    fn circle(&mut self, centre: Vec2, radius: f32, color: Color);

    /// An axis-aligned rectangle with its top-left corner at `pos`.
    fn rect(&mut self, pos: Vec2, size: Vec2, color: Color);

    fn triangle(&mut self, a: Vec2, b: Vec2, c: Vec2, color: Color);

    /// Uploads a new texture and returns a handle to it. Call once and keep the
    /// handle; uploading every frame is exactly what this is designed to avoid.
    fn create_texture(&mut self, image: &ImageBuffer) -> TextureId;

    /// Replaces the contents of an existing texture. The image must match the
    /// dimensions the texture was created with.
    fn update_texture(&mut self, texture: TextureId, image: &ImageBuffer);

    /// Draws a texture stretched to `size`, multiplied by `tint`.
    fn draw_texture(&mut self, texture: TextureId, pos: Vec2, size: Vec2, tint: Color);

    /// Composes `draw` into an offscreen layer of `size`, then draws that layer
    /// at `pos` with `effect` applied.
    ///
    /// The layer starts transparent and shares the caller's coordinate space.
    fn with_layer(
        &mut self,
        pos: Vec2,
        size: Vec2,
        effect: Effect,
        draw: &mut dyn FnMut(&mut dyn Canvas),
    );
}
