/// A straight RGBA colour, each channel in `0.0..=1.0`.
///
/// Deliberately not a re-export: a simulation saying "draw this red" should not
/// have to name a rendering library to do it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Fully opaque, from the given channels.
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(r, g, b, 1.0)
    }

    /// An opaque colour from a packed `0xRRGGBB` value.
    pub const fn from_hex(hex: u32) -> Self {
        Self::new(
            ((hex >> 16) & 0xff) as f32 / 255.,
            ((hex >> 8) & 0xff) as f32 / 255.,
            (hex & 0xff) as f32 / 255.,
            1.0,
        )
    }

    /// The same colour at a different opacity.
    #[must_use]
    pub const fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }

    /// Whether this colour would draw nothing at all.
    pub fn is_transparent(self) -> bool {
        self.a <= 0.
    }

    /// The colour as 8-bit RGBA, for writing into an [`ImageBuffer`].
    ///
    /// [`ImageBuffer`]: crate::engine::ImageBuffer
    pub fn to_rgba8(self) -> [u8; 4] {
        [
            (self.r.clamp(0., 1.) * 255.) as u8,
            (self.g.clamp(0., 1.) * 255.) as u8,
            (self.b.clamp(0., 1.) * 255.) as u8,
            (self.a.clamp(0., 1.) * 255.) as u8,
        ]
    }
}

// The basic palette. Part of the engine's public vocabulary, so not every
// entry has a caller in-tree.
pub const BLANK: Color = Color::new(0., 0., 0., 0.);
pub const BLACK: Color = Color::rgb(0., 0., 0.);
pub const WHITE: Color = Color::rgb(1., 1., 1.);
pub const RED: Color = Color::rgb(1., 0., 0.);
pub const GREEN: Color = Color::rgb(0., 1., 0.);
pub const BLUE: Color = Color::rgb(0., 0., 1.);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_is_transparent_and_white_is_not() {
        assert!(BLANK.is_transparent());
        assert!(!WHITE.is_transparent());
    }

    #[test]
    fn from_hex_unpacks_channels() {
        assert_eq!(RED, Color::from_hex(0xff0000));
        assert_eq!(WHITE, Color::from_hex(0xffffff));
        assert_eq!(BLACK, Color::from_hex(0x000000));
        assert_eq!(
            [0x12, 0x34, 0x56, 255],
            Color::from_hex(0x123456).to_rgba8()
        );
    }

    #[test]
    fn rgba8_round_trips_the_extremes() {
        assert_eq!([255, 0, 0, 255], RED.to_rgba8());
        assert_eq!([0, 0, 0, 0], BLANK.to_rgba8());
    }

    #[test]
    fn rgba8_clamps_out_of_range_channels() {
        assert_eq!([255, 0, 0, 255], Color::new(1.5, -0.2, 0., 1.).to_rgba8());
    }
}
