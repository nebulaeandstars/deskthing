use super::Color;

/// A CPU-side RGBA8 pixel buffer.
///
/// Simulations rasterise into one of these and hand it to a [`Canvas`] to be
/// uploaded; they never hold a GPU texture themselves.
///
/// [`Canvas`]: super::Canvas
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageBuffer {
    bytes: Vec<u8>,
    width: u16,
    height: u16,
}

impl ImageBuffer {
    /// A buffer of the given size, filled with `color`.
    pub fn filled(width: u16, height: u16, color: Color) -> Self {
        let pixel = color.to_rgba8();
        let bytes = pixel.repeat(width as usize * height as usize);

        Self {
            bytes,
            width,
            height,
        }
    }

    /// Wraps existing RGBA8 bytes, row-major from the top left.
    ///
    /// # Panics
    /// If `bytes` is not exactly `width * height * 4` long.
    pub fn from_rgba8(bytes: Vec<u8>, width: u16, height: u16) -> Self {
        assert_eq!(
            width as usize * height as usize * 4,
            bytes.len(),
            "byte count does not match {width}x{height} RGBA"
        );

        Self {
            bytes,
            width,
            height,
        }
    }

    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }

    /// The raw RGBA8 bytes, row-major from the top left.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The pixels as mutable 4-byte chunks, row-major from the top left.
    pub fn pixels_mut(&mut self) -> impl Iterator<Item = &mut [u8; 4]> {
        self.bytes.as_chunks_mut::<4>().0.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::super::color::{BLANK, RED};
    use super::*;

    #[test]
    fn filled_has_one_rgba_quad_per_pixel() {
        let image = ImageBuffer::filled(3, 2, RED);

        assert_eq!(3, image.width());
        assert_eq!(2, image.height());
        assert_eq!(3 * 2 * 4, image.bytes().len());
        assert!(image.bytes().chunks(4).all(|px| px == [255, 0, 0, 255]));
    }

    #[test]
    fn pixels_mut_visits_every_pixel_once() {
        let mut image = ImageBuffer::filled(4, 4, BLANK);

        let mut visited = 0;
        for pixel in image.pixels_mut() {
            *pixel = [1, 2, 3, 4];
            visited += 1;
        }

        assert_eq!(16, visited);
        assert!(image.bytes().chunks(4).all(|px| px == [1, 2, 3, 4]));
    }
}
