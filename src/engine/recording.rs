//! A [`Canvas`] that records instead of rendering.
//!
//! This is the other half of the point of the canvas abstraction: a simulation
//! that draws through a trait can be drawn in a test, with no window, no GPU
//! and no graphics context, and the test can assert on exactly what it asked
//! for.

use super::{Canvas, Color, Effect, ImageBuffer, TextureId, Vec2};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DrawCall {
    SetView {
        pos: Vec2,
        size: Vec2,
    },
    Clear(Color),
    Circle {
        centre: Vec2,
        radius: f32,
        color: Color,
    },
    Rect {
        pos: Vec2,
        size: Vec2,
        color: Color,
    },
    Triangle {
        a: Vec2,
        b: Vec2,
        c: Vec2,
        color: Color,
    },
    CreateTexture {
        texture: TextureId,
        width: u16,
        height: u16,
    },
    UpdateTexture {
        texture: TextureId,
        width: u16,
        height: u16,
    },
    DrawTexture {
        texture: TextureId,
        pos: Vec2,
        size: Vec2,
        tint: Color,
    },
    /// Emitted before the layer's contents; the matching [`Self::EndLayer`]
    /// follows them.
    BeginLayer {
        pos: Vec2,
        size: Vec2,
        effect: Effect,
    },
    EndLayer,
}

#[derive(Debug, Default)]
pub struct RecordingCanvas {
    pub calls: Vec<DrawCall>,
    next_texture: usize,
}

impl RecordingCanvas {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every circle drawn, in order.
    pub fn circles(&self) -> impl Iterator<Item = (Vec2, f32, Color)> + '_ {
        self.calls.iter().filter_map(|call| match call {
            DrawCall::Circle {
                centre,
                radius,
                color,
            } => Some((*centre, *radius, *color)),
            _ => None,
        })
    }

    /// Every rectangle drawn, in order.
    pub fn rects(&self) -> impl Iterator<Item = (Vec2, Vec2, Color)> + '_ {
        self.calls.iter().filter_map(|call| match call {
            DrawCall::Rect { pos, size, color } => Some((*pos, *size, *color)),
            _ => None,
        })
    }

    /// Every triangle drawn, in order.
    pub fn triangles(&self) -> impl Iterator<Item = (Vec2, Vec2, Vec2, Color)> + '_ {
        self.calls.iter().filter_map(|call| match call {
            DrawCall::Triangle { a, b, c, color } => Some((*a, *b, *c, *color)),
            _ => None,
        })
    }

    /// The view the simulation last asked for, if it asked at all.
    pub fn view(&self) -> Option<(Vec2, Vec2)> {
        self.calls.iter().rev().find_map(|call| match call {
            DrawCall::SetView { pos, size } => Some((*pos, *size)),
            _ => None,
        })
    }

    /// Effects requested this recording, in order.
    pub fn effects(&self) -> impl Iterator<Item = Effect> + '_ {
        self.calls.iter().filter_map(|call| match call {
            DrawCall::BeginLayer { effect, .. } => Some(*effect),
            _ => None,
        })
    }

    pub fn clear_calls(&mut self) {
        self.calls.clear();
    }
}

impl Canvas for RecordingCanvas {
    fn set_view(&mut self, pos: Vec2, size: Vec2) {
        self.calls.push(DrawCall::SetView { pos, size });
    }

    fn clear(&mut self, color: Color) {
        self.calls.push(DrawCall::Clear(color));
    }

    fn circle(&mut self, centre: Vec2, radius: f32, color: Color) {
        self.calls.push(DrawCall::Circle {
            centre,
            radius,
            color,
        });
    }

    fn rect(&mut self, pos: Vec2, size: Vec2, color: Color) {
        self.calls.push(DrawCall::Rect { pos, size, color });
    }

    fn triangle(&mut self, a: Vec2, b: Vec2, c: Vec2, color: Color) {
        self.calls.push(DrawCall::Triangle { a, b, c, color });
    }

    fn create_texture(&mut self, image: &ImageBuffer) -> TextureId {
        let texture = TextureId(self.next_texture);
        self.next_texture += 1;

        self.calls.push(DrawCall::CreateTexture {
            texture,
            width: image.width(),
            height: image.height(),
        });

        texture
    }

    fn update_texture(&mut self, texture: TextureId, image: &ImageBuffer) {
        self.calls.push(DrawCall::UpdateTexture {
            texture,
            width: image.width(),
            height: image.height(),
        });
    }

    fn draw_texture(&mut self, texture: TextureId, pos: Vec2, size: Vec2, tint: Color) {
        self.calls.push(DrawCall::DrawTexture {
            texture,
            pos,
            size,
            tint,
        });
    }

    fn with_layer(
        &mut self,
        pos: Vec2,
        size: Vec2,
        effect: Effect,
        draw: &mut dyn FnMut(&mut dyn Canvas),
    ) {
        self.calls.push(DrawCall::BeginLayer { pos, size, effect });
        draw(self);
        self.calls.push(DrawCall::EndLayer);
    }
}

#[cfg(test)]
mod tests {
    use super::super::{BLANK, RED, vec2};
    use super::*;

    #[test]
    fn primitives_are_recorded_in_order() {
        let mut canvas = RecordingCanvas::new();

        canvas.circle(vec2(1., 2.), 3., RED);
        canvas.rect(vec2(4., 5.), vec2(6., 7.), BLANK);

        assert_eq!(2, canvas.calls.len());
        assert_eq!(
            vec![(vec2(1., 2.), 3., RED)],
            canvas.circles().collect::<Vec<_>>()
        );
        assert_eq!(
            vec![(vec2(4., 5.), vec2(6., 7.), BLANK)],
            canvas.rects().collect::<Vec<_>>()
        );
    }

    #[test]
    fn texture_handles_are_distinct() {
        let mut canvas = RecordingCanvas::new();
        let image = ImageBuffer::filled(2, 2, BLANK);

        let first = canvas.create_texture(&image);
        let second = canvas.create_texture(&image);

        assert_ne!(first, second);
    }

    #[test]
    fn layer_contents_are_bracketed() {
        let mut canvas = RecordingCanvas::new();

        canvas.with_layer(Vec2::ZERO, vec2(10., 10.), Effect::Liquid, &mut |layer| {
            layer.circle(Vec2::ZERO, 1., RED);
        });

        assert!(matches!(canvas.calls[0], DrawCall::BeginLayer { .. }));
        assert!(matches!(canvas.calls[1], DrawCall::Circle { .. }));
        assert!(matches!(canvas.calls[2], DrawCall::EndLayer));
        assert_eq!(vec![Effect::Liquid], canvas.effects().collect::<Vec<_>>());
    }

    #[test]
    fn view_reports_the_most_recent_one() {
        let mut canvas = RecordingCanvas::new();

        canvas.set_view(Vec2::ZERO, vec2(1., 1.));
        canvas.set_view(vec2(5., 5.), vec2(2., 2.));

        assert_eq!(Some((vec2(5., 5.), vec2(2., 2.))), canvas.view());
    }
}
