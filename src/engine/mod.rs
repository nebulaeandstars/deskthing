//! The boundary between the simulations and whatever is drawing them.
//!
//! Nothing in this module may reference a rendering backend. A simulation
//! describes *what* it wants on screen — shapes, images, an effect over a
//! layer — and a [`Canvas`] implementation decides *how* that happens. Swapping
//! macroquad for something else should mean writing one new `Canvas`, not
//! touching a simulation.

// The engine module is an API boundary: it defines the full vocabulary a
// simulation may use, not just the subset the current simulations happen to
// reach for. Unused entries here are surface, not dead weight.
#![allow(dead_code, unused_imports)]

mod canvas;
mod color;
mod image;
mod input;
mod macroquad_backend;
#[cfg(test)]
mod recording;

pub use canvas::{Canvas, Effect, LayerId, TextureId};
pub use color::{BLACK, BLANK, BLUE, Color, GREEN, RED, WHITE};
pub use image::ImageBuffer;
pub use input::{Input, MouseButton};
pub use macroquad_backend::{MacroquadCanvas, to_mq_color};
#[cfg(test)]
pub use recording::{DrawCall, RecordingCanvas};

/// Re-exported so simulations get their vector maths from here rather than
/// from a rendering library. This is the same `glam` version macroquad uses,
/// so the backend needs no conversions.
pub use glam::{Vec2, vec2};
