#![allow(
    clippy::wildcard_imports,
    clippy::enum_glob_use,
    clippy::unused_self,
    clippy::struct_field_names,
    clippy::match_same_arms,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::unreadable_literal,
    clippy::needless_raw_string_hashes
)]
#![warn(clippy::style, clippy::perf, clippy::complexity)]
#![deny(clippy::correctness, clippy::suspicious)]

mod buffer;
mod component;
mod grid;
mod shaders;
mod simulations;
mod traits;

use component::ComponentFrame;
use simulations::*;
use traits::*;

use macroquad::prelude::*;
use std::time::{Duration, Instant};

pub const BG_COLOR: Color = Color::new(0.18, 0.18, 0.18, 1.0);
pub const OUTLINE_COLOR: Color = Color::new(0.8, 0.8, 0.8, 1.0);
pub const OUTLINE_THICKNESS: f32 = 4.0;

const AUTOMATA_WIDTH: usize = 200;
const AUTOMATA_HEIGHT: usize = 100;
const SIM_WIDTH: f32 = 500.;
const SIM_HEIGHT: f32 = 300.;

const NUM_BOIDS: usize = 500;
const COLORLIFE_PARTICLES: usize = 3000;
const FLUID_PARTICLES: usize = 1000;

struct Video {
    pub frames: Vec<Image>,
    pub framerate: u32,
}

#[macroquad::main("window_config")]
async fn main() {
    let mut sim = default_sim();

    let bad_apple = load_bad_apple().await;
    let start = Instant::now();
    loop {
        clear_background(BG_COLOR);

        let now = Instant::now();
        let time_since_start = now - start;
        let frame = (time_since_start.as_secs_f32() * 30.).round() as usize; // 30 fps

        // TODO: Remove
        let image = &bad_apple.frames[frame];
        let bitmap = BinaryBitmap::from_image(&image);
        let mut obstacle = BitmapObstacle::from_bitmap(bitmap, vec2(0., 0.), vec2(300., 200.));
        obstacle.draw();

        handle_sim_selection(&mut sim);
        // update(&mut sim);
        // draw(&mut sim);
        next_frame().await;
    }
}

#[allow(dead_code)]
fn window_config() -> Conf {
    Conf {
        window_title: "Deskthing".to_owned(),
        sample_count: 4,
        ..Default::default()
    }
}

async fn load_bad_apple() -> Video {
    let mut files = std::fs::read_dir("./resources/badapple/frames/")
        .unwrap()
        .collect::<Vec<_>>();

    files.sort_by_key(|file| file.as_ref().unwrap().file_name());
    let mut frames = Vec::with_capacity(files.len());

    for file in files {
        let file = file.unwrap();
        let path = file.path();

        let image = load_image(path.to_str().unwrap()).await.unwrap();

        frames.push(image);
    }

    Video {
        frames,
        framerate: 30,
    }
}

fn default_sim() -> ComponentFrame {
    let default_sim = FluidSim::init(FLUID_PARTICLES, SIM_WIDTH, SIM_HEIGHT);
    ComponentFrame::relative_to_screen(default_sim, vec2(0.2, 0.2), vec2(0.6, 0.6))
}

fn handle_sim_selection(sim: &mut ComponentFrame) {
    if is_key_pressed(KeyCode::A) {
        sim.set_component(Conway::random(
            _CONWAY,
            0.6,
            AUTOMATA_WIDTH,
            AUTOMATA_HEIGHT,
        ));
    } else if is_key_pressed(KeyCode::B) {
        sim.set_component(Boids::init(NUM_BOIDS, SIM_WIDTH, SIM_HEIGHT));
    } else if is_key_pressed(KeyCode::C) {
        sim.set_component(Colorlife::init(COLORLIFE_PARTICLES, SIM_WIDTH, SIM_HEIGHT));
    } else if is_key_pressed(KeyCode::D) {
        sim.set_component(FluidSim::init(FLUID_PARTICLES, SIM_WIDTH, SIM_HEIGHT));
    }
}

fn update(sim: &mut ComponentFrame) {
    sim.refit_to_screen(vec2(0.2, 0.2), vec2(0.6, 0.6));
    sim.refit_to_component();
    sim.update();
}

fn draw(sim: &mut ComponentFrame) {
    clear_background(BG_COLOR);
    sim.draw();
    sim.draw_outline(4., WHITE);
}
