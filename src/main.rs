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

use grid::Grid;
use macroquad::prelude::*;
use std::fs::File;
use std::io::{Read, Write};

const BADAPPLE_X: usize = 320;
const BADAPPLE_Y: usize = 240;

pub const BG_COLOR: Color = Color::new(0.18, 0.18, 0.18, 1.0);
pub const OUTLINE_COLOR: Color = Color::new(0.8, 0.8, 0.8, 1.0);
pub const OUTLINE_THICKNESS: f32 = 4.0;

const AUTOMATA_WIDTH: usize = 200;
const AUTOMATA_HEIGHT: usize = 100;
const SIM_WIDTH: f32 = 500.;
const SIM_HEIGHT: f32 = 300.;

const NUM_BOIDS: usize = 500;
const COLORLIFE_PARTICLES: usize = 3000;
const FLUID_PARTICLES: usize = 2000;

struct Video {
    pub frames: Vec<Image>,
    pub framerate: u32,
}

#[macroquad::main("window_config")]
async fn main() {
    // let mut sim = default_sim();

    // generate_bitmaps().await;
    // generate_distance_fields().await;
    let bad_apple = load_distance_fields().await;

    let fluid = FluidSim::init(FLUID_PARTICLES, SIM_WIDTH, SIM_HEIGHT, bad_apple);
    let mut sim = ComponentFrame::relative_to_screen(fluid, vec2(0.2, 0.2), vec2(0.6, 0.6));

    // let start = Instant::now();
    loop {
        clear_background(BG_COLOR);

        // let now = Instant::now();
        // let time_since_start = now - start;
        // let frame = (time_since_start.as_secs_f32() * 30.).round() as usize; // 30 fps

        // TODO: Remove
        // let frame = frame % bad_apple.len();
        // let bitmap = &mut bad_apple[frame];
        // bitmap.draw();

        handle_sim_selection(&mut sim);
        update(&mut sim);
        draw(&mut sim);
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

        println!("loading {}", file.file_name().to_str().unwrap());
        let image = load_image(path.to_str().unwrap()).await.unwrap();

        frames.push(image);
    }

    Video {
        frames,
        framerate: 30,
    }
}

async fn generate_bitmaps() {
    let mut out = File::create("./resources/badapple/bitmaps.bin").unwrap();

    let mut files = std::fs::read_dir("./resources/badapple/frames/")
        .unwrap()
        .map(|f| f.unwrap())
        .collect::<Vec<_>>();
    files.sort_by_key(|file| file.file_name());

    for file in files.iter() {
        let path = file.path();
        let image = load_image(path.to_str().unwrap()).await.unwrap();
        println!("writing {}", file.file_name().to_str().unwrap());

        let bitmap = BinaryBitmap::from_image(&image);
        let bin = bitmap.grid.iter().map(|b| *b as u8).collect::<Vec<_>>();

        out.write_all(&bin).unwrap();
    }
}

async fn generate_distance_fields() {
    let mut out = File::create("./resources/badapple/distance_fields.bin").unwrap();
    let bitmaps = load_bitmaps().await;

    for (i, bitmap) in bitmaps.iter().enumerate() {
        println!("generating distance field #{i}",);

        let distance_field = DistanceField::from(bitmap);
        let bin = distance_field
            .grid
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>();

        out.write_all(&bin).unwrap();
    }
}

async fn load_bitmaps() -> Vec<BinaryBitmap> {
    println!("loading bitmaps.bin...");
    let mut file = File::open("./resources/badapple/bitmaps.bin").unwrap();
    let mut obstacles = Vec::new();
    let mut buffer = [0; BADAPPLE_X * BADAPPLE_Y];

    while let Ok(()) = file.read_exact(&mut buffer) {
        let data: Vec<bool> = buffer.iter().map(|byte| *byte >= 1).collect();
        let grid = Grid::new(data, BADAPPLE_X, BADAPPLE_Y);
        let bitmap = BinaryBitmap::new(grid);
        obstacles.push(bitmap);
    }

    obstacles
}

async fn load_distance_fields() -> Vec<DistanceField> {
    println!("loading distance_fields.bin...");
    let mut file = File::open("./resources/badapple/distance_fields.bin").unwrap();
    let mut distance_fields = Vec::new();
    let mut buffer = [0; BADAPPLE_X * BADAPPLE_Y * 4];

    while let Ok(()) = file.read_exact(&mut buffer) {
        let data: Vec<f32> = buffer
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let grid = Grid::new(data, BADAPPLE_X, BADAPPLE_Y);
        let distance_field = DistanceField::new(grid);
        distance_fields.push(distance_field);
    }

    distance_fields
}

// fn default_sim() -> ComponentFrame {
//     let default_sim = FluidSim::init(FLUID_PARTICLES, SIM_WIDTH, SIM_HEIGHT);
//     ComponentFrame::relative_to_screen(default_sim, vec2(0.2, 0.2), vec2(0.6, 0.6))
// }

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
        // sim.set_component(FluidSim::init(FLUID_PARTICLES, SIM_WIDTH, SIM_HEIGHT));
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
