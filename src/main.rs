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
mod engine;
mod grid;
mod rng;
mod shaders;
mod simulations;
mod traits;

use component::ComponentFrame;
use engine::{Audio, ImageBuffer, MacroquadAudio, SoundId, to_mq_color};
use simulations::*;

use grid::Grid;
use macroquad::prelude::*;
use std::fs::File;
use std::io::{Read, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

const BADAPPLE_X: usize = 320;
const BADAPPLE_Y: usize = 240;

// The app palette, in engine colours so simulations can use it too. The host
// converts when it makes its own macroquad calls.
pub const BG_COLOR: engine::Color = engine::Color::new(0.18, 0.18, 0.18, 1.0);
pub const OUTLINE_COLOR: engine::Color = engine::Color::new(0.8, 0.8, 0.8, 1.0);
pub const OUTLINE_THICKNESS: f32 = 4.0;

const AUTOMATA_WIDTH: usize = 200;
const AUTOMATA_HEIGHT: usize = 100;
const SIM_WIDTH: f32 = 500.;
const SIM_HEIGHT: f32 = 300.;

const NUM_BOIDS: usize = 500;
const COLORLIFE_PARTICLES: usize = 3000;
const FLUID_PARTICLES: usize = 2000;

/// A plain tank of fluid under gravity, with no obstacle and no margin — the
/// whole simulated area is drawn, walls included. Useful for seeing what the
/// solver is actually doing, which the Bad Apple version hides behind its
/// margin.
const TANK_PARTICLES: usize = 2000;
const TANK_GRAVITY: f32 = 400.;

const BADAPPLE_AUDIO_PATH: &str = "./resources/badapple/badapple.ogg";
const SOUNDTRACK_VOLUME: f32 = 0.5;

#[macroquad::main("window_config")]
async fn main() {
    // generate_bitmaps().await;
    // generate_distance_fields().await;
    let bad_apple = load_distance_fields().await;

    let mut audio = MacroquadAudio::new();
    let soundtrack = load_soundtrack(&mut audio).await;

    let mut sim = default_sim(&bad_apple);
    play_soundtrack(&mut audio, soundtrack);

    // One clock for the whole app. Simulations are told how much time passed;
    // none of them reads a clock itself.
    let mut last_frame = Instant::now();

    loop {
        clear_background(to_mq_color(BG_COLOR));

        let now = Instant::now();
        let dt = now - last_frame;
        last_frame = now;

        handle_sim_selection(&mut sim, &bad_apple, &mut audio, soundtrack);
        update(&mut sim, dt);
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

// The asset pipeline, all of it derived from the committed badapple.mp4 and
// all of it gitignored, so a fresh clone has to regenerate it:
//
//   ffmpeg -i resources/badapple/badapple.mp4 \
//     resources/badapple/frames/%05d.png
//   ffmpeg -i resources/badapple/badapple.mp4 -vn -c:a libvorbis -q:a 4 \
//     resources/badapple/badapple.ogg
//
// then uncomment the two generator calls in `main` and run once, which turns
// the PNG frames into bitmaps.bin and then into distance_fields.bin. The
// distance fields are required to start; the soundtrack is not.
#[allow(dead_code)]
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

        let bitmap = BinaryBitmap::from_image(&to_image_buffer(&image));
        let bin = bitmap.grid.iter().map(|b| *b as u8).collect::<Vec<_>>();

        out.write_all(&bin).unwrap();
    }
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

const DISTANCE_FIELDS_PATH: &str = "./resources/badapple/distance_fields.bin";

async fn load_distance_fields() -> Arc<Vec<DistanceField>> {
    println!("loading {DISTANCE_FIELDS_PATH}...");

    let mut file = File::open(DISTANCE_FIELDS_PATH).unwrap_or_else(|error| {
        panic!(
            "could not open {DISTANCE_FIELDS_PATH}: {error}\n\
             It is gitignored — regenerate it by uncommenting the generator \
             calls in main, and run from the repository root."
        )
    });

    let mut distance_fields = Vec::new();
    let mut buffer = [0; BADAPPLE_X * BADAPPLE_Y * 4];

    loop {
        match file.read_exact(&mut buffer) {
            Ok(()) => {}
            // A clean end of file is the only acceptable way out. Anything
            // else — a truncated final frame, a read error — used to be
            // indistinguishable from "done", leaving a silently short video.
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => panic!(
                "failed reading {DISTANCE_FIELDS_PATH} after {} frames: {error}",
                distance_fields.len()
            ),
        }

        let data: Vec<f32> = buffer
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let grid = Grid::new(data, BADAPPLE_X, BADAPPLE_Y);
        distance_fields.push(DistanceField::new(grid));
    }

    assert!(
        !distance_fields.is_empty(),
        "{DISTANCE_FIELDS_PATH} held no complete frames"
    );

    println!("loaded {} frames", distance_fields.len());
    Arc::new(distance_fields)
}

/// Loads the video's soundtrack, if it is there.
///
/// Unlike the distance fields this is optional: a missing soundtrack costs you
/// the music, not the program. It is generated from the committed mp4 and
/// gitignored, so a fresh clone will not have it until the pipeline is run.
async fn load_soundtrack(audio: &mut MacroquadAudio) -> Option<SoundId> {
    match audio.load(BADAPPLE_AUDIO_PATH).await {
        Ok(sound) => Some(sound),
        Err(error) => {
            println!(
                "no soundtrack ({BADAPPLE_AUDIO_PATH}: {error}); \
                 regenerate it with:\n  \
                 ffmpeg -i resources/badapple/badapple.mp4 -vn -c:a libvorbis \
                 -q:a 4 {BADAPPLE_AUDIO_PATH}"
            );
            None
        }
    }
}

/// Starts the soundtrack from the top.
///
/// It loops, and the simulation's frame index is driven by accumulated real
/// time, so the two stay together without either being told about the other.
/// There is no way to read a sound's playback position, so this is a shared
/// start rather than genuine synchronisation.
fn play_soundtrack(audio: &mut MacroquadAudio, soundtrack: Option<SoundId>) {
    if let Some(sound) = soundtrack {
        audio.play(sound, true);
        audio.set_volume(sound, SOUNDTRACK_VOLUME);
    }
}

/// Bridges a macroquad-loaded image into the engine-agnostic buffer the
/// bitmap code expects. Only the asset pipeline needs this.
fn to_image_buffer(image: &Image) -> ImageBuffer {
    ImageBuffer::from_rgba8(image.bytes.clone(), image.width, image.height)
}

/// The video is shared by `Arc`, so rebuilding the fluid sim is a refcount
/// bump rather than a copy of every distance field.
fn fluid_sim(video: &Arc<Vec<DistanceField>>) -> FluidSim {
    FluidSim::init(
        FLUID_PARTICLES,
        SIM_WIDTH,
        SIM_HEIGHT,
        Arc::clone(video),
        rng::from_entropy(),
    )
}

fn tank_sim() -> FluidSim {
    FluidSim::init(
        TANK_PARTICLES,
        SIM_WIDTH,
        SIM_HEIGHT,
        Arc::new(Vec::new()),
        rng::from_entropy(),
    )
    .with_gravity(TANK_GRAVITY)
    .with_margin(0.)
}

fn default_sim(video: &Arc<Vec<DistanceField>>) -> ComponentFrame {
    ComponentFrame::relative_to_screen(fluid_sim(video), vec2(0.2, 0.2), vec2(0.6, 0.6))
}

/// The soundtrack belongs to the video, so it plays only while the simulation
/// showing that video is on screen.
fn handle_sim_selection(
    sim: &mut ComponentFrame,
    video: &Arc<Vec<DistanceField>>,
    audio: &mut MacroquadAudio,
    soundtrack: Option<SoundId>,
) {
    if is_key_pressed(KeyCode::A) {
        audio.stop_all();
        sim.set_component(Conway::random(
            _CONWAY,
            0.6,
            AUTOMATA_WIDTH,
            AUTOMATA_HEIGHT,
            rng::from_entropy(),
        ));
    } else if is_key_pressed(KeyCode::B) {
        audio.stop_all();
        sim.set_component(Boids::init(
            NUM_BOIDS,
            SIM_WIDTH,
            SIM_HEIGHT,
            rng::from_entropy(),
        ));
    } else if is_key_pressed(KeyCode::C) {
        audio.stop_all();
        sim.set_component(Colorlife::init(
            COLORLIFE_PARTICLES,
            SIM_WIDTH,
            SIM_HEIGHT,
            rng::from_entropy(),
        ));
    } else if is_key_pressed(KeyCode::D) {
        // Restarted from the top so the video and the music begin together.
        sim.set_component(fluid_sim(video));
        play_soundtrack(audio, soundtrack);
    } else if is_key_pressed(KeyCode::E) {
        audio.stop_all();
        sim.set_component(tank_sim());
    }
}

fn update(sim: &mut ComponentFrame, dt: Duration) {
    sim.refit_to_screen(vec2(0.2, 0.2), vec2(0.6, 0.6));
    sim.refit_to_component();
    sim.update(dt);
}

fn draw(sim: &mut ComponentFrame) {
    clear_background(to_mq_color(BG_COLOR));
    sim.draw();
    sim.draw_outline(4., WHITE);
}
