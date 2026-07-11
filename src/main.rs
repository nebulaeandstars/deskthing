mod buffer;
mod frame;
mod grid;
mod shaders;
mod simulations;
mod traits;

use simulations::*;
use traits::*;

use macroquad::prelude::*;

pub const BG_COLOR: Color = Color::new(0.18, 0.18, 0.18, 1.0);
pub const OUTLINE_COLOR: Color = Color::new(0.8, 0.8, 0.8, 1.0);
pub const OUTLINE_THICKNESS: f32 = 4.0;

#[macroquad::main("window_config")]
async fn main() {
    const CONWAY_DIMENSIONS: (usize, usize) = (200, 100);
    const CONWAY_FILL_PERCENT: f32 = 0.60;
    const NUM_BOIDS: usize = 500;
    const COLORLIFE_PARTICLES: usize = 3000;
    const FLUID_PARTICLES: usize = 1000;

    const SIM_WIDTH: f32 = 400.;
    const SIM_HEIGHT: f32 = 300.;

    let _conway = frame::Component::relative_to_screen(
        Conway::random(
            _CONWAY,
            CONWAY_FILL_PERCENT,
            CONWAY_DIMENSIONS.0,
            CONWAY_DIMENSIONS.1,
        ),
        vec2(0.2, 0.2),
        vec2(0.6, 0.6),
    );

    let _boids = frame::Component::relative_to_screen(
        Boids::init(NUM_BOIDS, SIM_WIDTH, SIM_HEIGHT),
        vec2(0.2, 0.2),
        vec2(0.6, 0.6),
    );

    let _colorlife = frame::Component::relative_to_screen(
        Colorlife::init(COLORLIFE_PARTICLES, SIM_WIDTH, SIM_HEIGHT),
        vec2(0.2, 0.2),
        vec2(0.6, 0.6),
    );

    let fluidsim = frame::Component::relative_to_screen(
        FluidSim::init(FLUID_PARTICLES, SIM_WIDTH, SIM_HEIGHT),
        vec2(0.2, 0.2),
        vec2(0.6, 0.6),
    );

    let target = render_target(512, 512);
    target.texture.set_filter(FilterMode::Nearest);

    let mut sim = fluidsim;

    loop {
        // if is_key_pressed(KeyCode::A) {
        //     components.add(
        //         "simulation-content",
        //         Conway::random(
        //             simulation,
        //             _CONWAY,
        //             CONWAY_FILL_PERCENT,
        //             CONWAY_DIMENSIONS.0,
        //             CONWAY_DIMENSIONS.1,
        //         ),
        //     );
        // }

        // if is_key_pressed(KeyCode::B) {
        //     sim_frame = frame::Component::relative_to_screen(
        //         Boids::init(NUM_BOIDS, SIM_WIDTH, SIM_HEIGHT),
        //         vec2(0.2, 0.2),
        //         vec2(0.6, 0.6),
        //     );
        // }

        // if is_key_pressed(KeyCode::C) {
        //     components.add(
        //         "simulation-content",
        //         Colorlife::init(simulation, COLORLIFE_PARTICLES),
        //     );
        // }

        // if is_key_pressed(KeyCode::D) {
        //     sim_frame = frame::Component::relative_to_screen(
        //         FluidSim::init(FLUID_PARTICLES, SIM_WIDTH, SIM_HEIGHT),
        //         vec2(0.2, 0.2),
        //         vec2(0.6, 0.6),
        //     );
        // }

        clear_background(BG_COLOR);

        // TODO: remove
        // simulation.refresh();
        // fluid_sim.update();

        sim.refit_to_screen(vec2(0.2, 0.2), vec2(0.6, 0.6));
        sim.update();
        sim.draw();
        sim.draw_outline(4., WHITE);

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
