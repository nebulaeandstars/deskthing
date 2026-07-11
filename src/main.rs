mod buffer;
mod frame;
mod grid;
mod shaders;
mod simulations;
mod traits;

use frame::{DrawFrame, DrawFrameLayout};
use simulations::*;
use traits::*;

use macroquad::prelude::*;

pub const BG_COLOR: Color = Color::new(0.18, 0.18, 0.18, 1.0);
pub const OUTLINE_COLOR: Color = Color::new(0.8, 0.8, 0.8, 1.0);
pub const OUTLINE_THICKNESS: f32 = 4.0;

#[macroquad::main("window_config")]
async fn main() {
    const CONWAY_DIMENSIONS: (usize, usize) = (200, 200);
    const CONWAY_FILL_PERCENT: f32 = 0.75;
    const NUM_BOIDS: usize = 500;
    const COLORLIFE_PARTICLES: usize = 3000;
    const FLUID_PARTICLES: usize = 1000;

    const SIM_WIDTH: f32 = 400.;
    const SIM_HEIGHT: f32 = 300.;

    // Frames
    let sidebar = DrawFrameLayout::new(None, 0.05, 0.05, 0.19, 0.9);
    let mut simulation = DrawFrameLayout::new(None, 0.25, 0.05, 0.7, 0.9);

    // Sidebar components
    // components.add(
    //     "sidebar-outline",
    //     FrameOutline::new(sidebar, OUTLINE_THICKNESS, OUTLINE_COLOR),
    // );

    // Simulation components
    // components.add(
    //     "simulation-outline",
    //     FrameOutline::new(simulation, OUTLINE_THICKNESS, OUTLINE_COLOR),
    // );
    // components.add("simulation-content", FluidSim::init(simulation, FLUID_PARTICLES));

    let fluid_sim = FluidSim::init(FLUID_PARTICLES, SIM_WIDTH, SIM_HEIGHT);
    let mut test_frame =
        frame::Component::relative_to_screen(fluid_sim, vec2(0.2, 0.2), vec2(0.6, 0.6));

    let target = render_target(512, 512);
    target.texture.set_filter(FilterMode::Nearest);

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
        //     components.add("simulation-content", Boids::init(simulation, NUM_BOIDS));
        // }

        // if is_key_pressed(KeyCode::C) {
        //     components.add(
        //         "simulation-content",
        //         Colorlife::init(simulation, COLORLIFE_PARTICLES),
        //     );
        // }

        // if is_key_pressed(KeyCode::D) {
        //     components.add(
        //         "simulation-content",
        //         FluidSim::init(simulation, FLUID_PARTICLES),
        //     );
        // }

        clear_background(BG_COLOR);

        // TODO: remove
        // simulation.refresh();
        // fluid_sim.update();

        test_frame.refit_to_screen(vec2(0.2, 0.2), vec2(0.6, 0.6));
        test_frame.update();
        test_frame.draw();
        test_frame.draw_outline(4., WHITE);

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
