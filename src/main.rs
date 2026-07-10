mod buffer;
mod frame;
mod grid;
mod simulations;
mod traits;

use frame::{Frame, FrameOutline, Layout};
use simulations::*;
use traits::*;

use macroquad::prelude::*;
use std::collections::HashMap;

pub const BG_COLOR: Color = Color::new(0.18, 0.18, 0.18, 1.0);
pub const OUTLINE_COLOR: Color = Color::new(0.8, 0.8, 0.8, 1.0);
pub const OUTLINE_THICKNESS: f32 = 4.0;

#[derive(Default)]
struct Components {
    inner: HashMap<&'static str, Box<dyn Component>>,
}

impl Components {
    pub fn init() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn add(&mut self, name: &'static str, component: impl Component + 'static) {
        self.inner.insert(name, Box::new(component));
    }

    pub fn update(&mut self) {
        for component in self.inner.values_mut() {
            component.update();
        }
    }

    pub async fn draw(&self) {
        for component in self.inner.values() {
            component.draw();
        }
    }
}

#[macroquad::main("window_config")]
async fn main() {
    const NUM_BOIDS: usize = 500;

    let mut components = Components::init();

    // Frames
    let sidebar = Layout::new(None, 0.05, 0.05, 0.19, 0.9);
    let simulation = Layout::new(None, 0.25, 0.05, 0.7, 0.9);

    // Sidebar components
    components.add(
        "sidebar-outline",
        FrameOutline::new(sidebar, OUTLINE_THICKNESS, OUTLINE_COLOR),
    );

    // Simulation components
    components.add(
        "simulation-outline",
        FrameOutline::new(simulation, OUTLINE_THICKNESS, OUTLINE_COLOR),
    );
    // components.add("simulation-content", Boids::init(simulation, NUM_BOIDS));
    // components.add("simulation-content", Colorlife::init(simulation, 3000));
    components.add("simulation-content", SimpleFluidSim::init(simulation, 1000));

    let target = render_target(512, 512);
    target.texture.set_filter(FilterMode::Nearest);

    loop {
        components.update();

        if is_key_pressed(KeyCode::Enter) {
            // components.add("simulation-content", Boids::init(simulation, NUM_BOIDS));
            // components.add("simulation-content", Colorlife::init(simulation, 3000));
            components.add("simulation-content", SimpleFluidSim::init(simulation, 1000));
        }

        if is_key_pressed(KeyCode::Space) {
            components.add(
                "simulation-content",
                Conway::random(simulation, _CONWAY, 0.75, 200, 200),
            );
        }

        if is_key_pressed(KeyCode::A) {
            components.add("simulation-content", Colorlife::init(simulation, 2000));
        }

        clear_background(BG_COLOR);
        components.draw().await;
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
