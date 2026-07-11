use crate::component::Frame;
use crate::grid::Grid;
use crate::shaders::liquid_material;
use crate::traits::*;

use macroquad::prelude::*;
use macroquad::rand::RandomRange;
use rayon::prelude::*;
use std::time::{Duration, Instant};

const ITERATIONS_PER_UPDATE: usize = 3;

const MIN_SPEED: f32 = 0.0;
const MAX_SPEED: f32 = 200.0;

const SMOOTHING_RADIUS: f32 = 50.;
const REST_DENSITY: f32 = 20.;
const VISCOSITY: f32 = 0.01;

const DELTA_DAMPENING_FACTOR: f32 = 0.8;
const VELOCITY_DAMPENING_FACTOR: f32 = 0.98;

const PARTICLE_MASS: f32 = 1.;
const GRAVITY: f32 = 0.;

const RESTITUTION_COEFFICIENT: f32 = 0.3;

const PARTICLE_DRAW_SIZE: f32 = 5.;

#[derive(Clone, Debug)]
pub struct FluidParticle {
    index: usize,
    pos: Vec2,
    vel: Vec2,
    predicted_pos: Vec2,
}

impl FluidParticle {
    #[allow(unused)]
    pub fn new(index: usize, x: f32, y: f32) -> Self {
        Self {
            index,
            pos: Vec2::new(x, y),
            vel: Vec2::new(0., 0.),
            predicted_pos: Vec2::new(x, y),
        }
    }

    pub fn clamp_speed(&mut self) {
        if self.vel.length() < MIN_SPEED {
            self.vel = self.vel.normalize() * MIN_SPEED;
        } else if self.vel.length() > MAX_SPEED {
            self.vel = self.vel.normalize() * MAX_SPEED;
        }
    }

    pub fn bounce(&mut self, sim_width: f32, sim_height: f32) {
        const MIN_DISTANCE_FROM_EDGE: f32 = 1. + PARTICLE_DRAW_SIZE / 2.;

        if self.pos.x < MIN_DISTANCE_FROM_EDGE {
            self.pos.x = MIN_DISTANCE_FROM_EDGE;
            self.vel.x *= -RESTITUTION_COEFFICIENT;
        } else if self.pos.x > sim_width - MIN_DISTANCE_FROM_EDGE {
            self.pos.x = sim_width - MIN_DISTANCE_FROM_EDGE;
            self.vel.x *= -RESTITUTION_COEFFICIENT;
        }

        if self.pos.y < MIN_DISTANCE_FROM_EDGE {
            self.pos.y = MIN_DISTANCE_FROM_EDGE;
            self.vel.y *= -RESTITUTION_COEFFICIENT;
        } else if self.pos.y > sim_height - MIN_DISTANCE_FROM_EDGE {
            self.pos.y = sim_height - MIN_DISTANCE_FROM_EDGE;
            self.vel.y *= -RESTITUTION_COEFFICIENT;
        }
    }

    pub fn apply_gravity(&mut self, deltatime: Duration) {
        self.vel += Vec2::new(0., GRAVITY) * deltatime.as_secs_f32();
    }

    pub fn reset_predicted_pos(&mut self, deltatime: Duration) {
        self.predicted_pos = self.pos + self.vel * deltatime.as_secs_f32();
    }

    pub fn update_velocity(&mut self, deltatime: Duration) {
        self.vel = (self.predicted_pos - self.pos) / deltatime.as_secs_f32();
        self.vel *= VELOCITY_DAMPENING_FACTOR;
        self.clamp_speed();
    }

    pub fn apply_viscosity(&mut self, viscosity_force: Vec2) {
        self.vel += viscosity_force * VISCOSITY;
        self.clamp_speed();
    }

    pub fn commit_new_position(&mut self, sim_width: f32, sim_height: f32) {
        self.pos = self.predicted_pos;
        self.bounce(sim_width, sim_height);
    }
}

#[derive(Clone, Debug)]
pub struct FluidSim {
    sim_width: f32,
    sim_height: f32,
    particles: Vec<FluidParticle>,
    densities: Vec<f32>,
    lambdas: Vec<f32>,
    position_deltas: Vec<Vec2>,
    viscosity_forces: Vec<Vec2>,
    chunks: Grid<Vec<usize>>,
    last_update: Instant,
    draw_material: Material,
    fluid_render_target: RenderTarget,
}

impl FluidSim {
    #[allow(unused)]
    pub fn new(particles: Vec<FluidParticle>, sim_width: f32, sim_height: f32) -> Self {
        let ideal_chunk_size = SMOOTHING_RADIUS;
        let columns = (sim_width / ideal_chunk_size).floor() as usize;
        let rows = (sim_height / ideal_chunk_size).floor() as usize;
        let num_particles = particles.len();

        let fluid_render_target = render_target(sim_width as u32, sim_height as u32);
        fluid_render_target.texture.set_filter(FilterMode::Nearest);

        Self {
            sim_width,
            sim_height,
            particles,
            densities: vec![0.; num_particles],
            lambdas: vec![0.; num_particles],
            position_deltas: vec![Vec2::ZERO; num_particles],
            viscosity_forces: vec![Vec2::ZERO; num_particles],
            chunks: Grid::with_defaults(columns, rows),
            last_update: Instant::now(),
            draw_material: liquid_material(),
            fluid_render_target,
        }
    }

    #[allow(unused)]
    pub fn init(num_particles: usize, sim_width: f32, sim_height: f32) -> Self {
        let mut particles = Vec::new();

        for i in 0..num_particles {
            let x = rand::gen_range(10., sim_width - 10.);
            let y = rand::gen_range(sim_height * 0.25 - 10., sim_height - 10.);
            particles.push(FluidParticle::new(i, x, y));
        }

        Self::new(particles, sim_width, sim_height)
    }

    fn update_chunks(&mut self) {
        let ideal_chunk_size = SMOOTHING_RADIUS;
        let columns = (self.sim_width / ideal_chunk_size).floor();
        let rows = (self.sim_height / ideal_chunk_size).floor();

        // Reset all chunks.
        for chunk in self.chunks.iter_mut() {
            chunk.clear();
        }

        // Grow the grid if needed (eg. if the frame size increases)
        self.chunks
            .resize_with_defaults(columns as usize, rows as usize);

        // Register all particles within their current chunks.
        for particle in &self.particles {
            if let Some(chunk) = self.chunks.get_mut_by_pos(
                particle.predicted_pos,
                vec2(0., 0.),
                vec2(self.sim_width, self.sim_height),
            ) {
                chunk.push(particle.index);
            }
        }
    }

    fn reset_predicted_positions(&mut self, deltatime: Duration) {
        self.particles.par_iter_mut().for_each(|particle| {
            particle.apply_gravity(deltatime);
            particle.reset_predicted_pos(deltatime);
        });
    }

    fn update_densities(&mut self) {
        const EPSILON: f32 = 1e-6;

        let particles = &self.particles;
        let chunks = &self.chunks;

        self.densities
            .par_iter_mut()
            .zip(self.lambdas.par_iter_mut())
            .enumerate()
            .for_each(|(i, (density, lambda))| {
                let position = particles[i].predicted_pos;
                let mut local_density = 0.0;

                let mut gradient_sum = 0.;
                let mut self_gradient = Vec2::ZERO;

                for chunk in chunks.get_neighbourhood_at_pos(
                    position,
                    1,
                    vec2(0., 0.),
                    vec2(self.sim_width, self.sim_height),
                ) {
                    for j in chunk.iter().copied() {
                        let displacement = particles[j].predicted_pos - particles[i].predicted_pos;
                        let distance_squared = displacement.length_squared();

                        if distance_squared < SMOOTHING_RADIUS * SMOOTHING_RADIUS {
                            let distance = distance_squared.sqrt();
                            local_density += PARTICLE_MASS * Self::smoothing_kernel(distance);

                            let gradient = Self::pressure_gradient(displacement) / REST_DENSITY;
                            gradient_sum += gradient.length_squared();
                            self_gradient += gradient;
                        }
                    }
                }

                gradient_sum += self_gradient.length_squared();

                let constraint = local_density / REST_DENSITY - 1.;

                let calculated_lambda = -constraint / (gradient_sum + EPSILON);

                *density = local_density;
                *lambda = calculated_lambda;
            });
    }

    fn update_position_deltas(&mut self) {
        let particles = &self.particles;
        let lambdas = &self.lambdas;
        let chunks = &self.chunks;

        self.position_deltas
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, delta)| {
                *delta = Vec2::ZERO;
                let position = particles[i].predicted_pos;

                for chunk in chunks.get_neighbourhood_at_pos(
                    position,
                    1,
                    vec2(0., 0.),
                    vec2(self.sim_width, self.sim_height),
                ) {
                    for j in chunk.iter().copied() {
                        if i == j {
                            continue;
                        }
                        let displacement = particles[j].predicted_pos - position;

                        let s_corr = -0.001
                            * (Self::smoothing_kernel(displacement.length())
                                / Self::smoothing_kernel(0.3 * SMOOTHING_RADIUS))
                            .powi(4);

                        let gradient = Self::pressure_gradient(displacement);
                        *delta += ((lambdas[i] + lambdas[j] + s_corr) * gradient) / REST_DENSITY;
                    }
                }
            });
    }

    fn apply_position_deltas(&mut self) {
        let position_deltas = &self.position_deltas;
        self.particles
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, particle)| {
                particle.predicted_pos += position_deltas[i] * DELTA_DAMPENING_FACTOR;
            });
    }

    fn calculate_viscosity_forces(&mut self) {
        const EPSILON: f32 = 1e-3;

        let particles = &self.particles;
        let chunks = &self.chunks;

        self.viscosity_forces
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, viscosity)| {
                let position = particles[i].predicted_pos;
                let mut viscosity_force = Vec2::ZERO;

                let mut weighted_velocity = Vec2::ZERO;
                let mut sum_of_weights = 0.0;

                for chunk in chunks.get_neighbourhood_at_pos(
                    position,
                    1,
                    vec2(0., 0.),
                    vec2(self.sim_width, self.sim_height),
                ) {
                    for j in chunk.iter().copied() {
                        let displacement = particles[j].pos - particles[i].pos;
                        let distance_squared = displacement.length_squared();

                        if i != j {
                            // Nudge overlapping particles
                            if distance_squared == 0. {
                                let nudge_x = RandomRange::gen_range(-EPSILON, EPSILON);
                                let nudge_y = RandomRange::gen_range(-EPSILON, EPSILON);
                                let nudge = vec2(nudge_x, nudge_y);
                                weighted_velocity += nudge;
                            } else {
                                let distance = distance_squared.sqrt();

                                if distance < SMOOTHING_RADIUS {
                                    let w = Self::smoothing_kernel(distance);
                                    weighted_velocity += particles[j].vel * w;
                                    sum_of_weights += w;
                                }
                            }
                        }
                    }
                }

                if sum_of_weights > 0.0 {
                    let average_velocity = weighted_velocity / sum_of_weights;
                    viscosity_force = average_velocity - particles[i].vel;
                }

                *viscosity = viscosity_force;
            });
    }

    fn smoothing_kernel(distance: f32) -> f32 {
        Self::poly6_smoothing_kernel(distance)
    }

    #[allow(unused)]
    fn quadratic_smoothing_kernel(distance: f32) -> f32 {
        if distance >= SMOOTHING_RADIUS {
            return 0.0;
        }

        (1.0 - distance / SMOOTHING_RADIUS).powi(2)
    }

    fn poly6_smoothing_kernel(distance: f32) -> f32 {
        // Can't use f32::powi(8) here as it is not const
        const SMOOTHING_RADIUS_POW8: f32 = SMOOTHING_RADIUS
            * SMOOTHING_RADIUS
            * SMOOTHING_RADIUS
            * SMOOTHING_RADIUS
            * SMOOTHING_RADIUS
            * SMOOTHING_RADIUS
            * SMOOTHING_RADIUS
            * SMOOTHING_RADIUS;

        const SMOOTHING_CONSTANT: f32 = 4.0 / (std::f32::consts::PI * SMOOTHING_RADIUS_POW8);

        if distance >= SMOOTHING_RADIUS {
            return 0.0;
        }

        let smoothing_radius_squared = SMOOTHING_RADIUS.powi(2);
        let x = smoothing_radius_squared - distance.powi(2);

        SMOOTHING_CONSTANT * x.powi(3)
    }

    fn pressure_gradient(displacement: Vec2) -> Vec2 {
        let distance = displacement.length();
        if distance == 0.0 || distance >= SMOOTHING_RADIUS {
            return Vec2::ZERO;
        }

        let direction = displacement / distance;
        let strength = SMOOTHING_RADIUS - distance;

        direction * strength * strength
    }

    fn apply_external_forces(&mut self, mouse_pos: Vec2) {
        self.apply_mouse_interaction_forces(mouse_pos);
    }

    fn apply_mouse_interaction_forces(&mut self, mouse_pos: Vec2) {
        const MOUSE_FORCE: f32 = 1000.;
        if is_mouse_button_down(MouseButton::Left) {
            for particle in &mut self.particles {
                let displacement = mouse_pos - particle.pos;

                if (mouse_pos - particle.pos).length() < SMOOTHING_RADIUS {
                    particle.vel -= (displacement / SMOOTHING_RADIUS) * -MOUSE_FORCE;
                }
            }
        } else if is_mouse_button_down(MouseButton::Right) {
            for particle in &mut self.particles {
                let displacement = mouse_pos - particle.pos;

                if (mouse_pos - particle.pos).length() < SMOOTHING_RADIUS {
                    particle.vel -= (displacement / SMOOTHING_RADIUS) * MOUSE_FORCE;
                }
            }
        }
    }

    #[allow(unused)]
    fn draw_fluid_texture(&self, camera: &mut Camera2D) {
        let target = camera.render_target.take();
        camera.render_target = Some(self.fluid_render_target.clone());
        set_camera(camera);

        clear_background(BLANK);
        for particle in &self.particles {
            draw_circle(
                particle.pos.x,
                particle.pos.y,
                30.,
                Color::new(1.0, 1.0, 1.0, 0.04),
            );
        }

        camera.render_target = target;
        set_camera(camera);
        gl_use_material(&self.draw_material);
        draw_texture_ex(
            &self.fluid_render_target.texture,
            0.,
            0.,
            WHITE,
            DrawTextureParams {
                flip_y: true,
                ..Default::default()
            },
        );
        gl_use_default_material();
    }

    #[allow(unused)]
    fn draw_particles(&self, _relative_mouse_pos: Vec2) {
        for (i, particle) in self.particles.iter().enumerate() {
            let density = self.densities[i];
            let speed = particle.vel.length();
            let red = (density * 60.).clamp(0., 1.);
            let green = (speed / 100.).clamp(0., 1.);

            // if (relative_mouse_pos - particle.pos).length() < SMOOTHING_RADIUS {
            //     draw_circle(
            //         particle.pos.x,
            //         particle.pos.y,
            //         3.,
            //         Color::new(0.8, 0.2, 0.2, 0.6),
            //     );
            // } else {
            draw_circle(
                particle.pos.x,
                particle.pos.y,
                3.,
                Color::new(red, green, 1., 0.8),
            );
            // }
        }
    }

    #[allow(unused)]
    fn draw_data_particles(&self) {
        let min_density = self
            .densities
            .iter()
            .reduce(|a, b| if a < b { a } else { b })
            .unwrap();
        let max_density = self
            .densities
            .iter()
            .reduce(|a, b| if a > b { a } else { b })
            .unwrap();
        let min_speed = self
            .particles
            .iter()
            .map(|particle| particle.vel.length_squared())
            .reduce(|a, b| if a < b { a } else { b })
            .unwrap()
            .sqrt();
        let max_speed = self
            .particles
            .iter()
            .map(|particle| particle.vel.length_squared())
            .reduce(|a, b| if a > b { a } else { b })
            .unwrap()
            .sqrt();

        clear_background(BLANK);
        for (i, particle) in self.particles.iter().enumerate() {
            let density = self.densities[i];
            let speed = particle.vel.length();

            let red = ((speed - min_speed) / max_speed).clamp(0., 1.);
            let green = ((density - min_density) / max_density).clamp(0., 1.);
            let blue = 1.;

            draw_circle(
                particle.pos.x,
                particle.pos.y,
                3.,
                Color::new(red, green, blue, 1.),
            );
        }
    }
}

impl Update for FluidSim {
    fn update(&mut self, frame: &Frame) {
        let update_start = Instant::now();
        let deltatime = update_start - self.last_update;

        let mouse_pos = frame.relative_mouse_pos();
        self.apply_external_forces(mouse_pos);

        self.reset_predicted_positions(deltatime);
        self.update_chunks();

        for _iteration in 1..=ITERATIONS_PER_UPDATE {
            self.update_densities();
            self.update_position_deltas();
            self.apply_position_deltas();
        }

        self.particles
            .par_iter_mut()
            .for_each(|particle| particle.update_velocity(deltatime));

        self.calculate_viscosity_forces();

        self.particles
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, particle)| particle.apply_viscosity(self.viscosity_forces[i]));

        self.particles
            .par_iter_mut()
            .for_each(|particle| particle.commit_new_position(self.sim_width, self.sim_height));

        self.last_update = update_start;
    }
}

impl Draw for FluidSim {
    fn draw(&self, frame: &mut Frame) {
        let mouse_pos = frame.relative_mouse_pos();
        // self.draw_fluid_texture(frame.camera());
        // self.draw_particles(mouse_pos);
        self.draw_data_particles();
    }
}

impl HasSize for FluidSim {
    fn size(&self) -> Vec2 {
        vec2(self.sim_width, self.sim_height)
    }
}
