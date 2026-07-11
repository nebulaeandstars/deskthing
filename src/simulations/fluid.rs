use crate::frame::Layout;
use crate::grid::Grid;
use crate::shaders::liquid_material;
use crate::traits::*;
use crate::Frame;

use macroquad::prelude::*;
use rayon::prelude::*;
use std::time::{Duration, Instant};

const ITERATIONS_PER_UPDATE: usize = 3;

const MIN_SPEED: f32 = 0.0;
const MAX_SPEED: f32 = 200.0;

const SMOOTHING_RADIUS: f32 = 50.;
const REST_DENSITY: f32 = 50.;
const VISCOSITY: f32 = 0.01;

const DELTA_DAMPENING_FACTOR: f32 = 0.8;
const VELOCITY_DAMPENING_FACTOR: f32 = 0.98;

const PARTICLE_MASS: f32 = 1.;
const GRAVITY: f32 = 100.;

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
            self.vel = self.vel.normalize() * MIN_SPEED
        } else if self.vel.length() > MAX_SPEED {
            self.vel = self.vel.normalize() * MAX_SPEED
        }
    }

    pub fn bounce(&mut self, frame: &Frame) {
        const MIN_DISTANCE_FROM_EDGE: f32 = 1. + PARTICLE_DRAW_SIZE / 2.;

        if self.pos.x < MIN_DISTANCE_FROM_EDGE {
            self.pos.x = MIN_DISTANCE_FROM_EDGE;
            self.vel.x *= -RESTITUTION_COEFFICIENT;
        } else if self.pos.x > frame.width() - MIN_DISTANCE_FROM_EDGE {
            self.pos.x = frame.width() - MIN_DISTANCE_FROM_EDGE;
            self.vel.x *= -RESTITUTION_COEFFICIENT;
        }

        if self.pos.y < MIN_DISTANCE_FROM_EDGE {
            self.pos.y = MIN_DISTANCE_FROM_EDGE;
            self.vel.y *= -RESTITUTION_COEFFICIENT;
        } else if self.pos.y > frame.height() - MIN_DISTANCE_FROM_EDGE {
            self.pos.y = frame.height() - MIN_DISTANCE_FROM_EDGE;
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

    pub fn commit_new_position(&mut self, frame: &Frame) {
        self.pos = self.predicted_pos;
        self.bounce(frame);
    }
}

pub struct FluidSim {
    layout: Layout,
    particles: Vec<FluidParticle>,
    densities: Vec<f32>,
    lambdas: Vec<f32>,
    position_deltas: Vec<Vec2>,
    viscosity_forces: Vec<Vec2>,
    chunks: Grid<Vec<usize>>,
    last_update: Instant,
    draw_material: Material,
}

impl FluidSim {
    pub fn new(mut layout: Layout, particles: Vec<FluidParticle>) -> Self {
        layout.refresh();

        let ideal_chunk_size = SMOOTHING_RADIUS;
        let columns = (layout.frame.width() / ideal_chunk_size).floor() as usize;
        let rows = (layout.frame.height() / ideal_chunk_size).floor() as usize;
        let num_particles = particles.len();

        Self {
            layout,
            particles,
            densities: vec![0.; num_particles],
            lambdas: vec![0.; num_particles],
            position_deltas: vec![Vec2::ZERO; num_particles],
            viscosity_forces: vec![Vec2::ZERO; num_particles],
            chunks: Grid::with_defaults(columns, rows),
            last_update: Instant::now(),
            draw_material: liquid_material(),
        }
    }

    pub fn init(mut layout: Layout, num_particles: usize) -> Self {
        layout.refresh();

        let mut particles = Vec::new();

        for i in 0..num_particles {
            let x = rand::gen_range(100., layout.frame.width() - 100.);
            let y = rand::gen_range(100., layout.frame.height() - 100.);
            particles.push(FluidParticle::new(i, x, y));
        }

        Self::new(layout, particles)
    }

    fn frame(&self) -> Frame {
        self.layout.frame
    }

    fn update_chunks(&mut self) {
        let ideal_chunk_size = SMOOTHING_RADIUS;
        let columns = (self.frame().width() / ideal_chunk_size).floor();
        let rows = (self.frame().height() / ideal_chunk_size).floor();

        // Reset all chunks.
        for chunk in self.chunks.iter_mut() {
            chunk.clear()
        }

        // Grow the grid if needed (eg. if the frame size increases)
        self.chunks
            .resize_with_defaults(columns as usize, rows as usize);

        let frame = Frame::new(0., 0., screen_width(), screen_height());
        // Register all particles within their current chunks.
        for particle in &self.particles {
            if let Some(chunk) = self.chunks.get_mut_by_pos(particle.predicted_pos, frame) {
                chunk.push(particle.index);
            }
        }
    }

    fn reset_predicted_positions(&mut self, deltatime: Duration) {
        self.particles.par_iter_mut().for_each(|particle| {
            particle.apply_gravity(deltatime);
            particle.reset_predicted_pos(deltatime);
        })
    }

    fn update_densities(&mut self) {
        let particles = &self.particles;
        let chunks = &self.chunks;
        let frame = Frame::new(0., 0., screen_width(), screen_height());

        self.densities
            .par_iter_mut()
            .zip(self.lambdas.par_iter_mut())
            .enumerate()
            .for_each(|(i, (density, lambda))| {
                let position = particles[i].predicted_pos;
                let mut local_density = 0.0;

                let mut gradient_sum = 0.;
                let mut self_gradient = Vec2::ZERO;

                for chunk in chunks.get_neighbourhood_at_pos(position, 1, frame) {
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

                const EPSILON: f32 = 1e-6;
                let calculated_lambda = -constraint / (gradient_sum + EPSILON);

                *density = local_density;
                *lambda = calculated_lambda;
            });
    }

    fn update_position_deltas(&mut self) {
        let particles = &self.particles;
        let lambdas = &self.lambdas;
        let chunks = &self.chunks;
        let frame = Frame::new(0., 0., screen_width(), screen_height());

        self.position_deltas
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, delta)| {
                *delta = Vec2::ZERO;
                let position = particles[i].predicted_pos;

                for chunk in chunks.get_neighbourhood_at_pos(position, 1, frame) {
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
            })
    }

    fn calculate_viscosity_forces(&mut self) {
        let particles = &self.particles;
        let chunks = &self.chunks;
        let frame = Frame::new(0., 0., screen_width(), screen_height());

        self.viscosity_forces
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, viscosity)| {
                let position = particles[i].predicted_pos;
                let mut viscosity_force = Vec2::ZERO;

                let mut weighted_velocity = Vec2::ZERO;
                let mut sum_of_weights = 0.0;

                for chunk in chunks.get_neighbourhood_at_pos(position, 1, frame) {
                    for j in chunk.iter().copied() {
                        let displacement = particles[j].pos - particles[i].pos;
                        let distance_squared = displacement.length_squared();

                        if i != j {
                            let distance = distance_squared.sqrt();

                            if distance < SMOOTHING_RADIUS {
                                let w = Self::smoothing_kernel(distance);
                                weighted_velocity += particles[j].vel * w;
                                sum_of_weights += w;
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
        // let strength = ((SMOOTHING_RADIUS - distance) / SMOOTHING_RADIUS).powi(2);
        // let strength = 1.0 - (distance / SMOOTHING_RADIUS);
        let strength = SMOOTHING_RADIUS - distance;

        direction * strength * strength
    }

    fn apply_external_forces(&mut self) {
        self.apply_mouse_interaction_forces()
    }

    fn apply_mouse_interaction_forces(&mut self) {
        let frame = self.frame();

        const MOUSE_FORCE: f32 = 1000.;
        if is_mouse_button_down(MouseButton::Left) {
            let mouse_pos = Vec2::from(mouse_position());
            for particle in self.particles.iter_mut() {
                let displacement = mouse_pos - frame.pos() - particle.pos;

                if (Vec2::from(mouse_pos) - frame.pos() - particle.pos).length() < SMOOTHING_RADIUS
                {
                    particle.vel -= (displacement / SMOOTHING_RADIUS) * -MOUSE_FORCE;
                }
            }
        } else if is_mouse_button_down(MouseButton::Right) {
            let mouse_pos = Vec2::from(mouse_position());
            for particle in self.particles.iter_mut() {
                let displacement = mouse_pos - frame.pos() - particle.pos;

                if (Vec2::from(mouse_pos) - frame.pos() - particle.pos).length() < SMOOTHING_RADIUS
                {
                    particle.vel -= (displacement / SMOOTHING_RADIUS) * MOUSE_FORCE;
                }
            }
        }
    }

    fn draw_fluid_texture(&self) {
        let target = render_target(self.frame().width() as u32, self.frame().height() as u32);
        target.texture.set_filter(FilterMode::Nearest);

        // Draw into the target
        set_camera(&Camera2D {
            render_target: Some(target.clone()),
            zoom: vec2(2.0 / target.texture.width(), -2.0 / target.texture.height()),
            target: vec2(target.texture.width() / 2.0, target.texture.height() / 2.0),
            ..Default::default()
        });

        clear_background(Color::new(0., 0., 0., 0.));
        for particle in &self.particles {
            draw_circle(
                particle.pos.x,
                particle.pos.y,
                30.,
                Color::new(1.0, 1.0, 1.0, 0.04),
            );
        }

        // Draw to the screen
        set_default_camera();

        // Draw the generated texture
        gl_use_material(&self.draw_material);
        let offset = crate::OUTLINE_THICKNESS / 2.;
        draw_texture_ex(
            &target.texture,
            self.frame().x() + offset,
            self.frame().y() + offset,
            Color::new(0., 0., 0., 0.),
            DrawTextureParams {
                dest_size: Some(vec2(
                    self.frame().width() - offset * 2.,
                    self.frame().height() - offset * 2.,
                )),
                flip_y: true,
                ..Default::default()
            },
        );
        gl_use_default_material();
    }

    fn draw_particles(&self) {
        let mouse_pos = Vec2::from(mouse_position());
        for particle in &self.particles {
            if (Vec2::from(mouse_pos) - self.frame().pos() - particle.pos).length()
                < SMOOTHING_RADIUS
            {
                draw_circle(
                    particle.pos.x + self.frame().x(),
                    particle.pos.y + self.frame().y(),
                    3.,
                    Color::new(0.8, 0.2, 0.2, 0.6),
                );
            } else {
                draw_circle(
                    particle.pos.x + self.frame().x(),
                    particle.pos.y + self.frame().y(),
                    3.,
                    Color::new(0.2, 0.6, 0.8, 0.6),
                );
            }
        }
    }
}

impl Update for FluidSim {
    fn update(&mut self) {
        let update_start = Instant::now();
        let deltatime = update_start - self.last_update;
        self.layout.refresh();
        let frame = self.frame();

        self.apply_external_forces();

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
            .for_each(|particle| particle.commit_new_position(&frame));

        self.last_update = update_start;
    }
}

impl Draw for FluidSim {
    fn draw(&self) {
        self.draw_fluid_texture();
        self.draw_particles();
    }
}
