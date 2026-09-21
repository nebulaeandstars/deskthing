use crate::engine::{
    BLANK, Canvas, Color, Effect, ImageBuffer, Input, MouseButton, TextureId, Vec2, vec2,
};
use crate::grid::Grid;
use crate::rng::{Rng, RngExt};
use crate::traits::{HasSize, Simulation};

use rayon::prelude::*;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;

const ITERATIONS_PER_UPDATE: usize = 4;

const MIN_SPEED: f32 = 0.0;
const MAX_SPEED: f32 = 200.0;

const SMOOTHING_RADIUS: f32 = 30.;
const REST_DENSITY: f32 = 2.;
const VISCOSITY: f32 = 0.1;

const DELTA_DAMPENING_FACTOR: f32 = 0.8;
const VELOCITY_DAMPENING_FACTOR: f32 = 0.98;

const PARTICLE_MASS: f32 = 1.;
const GRAVITY: f32 = 0.;

const VIDEO_FRAMERATE: f32 = 30.;

const RESTITUTION_COEFFICIENT: f32 = 0.3;

const PARTICLE_DRAW_SIZE: f32 = 5.;
// const OBSTACLE_COLOR: Color = Color::new(0.6, 0.6, 0.6, 1.);
const OBSTACLE_COLOR: Color = BLANK;
const OBSTACLE_BORDER: f32 = SMOOTHING_RADIUS;

// Large "infine" number to avoid NaNs
const INF: f32 = 1e20;

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

#[derive(Debug)]
pub struct FluidSim {
    sim_width: f32,
    sim_height: f32,
    video: Arc<Vec<DistanceField>>,
    particles: Vec<FluidParticle>,
    densities: Vec<f32>,
    lambdas: Vec<f32>,
    position_deltas: Vec<Vec2>,
    viscosity_forces: Vec<Vec2>,
    chunks: Grid<Vec<usize>>,
    rng: Rng,
    /// Time since the simulation started, accumulated from the `dt` handed to
    /// `update`. The simulation never reads a clock itself.
    elapsed: Duration,
    /// Index into `video`, pinned once per update so that every solver
    /// iteration and the draw pass all see the same obstacle.
    current_frame: usize,
    /// Scratch buffer for the obstacle, reused every frame, plus the canvas
    /// handle it is uploaded to. The handle is claimed on first draw because
    /// there is no canvas to ask at construction time.
    obstacle_image: ImageBuffer,
    obstacle_texture: Option<TextureId>,
}

impl FluidSim {
    pub fn new(
        particles: Vec<FluidParticle>,
        sim_width: f32,
        sim_height: f32,
        video: Arc<Vec<DistanceField>>,
        rng: Rng,
    ) -> Self {
        let ideal_chunk_size = SMOOTHING_RADIUS;
        let columns = (sim_width / ideal_chunk_size).floor() as usize;
        let rows = (sim_height / ideal_chunk_size).floor() as usize;
        let num_particles = particles.len();

        // Sized from the video, which is uniform across frames. Falls back to
        // a 1x1 placeholder when there is no video to draw.
        let obstacle_image = video.first().map_or_else(
            || ImageBuffer::filled(1, 1, BLANK),
            DistanceField::blank_image,
        );

        Self {
            sim_width,
            sim_height,
            video,
            particles,
            densities: vec![0.; num_particles],
            lambdas: vec![0.; num_particles],
            position_deltas: vec![Vec2::ZERO; num_particles],
            viscosity_forces: vec![Vec2::ZERO; num_particles],
            chunks: Grid::with_defaults(columns, rows),
            rng,
            elapsed: Duration::ZERO,
            current_frame: 0,
            obstacle_image,
            obstacle_texture: None,
        }
    }

    /// Scatters `num_particles` through the lower part of the sim area.
    ///
    /// The generator is taken by value so the caller decides whether this run
    /// is reproducible: `rng::seeded` for a test, `rng::from_entropy` for the app.
    pub fn init(
        num_particles: usize,
        sim_width: f32,
        sim_height: f32,
        video: Arc<Vec<DistanceField>>,
        mut rng: Rng,
    ) -> Self {
        let mut particles = Vec::with_capacity(num_particles);

        for i in 0..num_particles {
            let x = rng.random_range(10.0..sim_width - 10.);
            let y = rng.random_range(sim_height * 0.25 - 10.0..sim_height - 10.);
            particles.push(FluidParticle::new(i, x, y));
        }

        Self::new(particles, sim_width, sim_height, video, rng)
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

    /// Advances `current_frame` to match wall-clock time. Called once per
    /// update so the obstacle cannot shift between solver iterations.
    fn update_current_frame(&mut self) {
        if self.video.is_empty() {
            return;
        }

        let frame = (self.elapsed.as_secs_f32() * VIDEO_FRAMERATE).round() as usize;
        self.current_frame = frame % self.video.len();
    }

    /// Draws the current video frame into the persistent obstacle texture and
    /// blits it. Skipped entirely while the obstacle is drawn transparent.
    fn draw_video_frame(&mut self, canvas: &mut dyn Canvas) {
        if OBSTACLE_COLOR.is_transparent() || self.video.is_empty() {
            return;
        }

        let (field_pos, field_size) = self.video_bounds();

        self.video[self.current_frame].write_to_image(&mut self.obstacle_image);

        // Claimed lazily: construction happens before there is any canvas to
        // ask, so the first draw is what allocates the texture.
        let texture = match self.obstacle_texture {
            Some(texture) => {
                canvas.update_texture(texture, &self.obstacle_image);
                texture
            }
            None => {
                let texture = canvas.create_texture(&self.obstacle_image);
                self.obstacle_texture = Some(texture);
                texture
            }
        };

        canvas.draw_texture(texture, field_pos, field_size, OBSTACLE_COLOR);
    }

    /// Where the video obstacle sits within the sim, in sim coordinates.
    fn video_bounds(&self) -> (Vec2, Vec2) {
        (
            vec2(OBSTACLE_BORDER, OBSTACLE_BORDER),
            vec2(
                self.sim_width - OBSTACLE_BORDER * 2.,
                self.sim_height - OBSTACLE_BORDER * 2.,
            ),
        )
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

    fn solve_video_frame_obstacle(&mut self) {
        let Some(distance_field) = self.video.get(self.current_frame) else {
            return;
        };

        let (field_pos, field_size) = self.video_bounds();

        // `video` and `particles` are borrowed as disjoint fields, so the
        // distance field is shared by reference rather than deep-copied once
        // per solver iteration.

        self.particles.par_iter_mut().for_each(|particle| {
            let escape_displacement =
                distance_field.escape_displacement(particle.predicted_pos, field_pos, field_size);

            if let Some(displacement) = escape_displacement
                && displacement.is_finite()
            {
                particle.predicted_pos += displacement;

                let direction = displacement.normalize();
                let normal_speed = particle.vel.dot(direction);
                if normal_speed < 0.0 {
                    particle.vel -= direction * (1.0 + RESTITUTION_COEFFICIENT) * normal_speed;
                }
            }
        });
    }

    fn calculate_viscosity_forces(&mut self) {
        const EPSILON: f32 = 1e-3;

        // Drawn up front rather than inside the loop: the generator is not
        // shareable across rayon's threads, and pre-drawing keeps a seeded run
        // reproducible regardless of how the work happens to be scheduled.
        let nudges: Vec<Vec2> = (0..self.particles.len())
            .map(|_| {
                vec2(
                    self.rng.random_range(-EPSILON..EPSILON),
                    self.rng.random_range(-EPSILON..EPSILON),
                )
            })
            .collect();

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
                                weighted_velocity += nudges[i];
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

    fn apply_external_forces(&mut self, input: &Input) {
        self.apply_mouse_interaction_forces(input);
    }

    /// Left button pulls particles toward the cursor, right button pushes them
    /// away. Only particles within one smoothing radius are affected.
    fn apply_mouse_interaction_forces(&mut self, input: &Input) {
        const MOUSE_FORCE: f32 = 1000.;

        let sign = if input.is_down(MouseButton::Left) {
            -1.
        } else if input.is_down(MouseButton::Right) {
            1.
        } else {
            return;
        };

        let mouse_pos = input.mouse_pos;

        self.particles.par_iter_mut().for_each(|particle| {
            let displacement = mouse_pos - particle.pos;

            if displacement.length() < SMOOTHING_RADIUS {
                particle.vel -= (displacement / SMOOTHING_RADIUS) * sign * MOUSE_FORCE;
            }
        });
    }

    /// The metaball look: accumulate wide, faint marks into a layer and let the
    /// liquid effect turn that density field into a shaded surface.
    ///
    /// The simulation says what it wants drawn and which effect to apply; it
    /// has no idea whether that is a shader, a filter, or nothing at all.
    #[allow(dead_code)]
    fn draw_fluid_layer(&self, canvas: &mut dyn Canvas) {
        const DENSITY_MARK_RADIUS: f32 = 30.;
        const DENSITY_MARK_COLOR: Color = Color::new(1.0, 1.0, 1.0, 0.04);

        let size = vec2(self.sim_width, self.sim_height);
        let particles = &self.particles;

        canvas.with_layer(Vec2::ZERO, size, Effect::Liquid, &mut |layer| {
            for particle in particles {
                layer.circle(particle.pos, DENSITY_MARK_RADIUS, DENSITY_MARK_COLOR);
            }
        });
    }

    /// One mark per particle, tinted by local density and speed.
    fn draw_particles(&self, canvas: &mut dyn Canvas) {
        const PARTICLE_RADIUS: f32 = 3.;

        for (i, particle) in self.particles.iter().enumerate() {
            let density = self.densities[i];
            let speed = particle.vel.length();
            let red = (density * 60.).clamp(0., 1.);
            let green = (speed / 100.).clamp(0., 1.);
            let blue = (1.0 - density * 0.3).clamp(0.7, 1.0);

            canvas.circle(
                particle.pos,
                PARTICLE_RADIUS,
                Color::new(red, (green - red * 0.5).clamp(0.0, 1.0), blue, 0.8),
            );
        }
    }
}

impl Simulation for FluidSim {
    fn update(&mut self, dt: Duration, input: &Input) {
        // A zero step would divide by zero in `update_velocity`, and there is
        // nothing to advance anyway.
        if dt.is_zero() {
            return;
        }

        self.elapsed += dt;
        self.update_current_frame();

        self.apply_external_forces(input);

        self.reset_predicted_positions(dt);
        self.update_chunks();

        for _iteration in 1..=ITERATIONS_PER_UPDATE {
            self.update_densities();
            self.update_position_deltas();
            self.apply_position_deltas();
            self.solve_video_frame_obstacle();
        }

        self.particles
            .par_iter_mut()
            .for_each(|particle| particle.update_velocity(dt));

        self.calculate_viscosity_forces();

        self.particles
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, particle)| particle.apply_viscosity(self.viscosity_forces[i]));

        self.particles
            .par_iter_mut()
            .for_each(|particle| particle.commit_new_position(self.sim_width, self.sim_height));
    }

    fn draw(&mut self, canvas: &mut dyn Canvas) {
        // Inset by the obstacle border so the video fills the frame.
        let (field_pos, field_size) = self.video_bounds();
        canvas.set_view(field_pos, field_size);

        self.draw_video_frame(canvas);

        // self.draw_fluid_layer(canvas);
        self.draw_particles(canvas);
    }
}

impl HasSize for FluidSim {
    fn size(&self) -> Vec2 {
        vec2(self.sim_width, self.sim_height)
    }
}

/// The extension point for static obstacles. The fluid sim currently drives
/// its obstacle straight off the video's distance fields, so nothing dispatches
/// through this today — it stays for the next obstacle that isn't the video.
#[allow(dead_code)]
pub trait Obstacle: Debug + 'static {
    /// Returns the minimum translation vector needed to move a particle
    /// centred at `pos` out of the obstacle. Returns `None` if no collision.
    fn escape_displacement(&self, pos: Vec2) -> Option<Vec2>;
}

#[derive(Clone, Debug)]
pub struct BinaryBitmap {
    pub grid: Grid<bool>,
}

impl BinaryBitmap {
    pub fn new(grid: Grid<bool>) -> Self {
        Self { grid }
    }

    pub fn from_image(image: &ImageBuffer) -> Self {
        let width = image.width() as usize;
        let height = image.height() as usize;
        let bytes = image.bytes();

        let grid = Grid::from_generator(width, height, |x, y| {
            let i = (y * width + x) * 4;

            let r = bytes[i];
            let g = bytes[i + 1];
            let b = bytes[i + 2];

            let brightness = (r as f32 + g as f32 + b as f32) / (3.0 * 255.0);

            // true = obstacle
            brightness < 0.5
        });

        Self { grid }
    }

    #[allow(unused)]
    pub fn example_bitmap(width: usize, height: usize) -> Self {
        let mut grid = Grid::from_generator(width, height, |_, _| false);

        // Rectangle
        for y in 20..60 {
            for x in 20..80 {
                *grid.get_mut(x as isize, y as isize).unwrap() = true;
            }
        }

        // Circle
        let centre = vec2(120., 80.);
        let radius = 25.;

        for y in 0..height {
            for x in 0..width {
                let pos = vec2(x as f32, y as f32);

                if pos.distance(centre) < radius {
                    *grid.get_mut(x as isize, y as isize).unwrap() = true;
                }
            }
        }

        Self { grid }
    }

    /// A transparent buffer matching this bitmap's dimensions.
    #[allow(dead_code)]
    fn blank_image(&self) -> ImageBuffer {
        ImageBuffer::filled(self.grid.columns() as u16, self.grid.rows() as u16, BLANK)
    }

    /// Rasterises occupancy into an existing buffer: solid white where the
    /// bitmap is set, opaque black elsewhere.
    #[allow(dead_code)]
    fn write_to_image(&self, image: &mut ImageBuffer) {
        debug_assert_eq!(image.width() as usize, self.grid.columns());
        debug_assert_eq!(image.height() as usize, self.grid.rows());

        for (pixel, occupied) in image.pixels_mut().zip(self.grid.iter()) {
            *pixel = if *occupied {
                [255, 255, 255, 255]
            } else {
                [0, 0, 0, 255]
            };
        }
    }
}

#[derive(Clone, Debug)]
pub struct DistanceField {
    pub grid: Grid<f32>,
}

impl DistanceField {
    pub fn new(grid: Grid<f32>) -> Self {
        Self { grid }
    }

    pub fn sample(&self, pos: Vec2) -> f32 {
        *self
            .grid
            .get(pos.x.floor() as isize, pos.y.floor() as isize)
            .unwrap_or(&0.)
    }

    pub fn gradient(&self, pos: Vec2) -> Vec2 {
        let dx = self.sample(pos + vec2(1., 0.)) - self.sample(pos - vec2(1., 0.));
        let dy = self.sample(pos + vec2(0., 1.)) - self.sample(pos - vec2(0., 1.));
        vec2(dx, dy).normalize_or_zero()
    }

    /// Minimum translation vector needed to move a particle at `pos` out of
    /// this field, where the field is laid out in sim space at `field_pos`
    /// spanning `field_size`. Returns `None` if `pos` is not inside.
    ///
    /// Taking the placement as arguments rather than baking it into an owning
    /// obstacle lets callers share one field by reference across many
    /// particles and solver iterations.
    pub fn escape_displacement(
        &self,
        pos: Vec2,
        field_pos: Vec2,
        field_size: Vec2,
    ) -> Option<Vec2> {
        let pixel_width = field_size.x / self.grid.columns() as f32;
        let pixel_height = field_size.y / self.grid.rows() as f32;

        let mut relative_pos = pos - field_pos;
        relative_pos.x /= pixel_width;
        relative_pos.y /= pixel_height;

        let distance = self.sample(relative_pos);
        if distance >= 0. {
            return None;
        }

        let mut gradient = self.gradient(relative_pos);
        gradient.x *= pixel_width;
        gradient.y *= pixel_height;

        Some(gradient * -distance)
    }

    fn edt_from_bitmap(bitmap: &BinaryBitmap) -> Self {
        let outside = Self::unsigned_edt_from_bitmap(bitmap, true);
        let inside = Self::unsigned_edt_from_bitmap(bitmap, false);

        let sdf = Grid::from_generator(bitmap.grid.columns(), bitmap.grid.rows(), |x, y| {
            let inside_obstacle = *bitmap.grid.get(x as isize, y as isize).unwrap();

            if inside_obstacle {
                -*inside.get(x as isize, y as isize).unwrap()
            } else {
                *outside.get(x as isize, y as isize).unwrap()
            }
        });

        Self { grid: sdf }
    }

    // The explicit indices are the algorithm here: the transform runs over
    // columns and then rows of the same grid, so the loop variable addresses
    // the grid, not the scratch vector clippy sees it indexing.
    #[allow(clippy::needless_range_loop)]
    fn unsigned_edt_from_bitmap(bitmap: &BinaryBitmap, target: bool) -> Grid<f32> {
        let width = bitmap.grid.columns();
        let height = bitmap.grid.rows();

        let mut grid = Grid::from_generator(width, height, |x, y| {
            if *bitmap.grid.get(x as isize, y as isize).unwrap() == target {
                0.0
            } else {
                INF
            }
        });

        for x in 0..width {
            let mut column = vec![0.0; height];

            for y in 0..height {
                column[y] = *grid.get(x as isize, y as isize).unwrap();
            }

            let column = Self::edt_1d(&column);

            for y in 0..height {
                *grid.get_mut(x as isize, y as isize).unwrap() = column[y];
            }
        }

        for y in 0..height {
            let mut row = vec![0.0; width];

            for x in 0..width {
                row[x] = *grid.get(x as isize, y as isize).unwrap();
            }

            let row = Self::edt_1d(&row);

            for x in 0..width {
                *grid.get_mut(x as isize, y as isize).unwrap() = row[x].sqrt();
            }
        }

        grid
    }

    /// Felzenszwalb & Huttenlocher's exact 1D squared-distance transform: the
    /// lower envelope of parabolas rooted at each sample. `q` indexes the
    /// input alongside the output, so an iterator rewrite would obscure it.
    #[allow(clippy::needless_range_loop)]
    fn edt_1d(f: &[f32]) -> Vec<f32> {
        let n = f.len();

        let mut d = vec![0.0; n];

        // Locations of parabolas in the lower envelope
        let mut v = vec![0usize; n];

        // Locations where one parabola overtakes the previous one
        let mut z = vec![0.0; n + 1];

        let mut k = 0;

        v[0] = 0;
        z[0] = -INF;
        z[1] = INF;

        // Construct lower envelope
        for q in 1..n {
            if f[q] >= INF {
                continue;
            }

            let mut s = ((f[q] + (q * q) as f32) - (f[v[k]] + (v[k] * v[k]) as f32))
                / (2.0 * (q as f32 - v[k] as f32));

            while k > 0 && s <= z[k] {
                k -= 1;

                s = ((f[q] + (q * q) as f32) - (f[v[k]] + (v[k] * v[k]) as f32))
                    / (2.0 * (q as f32 - v[k] as f32));
            }

            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = INF;
        }

        // Evaluate lower envelope
        k = 0;

        for q in 0..n {
            while z[k + 1] < q as f32 {
                k += 1;
            }

            let dx = q as f32 - v[k] as f32;
            d[q] = dx * dx + f[v[k]];
        }

        d
    }

    /// An RGBA image the size of this field: interior white, a red band along
    /// the surface, transparent outside.
    fn blank_image(&self) -> ImageBuffer {
        ImageBuffer::filled(self.grid.columns() as u16, self.grid.rows() as u16, BLANK)
    }

    /// Rasterises this field into an existing buffer, so callers can reuse one
    /// allocation across frames instead of building a new image each time.
    /// The image must already match the field's dimensions.
    fn write_to_image(&self, image: &mut ImageBuffer) {
        debug_assert_eq!(image.width() as usize, self.grid.columns());
        debug_assert_eq!(image.height() as usize, self.grid.rows());

        for (pixel, value) in image.pixels_mut().zip(self.grid.iter()) {
            let occupied = *value < 0.;
            let on_border = occupied && *value > -5.;

            *pixel = if on_border {
                [255, 0, 0, 255]
            } else if occupied {
                [255, 255, 255, 255]
            } else {
                [0, 0, 0, 0]
            };
        }
    }

    /// Rasterises this field into a fresh buffer. Allocates, so prefer
    /// [`Self::write_to_image`] into a buffer you keep around on a hot path.
    #[allow(dead_code)]
    fn to_image(&self) -> ImageBuffer {
        let mut image = self.blank_image();
        self.write_to_image(&mut image);
        image
    }
}

impl From<&BinaryBitmap> for DistanceField {
    fn from(bitmap: &BinaryBitmap) -> Self {
        Self::edt_from_bitmap(bitmap)
    }
}

/// Wraps either representation of a video frame as an `Obstacle`. Retained as
/// the bridge between the bitmap/SDF types and the `Obstacle` trait.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct BitmapObstacle {
    pos: Vec2,
    size: Vec2,
    bitmap: Option<BinaryBitmap>,
    distance_field: Option<DistanceField>,
}

#[allow(dead_code)]
impl BitmapObstacle {
    pub fn from_bitmap(bitmap: BinaryBitmap, pos: Vec2, size: Vec2) -> Self {
        Self {
            pos,
            size,
            bitmap: Some(bitmap),
            distance_field: None,
        }
    }

    pub fn from_distance_field(distance_field: DistanceField, pos: Vec2, size: Vec2) -> Self {
        Self {
            pos,
            size,
            bitmap: None,
            distance_field: Some(distance_field),
        }
    }

    fn pixel_dimensions(&self) -> (f32, f32) {
        if let Some(bitmap) = &self.bitmap {
            let pixel_width = self.size.x / bitmap.grid.columns() as f32;
            let pixel_height = self.size.y / bitmap.grid.rows() as f32;
            (pixel_width, pixel_height)
        } else if let Some(distance_field) = &self.distance_field {
            let pixel_width = self.size.x / distance_field.grid.columns() as f32;
            let pixel_height = self.size.y / distance_field.grid.rows() as f32;
            (pixel_width, pixel_height)
        } else {
            unreachable!()
        }
    }

    fn relative_pos(&self, pos: Vec2) -> Vec2 {
        let (pixel_width, pixel_height) = self.pixel_dimensions();

        let mut relative_pos = pos - self.pos;
        relative_pos.x /= pixel_width;
        relative_pos.y /= pixel_height;

        relative_pos
    }

    fn escape_displacement(&self, pos: Vec2) -> Option<Vec2> {
        if self.distance_field.is_some() {
            self.escape_displacement_sdf(pos)
        } else if self.bitmap.is_some() {
            self.escape_displacement_cheap(pos)
        } else {
            unreachable!()
        }
    }

    fn escape_displacement_sdf(&self, pos: Vec2) -> Option<Vec2> {
        self.distance_field
            .as_ref()
            .unwrap()
            .escape_displacement(pos, self.pos, self.size)
    }

    fn escape_displacement_cheap(&self, pos: Vec2) -> Option<Vec2> {
        let bitmap = self.bitmap.as_ref().unwrap();

        let relative_pos = self.relative_pos(pos);

        let x = relative_pos.x as isize;
        let y = relative_pos.y as isize;

        let mut displacement = Vec2::ZERO;

        if bitmap.grid.get(x, y).is_some_and(|exists| *exists) {
            for dy in -5..=5 {
                for dx in -5..=5 {
                    let sample_x = x + dx;
                    let sample_y = y + dy;

                    if bitmap
                        .grid
                        .get(sample_x, sample_y)
                        .is_some_and(|exists| *exists)
                    {
                        let away = relative_pos - vec2(sample_x as f32, sample_y as f32);

                        if away.is_finite() && away.length_squared() > 0.0 {
                            displacement += away.normalize() / away.length();
                        }
                    }
                }
            }
        }

        if displacement.length_squared() > 0.0 {
            Some(displacement.normalize())
        } else {
            None
        }
    }
}

impl Obstacle for BitmapObstacle {
    fn escape_displacement(&self, pos: Vec2) -> Option<Vec2> {
        // Delegate to the inherent method, which picks the right collision
        // path for whichever representation this obstacle actually holds.
        BitmapObstacle::escape_displacement(self, pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{DrawCall, Input, MouseButton, RecordingCanvas};
    use crate::rng;

    const TEST_WIDTH: f32 = 240.;
    const TEST_HEIGHT: f32 = 240.;
    const TEST_PARTICLES: usize = 60;
    const STEP: Duration = Duration::from_millis(16);

    /// A simulation with no obstacle at all.
    fn sim() -> FluidSim {
        FluidSim::init(
            TEST_PARTICLES,
            TEST_WIDTH,
            TEST_HEIGHT,
            Arc::new(Vec::new()),
            rng::seeded(1),
        )
    }

    /// A simulation whose obstacle is solid across its left half.
    fn sim_with_obstacle() -> FluidSim {
        let video = vec![DistanceField::from(&left_half_solid())];

        FluidSim::init(
            TEST_PARTICLES,
            TEST_WIDTH,
            TEST_HEIGHT,
            Arc::new(video),
            rng::seeded(1),
        )
    }

    fn step(sim: &mut FluidSim, times: usize) {
        for _ in 0..times {
            sim.update(STEP, &Input::none());
        }
    }

    fn positions(sim: &FluidSim) -> Vec<Vec2> {
        sim.particles.iter().map(|p| p.pos).collect()
    }

    /// Mean distance between every pair of particles — a coarse measure of how
    /// spread out the fluid is.
    fn mean_pairwise_distance(sim: &FluidSim) -> f32 {
        let positions = positions(sim);
        let mut total = 0.;
        let mut pairs = 0;

        for (i, a) in positions.iter().enumerate() {
            for b in &positions[i + 1..] {
                total += a.distance(*b);
                pairs += 1;
            }
        }

        total / pairs as f32
    }

    // ---- stepping -------------------------------------------------------

    #[test]
    fn particles_are_neither_created_nor_destroyed() {
        let mut sim = sim();
        step(&mut sim, 30);

        assert_eq!(TEST_PARTICLES, sim.particles.len());
        assert_eq!(TEST_PARTICLES, sim.densities.len());
        assert_eq!(TEST_PARTICLES, sim.lambdas.len());
    }

    #[test]
    fn particles_stay_inside_the_sim_area() {
        let mut sim = sim();
        step(&mut sim, 60);

        for (i, pos) in positions(&sim).iter().enumerate() {
            assert!(
                (0.0..=TEST_WIDTH).contains(&pos.x) && (0.0..=TEST_HEIGHT).contains(&pos.y),
                "particle {i} escaped to {pos:?}"
            );
        }
    }

    #[test]
    fn nothing_goes_to_nan() {
        let mut sim = sim();
        step(&mut sim, 60);

        for (i, particle) in sim.particles.iter().enumerate() {
            assert!(
                particle.pos.is_finite(),
                "particle {i} position {:?}",
                particle.pos
            );
            assert!(
                particle.vel.is_finite(),
                "particle {i} velocity {:?}",
                particle.vel
            );
        }

        assert!(sim.densities.iter().all(|d| d.is_finite()));
        assert!(sim.lambdas.iter().all(|l| l.is_finite()));
    }

    #[test]
    fn a_zero_length_step_changes_nothing() {
        let mut sim = sim();
        step(&mut sim, 5);

        let before = positions(&sim);
        sim.update(Duration::ZERO, &Input::none());

        assert_eq!(before, positions(&sim));
    }

    #[test]
    fn speed_stays_within_its_clamp() {
        let mut sim = sim();
        step(&mut sim, 60);

        for particle in &sim.particles {
            assert!(
                particle.vel.length() <= MAX_SPEED + 1e-3,
                "{} exceeds the clamp",
                particle.vel.length()
            );
        }
    }

    // ---- physics --------------------------------------------------------

    #[test]
    fn an_overlapping_clump_pushes_itself_apart() {
        // A tight cluster, the case incompressibility exists to resolve.
        let mut rng = rng::seeded(2);
        let centre = vec2(TEST_WIDTH / 2., TEST_HEIGHT / 2.);

        let particles = (0..TEST_PARTICLES)
            .map(|i| {
                let jitter = vec2(rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0));
                let pos = centre + jitter;
                FluidParticle::new(i, pos.x, pos.y)
            })
            .collect();

        let mut sim = FluidSim::new(
            particles,
            TEST_WIDTH,
            TEST_HEIGHT,
            Arc::new(Vec::new()),
            rng::seeded(2),
        );

        let before = mean_pairwise_distance(&sim);
        step(&mut sim, 30);
        let after = mean_pairwise_distance(&sim);

        assert!(
            after > before * 2.,
            "cluster barely expanded: {before} -> {after}"
        );
        assert!(sim.particles.iter().all(|p| p.pos.is_finite()));
    }

    /// Perfectly coincident particles are a degenerate case: every pairwise
    /// displacement is zero, so the pressure gradient is zero and the solver
    /// has no direction to push in. It must at least stay numerically sound.
    #[test]
    fn perfectly_coincident_particles_stay_numerically_sound() {
        let particles = (0..8)
            .map(|i| FluidParticle::new(i, TEST_WIDTH / 2., TEST_HEIGHT / 2.))
            .collect();

        let mut sim = FluidSim::new(
            particles,
            TEST_WIDTH,
            TEST_HEIGHT,
            Arc::new(Vec::new()),
            rng::seeded(2),
        );

        step(&mut sim, 30);

        for particle in &sim.particles {
            assert!(particle.pos.is_finite(), "{:?}", particle.pos);
            assert!(particle.vel.is_finite(), "{:?}", particle.vel);
        }
    }

    #[test]
    fn density_responds_to_crowding() {
        let mut crowded = sim();
        step(&mut crowded, 1);
        let crowded_peak = crowded.densities.iter().cloned().fold(0., f32::max);

        // Two particles at opposite corners have no neighbours at all.
        let sparse_particles = vec![
            FluidParticle::new(0, 5., 5.),
            FluidParticle::new(1, TEST_WIDTH - 5., TEST_HEIGHT - 5.),
        ];
        let mut sparse = FluidSim::new(
            sparse_particles,
            TEST_WIDTH,
            TEST_HEIGHT,
            Arc::new(Vec::new()),
            rng::seeded(3),
        );
        step(&mut sparse, 1);
        let sparse_peak = sparse.densities.iter().cloned().fold(0., f32::max);

        assert!(
            crowded_peak > sparse_peak,
            "crowded {crowded_peak} should exceed sparse {sparse_peak}"
        );
    }

    // ---- obstacles ------------------------------------------------------

    #[test]
    fn the_obstacle_pushes_particles_out_of_solid_ground() {
        let mut sim = sim_with_obstacle();

        // Plant a particle deep inside the solid half.
        let (field_pos, field_size) = sim.video_bounds();
        let field = sim.video[0].clone();
        let pixel_width = field_size.x / field.grid.columns() as f32;
        let inside = field_pos + vec2(pixel_width * 1.5, field_size.y / 2.);

        sim.particles[0].pos = inside;
        sim.particles[0].predicted_pos = inside;

        step(&mut sim, 20);

        let escaped = sim.particles[0].pos;
        assert!(
            field
                .escape_displacement(escaped, field_pos, field_size)
                .is_none(),
            "particle remained inside the obstacle at {escaped:?}"
        );
    }

    #[test]
    fn an_empty_video_is_not_an_obstacle() {
        // Regression: the solver used to index the video unconditionally.
        let mut sim = sim();
        step(&mut sim, 5);

        assert_eq!(TEST_PARTICLES, sim.particles.len());
    }

    #[test]
    fn elapsed_time_advances_the_video_frame() {
        let video = (0..90)
            .map(|_| DistanceField::from(&left_half_solid()))
            .collect();
        let mut sim = FluidSim::init(4, TEST_WIDTH, TEST_HEIGHT, Arc::new(video), rng::seeded(4));

        assert_eq!(0, sim.current_frame);

        // One second at 30fps is thirty frames.
        sim.update(Duration::from_secs(1), &Input::none());
        assert_eq!(30, sim.current_frame);
    }

    #[test]
    fn the_video_loops_rather_than_running_off_the_end() {
        let video = (0..3)
            .map(|_| DistanceField::from(&left_half_solid()))
            .collect();
        let mut sim = FluidSim::init(4, TEST_WIDTH, TEST_HEIGHT, Arc::new(video), rng::seeded(5));

        sim.update(Duration::from_secs(10), &Input::none());
        assert!(
            sim.current_frame < 3,
            "frame {} out of range",
            sim.current_frame
        );
    }

    // ---- input ----------------------------------------------------------

    #[test]
    fn the_right_mouse_button_pushes_particles_away() {
        let mut sim = sim();
        let centre = vec2(TEST_WIDTH / 2., TEST_HEIGHT / 2.);

        // A particle right next to the cursor, at rest.
        sim.particles[0].pos = centre + vec2(5., 0.);
        sim.particles[0].vel = Vec2::ZERO;

        let input = Input::at(centre).with_held(MouseButton::Right);
        sim.apply_external_forces(&input);

        assert!(
            sim.particles[0].vel.x > 0.,
            "expected a push to the right, got {:?}",
            sim.particles[0].vel
        );
    }

    #[test]
    fn the_left_mouse_button_pulls_particles_in() {
        let mut sim = sim();
        let centre = vec2(TEST_WIDTH / 2., TEST_HEIGHT / 2.);

        sim.particles[0].pos = centre + vec2(5., 0.);
        sim.particles[0].vel = Vec2::ZERO;

        let input = Input::at(centre).with_held(MouseButton::Left);
        sim.apply_external_forces(&input);

        assert!(
            sim.particles[0].vel.x < 0.,
            "expected a pull to the left, got {:?}",
            sim.particles[0].vel
        );
    }

    #[test]
    fn particles_beyond_the_cursors_reach_are_untouched() {
        let mut sim = sim();

        sim.particles[0].pos = vec2(0., 0.);
        sim.particles[0].vel = Vec2::ZERO;

        let far_away = vec2(TEST_WIDTH, TEST_HEIGHT);
        let input = Input::at(far_away).with_held(MouseButton::Right);
        sim.apply_external_forces(&input);

        assert_eq!(Vec2::ZERO, sim.particles[0].vel);
    }

    #[test]
    fn an_idle_mouse_applies_no_force() {
        let mut sim = sim();
        step(&mut sim, 3);

        let before: Vec<Vec2> = sim.particles.iter().map(|p| p.vel).collect();
        sim.apply_external_forces(&Input::at(vec2(TEST_WIDTH / 2., TEST_HEIGHT / 2.)));
        let after: Vec<Vec2> = sim.particles.iter().map(|p| p.vel).collect();

        assert_eq!(before, after);
    }

    // ---- determinism ----------------------------------------------------

    #[test]
    fn the_same_seed_replays_the_same_run() {
        let mut a = FluidSim::init(
            TEST_PARTICLES,
            TEST_WIDTH,
            TEST_HEIGHT,
            Arc::new(Vec::new()),
            rng::seeded(99),
        );
        let mut b = FluidSim::init(
            TEST_PARTICLES,
            TEST_WIDTH,
            TEST_HEIGHT,
            Arc::new(Vec::new()),
            rng::seeded(99),
        );

        step(&mut a, 20);
        step(&mut b, 20);

        assert_eq!(positions(&a), positions(&b));
    }

    #[test]
    fn different_seeds_start_differently() {
        let a = FluidSim::init(
            TEST_PARTICLES,
            TEST_WIDTH,
            TEST_HEIGHT,
            Arc::new(Vec::new()),
            rng::seeded(1),
        );
        let b = FluidSim::init(
            TEST_PARTICLES,
            TEST_WIDTH,
            TEST_HEIGHT,
            Arc::new(Vec::new()),
            rng::seeded(2),
        );

        assert_ne!(positions(&a), positions(&b));
    }

    // ---- drawing --------------------------------------------------------

    #[test]
    fn drawing_emits_one_circle_per_particle() {
        let mut sim = sim();
        step(&mut sim, 3);

        let mut canvas = RecordingCanvas::new();
        sim.draw(&mut canvas);

        assert_eq!(TEST_PARTICLES, canvas.circles().count());
    }

    #[test]
    fn drawn_circles_sit_on_the_particles() {
        let mut sim = sim();
        step(&mut sim, 3);

        let mut canvas = RecordingCanvas::new();
        sim.draw(&mut canvas);

        let drawn: Vec<Vec2> = canvas.circles().map(|(centre, _, _)| centre).collect();
        assert_eq!(positions(&sim), drawn);
    }

    #[test]
    fn drawing_sets_the_view_before_anything_else() {
        let mut sim = sim();

        let mut canvas = RecordingCanvas::new();
        sim.draw(&mut canvas);

        assert!(
            matches!(canvas.calls.first(), Some(DrawCall::SetView { .. })),
            "first call was {:?}",
            canvas.calls.first()
        );

        // The view is inset by the obstacle border on every side.
        let (pos, size) = canvas.view().unwrap();
        assert_eq!(vec2(OBSTACLE_BORDER, OBSTACLE_BORDER), pos);
        assert_eq!(
            vec2(
                TEST_WIDTH - OBSTACLE_BORDER * 2.,
                TEST_HEIGHT - OBSTACLE_BORDER * 2.
            ),
            size
        );
    }

    #[test]
    fn drawing_does_not_advance_the_simulation() {
        let mut sim = sim();
        step(&mut sim, 5);

        let before = positions(&sim);

        let mut first = RecordingCanvas::new();
        sim.draw(&mut first);
        let mut second = RecordingCanvas::new();
        sim.draw(&mut second);

        assert_eq!(before, positions(&sim));
        assert_eq!(first.calls, second.calls);
    }

    #[test]
    fn a_transparent_obstacle_colour_skips_the_texture_entirely() {
        // OBSTACLE_COLOR is BLANK, so the video costs nothing to "draw".
        assert!(OBSTACLE_COLOR.is_transparent());

        let mut sim = sim_with_obstacle();
        step(&mut sim, 1);

        let mut canvas = RecordingCanvas::new();
        sim.draw(&mut canvas);

        assert!(
            !canvas
                .calls
                .iter()
                .any(|call| matches!(call, DrawCall::CreateTexture { .. })),
            "uploaded a texture that would draw nothing"
        );
    }

    #[test]
    fn the_metaball_path_requests_the_liquid_effect() {
        let mut sim = sim();
        step(&mut sim, 1);

        let mut canvas = RecordingCanvas::new();
        sim.draw_fluid_layer(&mut canvas);

        assert_eq!(vec![Effect::Liquid], canvas.effects().collect::<Vec<_>>());
        assert_eq!(TEST_PARTICLES, canvas.circles().count());
    }

    /// Builds a bitmap from row-major `true`/`false` values.
    fn bitmap(cells: Vec<bool>, columns: usize, rows: usize) -> BinaryBitmap {
        BinaryBitmap::new(Grid::new(cells, columns, rows))
    }

    /// 8x8 with the left half solid, so the escape direction is unambiguous.
    fn left_half_solid() -> BinaryBitmap {
        bitmap((0..64).map(|i| i % 8 < 4).collect(), 8, 8)
    }

    #[test]
    fn edt_1d_measures_squared_distance_from_a_single_seed() {
        let squared = DistanceField::edt_1d(&[0.0, INF, INF, INF]);
        assert_eq!(vec![0.0, 1.0, 4.0, 9.0], squared);
    }

    #[test]
    fn edt_1d_takes_the_nearest_of_several_seeds() {
        let squared = DistanceField::edt_1d(&[0.0, INF, INF, 0.0]);
        assert_eq!(vec![0.0, 1.0, 1.0, 0.0], squared);
    }

    #[test]
    fn edt_is_euclidean_not_manhattan() {
        // Only the centre is set, so the corners sit a true diagonal away.
        let centre_only = bitmap(
            vec![
                false, false, false, //
                false, true, false, //
                false, false, false,
            ],
            3,
            3,
        );

        let distances = DistanceField::unsigned_edt_from_bitmap(&centre_only, true);

        assert_eq!(0.0, *distances.get(1, 1).unwrap());
        assert_eq!(1.0, *distances.get(1, 0).unwrap());
        assert_eq!(1.0, *distances.get(0, 1).unwrap());

        // Manhattan would give 2.0 here.
        let corner = *distances.get(0, 0).unwrap();
        assert!(
            (corner - std::f32::consts::SQRT_2).abs() < 1e-5,
            "expected sqrt(2), got {corner}"
        );
    }

    #[test]
    fn distance_field_is_negative_inside_and_positive_outside() {
        let field = DistanceField::from(&left_half_solid());

        assert!(
            field.sample(vec2(0., 4.)) < 0.,
            "left half should be inside"
        );
        assert!(
            field.sample(vec2(7., 4.)) > 0.,
            "right half should be outside"
        );

        // Distance grows with depth into the solid region.
        assert!(field.sample(vec2(0., 4.)) < field.sample(vec2(3., 4.)));
    }

    #[test]
    fn escape_displacement_pushes_out_of_the_solid_region() {
        let field = DistanceField::from(&left_half_solid());

        // One sim unit per pixel, so sim and field coordinates coincide.
        let escape = field
            .escape_displacement(vec2(1.5, 4.0), Vec2::ZERO, vec2(8., 8.))
            .expect("a point inside the solid half should be pushed out");

        assert!(escape.x > 0., "should be pushed right, toward open space");
        assert!(escape.y.abs() < 1e-5, "should not be pushed vertically");

        assert_eq!(
            None,
            field.escape_displacement(vec2(7.5, 4.0), Vec2::ZERO, vec2(8., 8.)),
            "a point outside should not be displaced"
        );
    }

    /// Regression test: the `Obstacle` impl used to hardcode the bitmap-only
    /// collision path, which panicked on a distance-field-backed obstacle.
    #[test]
    fn obstacle_trait_handles_a_distance_field_obstacle() {
        let field = DistanceField::from(&left_half_solid());
        let obstacle = BitmapObstacle::from_distance_field(field, Vec2::ZERO, vec2(8., 8.));

        let via_trait: &dyn Obstacle = &obstacle;
        let escape = via_trait
            .escape_displacement(vec2(1.5, 4.0))
            .expect("the trait path should use the SDF, not panic on a missing bitmap");

        assert!(escape.x > 0.);
    }

    /// Builds a sim from explicit particle positions.
    fn sim_from(positions: &[Vec2]) -> FluidSim {
        let particles = positions
            .iter()
            .enumerate()
            .map(|(i, p)| FluidParticle::new(i, p.x, p.y))
            .collect();

        FluidSim::new(
            particles,
            TEST_WIDTH,
            TEST_HEIGHT,
            Arc::new(Vec::new()),
            rng::seeded(2),
        )
    }

    /// Two coincident particles with other neighbours around them: the nudge
    /// survives, because `sum_of_weights` is non-zero thanks to those
    /// neighbours. Differing nudges break the symmetry, and once the pair is
    /// even slightly apart the pressure gradient separates them properly.
    #[test]
    fn a_coincident_pair_in_a_crowd_comes_apart() {
        let centre = vec2(TEST_WIDTH / 2., TEST_HEIGHT / 2.);

        let mut positions = vec![centre, centre];
        for i in 0..10 {
            let angle = i as f32 / 10. * std::f32::consts::TAU;
            positions.push(centre + Vec2::from_angle(angle) * 12.);
        }

        let mut sim = sim_from(&positions);
        assert_eq!(0., sim.particles[0].pos.distance(sim.particles[1].pos));

        step(&mut sim, 30);

        let apart = sim.particles[0].pos.distance(sim.particles[1].pos);
        assert!(apart > 1., "pair stayed together, {apart} apart");
    }

    /// The same pair with nothing else nearby stays stuck. Every pairwise
    /// displacement is zero, so the pressure gradient is zero; and with no
    /// non-coincident neighbour, `sum_of_weights` stays zero too, so the nudge
    /// meant to break the tie is computed and then discarded.
    ///
    /// This pins current behaviour, not desired behaviour.
    #[test]
    fn an_isolated_coincident_pair_stays_stuck() {
        let centre = vec2(TEST_WIDTH / 2., TEST_HEIGHT / 2.);
        let mut sim = sim_from(&[centre, centre]);

        step(&mut sim, 60);

        assert_eq!(
            0.,
            sim.particles[0].pos.distance(sim.particles[1].pos),
            "unexpectedly separated — the nudge may now be reaching the solver"
        );
    }

    /// How far apart a coincident pair gets after one step, as the number of
    /// unrelated neighbours varies. Printed rather than asserted: it shows the
    /// nudge's strength is a function of the crowd, which is the real problem.
    #[test]
    #[ignore = "diagnostic; run with --ignored --nocapture"]
    fn coincident_pair_separation_by_crowd_size() {
        let centre = vec2(TEST_WIDTH / 2., TEST_HEIGHT / 2.);

        for neighbours in [0, 1, 2, 5, 10, 20] {
            let mut positions = vec![centre, centre];
            for i in 0..neighbours {
                let angle = i as f32 / neighbours.max(1) as f32 * std::f32::consts::TAU;
                positions.push(centre + Vec2::from_angle(angle) * 12.);
            }

            let mut sim = sim_from(&positions);
            step(&mut sim, 1);
            let after_one = sim.particles[0].pos.distance(sim.particles[1].pos);

            step(&mut sim, 29);
            let after_thirty = sim.particles[0].pos.distance(sim.particles[1].pos);

            println!(
                "  {neighbours:>2} neighbours: {after_one:>12.3e} after 1 step, {after_thirty:>10.4} after 30"
            );
        }
    }
}
