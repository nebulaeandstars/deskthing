use crate::component::Frame;
use crate::grid::Grid;
use crate::shaders::liquid_material;
use crate::traits::*;

use macroquad::prelude::*;
use macroquad::rand::RandomRange;
use rayon::prelude::*;
use std::fmt::Debug;
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
const OBSTACLE_COLOR: Color = Color::new(0.6, 0.6, 0.6, 1.);

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
    obstacles: Vec<Box<dyn Obstacle>>,
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

        let mut obstacles: Vec<Box<dyn Obstacle>> = Vec::new();
        // obstacles.push(Box::new(CircleObstacle {
        //     pos: vec2(sim_width / 2., sim_height / 2.),
        //     radius: 40.,
        // }));
        // obstacles.push(Box::new(RectangleObstacle {
        //     pos: vec2(sim_width - sim_width / 4., sim_height / 2.),
        //     size: vec2(40., 80.),
        // }));

        let bitmap = BinaryBitmap::example_bitmap(200, 150);
        let bitmap_obstacle =
            BitmapObstacle::from_bitmap(bitmap, vec2(0., 0.), vec2(sim_width, sim_height));
        obstacles.push(Box::new(bitmap_obstacle));

        Self {
            sim_width,
            sim_height,
            obstacles,
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

    fn solve_obstacles(&mut self) {
        for particle in &mut self.particles {
            for obstacle in &self.obstacles {
                let escape_displacement = obstacle.escape_displacement(particle.predicted_pos);
                if let Some(displacement) = escape_displacement {
                    particle.predicted_pos += displacement;

                    let direction = displacement.normalize();
                    let normal_speed = particle.vel.dot(direction);
                    if normal_speed < 0.0 {
                        particle.vel -= direction * (1.0 + RESTITUTION_COEFFICIENT) * normal_speed;
                    }
                }
            }
        }
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
            let blue = (1.0 - density * 0.3).clamp(0.7, 1.0);

            draw_circle(
                particle.pos.x,
                particle.pos.y,
                3.,
                Color::new(red, (green - red * 0.5).clamp(0.0, 1.0), blue, 0.8),
            );
        }
    }
}

impl UpdateWithContext for FluidSim {
    fn update_with_context(&mut self, frame: &Frame) {
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
            self.solve_obstacles();
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

impl DrawWithContext for FluidSim {
    fn draw_with_context(&mut self, frame: &mut Frame) {
        let mouse_pos = frame.relative_mouse_pos();

        for obstacle in &mut self.obstacles {
            obstacle.draw();
        }

        // self.draw_fluid_texture(frame.camera());
        // self.draw_data_particles();
        self.draw_particles(mouse_pos);
    }
}

impl HasSize for FluidSim {
    fn size(&self) -> Vec2 {
        vec2(self.sim_width, self.sim_height)
    }
}

trait Obstacle: Draw + Debug + 'static {
    /// Returns the minimum translation vector needed to move a particle
    /// centred at `pos` out of the obstacle. Returns `None` if no collision.
    fn escape_displacement(&self, pos: Vec2) -> Option<Vec2>;
}

#[derive(Clone, Debug)]
struct CircleObstacle {
    pos: Vec2,
    radius: f32,
}

impl Obstacle for CircleObstacle {
    fn escape_displacement(&self, pos: Vec2) -> Option<Vec2> {
        let displacement = pos - self.pos;
        let distance = displacement.length();

        if distance <= self.radius {
            let direction = displacement.normalize_or(Vec2::X);
            let distance_from_edge = self.radius - distance;
            Some(direction * distance_from_edge)
        } else {
            return None;
        }
    }
}

impl Draw for CircleObstacle {
    fn draw(&mut self) {
        draw_circle_lines(self.pos.x, self.pos.y, self.radius, 4., OBSTACLE_COLOR);
    }
}

#[derive(Clone, Debug)]
struct RectangleObstacle {
    pos: Vec2,
    size: Vec2,
}

impl Obstacle for RectangleObstacle {
    fn escape_displacement(&self, pos: Vec2) -> Option<Vec2> {
        if pos.x < self.pos.x
            || pos.x > self.pos.x + self.size.x
            || pos.y < self.pos.y
            || pos.y > self.pos.y + self.size.y
        {
            return None;
        }

        let dist_from_left = pos.x - self.pos.x;
        let dist_from_right = self.pos.x + self.size.x - pos.x;
        let dist_from_top = pos.y - self.pos.y;
        let dist_from_bottom = self.pos.y + self.size.y - pos.y;

        let min_distance = dist_from_left
            .min(dist_from_right)
            .min(dist_from_top)
            .min(dist_from_bottom);

        if min_distance == dist_from_left {
            Some(Vec2::new(-dist_from_left, 0.0))
        } else if min_distance == dist_from_right {
            Some(Vec2::new(dist_from_right, 0.0))
        } else if min_distance == dist_from_top {
            Some(Vec2::new(0.0, -dist_from_top))
        } else {
            Some(Vec2::new(0.0, dist_from_bottom))
        }
    }
}

impl Draw for RectangleObstacle {
    fn draw(&mut self) {
        draw_rectangle_lines(
            self.pos.x,
            self.pos.y,
            self.size.x,
            self.size.y,
            4.,
            OBSTACLE_COLOR,
        );
    }
}

#[derive(Clone, Debug)]
pub struct BinaryBitmap {
    grid: Grid<bool>,
}

impl BinaryBitmap {
    pub fn new(grid: Grid<bool>) -> Self {
        Self { grid }
    }

    pub fn from_image(image: &Image) -> Self {
        let width = image.width as usize;
        let height = image.height as usize;

        let grid = Grid::from_generator(width, height, |x, y| {
            let i = (y * width + x) * 4;

            let r = image.bytes[i];
            let g = image.bytes[i + 1];
            let b = image.bytes[i + 2];

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

    fn to_texture(&self) -> Texture2D {
        let width = self.grid.columns();
        let height = self.grid.rows();

        let mut bytes = Vec::with_capacity(width * height * 4);

        for y in 0..height {
            for x in 0..width {
                let occupied = *self.grid.get(x as isize, y as isize).unwrap();

                if occupied {
                    bytes.extend_from_slice(&[255, 255, 255, 255]);
                } else {
                    bytes.extend_from_slice(&[0, 0, 0, 255]);
                }
            }
        }

        let image = Image {
            bytes,
            width: width as u16,
            height: height as u16,
        };

        let texture = Texture2D::from_image(&image);
        texture.set_filter(FilterMode::Nearest);

        texture
    }
}

impl From<BinaryBitmap> for Texture2D {
    fn from(bitmap: BinaryBitmap) -> Self {
        bitmap.to_texture()
    }
}

#[derive(Clone, Debug)]
pub struct DistanceField {
    grid: Grid<f32>,
}

impl DistanceField {
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

    fn calculate_distance_field(bitmap: &Grid<bool>, target: bool) -> Grid<f32> {
        let mut distances =
            Grid::from_generator(bitmap.columns(), bitmap.rows(), |_, _| f32::INFINITY);

        // Pixels matching target are distance 0
        for y in 0..bitmap.rows() as isize {
            for x in 0..bitmap.columns() as isize {
                if *bitmap.get(x, y).unwrap() == target {
                    *distances.get_mut(x, y).unwrap() = 0.0;
                }
            }
        }

        // Relax neighbours
        for _ in 0..100 {
            for y in 1..bitmap.rows() as isize - 1 {
                for x in 1..bitmap.columns() as isize - 1 {
                    let i = bitmap.index(x, y).unwrap();

                    let best = distances
                        .get_by_index(i)
                        .unwrap()
                        .min(distances.get_by_index(i - 1).unwrap() + 1.0)
                        .min(distances.get_by_index(i + 1).unwrap() + 1.0)
                        .min(distances.get_by_index(i - bitmap.columns()).unwrap() + 1.0)
                        .min(distances.get_by_index(i + bitmap.columns()).unwrap() + 1.0);

                    *distances.get_mut_by_index(i).unwrap() = best;
                }
            }

            // Reverse pass (important for convergence)
            for y in (1..bitmap.rows() as isize - 1).rev() {
                for x in (1..bitmap.columns() as isize - 1).rev() {
                    let i = bitmap.index(x, y).unwrap();

                    let best = distances
                        .get_by_index(i)
                        .unwrap()
                        .min(distances.get_by_index(i - 1).unwrap() + 1.0)
                        .min(distances.get_by_index(i + 1).unwrap() + 1.0)
                        .min(distances.get_by_index(i - bitmap.columns()).unwrap() + 1.0)
                        .min(distances.get_by_index(i + bitmap.columns()).unwrap() + 1.0);

                    *distances.get_mut_by_index(i).unwrap() = best;
                }
            }
        }

        distances
    }
}

impl From<&BinaryBitmap> for DistanceField {
    fn from(bitmap: &BinaryBitmap) -> Self {
        let obstacle_distances = Self::calculate_distance_field(&bitmap.grid, true);
        let empty_distances = Self::calculate_distance_field(&bitmap.grid, false);

        let distances = Grid::from_generator(bitmap.grid.columns(), bitmap.grid.rows(), |x, y| {
            if *bitmap.grid.get(x as isize, y as isize).unwrap() {
                // Inside obstacle: negative distance to escape
                -empty_distances.get(x as isize, y as isize).unwrap()
            } else {
                // Outside obstacle: positive distance to obstacle
                *obstacle_distances.get(x as isize, y as isize).unwrap()
            }
        });

        Self { grid: distances }
    }
}

#[derive(Clone, Debug)]
pub struct BitmapObstacle {
    pos: Vec2,
    size: Vec2,
    bitmap: BinaryBitmap,
    // distance_field: Option<DistanceField>,
}

impl BitmapObstacle {
    pub fn from_bitmap(bitmap: BinaryBitmap, pos: Vec2, size: Vec2) -> Self {
        // let distance_field = DistanceField::from(&bitmap);
        Self {
            pos,
            size,
            bitmap,
            // distance_field,
        }
    }

    fn relative_pos(&self, pos: Vec2) -> Vec2 {
        let pixel_width = self.size.x / self.bitmap.grid.columns() as f32;
        let pixel_height = self.size.y / self.bitmap.grid.rows() as f32;

        let mut relative_pos = pos;
        relative_pos.x /= pixel_width;
        relative_pos.y /= pixel_height;

        relative_pos
    }

    // fn escape_displacement_sdf(&self, pos: Vec2) -> Option<Vec2> {
    //     let pixel_width = self.size.x / self.bitmap.grid.columns() as f32;
    //     let pixel_height = self.size.y / self.bitmap.grid.rows() as f32;
    //     let relative_pos = self.relative_pos(pos);

    //     let distance = self.distance_field.sample(relative_pos);
    //     if distance >= 0. {
    //         return None;
    //     }

    //     let mut gradient = self.distance_field.gradient(relative_pos);
    //     gradient.x *= pixel_width;
    //     gradient.y *= pixel_height;

    //     Some(gradient * -distance)
    // }

    fn escape_displacement_cheap(&self, pos: Vec2) -> Option<Vec2> {
        let relative_pos = self.relative_pos(pos);

        let x = relative_pos.x as isize;
        let y = relative_pos.y as isize;

        let mut displacement = Vec2::ZERO;

        if self.bitmap.grid.get(x, y).is_some_and(|exists| *exists) {
            for dy in -5..=5 {
                for dx in -5..=5 {
                    let sample_x = x + dx;
                    let sample_y = y + dy;

                    if self
                        .bitmap
                        .grid
                        .get(sample_x, sample_y)
                        .is_some_and(|exists| *exists)
                    {
                        let away = relative_pos - vec2(sample_x as f32, sample_y as f32);

                        if away.length_squared() > 0.0 {
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
        self.escape_displacement_cheap(pos)
    }
}

impl Draw for BitmapObstacle {
    fn draw(&mut self) {
        draw_texture_ex(
            &self.bitmap.to_texture(),
            self.pos.x,
            self.pos.y,
            WHITE,
            DrawTextureParams {
                flip_y: false,
                dest_size: Some(self.size),
                ..Default::default()
            },
        );
    }
}
