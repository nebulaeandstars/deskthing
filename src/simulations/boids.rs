use crate::buffer::DoubleBuffer;
use crate::component::Frame;
use crate::grid::Grid;
use crate::traits::*;

use macroquad::prelude::*;
use rand::RandomRange;
use rayon::prelude::*;
use std::f32::consts::{PI, TAU};
use std::time::{Duration, Instant};

const BOID_COLOR: Color = crate::OUTLINE_COLOR;
const BOID_HEIGHT: f32 = 15.0;
const BOID_WIDTH: f32 = 10.0;
const BOID_MIN_SPEED: f32 = 100.0;
const BOID_MAX_SPEED: f32 = 300.0;

const BOID_AVOIDANCE_FACTOR: f32 = 500.;
const BOID_AVOIDANCE_DISTANCE: f32 = 50.;
const BOID_ALIGNMENT_FACTOR: f32 = 10.;
const BOID_COHESION_FACTOR: f32 = 10.;
const BOID_EDGE_AVOIDANCE_FACTOR: f32 = 100.;
const BOID_EDGE_AVOIDANCE_DISTANCE: f32 = 50.;
const BOID_WANDER_FACTOR: f32 = 1000.;

const MAX_VISION_DISTANCE: f32 = 100.;
const DEFAULT_VISION_DISTANCE: f32 = 50.;
const MIN_FLOCKMATES: usize = 3;
const MAX_FLOCKMATES: usize = 20;
const VISION_DISTANCE_CHANGE_RATE: f32 = 10.;
const VISION_ANGLE: f32 = TAU * 2. / 3.;

#[derive(Clone, Debug)]
pub struct Boid {
    index: usize,
    pos: Vec2,
    vel: Vec2,
    vision_distance: f32,
}

impl Boid {
    pub fn new(index: usize, x: f32, y: f32, heading: f32) -> Self {
        let vel = Vec2::from_angle(heading) * (BOID_MIN_SPEED + BOID_MAX_SPEED) / 2.;

        Self {
            index,
            pos: Vec2::new(x, y),
            vel,
            vision_distance: DEFAULT_VISION_DISTANCE,
        }
    }

    fn heading(&self) -> Vec2 {
        self.vel.normalize()
    }

    pub fn clamp_to_frame(&mut self, sim_width: f32, sim_height: f32) {
        if self.pos.x < 1. {
            self.pos.x = 1.;
        } else if self.pos.x > sim_width - 1. {
            self.pos.x = sim_width - 1.;
        }

        if self.pos.y < 1. {
            self.pos.y = 1.;
        } else if self.pos.y > sim_height - 1. {
            self.pos.y = sim_height - 1.;
        }
    }

    pub fn clamp_speed(&mut self) {
        if self.vel.length() < BOID_MIN_SPEED {
            self.vel = self.vel.normalize() * BOID_MIN_SPEED;
        } else if self.vel.length() > BOID_MAX_SPEED {
            self.vel = self.vel.normalize() * BOID_MAX_SPEED;
        }
    }

    pub fn update<'a>(
        &self,
        deltatime: Duration,
        neighbours: impl Iterator<Item = &'a Boid>,
        sim_width: f32,
        sim_height: f32,
    ) -> Self {
        let mut new_boid = self.clone();
        let mut acceleration = Vec2::new(0., 0.);

        let mut flock_members: usize = 0;
        let mut flock_pos_sum: Vec2 = Vec2::ZERO;
        let mut flock_vel_sum: Vec2 = Vec2::ZERO;

        for other in neighbours {
            let displacement = other.pos - self.pos;
            let distance = displacement.length();

            // Avoid getting too close to other boids
            if self.pos != other.pos && distance < BOID_AVOIDANCE_DISTANCE {
                let displacement = self.pos - other.pos;
                acceleration +=
                    BOID_AVOIDANCE_FACTOR * (displacement / displacement.length_squared());
            }

            if distance < self.vision_distance
                && self.heading().angle_between(displacement).abs() < (VISION_ANGLE / 2.)
                && other.pos.is_finite()
                && other.pos.is_finite()
                && other.vel.is_finite()
            {
                flock_members += 1;
                flock_pos_sum += other.pos;
                flock_vel_sum += other.vel;
            }
        }

        // Try to achieve cohesion and alignment with observed flockmates.
        if flock_members > 0 {
            let flock_pos = flock_pos_sum / flock_members as f32;
            let flock_vel = flock_vel_sum / flock_members as f32;
            acceleration += (flock_pos - self.pos) * BOID_COHESION_FACTOR;
            acceleration += (flock_vel - self.vel) * BOID_ALIGNMENT_FACTOR;
        }

        // Wander slightly
        let wander = Vec2::new(
            RandomRange::gen_range(-1., 1.),
            RandomRange::gen_range(-1., 1.),
        );
        acceleration += wander * BOID_WANDER_FACTOR;

        // Reduce/expand flock distance if there are too few/many members.
        if flock_members < MIN_FLOCKMATES {
            new_boid.vision_distance += VISION_DISTANCE_CHANGE_RATE * deltatime.as_secs_f32();
        } else if flock_members > MAX_FLOCKMATES {
            new_boid.vision_distance -= VISION_DISTANCE_CHANGE_RATE * deltatime.as_secs_f32();
        }

        // Avoid edges
        if self.pos.x < BOID_EDGE_AVOIDANCE_DISTANCE {
            acceleration.x +=
                BOID_EDGE_AVOIDANCE_FACTOR * (BOID_EDGE_AVOIDANCE_DISTANCE / self.pos.x).powi(2);
        } else if self.pos.x > sim_width - BOID_EDGE_AVOIDANCE_DISTANCE {
            acceleration.x -= BOID_EDGE_AVOIDANCE_FACTOR
                * (BOID_EDGE_AVOIDANCE_DISTANCE / (sim_width - self.pos.x)).powi(2);
        }
        if self.pos.y < BOID_EDGE_AVOIDANCE_DISTANCE {
            acceleration.y +=
                BOID_EDGE_AVOIDANCE_FACTOR * (BOID_EDGE_AVOIDANCE_DISTANCE / self.pos.y).powi(2);
        } else if self.pos.y > sim_height - BOID_EDGE_AVOIDANCE_DISTANCE {
            acceleration.y -= BOID_EDGE_AVOIDANCE_FACTOR
                * (BOID_EDGE_AVOIDANCE_DISTANCE / (sim_height - self.pos.y)).powi(2);
        }

        // Update velocity
        new_boid.vel += acceleration * deltatime.as_secs_f32();
        new_boid.clamp_speed();

        // Update position
        new_boid.pos += new_boid.vel * deltatime.as_secs_f32();
        new_boid.clamp_to_frame(sim_width, sim_height);

        new_boid
    }

    pub fn draw(&self) {
        let heading = self.heading();

        let v1 = self.pos + heading.rotate(Vec2::new(BOID_HEIGHT / 2.0, 0.0));
        let v2 = self.pos + heading.rotate(Vec2::new(-BOID_HEIGHT / 2.0, BOID_WIDTH / 2.0));
        let v3 = self.pos + heading.rotate(Vec2::new(-BOID_HEIGHT / 2.0, -BOID_WIDTH / 2.0));

        draw_triangle(v1, v2, v3, BOID_COLOR);
    }
}

impl PartialEq for Boid {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

#[derive(Clone, Debug)]
pub struct Boids {
    sim_width: f32,
    sim_height: f32,
    boids: DoubleBuffer<Vec<Boid>>,
    chunks: Grid<Vec<usize>>,
    last_update: Instant,
}

impl Boids {
    pub fn new(boids: Vec<Boid>, sim_width: f32, sim_height: f32) -> Self {
        let boids = DoubleBuffer::new(boids);

        let ideal_chunk_size = MAX_VISION_DISTANCE;
        let columns = (sim_width / ideal_chunk_size).floor() as usize;
        let rows = (sim_height / ideal_chunk_size).floor() as usize;

        Self {
            sim_width,
            sim_height,
            boids,
            chunks: Grid::with_defaults(columns, rows),
            last_update: Instant::now(),
        }
    }

    pub fn init(num_boids: usize, sim_width: f32, sim_height: f32) -> Self {
        let mut boids = Vec::new();

        for i in 0..num_boids {
            let x = rand::gen_range(10., sim_width - 10.);
            let y = rand::gen_range(10., sim_height - 10.);
            let heading = rand::gen_range(0.0, PI * 2.0);
            boids.push(Boid::new(i, x, y, heading));
        }

        Self::new(boids, sim_width, sim_height)
    }

    fn update_chunks(&mut self) {
        let ideal_chunk_size = MAX_VISION_DISTANCE;
        let columns = (self.sim_width / ideal_chunk_size).floor();
        let rows = (self.sim_height / ideal_chunk_size).floor();

        // Reset all chunks.
        for chunk in self.chunks.iter_mut() {
            chunk.clear();
        }

        // Grow the grid if needed (eg. if the frame size increases)
        self.chunks
            .resize_with_defaults(columns as usize, rows as usize);

        // Register all boids within their current chunks.
        for boid in self.boids.state() {
            if let Some(chunk) = self.chunks.get_mut_by_pos(
                boid.pos,
                vec2(0., 0.),
                vec2(self.sim_width, self.sim_height),
            ) {
                chunk.push(boid.index);
            }
        }
    }
}

impl Draw for Boids {
    fn draw(&self, _frame: &mut Frame) {
        clear_background(BLANK);
        for boid in self.boids.state() {
            boid.draw();
        }
    }
}

impl Update for Boids {
    fn update(&mut self, _frame: &Frame) {
        let update_start = Instant::now();
        let deltatime = update_start - self.last_update;

        let sim_width = self.sim_width;
        let sim_height = self.sim_height;
        self.update_chunks();

        let (boids, chunks) = (&mut self.boids, &self.chunks);

        let (state, next) = boids.states();

        next.par_iter_mut().enumerate().for_each(|(i, new_boid)| {
            let old_boid = &state[i];

            let neighbours = chunks
                .get_neighbourhood_at_pos(
                    old_boid.pos,
                    1,
                    vec2(0., 0.),
                    vec2(self.sim_width, self.sim_height),
                )
                .flat_map(|chunk| chunk.iter())
                .copied()
                .filter(|&j| j != i)
                .map(|j| &state[j]);

            *new_boid = old_boid.update(deltatime, neighbours, sim_width, sim_height);
        });

        self.boids.swap();
        self.last_update = update_start;
    }
}

impl HasSize for Boids {
    fn size(&self) -> Vec2 {
        vec2(self.sim_width, self.sim_height)
    }
}
