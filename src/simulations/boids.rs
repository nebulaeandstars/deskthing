use crate::buffer::DoubleBuffer;
use crate::grid::Grid;

use crate::engine::{Canvas, Color, Input, Vec2, vec2};
use crate::rng::{Rng, RngExt};
use crate::traits::{HasSize, Simulation};

use rayon::prelude::*;
use std::f32::consts::{PI, TAU};
use std::time::Duration;

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

    /// Falls back to a fixed direction when stationary: `normalize` on a zero
    /// vector is NaN, and a NaN heading poisons the boid permanently.
    fn heading(&self) -> Vec2 {
        self.vel.normalize_or(Vec2::X)
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
        // Boids have a non-zero minimum speed, so this branch is live — and a
        // boid that came to a dead stop would normalize to NaN without the
        // fallback direction.
        let speed = self.vel.length();

        if speed < BOID_MIN_SPEED {
            self.vel = self.vel.normalize_or(Vec2::X) * BOID_MIN_SPEED;
        } else if speed > BOID_MAX_SPEED {
            self.vel *= BOID_MAX_SPEED / speed;
        }
    }

    pub fn update<'a>(
        &self,
        deltatime: Duration,
        neighbours: impl Iterator<Item = &'a Boid>,
        sim_width: f32,
        sim_height: f32,
        wander: Vec2,
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

        // Wander slightly. The vector is supplied rather than drawn here so
        // this stays a pure function of its inputs, and so a seeded run
        // reproduces regardless of how rayon schedules the work.
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

    pub fn draw(&self, canvas: &mut dyn Canvas) {
        let heading = self.heading();

        let v1 = self.pos + heading.rotate(Vec2::new(BOID_HEIGHT / 2.0, 0.0));
        let v2 = self.pos + heading.rotate(Vec2::new(-BOID_HEIGHT / 2.0, BOID_WIDTH / 2.0));
        let v3 = self.pos + heading.rotate(Vec2::new(-BOID_HEIGHT / 2.0, -BOID_WIDTH / 2.0));

        canvas.triangle(v1, v2, v3, BOID_COLOR);
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
    rng: Rng,
}

impl Boids {
    pub fn new(boids: Vec<Boid>, sim_width: f32, sim_height: f32, rng: Rng) -> Self {
        let boids = DoubleBuffer::new(boids);

        let ideal_chunk_size = MAX_VISION_DISTANCE;
        let columns = (sim_width / ideal_chunk_size).floor() as usize;
        let rows = (sim_height / ideal_chunk_size).floor() as usize;

        Self {
            sim_width,
            sim_height,
            boids,
            chunks: Grid::with_defaults(columns, rows),
            rng,
        }
    }

    pub fn init(num_boids: usize, sim_width: f32, sim_height: f32, mut rng: Rng) -> Self {
        let mut boids = Vec::with_capacity(num_boids);

        for i in 0..num_boids {
            let x = rng.random_range(10.0..sim_width - 10.);
            let y = rng.random_range(10.0..sim_height - 10.);
            let heading = rng.random_range(0.0..PI * 2.0);
            boids.push(Boid::new(i, x, y, heading));
        }

        Self::new(boids, sim_width, sim_height, rng)
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

impl Simulation for Boids {
    fn draw(&mut self, canvas: &mut dyn Canvas) {
        for boid in self.boids.state() {
            boid.draw(canvas);
        }
    }

    fn update(&mut self, deltatime: Duration, _input: &Input) {
        if deltatime.is_zero() {
            return;
        }

        // Drawn up front: the generator cannot cross rayon's threads.
        let wanders: Vec<Vec2> = (0..self.boids.state().len())
            .map(|_| {
                vec2(
                    self.rng.random_range(-1.0..1.0),
                    self.rng.random_range(-1.0..1.0),
                )
            })
            .collect();

        let sim_width = self.sim_width;
        let sim_height = self.sim_height;
        self.update_chunks();

        let chunks = &self.chunks;

        self.boids.generate(|state, next| {
            next.par_iter_mut().enumerate().for_each(|(i, new_boid)| {
                let old_boid = &state[i];

                let neighbours = chunks
                    .get_neighbourhood_at_pos(
                        old_boid.pos,
                        1,
                        vec2(0., 0.),
                        vec2(sim_width, sim_height),
                    )
                    .flat_map(|chunk| chunk.iter())
                    .copied()
                    .filter(|&j| j != i)
                    .map(|j| &state[j]);

                *new_boid =
                    old_boid.update(deltatime, neighbours, sim_width, sim_height, wanders[i]);
            });
        });
    }
}

impl HasSize for Boids {
    fn size(&self) -> Vec2 {
        vec2(self.sim_width, self.sim_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Input, RecordingCanvas};
    use crate::rng;

    const W: f32 = 400.;
    const H: f32 = 400.;
    const COUNT: usize = 40;
    const STEP: Duration = Duration::from_millis(16);

    fn sim(seed: u64) -> Boids {
        Boids::init(COUNT, W, H, rng::seeded(seed))
    }

    fn step(sim: &mut Boids, times: usize) {
        for _ in 0..times {
            sim.update(STEP, &Input::none());
        }
    }

    #[test]
    fn boids_stay_on_screen() {
        let mut sim = sim(1);
        step(&mut sim, 120);

        for boid in sim.boids.state() {
            assert!(
                (0.0..=W).contains(&boid.pos.x) && (0.0..=H).contains(&boid.pos.y),
                "boid escaped to {:?}",
                boid.pos
            );
        }
    }

    #[test]
    fn boids_keep_moving_within_their_speed_limits() {
        let mut sim = sim(2);
        step(&mut sim, 60);

        for boid in sim.boids.state() {
            let speed = boid.vel.length();
            assert!(
                (BOID_MIN_SPEED - 1e-2..=BOID_MAX_SPEED + 1e-2).contains(&speed),
                "speed {speed} outside its clamp"
            );
        }
    }

    #[test]
    fn nothing_goes_to_nan() {
        let mut sim = sim(3);
        step(&mut sim, 120);

        assert!(
            sim.boids
                .state()
                .iter()
                .all(|b| b.pos.is_finite() && b.vel.is_finite())
        );
    }

    #[test]
    fn the_flock_is_neither_gained_nor_lost() {
        let mut sim = sim(4);
        step(&mut sim, 30);

        assert_eq!(COUNT, sim.boids.state().len());
    }

    #[test]
    fn two_boids_on_top_of_each_other_push_apart() {
        let boids = vec![
            Boid::new(0, W / 2., H / 2., 0.),
            Boid::new(1, W / 2. + 1., H / 2., PI),
        ];
        let mut sim = Boids::new(boids, W, H, rng::seeded(5));

        let before = sim.boids.state()[0].pos.distance(sim.boids.state()[1].pos);
        step(&mut sim, 10);
        let after = sim.boids.state()[0].pos.distance(sim.boids.state()[1].pos);

        assert!(after > before, "separation went {before} -> {after}");
    }

    #[test]
    fn a_zero_length_step_changes_nothing() {
        let mut sim = sim(6);
        step(&mut sim, 5);

        let before: Vec<Vec2> = sim.boids.state().iter().map(|b| b.pos).collect();
        sim.update(Duration::ZERO, &Input::none());
        let after: Vec<Vec2> = sim.boids.state().iter().map(|b| b.pos).collect();

        assert_eq!(before, after);
    }

    #[test]
    fn the_same_seed_replays_the_same_flock() {
        let (mut a, mut b) = (sim(42), sim(42));
        step(&mut a, 30);
        step(&mut b, 30);

        let positions = |s: &Boids| s.boids.state().iter().map(|x| x.pos).collect::<Vec<_>>();
        assert_eq!(positions(&a), positions(&b));
    }

    #[test]
    fn drawing_emits_one_triangle_per_boid() {
        let mut sim = sim(7);
        step(&mut sim, 3);

        let mut canvas = RecordingCanvas::new();
        sim.draw(&mut canvas);

        assert_eq!(COUNT, canvas.triangles().count());
    }
}
