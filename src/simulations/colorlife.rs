use crate::buffer::DoubleBuffer;
use crate::engine::{Canvas, Color, Input, Vec2, vec2};
use crate::grid::Grid;
use crate::rng::{Rng, RngExt};
use crate::traits::{HasSize, Simulation};

use rayon::prelude::*;
use std::time::Duration;

const MIN_SPEED: f32 = 0.0;
const MAX_SPEED: f32 = 100.0;
const MAX_VISION_DISTANCE: f32 = 50.;

const AVOIDANCE_DISTANCE: f32 = 10.;
const AVOIDANCE_FACTOR: f32 = 20.;
const INTERACTION_FACTOR: f32 = 1000.;

const EDGE_AVOIDANCE_FACTOR: f32 = 10.;
const EDGE_AVOIDANCE_DISTANCE: f32 = 50.;

#[derive(Clone, Copy, Debug)]
pub enum CreatureType {
    Red,
    Green,
    Blue,
}

impl CreatureType {
    pub fn random(rng: &mut Rng) -> Self {
        match rng.random_range(0..3u32) {
            0 => CreatureType::Red,
            1 => CreatureType::Green,
            _ => CreatureType::Blue,
        }
    }

    pub fn force_on(self, other: CreatureType) -> f32 {
        use CreatureType::*;

        match (self, other) {
            (Red, Red) => 0.2,
            (Red, Green) => 0.8,
            (Red, Blue) => -0.5,
            (Green, Red) => -0.8,
            (Green, Green) => 0.2,
            (Green, Blue) => 0.8,
            (Blue, Red) => 0.8,
            (Blue, Green) => -0.8,
            (Blue, Blue) => 0.2,
        }
    }

    pub fn color(self) -> Color {
        match self {
            Self::Red => Color::from_hex(0xaa4444),
            Self::Green => Color::from_hex(0x44aa44),
            Self::Blue => Color::from_hex(0x4444aa),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Creature {
    index: usize,
    pos: Vec2,
    vel: Vec2,
    size: f32,
    vision_distance: f32,
    species: CreatureType,
}

impl Creature {
    pub fn new(index: usize, x: f32, y: f32, size: f32, species: CreatureType) -> Self {
        Self {
            index,
            pos: Vec2::new(x, y),
            vel: Vec2::new(0., 0.),
            size,
            vision_distance: MAX_VISION_DISTANCE,
            species,
        }
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
        // Compared squared so an in-range speed costs no square root.
        let speed_squared = self.vel.length_squared();

        if speed_squared > MAX_SPEED * MAX_SPEED {
            self.vel *= MAX_SPEED / speed_squared.sqrt();
        } else if MIN_SPEED > 0. && speed_squared < MIN_SPEED * MIN_SPEED {
            // Unreachable while MIN_SPEED is zero, and folded away when it is.
            self.vel = self.vel.normalize_or(Vec2::X) * MIN_SPEED;
        }
    }

    pub fn update<'a>(
        &self,
        deltatime: Duration,
        neighbours: impl Iterator<Item = &'a Creature>,
        sim_width: f32,
        sim_height: f32,
    ) -> Self {
        let mut new_creature = self.clone();
        let mut acceleration = Vec2::new(0., 0.);

        for other in neighbours {
            let displacement = other.pos - self.pos;
            let distance = displacement.length();

            // Avoid getting too close to other creatures
            if self.pos != other.pos && distance < AVOIDANCE_DISTANCE {
                let displacement = self.pos - other.pos;
                acceleration += AVOIDANCE_FACTOR * (displacement / displacement.length_squared());
            }

            if distance < self.vision_distance
                && other.pos.is_finite()
                && other.pos.is_finite()
                && other.vel.is_finite()
            {
                acceleration -= (displacement / displacement.length_squared())
                    * other.species.force_on(self.species)
                    * INTERACTION_FACTOR;
                // flock_members += 1;
                // flock_pos_sum += other.pos;
                // flock_vel_sum += other.vel;
            }
        }

        // Avoid edges
        if self.pos.x < EDGE_AVOIDANCE_DISTANCE {
            acceleration.x +=
                EDGE_AVOIDANCE_FACTOR * (EDGE_AVOIDANCE_DISTANCE / self.pos.x).powi(2);
        } else if self.pos.x > sim_width - EDGE_AVOIDANCE_DISTANCE {
            acceleration.x -= EDGE_AVOIDANCE_FACTOR
                * (EDGE_AVOIDANCE_DISTANCE / (sim_width - self.pos.x)).powi(2);
        }
        if self.pos.y < EDGE_AVOIDANCE_DISTANCE {
            acceleration.y +=
                EDGE_AVOIDANCE_FACTOR * (EDGE_AVOIDANCE_DISTANCE / self.pos.y).powi(2);
        } else if self.pos.y > sim_height - EDGE_AVOIDANCE_DISTANCE {
            acceleration.y -= EDGE_AVOIDANCE_FACTOR
                * (EDGE_AVOIDANCE_DISTANCE / (sim_height - self.pos.y)).powi(2);
        }

        // Update velocity
        new_creature.vel += acceleration * deltatime.as_secs_f32();
        new_creature.clamp_speed();

        // Update position
        new_creature.pos += new_creature.vel * deltatime.as_secs_f32();
        new_creature.clamp_to_frame(sim_width, sim_height);

        new_creature
    }

    pub fn draw(&self, canvas: &mut dyn Canvas) {
        canvas.rect(
            self.pos - Vec2::splat(self.size / 2.),
            Vec2::splat(self.size),
            self.species.color().with_alpha(0.75),
        );
    }
}

#[derive(Clone, Debug)]
pub struct Colorlife {
    sim_width: f32,
    sim_height: f32,
    creatures: DoubleBuffer<Vec<Creature>>,
    chunks: Grid<Vec<usize>>,
}

impl Colorlife {
    pub fn new(creatures: Vec<Creature>, sim_width: f32, sim_height: f32) -> Self {
        let creatures = DoubleBuffer::new(creatures);

        let ideal_chunk_size = MAX_VISION_DISTANCE;
        let columns = (sim_width / ideal_chunk_size).floor() as usize;
        let rows = (sim_height / ideal_chunk_size).floor() as usize;

        Self {
            sim_width,
            sim_height,
            creatures,
            chunks: Grid::with_defaults(columns, rows),
        }
    }

    pub fn init(num_creatures: usize, sim_width: f32, sim_height: f32, mut rng: Rng) -> Self {
        const CREATURE_SIZE: f32 = 5.;

        let mut creatures = Vec::with_capacity(num_creatures);

        for i in 0..num_creatures {
            let x = rng.random_range(100.0..sim_width - 100.);
            let y = rng.random_range(100.0..sim_height - 100.);
            let species = CreatureType::random(&mut rng);
            creatures.push(Creature::new(i, x, y, CREATURE_SIZE, species));
        }

        Self::new(creatures, sim_width, sim_height)
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

        // Register all creatures within their current chunks.
        for creature in self.creatures.state() {
            if let Some(chunk) = self.chunks.get_mut_by_pos(
                creature.pos,
                vec2(0., 0.),
                vec2(self.sim_width, self.sim_height),
            ) {
                chunk.push(creature.index);
            }
        }
    }
}

impl Simulation for Colorlife {
    fn draw(&mut self, canvas: &mut dyn Canvas) {
        for creature in self.creatures.state() {
            creature.draw(canvas);
        }
    }

    fn update(&mut self, deltatime: Duration, _input: &Input) {
        if deltatime.is_zero() {
            return;
        }

        let sim_width = self.sim_width;
        let sim_height = self.sim_height;
        self.update_chunks();

        let chunks = &self.chunks;

        self.creatures.generate(|state, next| {
            next.par_iter_mut()
                .enumerate()
                .for_each(|(i, new_creature)| {
                    let old_creature = &state[i];

                    let neighbours = chunks
                        .get_neighbourhood_at_pos(
                            old_creature.pos,
                            1,
                            vec2(0., 0.),
                            vec2(sim_width, sim_height),
                        )
                        .flat_map(|chunk| chunk.iter())
                        .copied()
                        .filter(|&j| j != i)
                        .map(|j| &state[j]);

                    *new_creature =
                        old_creature.update(deltatime, neighbours, sim_width, sim_height);
                });
        });
    }
}

impl HasSize for Colorlife {
    fn size(&self) -> Vec2 {
        vec2(self.sim_width, self.sim_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Input, RecordingCanvas};
    use crate::rng;

    const W: f32 = 500.;
    const H: f32 = 500.;
    const COUNT: usize = 40;
    const STEP: Duration = Duration::from_millis(16);

    fn sim(seed: u64) -> Colorlife {
        Colorlife::init(COUNT, W, H, rng::seeded(seed))
    }

    fn step(sim: &mut Colorlife, times: usize) {
        for _ in 0..times {
            sim.update(STEP, &Input::none());
        }
    }

    #[test]
    fn creatures_stay_on_screen() {
        let mut sim = sim(1);
        step(&mut sim, 120);

        for creature in sim.creatures.state() {
            assert!(
                (0.0..=W).contains(&creature.pos.x) && (0.0..=H).contains(&creature.pos.y),
                "creature escaped to {:?}",
                creature.pos
            );
        }
    }

    #[test]
    fn nothing_goes_to_nan() {
        let mut sim = sim(2);
        step(&mut sim, 120);

        assert!(
            sim.creatures
                .state()
                .iter()
                .all(|c| c.pos.is_finite() && c.vel.is_finite())
        );
    }

    #[test]
    fn the_population_is_stable() {
        let mut sim = sim(3);
        step(&mut sim, 30);

        assert_eq!(COUNT, sim.creatures.state().len());
    }

    #[test]
    fn a_zero_length_step_changes_nothing() {
        let mut sim = sim(4);
        step(&mut sim, 5);

        let before: Vec<Vec2> = sim.creatures.state().iter().map(|c| c.pos).collect();
        sim.update(Duration::ZERO, &Input::none());
        let after: Vec<Vec2> = sim.creatures.state().iter().map(|c| c.pos).collect();

        assert_eq!(before, after);
    }

    #[test]
    fn the_same_seed_replays_the_same_run() {
        let (mut a, mut b) = (sim(42), sim(42));
        step(&mut a, 30);
        step(&mut b, 30);

        let positions = |s: &Colorlife| {
            s.creatures
                .state()
                .iter()
                .map(|c| c.pos)
                .collect::<Vec<_>>()
        };
        assert_eq!(positions(&a), positions(&b));
    }

    #[test]
    fn species_forces_are_asymmetric() {
        // Particle life is only interesting because A pulling on B does not
        // imply B pulling on A.
        assert_ne!(
            CreatureType::Red.force_on(CreatureType::Green),
            CreatureType::Green.force_on(CreatureType::Red)
        );
    }

    #[test]
    fn all_three_species_get_generated() {
        let mut rng = rng::seeded(9);
        let mut seen = [false; 3];

        for _ in 0..200 {
            match CreatureType::random(&mut rng) {
                CreatureType::Red => seen[0] = true,
                CreatureType::Green => seen[1] = true,
                CreatureType::Blue => seen[2] = true,
            }
        }

        assert_eq!([true, true, true], seen);
    }

    #[test]
    fn drawing_emits_one_rect_per_creature() {
        let mut sim = sim(5);
        step(&mut sim, 3);

        let mut canvas = RecordingCanvas::new();
        sim.draw(&mut canvas);

        assert_eq!(COUNT, canvas.rects().count());
    }
}
