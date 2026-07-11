use crate::buffer::DoubleBuffer;
use crate::frame::Frame;
use crate::grid::Grid;
use crate::traits::*;

use macroquad::prelude::*;
use rayon::prelude::*;
use std::time::{Duration, Instant};

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
    pub fn random() -> Self {
        let i = rand::rand() % 3;
        match i {
            0 => CreatureType::Red,
            1 => CreatureType::Green,
            2 => CreatureType::Blue,
            _ => panic!(),
        }
    }

    pub fn force_on(&self, other: CreatureType) -> f32 {
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

    pub fn color(&self) -> Color {
        match &self {
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
        if self.vel.length() < MIN_SPEED {
            self.vel = self.vel.normalize() * MIN_SPEED
        } else if self.vel.length() > MAX_SPEED {
            self.vel = self.vel.normalize() * MAX_SPEED
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

    pub fn draw(&self) {
        // draw_circle(
        //     self.pos.x + frame.x(),
        //     self.pos.y + frame.y(),
        //     self.size,
        //     self.species.color().with_alpha(0.75),
        // );

        draw_rectangle(
            self.pos.x - self.size / 2.,
            self.pos.y - self.size / 2.,
            self.size,
            self.size,
            self.species.color().with_alpha(0.75),
        );
    }
}

pub struct Colorlife {
    sim_width: f32,
    sim_height: f32,
    creatures: DoubleBuffer<Vec<Creature>>,
    chunks: Grid<Vec<usize>>,
    last_update: Instant,
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
            last_update: Instant::now(),
        }
    }

    pub fn init(num_creatures: usize, sim_width: f32, sim_height: f32) -> Self {
        let mut creatures = Vec::new();

        for i in 0..num_creatures {
            let x = rand::gen_range(100., sim_width - 100.);
            let y = rand::gen_range(100., sim_height - 100.);
            let size = rand::gen_range(5., 5.);
            let species = CreatureType::random();
            creatures.push(Creature::new(i, x, y, size, species));
        }

        Self::new(creatures, sim_width, sim_height)
    }

    fn update_chunks(&mut self) {
        let ideal_chunk_size = MAX_VISION_DISTANCE;
        let columns = (self.sim_width / ideal_chunk_size).floor();
        let rows = (self.sim_height / ideal_chunk_size).floor();

        // Reset all chunks.
        for chunk in self.chunks.iter_mut() {
            chunk.clear()
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

impl Draw for Colorlife {
    fn draw(&self, _frame: &mut Frame) {
        for creature in self.creatures.state() {
            creature.draw();
        }
    }
}

impl Update for Colorlife {
    fn update(&mut self, _frame: &Frame) {
        let update_start = Instant::now();
        let deltatime = update_start - self.last_update;

        self.update_chunks();

        let (creatures, chunks) = (&mut self.creatures, &self.chunks);

        let (state, next) = creatures.states();

        next.par_iter_mut()
            .enumerate()
            .for_each(|(i, new_creature)| {
                let old_creature = &state[i];

                let neighbours = chunks
                    .get_neighbourhood_at_pos(
                        old_creature.pos,
                        1,
                        vec2(0., 0.),
                        vec2(self.sim_width, self.sim_height),
                    )
                    .flat_map(|chunk| chunk.iter())
                    .copied()
                    .filter(|&j| j != i)
                    .map(|j| &state[j]);

                *new_creature =
                    old_creature.update(deltatime, neighbours, self.sim_width, self.sim_height);
            });

        self.creatures.swap();
        self.last_update = update_start;
    }
}

impl HasSize for Colorlife {
    fn size(&self) -> Vec2 {
        vec2(self.sim_width, self.sim_height)
    }
}
