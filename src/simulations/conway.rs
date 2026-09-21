use crate::buffer::DoubleBuffer;
use crate::engine::{Canvas, Color, Input, Vec2, vec2};
use crate::grid::Grid;
use crate::rng::{Rng, RngExt};
use crate::traits::{HasSize, Simulation};

use rayon::prelude::*;
use std::time::Duration;

const ALIVE_COLOR: Color = Color::new(0.8, 0.8, 0.8, 1.0);
const GHOST_COLOR: Color = Color::new(0.6, 0.6, 0.8, 1.0);
const UPDATE_INTERVAL: Duration = Duration::from_millis(100);

pub const _CONWAY: &str = "B3/S23";
pub const _MAZE: &str = "B3/S12345";
pub const _MAZECETRIC: &str = "B3/S1234";
pub const _CORAL: &str = "B3/S45678";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rule {
    birth_rule: [bool; 9],
    death_rule: [bool; 9],
}

impl Rule {
    fn from_rulestring(rulestring: &str) -> Self {
        let (births, survival) = rulestring.split_once('/').expect("invald rulestring!");
        let mut rule = Rule::default();

        births
            .chars()
            .filter(char::is_ascii_digit)
            .map(|c| c.to_digit(10).unwrap())
            .for_each(|d| rule.birth_rule[d as usize] = true);
        survival
            .chars()
            .filter(char::is_ascii_digit)
            .map(|c| c.to_digit(10).unwrap())
            .for_each(|d| rule.death_rule[d as usize] = true);

        rule
    }

    pub fn apply(&self, currently_alive: bool, num_neighbours: usize) -> bool {
        if currently_alive {
            self.death_rule[num_neighbours]
        } else {
            self.birth_rule[num_neighbours]
        }
    }
}

#[derive(Clone, Debug)]
struct Cell {
    alive: bool,
    age: Option<usize>,
    ghost: Option<usize>,
}

impl Cell {
    pub fn new(alive: bool) -> Self {
        Cell {
            alive,
            age: None,
            ghost: None,
        }
    }

    pub fn apply_rule(&self, rule: &Rule, neighbours: usize) -> Self {
        let mut new_cell = self.clone();
        new_cell.alive = rule.apply(self.alive, neighbours);

        if self.alive && !new_cell.alive {
            new_cell.age = None;
            new_cell.ghost = Some(0);
        } else if !self.alive && new_cell.alive {
            new_cell.ghost = None;
            new_cell.age = Some(0);
        }

        new_cell.age = new_cell.age.map(|age| age + 1);
        new_cell.ghost = new_cell.ghost.map(|age| age + 1);

        new_cell
    }

    pub fn random(rng: &mut Rng) -> Self {
        Cell::new(rng.random())
    }

    pub fn is_alive(&self) -> bool {
        self.alive
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self::new(false)
    }
}

#[derive(Clone, Debug)]
pub struct Conway {
    rule: Rule,
    buffer: DoubleBuffer<Grid<Cell>>,
    width: usize,
    height: usize,
    /// Time owed to the simulation but not yet spent on a tick. Conway runs on
    /// a fixed interval, so leftover time carries into the next update rather
    /// than being dropped.
    unspent: Duration,
}

impl Conway {
    fn new(rulestring: &str, grid: Grid<Cell>, width: usize, height: usize) -> Self {
        let buffer = DoubleBuffer::new(grid);

        Self {
            rule: Rule::from_rulestring(rulestring),
            buffer,
            width,
            height,
            unspent: Duration::ZERO,
        }
    }

    pub fn random(
        rulestring: &str,
        fill_percent: f32,
        width: usize,
        height: usize,
        mut rng: Rng,
    ) -> Self {
        let fill_divisor = 2. / (1. - fill_percent);

        let fill_offset_x = width as f32 / fill_divisor;
        let fill_offset_y = height as f32 / fill_divisor;

        let mut generator = |x, y| {
            let (x, y) = (x as f32, y as f32);

            let in_centre = x > fill_offset_x
                && x < width as f32 - fill_offset_x
                && y > fill_offset_y
                && y < height as f32 - fill_offset_y;

            if in_centre {
                Cell::random(&mut rng)
            } else {
                Cell::default()
            }
        };

        let grid = Grid::from_generator_mut(width, height, &mut generator);
        Self::new(rulestring, grid, width, height)
    }

    pub fn apply_rule(&mut self, rule: &Rule) {
        let (current, next) = self.buffer.states();

        next.par_iter_mut()
            .enumerate()
            .for_each(|(index, new_cell)| {
                let x = (index % self.width) as isize;
                let y = (index / self.width) as isize;

                let neighbours = current
                    .get_neighbours(x, y, 1)
                    .filter(|cell| cell.is_alive())
                    .count();

                let old_cell = current
                    .get(x, y)
                    .expect("conway: grid size changed unexpectedly between updates");

                *new_cell = old_cell.apply_rule(rule, neighbours);
            });

        self.buffer.swap();
    }

    fn cell(&self, x: isize, y: isize) -> Option<&Cell> {
        self.buffer.state().get(x, y)
    }
}

impl Simulation for Conway {
    fn update(&mut self, dt: Duration, _input: &Input) {
        self.unspent += dt;

        // Catch up if several intervals elapsed, but never spin forever on a
        // pathologically large `dt`.
        const MAX_TICKS_PER_UPDATE: u32 = 4;

        let mut ticks = 0;
        while self.unspent >= UPDATE_INTERVAL && ticks < MAX_TICKS_PER_UPDATE {
            self.apply_rule(&self.rule.clone());
            self.unspent -= UPDATE_INTERVAL;
            ticks += 1;
        }

        if ticks == MAX_TICKS_PER_UPDATE {
            self.unspent = Duration::ZERO;
        }
    }

    fn draw(&mut self, canvas: &mut dyn Canvas) {
        const CELL: Vec2 = Vec2::ONE;

        for column in 0..self.width as isize {
            for row in 0..self.height as isize {
                let cell = self.cell(column, row).unwrap();
                let pos = vec2(column as f32, row as f32);

                if cell.is_alive() {
                    canvas.rect(pos, CELL, ALIVE_COLOR);
                } else if cell.ghost.is_some_and(|ghost| ghost < 10) {
                    let alpha = 1.0 - ((cell.ghost.unwrap() as f32 + 1.) / 5.);
                    canvas.rect(pos, CELL, GHOST_COLOR.with_alpha(alpha));
                }
            }
        }
    }
}

impl HasSize for Conway {
    fn size(&self) -> Vec2 {
        vec2(self.width as f32, self.height as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Input, RecordingCanvas};
    use crate::rng;

    const TICK: Duration = UPDATE_INTERVAL;

    /// Builds a board from an ASCII picture: `#` alive, anything else dead.
    fn board(rows: &[&str]) -> Conway {
        let height = rows.len();
        let width = rows[0].len();

        let grid = Grid::from_generator(width, height, |x, y| {
            Cell::new(rows[y].as_bytes()[x] == b'#')
        });

        Conway::new(_CONWAY, grid, width, height)
    }

    /// Renders the live cells back out as an ASCII picture.
    fn render(sim: &Conway) -> Vec<String> {
        (0..sim.height as isize)
            .map(|y| {
                (0..sim.width as isize)
                    .map(|x| {
                        if sim.cell(x, y).unwrap().is_alive() {
                            '#'
                        } else {
                            '.'
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn tick(sim: &mut Conway, times: usize) {
        for _ in 0..times {
            sim.update(TICK, &Input::none());
        }
    }

    #[test]
    fn a_block_is_a_still_life() {
        let mut sim = board(&["....", ".##.", ".##.", "...."]);
        let before = render(&sim);

        tick(&mut sim, 4);

        assert_eq!(before, render(&sim));
    }

    #[test]
    fn a_blinker_oscillates_with_period_two() {
        let mut sim = board(&[".....", ".....", ".###.", ".....", "....."]);
        let horizontal = render(&sim);

        tick(&mut sim, 1);
        let vertical = render(&sim);

        assert_ne!(horizontal, vertical, "blinker did not move");
        assert_eq!(
            vec![".....", "..#..", "..#..", "..#..", "....."],
            vertical,
            "expected the vertical phase"
        );

        tick(&mut sim, 1);
        assert_eq!(horizontal, render(&sim), "did not return after two ticks");
    }

    #[test]
    fn a_glider_walks_one_cell_diagonally_every_four_ticks() {
        let mut sim = board(&[
            "..........",
            "..#.......",
            "...#......",
            ".###......",
            "..........",
            "..........",
            "..........",
            "..........",
        ]);

        tick(&mut sim, 4);

        // The same glider, translated one cell down and one cell right.
        assert_eq!(
            [
                "..........",
                "..........",
                "...#......",
                "....#.....",
                "..###.....",
                "..........",
                "..........",
                "..........",
            ]
            .map(str::to_string),
            render(&sim).as_slice()
        );
    }

    #[test]
    fn an_empty_board_stays_empty() {
        let mut sim = board(&["....", "....", "...."]);
        tick(&mut sim, 10);

        assert!(render(&sim).iter().all(|row| !row.contains('#')));
    }

    #[test]
    fn a_lone_cell_dies_of_loneliness() {
        let mut sim = board(&["...", ".#.", "..."]);
        tick(&mut sim, 1);

        assert!(!sim.cell(1, 1).unwrap().is_alive());
    }

    #[test]
    fn ticks_are_paced_by_the_interval_not_the_call_count() {
        let mut sim = board(&[".....", ".....", ".###.", ".....", "....."]);
        let start = render(&sim);

        // Ten updates that together add up to less than one interval.
        for _ in 0..10 {
            sim.update(UPDATE_INTERVAL / 20, &Input::none());
        }
        assert_eq!(start, render(&sim), "advanced before the interval elapsed");

        // One more takes the total past it.
        for _ in 0..11 {
            sim.update(UPDATE_INTERVAL / 20, &Input::none());
        }
        assert_ne!(start, render(&sim), "never advanced");
    }

    #[test]
    fn a_huge_step_does_not_spin_forever() {
        let mut sim = board(&[".....", ".....", ".###.", ".....", "....."]);

        // An hour of catch-up is capped, not simulated tick by tick.
        sim.update(Duration::from_secs(3600), &Input::none());

        assert_eq!(Duration::ZERO, sim.unspent);
    }

    #[test]
    fn a_rulestring_is_parsed_into_birth_and_survival_sets() {
        let rule = Rule::from_rulestring("B3/S23");

        assert!(rule.apply(false, 3), "three neighbours should birth");
        assert!(!rule.apply(false, 2), "two neighbours should not birth");
        assert!(rule.apply(true, 2), "two neighbours should survive");
        assert!(rule.apply(true, 3), "three neighbours should survive");
        assert!(!rule.apply(true, 4), "four neighbours should die");
    }

    #[test]
    fn drawing_emits_a_cell_for_every_live_cell() {
        let mut sim = board(&["....", ".##.", ".##.", "...."]);

        let mut canvas = RecordingCanvas::new();
        sim.draw(&mut canvas);

        // Four live cells, no ghosts yet.
        assert_eq!(4, canvas.rects().count());
        assert!(canvas.rects().all(|(_, size, _)| size == Vec2::ONE));
    }

    #[test]
    fn the_same_seed_produces_the_same_board() {
        let a = Conway::random(_CONWAY, 0.6, 20, 20, rng::seeded(7));
        let b = Conway::random(_CONWAY, 0.6, 20, 20, rng::seeded(7));

        assert_eq!(render(&a), render(&b));
    }
}
