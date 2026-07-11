use crate::buffer::DoubleBuffer;
use crate::component::Frame;
use crate::grid::Grid;
use crate::traits::*;

use macroquad::prelude::*;
use rayon::prelude::*;
use std::time::{Duration, Instant};

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
            .filter(|c| c.is_digit(10))
            .map(|c| c.to_digit(10).unwrap())
            .for_each(|d| rule.birth_rule[d as usize] = true);
        survival
            .chars()
            .filter(|c| c.is_digit(10))
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

    pub fn random() -> Self {
        Cell::new(rand::rand() % 2 == 0)
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
    last_update: Instant,
}

impl Conway {
    fn new(rulestring: &str, grid: Grid<Cell>, width: usize, height: usize) -> Self {
        let buffer = DoubleBuffer::new(grid);

        Self {
            rule: Rule::from_rulestring(rulestring),
            buffer,
            width,
            height,
            last_update: Instant::now(),
        }
    }

    pub fn random(rulestring: &str, fill_percent: f32, width: usize, height: usize) -> Self {
        let fill_divisor = 2. / (1. - fill_percent);

        let fill_offset_x = width as f32 / fill_divisor;
        let fill_offset_y = height as f32 / fill_divisor;

        let generator = |x, y| {
            let (x, y) = (x as f32, y as f32);

            let in_centre = x > fill_offset_x
                && x < width as f32 - fill_offset_x
                && y > fill_offset_y
                && y < height as f32 - fill_offset_y;

            if in_centre {
                Cell::random()
            } else {
                Cell::default()
            }
        };

        let grid = Grid::from_generator(width, height, generator);
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

impl Draw for Conway {
    fn draw(&self, _frame: &mut Frame) {
        for column in 0..self.width as isize {
            for row in 0..self.height as isize {
                let cell = self.cell(column, row).unwrap();

                if cell.is_alive() {
                    draw_rectangle(column as f32, row as f32, 1., 1., ALIVE_COLOR);
                } else {
                    if cell.ghost.is_some_and(|ghost| ghost < 10) {
                        let alpha = 1.0 - ((cell.ghost.unwrap() as f32 + 1.) / 5.);
                        let color = GHOST_COLOR.with_alpha(alpha);
                        draw_rectangle(column as f32, row as f32, 1., 1., color);
                    }
                }
            }
        }
    }
}

impl Update for Conway {
    fn update(&mut self, _frame: &Frame) {
        let update_start = Instant::now();
        if update_start - self.last_update > UPDATE_INTERVAL {
            self.apply_rule(&self.rule.clone());
            self.last_update = update_start;
        }
    }
}

impl HasSize for Conway {
    fn size(&self) -> Vec2 {
        vec2(self.width as f32, self.height as f32)
    }
}
