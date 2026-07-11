#![allow(unused)]

use crate::frame::Frame;
use crate::traits::*;

use macroquad::prelude::*;
use rayon::prelude::*;
use std::iter::IntoIterator;

#[derive(Clone, Debug)]
pub struct Grid<T> {
    inner: Vec<T>,
    columns: usize,
    rows: usize,
}

#[allow(unused)]
impl<T: Default> Grid<T> {
    /// Constructs a new, empty grid.
    pub fn with_defaults(columns: usize, rows: usize) -> Self {
        let mut inner = Vec::with_capacity(columns * rows);
        for _ in 0..(columns * rows) {
            inner.push(T::default());
        }

        Self::new(inner, columns, rows)
    }

    /// Updates the dimensions of the grid.
    pub fn resize_with_defaults(&mut self, columns: usize, rows: usize) {
        self.columns = columns;
        self.rows = rows;
        self.inner.reserve(columns * rows);

        for _ in self.inner.len()..(columns * rows) {
            self.inner.push(T::default())
        }
    }
}

#[allow(unused)]
impl<T> Grid<T> {
    /// Constructs a new grid with the given values.
    pub fn new(cells: Vec<T>, columns: usize, rows: usize) -> Self {
        let grid = Self {
            inner: cells,
            columns,
            rows,
        };

        debug_assert_eq!(grid.size(), grid.inner.len());
        grid
    }

    /// Constructs a new grid, generating values according to the generator
    /// function.
    pub fn from_generator<F: Fn(usize, usize) -> T>(columns: usize, rows: usize, f: F) -> Self {
        let mut cells = Vec::with_capacity(columns * rows);
        for row in 0..rows {
            for col in 0..columns {
                cells.push(f(col, row));
            }
        }

        let grid = Self {
            inner: cells,
            columns,
            rows,
        };

        debug_assert_eq!(grid.size(), grid.inner.len());
        grid
    }

    /// Returns the number of rows in the grid.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns in the grid.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the number of cells in the grid.
    pub fn size(&self) -> usize {
        debug_assert_eq!(self.inner.len(), self.columns * self.rows);
        self.columns * self.rows
    }

    /// Returns a reference to a grid cell, by column and row.
    pub fn get(&self, column: isize, row: isize) -> Option<&T> {
        self.index(column, row).and_then(|i| self.get_by_index(i))
    }

    /// Returns a mutable reference to a grid cell, by column and row.
    pub fn get_mut(&mut self, column: isize, row: isize) -> Option<&mut T> {
        self.index(column, row)
            .and_then(|i| self.get_mut_by_index(i))
    }

    /// Returns a reference to a grid cell, by index.
    pub fn get_by_index(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    /// Returns a mutable reference to a grid cell, by index.
    pub fn get_mut_by_index(&mut self, index: usize) -> Option<&mut T> {
        self.inner.get_mut(index)
    }

    pub fn get_by_pos(&self, pos: Vec2, grid_pos: Vec2, grid_size: Vec2) -> Option<&T> {
        let (column, row) = self.get_coords(pos, grid_pos, grid_size);
        self.get(column, row)
    }

    /// position.
    pub fn get_mut_by_pos(&mut self, pos: Vec2, grid_pos: Vec2, grid_size: Vec2) -> Option<&mut T> {
        let (column, row) = self.get_coords(pos, grid_pos, grid_size);
        self.get_mut(column, row)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.inner.iter_mut()
    }

    /// Calculates the row and column a cell, given a set of absolute screen
    /// coordinates and the frame that this grid spans.
    pub fn get_coords(&self, pos: Vec2, grid_pos: Vec2, grid_size: Vec2) -> (isize, isize) {
        let chunk_width = grid_size.x / self.columns as f32;
        let chunk_height = grid_size.y / self.rows as f32;

        let chunk_column = ((pos.x - grid_pos.x) / chunk_width).floor() as isize;
        let chunk_row = ((pos.y - grid_pos.y) / chunk_height).floor() as isize;
        (chunk_column, chunk_row)
    }

    /// Calculates the index of a cell, given its column and row.
    pub fn index(&self, column: isize, row: isize) -> Option<usize> {
        let index = column + (row * self.columns as isize);

        if self.coords_are_in_bounds(column, row) && index < self.inner.len() as isize {
            Some(index as usize)
        } else {
            None
        }
    }

    fn coords_are_in_bounds(&self, column: isize, row: isize) -> bool {
        column >= 0 && row >= 0 && (column as usize) < self.columns && (row as usize) < self.rows
    }

    /// Returns the given cell, plus all of its neighbours within the given distance.
    pub fn get_neighbourhood(
        &self,
        column: isize,
        row: isize,
        distance: usize,
    ) -> impl Iterator<Item = &T> {
        self.get_neighbourhood_coords(column, row, distance)
            .filter_map(|(column, row)| self.get(column, row))
    }

    /// Returns all neighbours within a given distance from the given cell. Does
    /// not return the given cell itself.
    pub fn get_neighbours(
        &self,
        column: isize,
        row: isize,
        distance: usize,
    ) -> impl Iterator<Item = &T> {
        self.get_neighbourhood_coords(column, row, distance)
            .filter(move |(x, y)| !(x == &column && y == &row))
            .filter_map(|(column, row)| self.get(column, row))
    }

    /// Returns the cell at the given absolute position, plus all of its
    /// neighbours within the given distance.
    pub fn get_neighbourhood_at_pos(
        &self,
        pos: Vec2,
        distance: usize,
        grid_pos: Vec2,
        grid_size: Vec2,
    ) -> impl Iterator<Item = &T> {
        let (column, row) = self.get_coords(pos, grid_pos, grid_size);
        self.get_neighbourhood(column, row, distance)
    }

    /// Returns the coords for the given cell, plus all surrounding cells.
    pub fn get_neighbourhood_coords(
        &self,
        column: isize,
        row: isize,
        distance: usize,
    ) -> impl Iterator<Item = (isize, isize)> {
        let distance = distance as isize;

        (-distance..=distance).flat_map(move |dx| {
            (-distance..=distance).map(move |dy| {
                let column = column + dx;
                let row = row + dy;
                (column, row)
            })
        })
    }

    /// Returns the coords for the given cell, plus all surrounding cells.
    pub fn get_neighbourhood_coords_at_pos(
        &self,
        pos: Vec2,
        distance: usize,
        grid_pos: Vec2,
        grid_size: Vec2,
    ) -> impl Iterator<Item = (isize, isize)> {
        let (column, row) = self.get_coords(pos, grid_pos, grid_size);
        self.get_neighbourhood_coords(column, row, distance)
    }

    pub fn draw_gridlines(&self, grid_pos: Vec2, grid_size: Vec2) {
        let width = grid_size.x / self.columns() as f32;
        let height = grid_size.y / self.rows() as f32;

        for column in 0..self.columns() {
            for row in 0..self.rows() {
                let x = grid_pos.x + column as f32 * width;
                let y = grid_pos.y + row as f32 * height;

                draw_rectangle_lines(x, y, width, height, 4., crate::OUTLINE_COLOR);
            }
        }
    }

    pub fn highlight_cell(&self, pos: Vec2, color: Color, grid_pos: Vec2, grid_size: Vec2) {
        let width = grid_size.x / self.columns() as f32;
        let height = grid_size.y / self.rows() as f32;

        let target_cell = self.get_coords(pos, grid_pos, grid_size);

        for column in 0..self.columns() {
            for row in 0..self.rows() {
                if target_cell == (column as isize, row as isize) {
                    let x = grid_pos.x + column as f32 * width;
                    let y = grid_pos.y + row as f32 * height;
                    draw_rectangle(x, y, width, height, color);
                }
            }
        }
    }

    pub fn highlight_neighbours(&self, pos: Vec2, color: Color, grid_pos: Vec2, grid_size: Vec2) {
        let width = grid_size.x / self.columns() as f32;
        let height = grid_size.y / self.rows() as f32;

        let target_cell = self.get_coords(pos, grid_pos, grid_size);

        for column in 0..self.columns() as isize {
            for row in 0..self.rows() as isize {
                let mut neighbourhood =
                    self.get_neighbourhood_coords_at_pos(pos, 1, grid_pos, grid_size);

                if target_cell != (column, row) && neighbourhood.any(|coord| coord == (column, row))
                {
                    let x = grid_pos.x + column as f32 * width;
                    let y = grid_pos.y + row as f32 * height;
                    draw_rectangle(x, y, width, height, color);
                }
            }
        }
    }
}

impl<T: Send + Sync> Grid<T> {
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T> {
        self.inner.par_iter()
    }

    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, T> {
        self.inner.par_iter_mut()
    }
}

impl<T> IntoIterator for Grid<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> <Self as IntoIterator>::IntoIter {
        self.inner.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cells() -> Vec<i32> {
        Vec::from([1, 2, 3, 4, 5, 6])
    }

    fn test_grid() -> Grid<i32> {
        Grid::new(test_cells(), 2, 3)
    }

    fn test_grid_pos() -> Vec2 {
        vec2(100., 50.)
    }

    fn test_grid_size() -> Vec2 {
        vec2(200., 300.)
    }

    #[test]
    fn grid_length_matches_cell_dimensions() {
        let grid = test_grid();
        assert_eq!(6, grid.size());
        assert_eq!(6, grid.inner.len());
    }

    #[test]
    #[should_panic]
    fn bad_dimensions_should_panic() {
        let _invalid = Grid::new(test_cells(), 2, 4);
    }

    #[test]
    fn grid_indexes_correctly() {
        let grid = Grid::new(test_cells(), 2, 3);
        assert_eq!(Some(0), grid.index(0, 0));
        assert_eq!(Some(1), grid.index(1, 0));
        assert_eq!(Some(2), grid.index(0, 1));
        assert_eq!(Some(5), grid.index(1, 2));

        let grid = Grid::new(test_cells(), 3, 2);
        assert_eq!(Some(0), grid.index(0, 0));
        assert_eq!(Some(1), grid.index(1, 0));
        assert_eq!(Some(2), grid.index(2, 0));
        assert_eq!(Some(5), grid.index(2, 1));
    }

    #[test]
    fn grid_coords_outside_bounds_are_not_indexed() {
        let grid = Grid::new(test_cells(), 2, 3);
        assert_eq!(None, grid.index(-1, 0));
        assert_eq!(None, grid.index(-1, 1));
        assert_eq!(None, grid.index(-1, 2));
        assert_eq!(None, grid.index(3, 0));
        assert_eq!(None, grid.index(0, 4));
        assert_eq!(None, grid.index(1, 4));
    }

    #[test]
    fn absolute_coords_outside_bounds_are_not_indexed() {
        let grid = Grid::new(test_cells(), 2, 3);

        assert_eq!(
            None,
            grid.get_by_pos(Vec2::new(0., 0.), test_grid_pos(), test_grid_size())
        );
        assert_eq!(
            None,
            grid.get_by_pos(Vec2::new(99., 49.), test_grid_pos(), test_grid_size())
        );
        assert_eq!(
            None,
            grid.get_by_pos(Vec2::new(200., 49.), test_grid_pos(), test_grid_size())
        );
        assert_eq!(
            None,
            grid.get_by_pos(Vec2::new(99., 200.), test_grid_pos(), test_grid_size())
        );

        assert_eq!(
            None,
            grid.get_by_pos(Vec2::new(200., 350.), test_grid_pos(), test_grid_size())
        );
        assert_eq!(
            None,
            grid.get_by_pos(Vec2::new(300., 200.), test_grid_pos(), test_grid_size())
        );
        assert_eq!(
            None,
            grid.get_by_pos(Vec2::new(300., 350.), test_grid_pos(), test_grid_size())
        );
    }

    #[test]
    fn absolute_coords_inside_bounds_are_indexed() {
        let grid = Grid::new(test_cells(), 2, 3);

        assert_eq!(
            Some(&1),
            grid.get_by_pos(Vec2::new(100., 50.), test_grid_pos(), test_grid_size())
        );
        assert_eq!(
            Some(&2),
            grid.get_by_pos(Vec2::new(200., 50.), test_grid_pos(), test_grid_size())
        );
        assert_eq!(
            Some(&3),
            grid.get_by_pos(Vec2::new(100., 150.), test_grid_pos(), test_grid_size())
        );
        assert_eq!(
            Some(&6),
            grid.get_by_pos(Vec2::new(299., 349.), test_grid_pos(), test_grid_size())
        );
    }

    #[test]
    fn into_iter_matches_values() {
        let grid = Grid::new(test_cells(), 2, 3);
        assert!(test_cells().iter().copied().eq(grid.into_iter()))
    }

    #[test]
    fn iter_matches_values() {
        let grid = Grid::new(test_cells(), 2, 3);
        assert!(test_cells().iter().eq(grid.iter()))
    }

    #[test]
    fn neighbourhood_coords() {
        let grid = Grid::new(test_cells(), 2, 3);
        let mut neighbours = grid.get_neighbourhood_coords(1, 0, 1);
        assert_eq!(Some((0, -1)), neighbours.next());
        assert_eq!(Some((0, 0)), neighbours.next());
        assert_eq!(Some((0, 1)), neighbours.next());
        assert_eq!(Some((1, -1)), neighbours.next());
        assert_eq!(Some((1, 0)), neighbours.next()); // self is included
        assert_eq!(Some((1, 1)), neighbours.next());
        assert_eq!(Some((2, -1)), neighbours.next());
        assert_eq!(Some((2, 0)), neighbours.next());
        assert_eq!(Some((2, 1)), neighbours.next());
        assert_eq!(None, neighbours.next());
    }
}
