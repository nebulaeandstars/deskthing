#![allow(unused)]

use crate::engine::{Canvas, Color, Vec2, vec2};
use crate::traits::HasSize;

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

    /// Resizes the grid, filling any new cells with the default value.
    ///
    /// Cell *contents* are not carried across a change in `columns`: the
    /// layout is row-major, so a different column count reinterprets every
    /// index. Callers that care must repopulate afterwards.
    pub fn resize_with_defaults(&mut self, columns: usize, rows: usize) {
        let len = columns * rows;

        self.columns = columns;
        self.rows = rows;

        // Both directions, or the length stops matching the dimensions and
        // every subsequent index is wrong.
        self.inner.truncate(len);
        self.inner.reserve(len);
        for _ in self.inner.len()..len {
            self.inner.push(T::default());
        }

        debug_assert_eq!(len, self.inner.len());
    }
}

#[allow(unused)]
impl<T> Grid<T> {
    /// Constructs a new grid with the given values.
    ///
    /// # Panics
    /// If `cells` is not exactly `columns * rows` long. This is a hard assert
    /// rather than a debug one: a grid whose length disagrees with its
    /// dimensions indexes wrongly for the rest of its life, and the check is a
    /// single multiply in a constructor.
    pub fn new(cells: Vec<T>, columns: usize, rows: usize) -> Self {
        assert_eq!(
            columns * rows,
            cells.len(),
            "grid of {columns}x{rows} needs {} cells, got {}",
            columns * rows,
            cells.len()
        );

        Self {
            inner: cells,
            columns,
            rows,
        }
    }

    /// Like [`Self::from_generator`], but for a generator that carries state —
    /// a random number generator, say.
    pub fn from_generator_mut<F: FnMut(usize, usize) -> T>(
        columns: usize,
        rows: usize,
        mut f: F,
    ) -> Self {
        let mut cells = Vec::with_capacity(columns * rows);
        for row in 0..rows {
            for col in 0..columns {
                cells.push(f(col, row));
            }
        }

        Self::new(cells, columns, rows)
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

    pub fn highlight_cell(
        &self,
        canvas: &mut dyn Canvas,
        pos: Vec2,
        color: Color,
        grid_pos: Vec2,
        grid_size: Vec2,
    ) {
        let width = grid_size.x / self.columns() as f32;
        let height = grid_size.y / self.rows() as f32;

        let target_cell = self.get_coords(pos, grid_pos, grid_size);

        for column in 0..self.columns() {
            for row in 0..self.rows() {
                if target_cell == (column as isize, row as isize) {
                    let x = grid_pos.x + column as f32 * width;
                    let y = grid_pos.y + row as f32 * height;
                    canvas.rect(vec2(x, y), vec2(width, height), color);
                }
            }
        }
    }

    pub fn highlight_neighbours(
        &self,
        canvas: &mut dyn Canvas,
        pos: Vec2,
        color: Color,
        grid_pos: Vec2,
        grid_size: Vec2,
    ) {
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
                    canvas.rect(vec2(x, y), vec2(width, height), color);
                }
            }
        }
    }
}

impl<T> HasSize for Grid<T> {
    fn size(&self) -> Vec2 {
        vec2(self.columns as f32, self.rows as f32)
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

    #[test]
    fn resizing_smaller_keeps_length_and_dimensions_in_step() {
        let mut grid: Grid<i32> = Grid::with_defaults(4, 4);
        grid.resize_with_defaults(2, 2);

        assert_eq!(2, grid.columns());
        assert_eq!(2, grid.rows());
        assert_eq!(4, grid.size());
        // Previously left at 16, silently breaking Grid's core invariant.
        assert_eq!(4, grid.inner.len());
    }

    #[test]
    fn resizing_larger_keeps_length_and_dimensions_in_step() {
        let mut grid: Grid<i32> = Grid::with_defaults(2, 2);
        grid.resize_with_defaults(4, 5);

        assert_eq!(20, grid.size());
        assert_eq!(20, grid.inner.len());
    }

    #[test]
    fn resizing_to_the_same_dimensions_is_a_no_op() {
        let mut grid = Grid::new(test_cells(), 2, 3);
        grid.resize_with_defaults(2, 3);

        assert!(test_cells().iter().eq(grid.iter()));
    }

    #[test]
    fn a_resized_grid_can_still_be_indexed_everywhere() {
        let mut grid: Grid<i32> = Grid::with_defaults(4, 4);
        grid.resize_with_defaults(3, 2);

        for row in 0..2 {
            for column in 0..3 {
                assert!(
                    grid.get(column, row).is_some(),
                    "({column}, {row}) unreachable after resize"
                );
            }
        }
        assert_eq!(None, grid.get(3, 0));
        assert_eq!(None, grid.get(0, 2));
    }
}
