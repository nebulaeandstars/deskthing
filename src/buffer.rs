#![allow(unused)]

use std::borrow::Borrow;
use std::borrow::BorrowMut;

#[derive(Clone, Debug)]
pub struct DoubleBuffer<T> {
    current: T,
    next: T,
}

impl<T: Clone> DoubleBuffer<T> {
    pub fn new(current: T) -> Self {
        let next = current.clone();
        Self { current, next }
    }

    pub fn apply<F: Fn(&T, &mut T)>(&mut self, f: F) {
        f(&self.current, &mut self.next);
        self.swap();
    }
}

impl<T> DoubleBuffer<T> {
    pub fn state(&self) -> &T {
        &self.current
    }

    pub fn next(&mut self) -> &mut T {
        &mut self.next
    }

    pub fn states(&mut self) -> (&T, &mut T) {
        (&self.current, &mut self.next)
    }

    pub fn set(&mut self, next: T) {
        self.current = next;
    }

    pub fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }

    pub fn update(&mut self) {}
}

impl<T> From<T> for DoubleBuffer<T>
where
    T: Clone,
{
    fn from(current: T) -> Self {
        Self::new(current)
    }
}

impl<T> Borrow<T> for DoubleBuffer<T> {
    fn borrow(&self) -> &T {
        &self.current
    }
}

impl<T> BorrowMut<T> for DoubleBuffer<T> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut self.next
    }
}
