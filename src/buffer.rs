//! A two-state buffer for simulations that build each generation by reading
//! the previous one.

/// Holds a committed state and a scratch state.
///
/// The only way to advance is [`Self::generate`]: read the committed state,
/// fill the scratch, commit. There is deliberately no way to reach the scratch
/// without committing it, and no way to mutate the committed state in place —
/// a generation that could observe its own partial output would not be a
/// generation.
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
}

impl<T> DoubleBuffer<T> {
    /// The committed state.
    pub fn state(&self) -> &T {
        &self.current
    }

    /// Reads the committed state to build the next one, then commits it.
    ///
    /// `generate` is handed the committed state to read and the scratch state
    /// to fill. Reads never observe the writes, so the whole generation sees a
    /// consistent snapshot.
    ///
    /// `generate` must write *every* element of the scratch. Committing swaps
    /// the two buffers rather than copying, so the scratch arrives holding the
    /// state from two generations ago — an element left untouched is not
    /// "unchanged", it is resurrected.
    pub fn generate<F: FnOnce(&T, &mut T)>(&mut self, generate: F) {
        generate(&self.current, &mut self.next);
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

impl<T: Clone> From<T> for DoubleBuffer<T> {
    fn from(current: T) -> Self {
        Self::new(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_new_buffer_starts_committed() {
        let buffer = DoubleBuffer::new(vec![1, 2, 3]);

        assert_eq!(&vec![1, 2, 3], buffer.state());
    }

    #[test]
    fn generate_reads_the_committed_state_and_commits_the_result() {
        let mut buffer = DoubleBuffer::new(vec![1, 2, 3]);

        buffer.generate(|current, next| {
            for (slot, value) in next.iter_mut().zip(current) {
                *slot = value * 10;
            }
        });

        assert_eq!(&vec![10, 20, 30], buffer.state());
    }

    #[test]
    fn generate_sees_the_previous_generation_not_its_own_output() {
        let mut buffer = DoubleBuffer::new(vec![1, 1, 1]);

        // Each slot reads its left neighbour. If writes were visible to later
        // reads, the 9 would cascade along the vector instead of shifting once.
        buffer.generate(|current, next| {
            for i in 0..next.len() {
                next[i] = if i == 0 { 9 } else { current[i - 1] };
            }
        });

        assert_eq!(&vec![9, 1, 1], buffer.state());
    }

    #[test]
    fn generations_compose() {
        let mut buffer = DoubleBuffer::new(0);

        for _ in 0..5 {
            buffer.generate(|current, next| *next = current + 1);
        }

        assert_eq!(&5, buffer.state());
    }

    /// Committing swaps rather than copies, so the scratch is two generations
    /// old. This pins what goes wrong when `generate` skips an element — the
    /// reason its contract demands all of them.
    #[test]
    fn a_partially_written_scratch_resurrects_an_older_generation() {
        let mut buffer = DoubleBuffer::new(vec![0, 0]);

        buffer.generate(|_, next| next[0] = 1);
        assert_eq!(&vec![1, 0], buffer.state());

        // Only slot 0 is written again. Slot 1 comes back holding generation
        // zero rather than the value committed a moment ago.
        buffer.generate(|_, next| next[0] = 2);
        assert_eq!(
            &vec![2, 0],
            buffer.state(),
            "slot 1 should have been resurrected from two generations ago"
        );
    }
}
