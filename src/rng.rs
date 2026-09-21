//! Random number generators for the simulations.

pub use rand::RngExt;
use rand::SeedableRng;

/// `SmallRng` rather than `StdRng`, as "mostly random" is more than enough.
pub type Rng = rand::rngs::SmallRng;

/// Generates a new random number generator using system entropy.
pub fn from_entropy() -> Rng {
    Rng::from_rng(&mut rand::rng())
}

/// Generates a new random number generator from a seed. Used by the tests,
/// which need a simulation to replay identically rather than to look random.
#[cfg_attr(not(test), allow(dead_code))]
pub fn seeded(seed: u64) -> Rng {
    Rng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(rng: &mut Rng) -> Vec<u32> {
        (0..16).map(|_| rng.random_range(0..1_000_000u32)).collect()
    }

    #[test]
    fn the_same_seed_replays_the_same_sequence() {
        assert_eq!(sample(&mut seeded(42)), sample(&mut seeded(42)));
    }

    #[test]
    fn different_seeds_diverge() {
        assert_ne!(sample(&mut seeded(1)), sample(&mut seeded(2)));
    }

    #[test]
    fn entropy_seeding_is_not_fixed() {
        assert_ne!(sample(&mut from_entropy()), sample(&mut from_entropy()));
    }
}
