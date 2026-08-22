use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    pub fn from_entropy() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let nanos = (now as u128) as u64;
        let mix = nanos
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0xBF58476D1CE4E5B9);
        Self::from_seed(mix)
    }

    pub fn seed(&self) -> u64 {
        self.state
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn range_f64(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    pub fn bool(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }

    pub fn gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * core::f64::consts::PI * u2).cos()
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            items.swap(i, j);
        }
    }

    pub fn sample_index(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let target = self.next_f64() * total;
        let mut acc = 0.0;
        for (i, w) in weights.iter().enumerate() {
            acc += *w;
            if target <= acc {
                return i;
            }
        }
        weights.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_seed() {
        let mut a = Rng::from_seed(42);
        let mut b = Rng::from_seed(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn f64_in_unit_interval() {
        let mut r = Rng::from_seed(1);
        for _ in 0..10_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn gaussian_distribution_moments() {
        let mut r = Rng::from_seed(7);
        let n = 20_000u32;
        let (sum, sumsq) = (0..n).fold((0.0, 0.0), |(s, q), _| {
            let x = r.gaussian();
            (s + x, q + x * x)
        });
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean {}", mean);
        assert!((var - 1.0).abs() < 0.1, "var {}", var);
    }
}
