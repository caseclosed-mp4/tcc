use std::fmt;

use crate::rng::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Uuid([u8; 16]);

impl Uuid {
    pub fn random() -> Self {
        let mut rng = Rng::from_entropy();
        Self::from_rng(&mut rng)
    }

    pub fn from_rng(rng: &mut Rng) -> Self {
        let mut bytes = [0u8; 16];
        for chunk in bytes.chunks_mut(8) {
            let v = rng.next_u64().to_be_bytes();
            chunk.copy_from_slice(&v[..chunk.len()]);
        }
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        Uuid(bytes)
    }

    pub fn from_bytes(bytes: [u8; 16]) -> Self {
        Uuid(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    pub fn nil() -> Self {
        Uuid([0u8; 16])
    }

    pub fn simple(&self) -> String {
        self.to_string().replace('-', "")
    }

    pub fn parse(s: &str) -> Option<Uuid> {
        let cleaned: String = s.chars().filter(|c| *c != '-').collect();
        if cleaned.len() != 32 {
            return None;
        }
        let mut bytes = [0u8; 16];
        for (i, pair) in cleaned.as_bytes().chunks(2).enumerate() {
            let hi = hex_val(pair[0])?;
            let lo = hex_val(pair[1])?;
            bytes[i] = (hi << 4) | lo;
        }
        Some(Uuid(bytes))
    }
}

impl fmt::Display for Uuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let b = &self.0;
        write!(
            f,
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]
        )
    }
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_string() {
        let u = Uuid::random();
        let s = u.to_string();
        assert_eq!(Uuid::parse(&s), Some(u));
        assert_eq!(s.len(), 36);
    }

    #[test]
    fn version_and_variant_bits() {
        let u = Uuid::random();
        assert_eq!(u.as_bytes()[6] >> 4, 4);
        assert!(u.as_bytes()[8] >> 6 == 2);
    }
}
