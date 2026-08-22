use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timestamp {
    millis: i64,
}

impl Timestamp {
    pub fn now() -> Self {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Timestamp { millis }
    }

    pub fn from_millis(millis: i64) -> Self {
        Timestamp { millis }
    }

    pub fn millis(&self) -> i64 {
        self.millis
    }

    pub fn seconds(&self) -> i64 {
        self.millis / 1000
    }

    pub fn add_seconds(&self, secs: i64) -> Self {
        Timestamp {
            millis: self.millis + secs * 1000,
        }
    }

    pub fn elapsed(&self) -> i64 {
        Self::now().millis - self.millis
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.millis.div_euclid(1000);
        let ms = self.millis.rem_euclid(1000);
        let days = secs.div_euclid(86_400) as i64;
        let rem = secs.rem_euclid(86_400);
        let hour = rem / 3600;
        let minute = (rem % 3600) / 60;
        let second = rem % 60;
        let z = days_to_ymd(days);
        write!(
            f,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
            z.0, z.1, z.2, hour, minute, second, ms
        )
    }
}

fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    let shifted = days + 719_468;
    let era = if shifted >= 0 { shifted } else { shifted - 146_096 } / 146_097;
    let era_of_year = (shifted - era * 146_097) as u64;
    let year = (era_of_year - era_of_year / 1460 + era_of_year / 36524 - era_of_year / 146_096)
        / 365;
    let year_day =
        era_of_year - (365 * year + year / 4 - year / 100 + year / 400);
    let month = (5 * year_day + 2) / 153;
    let day = year_day - (153 * month + 2) / 5 + 1;
    let month = if month < 10 { month + 3 } else { month - 9 };
    let year = year as i64 + era * 400 + if month <= 2 { 1 } else { 0 };
    (year as i32, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_format() {
        let t = Timestamp::from_millis(0);
        assert_eq!(t.to_string(), "1970-01-01T00:00:00.000Z");
    }

    #[test]
    fn known_date() {
        let t = Timestamp::from_millis(1_700_000_000_000);
        let s = t.to_string();
        assert!(s.starts_with("2023-11-14T"));
    }

    #[test]
    fn roundtrip_components() {
        let t = Timestamp::now();
        let t2 = t.add_seconds(60);
        assert_eq!(t2.millis() - t.millis(), 60_000);
    }
}
