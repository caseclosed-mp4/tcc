use tcc_types::rng::Rng;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        let cols = rows.first().map(|r| r.len()).unwrap_or(0);
        let mut m = Self::zeros(rows.len(), cols);
        for (i, row) in rows.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                m.set(i, j, *v);
            }
        }
        m
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }

    pub fn transpose(&self) -> Self {
        let mut m = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                m.set(j, i, self.get(i, j));
            }
        }
        m
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    out.data[i * other.cols + j] += a * other.get(k, j);
                }
            }
        }
        out
    }

    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len());
        (0..self.rows)
            .map(|i| {
                let mut s = 0.0;
                for j in 0..self.cols {
                    s += self.get(i, j) * v[j];
                }
                s
            })
            .collect()
    }

    pub fn inverse(&self) -> Option<Matrix> {
        if self.rows != self.cols {
            return None;
        }
        let n = self.rows;
        let mut aug = Matrix::zeros(n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, n + i, 1.0);
        }
        for col in 0..n {
            let mut pivot = col;
            let mut best = aug.get(col, col).abs();
            for r in (col + 1)..n {
                let v = aug.get(r, col).abs();
                if v > best {
                    best = v;
                    pivot = r;
                }
            }
            if best < 1e-12 {
                return None;
            }
            if pivot != col {
                for j in 0..2 * n {
                    aug.data.swap(col * 2 * n + j, pivot * 2 * n + j);
                }
            }
            let div = aug.get(col, col);
            for j in 0..2 * n {
                aug.set(col, j, aug.get(col, j) / div);
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let factor = aug.get(r, col);
                for j in 0..2 * n {
                    let v = aug.get(col, j);
                    aug.set(r, j, aug.get(r, j) - factor * v);
                }
            }
        }
        let mut inv = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                inv.set(i, j, aug.get(i, n + j));
            }
        }
        Some(inv)
    }
}

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

pub fn variance(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64
}

pub fn standard_deviation(xs: &[f64]) -> f64 {
    variance(xs).sqrt()
}

pub fn covariance(xs: &[f64], ys: &[f64]) -> f64 {
    assert_eq!(xs.len(), ys.len());
    if xs.len() < 2 {
        return 0.0;
    }
    let mx = mean(xs);
    let my = mean(ys);
    xs.iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum::<f64>()
        / (xs.len() - 1) as f64
}

pub fn correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let cov = covariance(xs, ys);
    let sx = standard_deviation(xs);
    let sy = standard_deviation(ys);
    if sx == 0.0 || sy == 0.0 {
        0.0
    } else {
        cov / (sx * sy)
    }
}

const T_TABLE: &[(f64, f64)] = &[
    (1.0, 12.706),
    (2.0, 4.303),
    (3.0, 3.182),
    (4.0, 2.776),
    (5.0, 2.571),
    (6.0, 2.447),
    (7.0, 2.365),
    (8.0, 2.306),
    (9.0, 2.262),
    (10.0, 2.228),
    (11.0, 2.201),
    (12.0, 2.179),
    (13.0, 2.160),
    (14.0, 2.145),
    (15.0, 2.131),
    (16.0, 2.120),
    (17.0, 2.110),
    (18.0, 2.101),
    (19.0, 2.093),
    (20.0, 2.086),
    (21.0, 2.080),
    (22.0, 2.074),
    (23.0, 2.069),
    (24.0, 2.064),
    (25.0, 2.060),
    (26.0, 2.056),
    (27.0, 2.052),
    (28.0, 2.048),
    (29.0, 2.045),
    (30.0, 2.042),
    (40.0, 2.021),
    (50.0, 2.009),
    (60.0, 2.000),
    (80.0, 1.990),
    (100.0, 1.984),
    (120.0, 1.980),
    (200.0, 1.972),
    (500.0, 1.965),
    (1000.0, 1.962),
    (f64::INFINITY, 1.960),
];

pub fn t_critical(df: f64, alpha: f64) -> f64 {
    if df <= 0.0 {
        return 1.96;
    }
    if alpha != 0.05 {
        return normal_critical(alpha);
    }
    let mut prev = T_TABLE[0];
    for &(d, v) in T_TABLE {
        if df <= d {
            if (df - prev.0).abs() < 1e-9 {
                return prev.1;
            }
            let frac = (df - prev.0) / (d - prev.0);
            return prev.1 + frac * (v - prev.1);
        }
        prev = (d, v);
    }
    1.96
}

pub fn normal_critical(alpha: f64) -> f64 {
    let target = 1.0 - alpha / 2.0;
    let mut lo = -10.0;
    let mut hi = 10.0;
    for _ in 0..100 {
        let mid = (lo + hi) / 2.0;
        if normal_cdf(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

pub fn t_cdf(t: f64, df: f64) -> f64 {
    if df >= 100.0 {
        return normal_cdf(t);
    }
    if t == 0.0 {
        return 0.5;
    }
    let x = df / (df + t * t);
    let ib = regularized_incomplete_beta(df / 2.0, 0.5, x);
    if t >= 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if x < (a + 1.0) / (a + b + 2.0) {
        beta_cf_series(a, b, x)
    } else {
        1.0 - beta_cf_series(b, a, 1.0 - x)
    }
}

fn beta_cf_series(a: f64, b: f64, x: f64) -> f64 {
    let lbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let prefix = (a * x.ln() + b * (1.0 - x).ln() - lbeta).exp() / a;
    prefix * betacf(a, b, x)
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-30;
    let max_iter = 400;
    let eps = 3e-14;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=max_iter {
        let m2 = 2 * m;
        let mf = m as f64;
        let numerator = mf * (b - mf) * x / ((qam + m2 as f64) * (a + m2 as f64));
        d = 1.0 + numerator * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + numerator / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let delta = (a + mf) * (qab + mf) * x / ((a + m2 as f64) * (qap + m2 as f64));
        d = 1.0 + delta * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + delta / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h
}


fn ln_gamma(x: f64) -> f64 {
    let coefs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let mut y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for c in &coefs {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub x: Matrix,
    pub y: Vec<f64>,
    pub feature_names: Vec<String>,
}

impl Dataset {
    pub fn new(x: Matrix, y: Vec<f64>, feature_names: Vec<String>) -> Self {
        assert_eq!(x.rows, y.len());
        Self {
            x,
            y,
            feature_names,
        }
    }

    pub fn n(&self) -> usize {
        self.y.len()
    }

    pub fn p(&self) -> usize {
        self.x.cols
    }

    pub fn fold_indices(&self, k: usize, rng: &mut Rng) -> Vec<Vec<usize>> {
        let mut idx: Vec<usize> = (0..self.n()).collect();
        rng.shuffle(&mut idx);
        let mut folds = vec![Vec::new(); k];
        for (i, v) in idx.into_iter().enumerate() {
            folds[i % k].push(v);
        }
        folds
    }

    pub fn select(&self, indices: &[usize]) -> Dataset {
        let rows: Vec<Vec<f64>> = indices
            .iter()
            .map(|&i| self.x.row(i).to_vec())
            .collect();
        let y = indices.iter().map(|&i| self.y[i]).collect();
        Dataset::new(Matrix::from_rows(&rows), y, self.feature_names.clone())
    }

    pub fn select_features(&self, cols: &[usize]) -> Dataset {
        let rows: Vec<Vec<f64>> = (0..self.n())
            .map(|i| cols.iter().map(|&j| self.x.get(i, j)).collect())
            .collect();
        let names = cols
            .iter()
            .map(|&j| self.feature_names[j].clone())
            .collect();
        Dataset::new(Matrix::from_rows(&rows), self.y.clone(), names)
    }
}

#[derive(Debug, Clone)]
pub struct LinearModel {
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub residuals: Vec<f64>,
    pub standard_errors: Vec<f64>,
    pub r_squared: f64,
}

impl LinearModel {
    pub fn fit(data: &Dataset) -> Option<Self> {
        let n = data.n();
        let p = data.p();
        if n <= p + 1 {
            return None;
        }
        let mut design = Matrix::zeros(n, p + 1);
        for i in 0..n {
            design.set(i, 0, 1.0);
            for j in 0..p {
                design.set(i, j + 1, data.x.get(i, j));
            }
        }
        let xt = design.transpose();
        let xtx = xt.multiply(&design);
        let xty = xt.matvec(&data.y);
        let xtx_inv = xtx.inverse()?;
        let beta = xtx_inv.matvec(&xty);
        let predictions = design.matvec(&beta);
        let residuals: Vec<f64> = data
            .y
            .iter()
            .zip(predictions.iter())
            .map(|(y, hat)| y - hat)
            .collect();
        let rss: f64 = residuals.iter().map(|r| r * r).sum();
        let y_mean = mean(&data.y);
        let tss: f64 = data.y.iter().map(|y| (y - y_mean).powi(2)).sum();
        let sigma2 = rss / (n - p - 1) as f64;
        let standard_errors = (0..=p)
            .map(|j| (sigma2 * xtx_inv.get(j, j)).sqrt())
            .collect();
        Some(Self {
            coefficients: beta[1..].to_vec(),
            intercept: beta[0],
            residuals,
            standard_errors,
            r_squared: if tss == 0.0 { 0.0 } else { 1.0 - rss / tss },
        })
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        self.intercept
            + self
                .coefficients
                .iter()
                .zip(x.iter())
                .map(|(b, v)| b * v)
                .sum::<f64>()
    }

    pub fn treatment_coefficient(&self) -> f64 {
        self.coefficients.first().copied().unwrap_or(0.0)
    }

    pub fn treatment_standard_error(&self) -> f64 {
        self.standard_errors.get(1).copied().unwrap_or(0.0)
    }
}

pub fn normal_pdf(x: f64) -> f64 {
    const SQRT_2PI: f64 = 2.5066282746310002;
    (-0.5 * x * x).exp() / SQRT_2PI
}

pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / core::f64::consts::SQRT_2))
}

pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

pub fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-9, 1.0 - 1e-9);
    (p / (1.0 - p)).ln()
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_inverse_identity() {
        let m = Matrix::identity(4);
        let inv = m.inverse().unwrap();
        for i in 0..4 {
            assert!((inv.get(i, i) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn linear_regression_recovers_slope() {
        let mut rng = Rng::from_seed(3);
        let rows: Vec<Vec<f64>> = (0..500)
            .map(|_| {
                let x = rng.range_f64(-2.0, 2.0);
                let z = rng.gaussian();
                let y = 2.0 * x + 0.5 * z + rng.gaussian() * 0.1;
                vec![x, z, y]
            })
            .collect();
        let x = Matrix::from_rows(&rows.iter().map(|r| vec![r[0], r[1]]).collect::<Vec<_>>());
        let y: Vec<f64> = rows.iter().map(|r| r[2]).collect();
        let data = Dataset::new(x, y, vec!["x".into(), "z".into()]);
        let model = LinearModel::fit(&data).unwrap();
        assert!((model.coefficients[0] - 2.0).abs() < 0.05);
        assert!((model.coefficients[1] - 0.5).abs() < 0.05);
        assert!(model.r_squared > 0.95);
    }

    #[test]
    fn t_critical_known_values() {
        assert!((t_critical(1000.0, 0.05) - 1.96).abs() < 0.03);
        assert!((t_critical(10.0, 0.05) - 2.228).abs() < 0.03);
        assert!((t_cdf(0.0, 1000.0) - 0.5).abs() < 1e-9);
        assert!((t_cdf(1.96, 1000.0) - 0.975).abs() < 0.005);
    }
}