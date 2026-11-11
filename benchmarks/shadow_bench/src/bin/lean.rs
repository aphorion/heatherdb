//! Shadow in memory, truth on disk.
//!
//! Only the int8 shadow of every vector is held in RAM: its codes, one scale,
//! its quantisation error and its norm. Exact vectors stay in the file on
//! disk. A query scores every shadow (certified bounds), and only the rows
//! whose bound can still reach the 10th-best guaranteed score are read from
//! disk and scored exactly. The full vectors are never loaded into memory.
//!
//! usage: lean <dir>     (dir holds base.f32, query.f32, truth.u32, shape.txt)
//! Reports peak RSS, warm latency (rows may be in the OS page cache) and cold
//! latency (F_NOCACHE: every exact row read goes to the disk).

use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::time::Instant;

use rayon::prelude::*;

const K: usize = 10;
const CHUNK: usize = 16384;

fn peak_rss_mb() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    ru.ru_maxrss as f64 / 1_048_576.0 // bytes on macOS
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn read_u32(path: &str) -> Vec<u32> {
    let b = std::fs::read(path).unwrap();
    b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(x, y)| *x as i32 * *y as i32).sum()
}

fn quantise(v: &[f32]) -> (Vec<i8>, f32, f32, f32) {
    let m = v.iter().fold(0f32, |a, x| a.max(x.abs()));
    let s = if m > 0.0 { m / 127.0 } else { 1.0 };
    let c: Vec<i8> = v.iter().map(|x| (x / s).round().clamp(-127.0, 127.0) as i8).collect();
    let (mut e, mut nq) = (0f64, 0f64);
    for (x, ci) in v.iter().zip(&c) {
        let r = *ci as f64 * s as f64;
        e += (*x as f64 - r).powi(2);
        nq += r * r;
    }
    (c, s, e.sqrt() as f32, nq.sqrt() as f32)
}

#[derive(Clone)]
struct TopK(Vec<(f32, u32)>);
impl TopK {
    fn new() -> Self {
        TopK(Vec::with_capacity(K + 1))
    }
    fn kth(&self) -> f32 {
        if self.0.len() < K { f32::NEG_INFINITY } else { self.0[K - 1].0 }
    }
    fn push(&mut self, s: f32, id: u32) {
        if self.0.len() == K && s <= self.0[K - 1].0 {
            return;
        }
        let pos = self.0.iter().position(|e| s > e.0).unwrap_or(self.0.len());
        self.0.insert(pos, (s, id));
        self.0.truncate(K);
    }
    fn merge(mut self, o: TopK) -> TopK {
        for (s, i) in o.0 {
            self.push(s, i);
        }
        self
    }
}

/// The int8 shadow: everything that lives in RAM.
struct Shadow {
    n: usize,
    d: usize,
    codes: Vec<i8>,
    scale: Vec<f32>,
    err: Vec<f32>,
    qnorm: Vec<f32>,
}

impl Shadow {
    /// Streams the vector file in chunks; never holds the full vectors.
    fn build(path: &str, n: usize, d: usize) -> Self {
        let mut f = File::open(path).unwrap();
        let mut sh = Shadow { n, d, codes: Vec::with_capacity(n * d), scale: Vec::with_capacity(n), err: Vec::with_capacity(n), qnorm: Vec::with_capacity(n) };
        let rows_per = ((16 << 20) / (d * 4)).max(1); // stream ~16 MB at a time
        let mut buf = vec![0u8; rows_per * d * 4];
        let mut done = 0;
        while done < n {
            let rows = rows_per.min(n - done);
            let bytes = &mut buf[..rows * d * 4];
            f.read_exact(bytes).unwrap();
            let floats: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let q: Vec<_> = floats.par_chunks(d).map(quantise).collect();
            for (c, s, e, nq) in q {
                sh.codes.extend_from_slice(&c);
                sh.scale.push(s);
                sh.err.push(e);
                sh.qnorm.push(nq);
            }
            done += rows;
        }
        sh
    }

    fn bytes(&self) -> usize {
        self.codes.len() + 12 * self.n
    }

    /// Rows that might be in the top K, by certified int8 bounds.
    fn survivors(&self, q: &[f32]) -> Vec<u32> {
        let d = self.d;
        let (cq, ts, eq, _) = quantise(q);
        let qn = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        let chunks = self.n.div_ceil(CHUNK);
        let parts: Vec<(usize, Vec<f32>, TopK)> = (0..chunks)
            .into_par_iter()
            .map(|c| {
                let (lo, hi) = (c * CHUNK, ((c + 1) * CHUNK).min(self.n));
                let mut ub = Vec::with_capacity(hi - lo);
                let mut lows = TopK::new();
                for i in lo..hi {
                    let approx = ts * self.scale[i] * dot_i8(&cq, &self.codes[i * d..(i + 1) * d]) as f32;
                    let rad = qn * self.err[i] + eq * self.qnorm[i] + 1e-5 * qn * (self.qnorm[i] + self.err[i]);
                    ub.push(approx + rad);
                    lows.push(approx - rad, i as u32);
                }
                (lo, ub, lows)
            })
            .collect();
        let floor = parts.iter().fold(TopK::new(), |a, p| a.merge(p.2.clone())).kth();
        let mut out = Vec::new();
        for (lo, ub, _) in &parts {
            for (j, u) in ub.iter().enumerate() {
                if *u >= floor {
                    out.push((lo + j) as u32);
                }
            }
        }
        out
    }
}

/// Exact scores for the survivors, each row read from the file.
fn exact(file: &File, d: usize, q: &[f32], rows: &[u32]) -> TopK {
    let mut t = TopK::new();
    let mut buf = vec![0u8; d * 4];
    let mut v = vec![0f32; d];
    for &i in rows {
        file.read_exact_at(&mut buf, i as u64 * d as u64 * 4).unwrap();
        for (x, c) in v.iter_mut().zip(buf.chunks_exact(4)) {
            *x = f32::from_le_bytes(c.try_into().unwrap());
        }
        t.push(dot(&v, q), i);
    }
    t
}

fn correct(t: &TopK, truth: &[u32]) -> bool {
    let mut a: Vec<u32> = t.0.iter().map(|e| e.1).collect();
    let mut b = truth.to_vec();
    a.sort();
    b.sort();
    a == b
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "data".into());
    let shape = std::fs::read_to_string(format!("{dir}/shape.txt")).unwrap();
    let s: Vec<usize> = shape.split_whitespace().map(|v| v.parse().unwrap()).collect();
    let (n, d, nq) = (s[0], s[1], s[2]);
    let qs = read_f32(&format!("{dir}/query.f32"));
    let truth = read_u32(&format!("{dir}/truth.u32"));
    let base = format!("{dir}/base.f32");
    let q = |i: usize| &qs[i * d..(i + 1) * d];
    let tr = |i: usize| &truth[i * K..(i + 1) * K];

    let t0 = Instant::now();
    let sh = Shadow::build(&base, n, d);
    println!(
        "n = {n}, d = {d}: int8 shadow built in {:.2} s; shadow {:.0} MB in RAM vs full f32 vectors {:.0} MB (on disk only)",
        t0.elapsed().as_secs_f64(),
        sh.bytes() as f64 / 1e6,
        (n * d * 4) as f64 / 1e6
    );

    let warm = File::open(&base).unwrap();
    let cold = File::open(&base).unwrap();
    unsafe { libc::fcntl(cold.as_raw_fd(), libc::F_NOCACHE, 1) };

    for (name, file) in [("warm (OS page cache allowed)", &warm), ("cold (F_NOCACHE: every exact row from disk)", &cold)] {
        let mut lat = Vec::with_capacity(nq);
        let (mut wrong, mut surv) = (0, 0usize);
        for i in 0..nq {
            let t = Instant::now();
            let rows = sh.survivors(q(i));
            let top = exact(file, d, q(i), &rows);
            lat.push(t.elapsed().as_secs_f64());
            surv += rows.len();
            wrong += !correct(&top, tr(i)) as usize;
        }
        lat.sort_by(|a, b| a.total_cmp(b));
        let p = |x: f64| lat[((lat.len() - 1) as f64 * x) as usize] * 1e3;
        println!(
            "{name:46} p50 {:6.3} ms  p95 {:6.3} ms  {:6.0} q/s  rows read from disk {:.1}/query  wrong {wrong}/{nq}",
            p(0.5),
            p(0.95),
            nq as f64 / lat.iter().sum::<f64>(),
            surv as f64 / nq as f64
        );
    }
    println!("peak RSS of the whole process: {:.0} MiB", peak_rss_mb());
}
