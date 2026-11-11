//! Batched exact scan: B queries share one pass over the int8 shadows.
//!
//! The int8 scan is limited by memory bandwidth: every query reads every
//! shadow. Here a thread loads each row's codes once and scores them against
//! all B queries of the batch, so B queries cost about one pass.
//!
//! Exactness per query is unchanged: certified bounds
//! |q.x - q~.x~| <= |q||x - x~| + |q - q~||x~|. Each chunk keeps, per query,
//! the rows whose upper bound reaches that chunk's own k-th best lower bound
//! (a superset of the global candidates, since a chunk's floor is never above
//! the global one); the merge applies the global floor and scores survivors
//! exactly in f32 against the full vectors.
//!
//! usage: batch <dir>

use std::time::Instant;

use rayon::prelude::*;

const K: usize = 10;
const CHUNK: usize = 8192;

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

/// Dot products of one row with four queries, with stable NEON: widening
/// int8 x int8 -> int16 multiplies, pairwise-accumulated into int32 lanes.
/// All slices are `dp` long, dp % 16 == 0.
#[cfg(target_arch = "aarch64")]
fn dots4(row: &[i8], q: [&[i8]; 4]) -> [i32; 4] {
    use std::arch::aarch64::*;
    // SAFETY: NEON is always available on aarch64; every load reads 16 bytes
    // at an offset k < len, and len is a multiple of 16.
    unsafe {
        let mut acc = [vdupq_n_s32(0); 4];
        let mut k = 0;
        while k < row.len() {
            let r = vld1q_s8(row.as_ptr().add(k));
            for t in 0..4 {
                let x = vld1q_s8(q[t].as_ptr().add(k));
                acc[t] = vpadalq_s16(acc[t], vmull_s8(vget_low_s8(r), vget_low_s8(x)));
                acc[t] = vpadalq_s16(acc[t], vmull_high_s8(r, x));
            }
            k += 16;
        }
        [vaddvq_s32(acc[0]), vaddvq_s32(acc[1]), vaddvq_s32(acc[2]), vaddvq_s32(acc[3])]
    }
}

/// codes, scale, |x - x~|, |x~|
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

/// K best (score, id), descending.
#[derive(Clone, Default)]
struct TopK(Vec<(f32, u32)>);
impl TopK {
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
}

struct Shadow {
    n: usize,
    d: usize,
    /// Row width in the code array, padded with zeros to a multiple of 16.
    dp: usize,
    codes: Vec<i8>,
    scale: Vec<f32>,
    err: Vec<f32>,
    qnorm: Vec<f32>,
}

impl Shadow {
    fn build(x: &[f32], n: usize, d: usize) -> Self {
        let rows: Vec<_> = (0..n).into_par_iter().map(|i| quantise(&x[i * d..(i + 1) * d])).collect();
        let dp = d.div_ceil(16) * 16;
        let mut s = Shadow { n, d, dp, codes: Vec::with_capacity(n * dp), scale: vec![], err: vec![], qnorm: vec![] };
        for (c, sc, e, nq) in rows {
            s.codes.extend_from_slice(&c);
            s.codes.resize(s.codes.len() + dp - d, 0);
            s.scale.push(sc);
            s.err.push(e);
            s.qnorm.push(nq);
        }
        s
    }
}

/// One query, quantised.
struct Q {
    codes: Vec<i8>,
    scale: f32,
    err: f32,
    norm: f32,
}

/// Exact top-K for every query in `qs`, sharing one pass over the shadows.
fn search_batch(sh: &Shadow, x: &[f32], qs: &[&[f32]]) -> (Vec<TopK>, usize) {
    let (n, d, dp, b) = (sh.n, sh.d, sh.dp, qs.len());
    let qq: Vec<Q> = qs
        .iter()
        .map(|q| {
            let (mut codes, scale, err, _) = quantise(q);
            codes.resize(sh.dp, 0);
            Q { codes, scale, err, norm: q.iter().map(|v| v * v).sum::<f32>().sqrt() }
        })
        .collect();

    // Per chunk: for each query, (chunk-local lows top-K, candidate (row, ub) list).
    let parts: Vec<Vec<(TopK, Vec<(u32, f32)>)>> = (0..n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| {
            let (lo, hi) = (c * CHUNK, ((c + 1) * CHUNK).min(n));
            let mut lows = vec![TopK::default(); b];
            // Running floor per query: it only rises, so a row admitted
            // under an older floor is a harmless extra (the merge re-filters
            // with the global floor), and nothing that could qualify is lost.
            let mut floors = vec![f32::NEG_INFINITY; b];
            let mut cands: Vec<Vec<(u32, f32)>> = vec![Vec::new(); b];
            let mut dots = vec![0i32; b];
            for i in lo..hi {
                let row = &sh.codes[i * dp..(i + 1) * dp]; // read once, used by every query
                let (si, ei, ni) = (sh.scale[i], sh.err[i], sh.qnorm[i]);
                let mut j = 0;
                while j + 4 <= b {
                    let r = dots4(row, [&qq[j].codes, &qq[j + 1].codes, &qq[j + 2].codes, &qq[j + 3].codes]);
                    dots[j..j + 4].copy_from_slice(&r);
                    j += 4;
                }
                for (t, q) in qq.iter().enumerate().skip(j) {
                    dots[t] = dot_i8(&q.codes, row);
                }
                for (j, q) in qq.iter().enumerate() {
                    let approx = q.scale * si * dots[j] as f32;
                    let rad = q.norm * ei + q.err * ni + 1e-5 * q.norm * (ni + ei);
                    let (lo_b, hi_b) = (approx - rad, approx + rad);
                    if lo_b > floors[j] {
                        lows[j].push(lo_b, i as u32);
                        floors[j] = lows[j].kth();
                    }
                    if hi_b >= floors[j] {
                        cands[j].push((i as u32, hi_b));
                    }
                }
            }
            lows.into_iter().zip(cands).collect()
        })
        .collect();

    let mut out = Vec::with_capacity(b);
    let mut scored = 0;
    for j in 0..b {
        let mut global = TopK::default();
        for p in &parts {
            for &(s, id) in &p[j].0 .0 {
                global.push(s, id);
            }
        }
        let floor = global.kth();
        let mut top = TopK::default();
        for p in &parts {
            for &(row, ub) in &p[j].1 {
                if ub >= floor {
                    scored += 1;
                    let r = row as usize;
                    top.push(dot(&x[r * d..(r + 1) * d], qs[j]), row);
                }
            }
        }
        out.push(top);
    }
    (out, scored)
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
    let x = read_f32(&format!("{dir}/base.f32"));
    let qs = read_f32(&format!("{dir}/query.f32"));
    let truth = read_u32(&format!("{dir}/truth.u32"));
    let sh = Shadow::build(&x, n, d);
    let queries: Vec<&[f32]> = (0..nq).map(|i| &qs[i * d..(i + 1) * d]).collect();
    println!("n = {n}, d = {d}, queries = {nq}, cores = {}", rayon::current_num_threads());

    for b in [1usize, 8, 32, 64] {
        let mut lat = Vec::new();
        let (mut wrong, mut scored) = (0, 0);
        let w0 = Instant::now();
        for (bi, batch) in queries.chunks(b).enumerate() {
            let t = Instant::now();
            let (res, sc) = search_batch(&sh, &x, batch);
            lat.push(t.elapsed().as_secs_f64());
            scored += sc;
            for (j, r) in res.iter().enumerate() {
                let qi = bi * b + j;
                wrong += !correct(r, &truth[qi * K..(qi + 1) * K]) as usize;
            }
        }
        let wall = w0.elapsed().as_secs_f64();
        lat.sort_by(|a, b| a.total_cmp(b));
        let p = |f: f64| lat[((lat.len() - 1) as f64 * f) as usize] * 1e3;
        println!(
            "batch {b:3}: {:8.0} q/s   per-batch p50 {:7.3} ms  p95 {:7.3} ms   exact scores {:.1}/query   wrong {wrong}/{nq}",
            nq as f64 / wall,
            p(0.5),
            p(0.95),
            scored as f64 / nq as f64
        );
    }
}
