//! Race: exact top-10 cosine search, Fashion-MNIST 60k x 784 (unit vectors).
//!
//! flat-1   f32 scan of every vector, one thread
//! flat-N   f32 scan, all cores (rayon) — the engine's current read path
//! shadow   k-d tree over each vector's 8-axis shadow, best-first with certified
//!          bounds; in reached leaves: 24-axis shadow bound -> 128-axis bound -> exact.
//!          One thread per query.
//! Every result is checked against the brute-force truth.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

use rayon::prelude::*;

const K: usize = 10;
const M1: usize = 8; // tree axes
const M2: usize = 24; // second shadow tier
// third shadow tier: width set at start-up from the data (shape.txt), min(128, d/2)
static MH_RT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(128);
#[inline]
fn mh() -> usize {
    MH_RT.load(std::sync::atomic::Ordering::Relaxed)
}
const LEAF: usize = 32;
const WARM: usize = 3;
/// Slack added to every upper bound to cover f32 rounding (scores are in [-1, 1]).
const SLACK: f32 = 1e-3;

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

/// The K best (score, id) seen so far.
#[derive(Clone)]
struct TopK {
    v: Vec<(f32, u32)>,
}
impl TopK {
    fn new() -> Self {
        TopK { v: Vec::with_capacity(K + 1) }
    }
    #[inline]
    fn kth(&self) -> f32 {
        if self.v.len() < K { f32::NEG_INFINITY } else { self.v[K - 1].0 }
    }
    #[inline]
    fn push(&mut self, s: f32, id: u32) {
        if self.v.len() == K && s <= self.v[K - 1].0 {
            return;
        }
        let pos = self.v.iter().position(|e| s > e.0).unwrap_or(self.v.len());
        self.v.insert(pos, (s, id));
        self.v.truncate(K);
    }
    fn merge(mut self, o: TopK) -> TopK {
        for (s, i) in o.v {
            self.push(s, i);
        }
        self
    }
}

struct Data {
    n: usize,
    d: usize,
    x: Vec<f32>,
}

fn flat_scan(data: &Data, q: &[f32], lo: usize, hi: usize) -> TopK {
    let mut t = TopK::new();
    for i in lo..hi {
        let s = dot(&data.x[i * data.d..(i + 1) * data.d], q);
        t.push(s, i as u32);
    }
    t
}

fn flat_parallel(data: &Data, q: &[f32]) -> TopK {
    const CHUNK: usize = 2048;
    (0..data.n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| flat_scan(data, q, c * CHUNK, ((c + 1) * CHUNK).min(data.n)))
        .reduce(TopK::new, TopK::merge)
}


// ------------------------------------------------------------- int8 scan
/// Exact top-k by an int8 pre-pass with certified per-vector bounds, then an
/// exact f32 re-score of only the rows the bounds cannot rule out.
struct Int8Index {
    d: usize,
    codes: Vec<i8>,
    scale: Vec<f32>,
    err: Vec<f32>,  // |x - x~|
    qnorm: Vec<f32>, // |x~|
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

impl Int8Index {
    fn build(data: &Data) -> Self {
        let d = data.d;
        let rows: Vec<(Vec<i8>, f32, f32, f32)> =
            (0..data.n).into_par_iter().map(|i| quantise(&data.x[i * d..(i + 1) * d])).collect();
        let mut t = Int8Index { d, codes: Vec::with_capacity(data.n * d), scale: vec![], err: vec![], qnorm: vec![] };
        for (c, s, e, nq) in rows {
            t.codes.extend_from_slice(&c);
            t.scale.push(s);
            t.err.push(e);
            t.qnorm.push(nq);
        }
        t
    }

    /// (upper bounds, k-th best lower bound) over rows lo..hi.
    fn pass1(&self, cq: &[i8], ts: f32, qn: f32, eq: f32, lo: usize, hi: usize) -> (Vec<f32>, TopK) {
        let d = self.d;
        let mut ub = Vec::with_capacity(hi - lo);
        let mut lows = TopK::new();
        for i in lo..hi {
            let approx = ts * self.scale[i] * dot_i8(cq, &self.codes[i * d..(i + 1) * d]) as f32;
            let rad = qn * self.err[i] + eq * self.qnorm[i] + 1e-5 * qn * (self.qnorm[i] + self.err[i]);
            ub.push(approx + rad);
            lows.push(approx - rad, i as u32);
        }
        (ub, lows)
    }

    fn search(&self, data: &Data, q: &[f32], parallel: bool, cand: &mut u64) -> TopK {
        let d = self.d;
        let (cq, ts, eq, _) = quantise(q);
        let qn = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        const CHUNK: usize = 16384;
        let chunks = data.n.div_ceil(CHUNK);
        let run = |c: usize| {
            let (lo, hi) = (c * CHUNK, ((c + 1) * CHUNK).min(data.n));
            let (ub, lows) = self.pass1(&cq, ts, qn, eq, lo, hi);
            (lo, ub, lows)
        };
        let parts: Vec<(usize, Vec<f32>, TopK)> =
            if parallel { (0..chunks).into_par_iter().map(run).collect() } else { (0..chunks).map(run).collect() };
        let floor = parts.iter().fold(TopK::new(), |a, p| a.merge(p.2.clone())).kth();
        let rescore = |(lo, ub, _): &(usize, Vec<f32>, TopK)| {
            let mut t = TopK::new();
            let mut n = 0u64;
            for (j, u) in ub.iter().enumerate() {
                if *u >= floor {
                    n += 1;
                    let i = lo + j;
                    t.push(dot(&data.x[i * d..(i + 1) * d], q), i as u32);
                }
            }
            (t, n)
        };
        let (top, n) = if parallel {
            parts.par_iter().map(rescore).reduce(|| (TopK::new(), 0), |a, b| (a.0.merge(b.0), a.1 + b.1))
        } else {
            parts.iter().map(rescore).fold((TopK::new(), 0), |a, b| (a.0.merge(b.0), a.1 + b.1))
        };
        *cand += n;
        top
    }
}

// ------------------------------------------------------------- shadow tree
struct Node {
    lo1: [f32; M1],
    hi1: [f32; M1],
    lo2: [f32; M2 - M1],
    hi2: [f32; M2 - M1],
    r2: f32,
    left: u32,
    right: u32,
    start: u32,
    end: u32,
}

struct ShadowTree {
    d: usize,
    axes: Vec<f32>, // mh() x d
    // all per-vector arrays in leaf order, so a leaf is contiguous
    x: Vec<f32>,
    xh: Vec<f32>, // n x mh()
    r2: Vec<f32>, // residual beyond M2 axes
    rh: Vec<f32>, // residual beyond mh() axes
    ids: Vec<u32>,
    nodes: Vec<Node>,
    leaves: Vec<u32>,
    leaf_centre: Vec<f32>, // leaves x mh(), unit
}

impl ShadowTree {
    fn build(data: &Data, axes: Vec<f32>) -> Self {
        let (n, d) = (data.n, data.d);
        let mut xh = vec![0f32; n * mh()];
        xh.par_chunks_mut(mh()).enumerate().for_each(|(i, row)| {
            let xi = &data.x[i * d..(i + 1) * d];
            for a in 0..mh() {
                row[a] = dot(xi, &axes[a * d..(a + 1) * d]);
            }
        });
        let resid = |i: usize, m: usize| -> f32 {
            let xi = &data.x[i * d..(i + 1) * d];
            let total: f64 = xi.iter().map(|v| (*v as f64).powi(2)).sum();
            let shadow: f64 = xh[i * mh()..i * mh() + m].iter().map(|v| (*v as f64).powi(2)).sum();
            (total - shadow).max(0.0).sqrt() as f32
        };
        let r2_all: Vec<f32> = (0..n).map(|i| resid(i, M2)).collect();
        let rh_all: Vec<f32> = (0..n).map(|i| resid(i, mh())).collect();

        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut nodes = Vec::new();
        Self::split(&mut order, 0, &xh, &r2_all, &mut nodes);

        let mut t = ShadowTree {
            d,
            axes,
            x: Vec::with_capacity(n * d),
            xh: Vec::with_capacity(n * mh()),
            r2: Vec::with_capacity(n),
            rh: Vec::with_capacity(n),
            ids: order.clone(),
            nodes,
            leaves: Vec::new(),
            leaf_centre: Vec::new(),
        };
        for &i in &order {
            let i = i as usize;
            t.x.extend_from_slice(&data.x[i * d..(i + 1) * d]);
            t.xh.extend_from_slice(&xh[i * mh()..(i + 1) * mh()]);
            t.r2.push(r2_all[i]);
            t.rh.push(rh_all[i]);
        }
        for (ni, nd) in t.nodes.iter().enumerate() {
            if nd.left == u32::MAX {
                t.leaves.push(ni as u32);
                let mut c = vec![0f32; mh()];
                for p in nd.start..nd.end {
                    for a in 0..mh() {
                        c[a] += t.xh[p as usize * mh() + a];
                    }
                }
                let norm = c.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-12;
                t.leaf_centre.extend(c.iter().map(|v| v / norm));
            }
        }
        t
    }

    /// Builds the subtree over `order[..]` (positions offset by `base`) and returns its index.
    fn split(order: &mut [u32], base: u32, xh: &[f32], r2: &[f32], nodes: &mut Vec<Node>) -> u32 {
        let mut nd = Node {
            lo1: [f32::INFINITY; M1],
            hi1: [f32::NEG_INFINITY; M1],
            lo2: [f32::INFINITY; M2 - M1],
            hi2: [f32::NEG_INFINITY; M2 - M1],
            r2: 0.0,
            left: u32::MAX,
            right: u32::MAX,
            start: base,
            end: base + order.len() as u32,
        };
        for &i in order.iter() {
            let s = &xh[i as usize * mh()..];
            for a in 0..M1 {
                nd.lo1[a] = nd.lo1[a].min(s[a]);
                nd.hi1[a] = nd.hi1[a].max(s[a]);
            }
            for a in M1..M2 {
                nd.lo2[a - M1] = nd.lo2[a - M1].min(s[a]);
                nd.hi2[a - M1] = nd.hi2[a - M1].max(s[a]);
            }
            nd.r2 = nd.r2.max(r2[i as usize]);
        }
        let me = nodes.len() as u32;
        if order.len() <= LEAF {
            nodes.push(nd);
            return me;
        }
        let ax = (0..M1)
            .max_by(|&a, &b| (nd.hi1[a] - nd.lo1[a]).total_cmp(&(nd.hi1[b] - nd.lo1[b])))
            .unwrap();
        let mid = order.len() / 2;
        order.select_nth_unstable_by(mid, |&a, &b| xh[a as usize * mh() + ax].total_cmp(&xh[b as usize * mh() + ax]));
        nodes.push(nd);
        let (l, r) = order.split_at_mut(mid);
        let li = Self::split(l, base, xh, r2, nodes);
        let ri = Self::split(r, base + mid as u32, xh, r2, nodes);
        nodes[me as usize].left = li;
        nodes[me as usize].right = ri;
        me
    }

    fn search(&self, q: &[f32], stats: &mut [u64; 4]) -> TopK {
        let d = self.d;
        let mut qh = vec![0f32; mh()];
        for a in 0..mh() {
            qh[a] = dot(q, &self.axes[a * d..(a + 1) * d]);
        }
        let qn2: f32 = q.iter().map(|v| v * v).sum();
        let q24n2: f32 = qh[..M2].iter().map(|v| v * v).sum();
        let qhn2: f32 = qh.iter().map(|v| v * v).sum();
        let qr2 = (qn2 - q24n2).max(0.0).sqrt();
        let qrh = (qn2 - qhn2).max(0.0).sqrt();
        let mut top = TopK::new();

        let ub_node = |nd: &Node| -> f32 {
            let mut u = 0f32;
            for a in 0..M1 {
                u += (qh[a] * nd.lo1[a]).max(qh[a] * nd.hi1[a]);
            }
            for a in M1..M2 {
                u += (qh[a] * nd.lo2[a - M1]).max(qh[a] * nd.hi2[a - M1]);
            }
            u + qr2 * nd.r2 + SLACK
        };
        let mut scan_leaf = |nd: &Node, top: &mut TopK, stats: &mut [u64; 4]| {
            for p in nd.start as usize..nd.end as usize {
                stats[0] += 1;
                let sh = &self.xh[p * mh()..(p + 1) * mh()];
                let ub1 = dot(&sh[..M2], &qh[..M2]) + qr2 * self.r2[p] + SLACK;
                if ub1 <= top.kth() {
                    continue;
                }
                stats[1] += 1;
                let ub2 = dot(sh, &qh) + qrh * self.rh[p] + SLACK;
                if ub2 <= top.kth() {
                    continue;
                }
                stats[2] += 1;
                top.push(dot(&self.x[p * d..(p + 1) * d], q), self.ids[p]);
            }
        };

        // warm start: the leaves whose centre's 128-axis shadow is closest to the query
        let mut warm: Vec<(f32, usize)> = (0..self.leaves.len())
            .map(|l| (dot(&self.leaf_centre[l * mh()..(l + 1) * mh()], &qh), l))
            .collect();
        warm.select_nth_unstable_by(WARM, |a, b| b.0.total_cmp(&a.0));
        let mut visited = [usize::MAX; WARM];
        for (w, &(_, l)) in warm[..WARM].iter().enumerate() {
            let ni = self.leaves[l] as usize;
            scan_leaf(&self.nodes[ni], &mut top, stats);
            visited[w] = ni;
        }

        let mut heap = BinaryHeap::new();
        heap.push(Ub(ub_node(&self.nodes[0]), 0));
        while let Some(Ub(u, ni)) = heap.pop() {
            if u <= top.kth() {
                break;
            }
            stats[3] += 1;
            let nd = &self.nodes[ni as usize];
            if nd.left == u32::MAX {
                if !visited.contains(&(ni as usize)) {
                    scan_leaf(nd, &mut top, stats);
                }
                continue;
            }
            for c in [nd.left, nd.right] {
                let cu = ub_node(&self.nodes[c as usize]);
                if cu > top.kth() {
                    heap.push(Ub(cu, c));
                }
            }
        }
        top
    }
}

struct Ub(f32, u32);
impl PartialEq for Ub {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for Ub {}
impl PartialOrd for Ub {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Ub {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

fn correct(t: &TopK, truth: &[u32]) -> bool {
    let mut a: Vec<u32> = t.v.iter().map(|e| e.1).collect();
    let mut b = truth.to_vec();
    a.sort();
    b.sort();
    a == b
}

fn report(name: &str, lat: &mut [f64], wrong: usize, nq: usize, wall: f64) {
    lat.sort_by(|a, b| a.total_cmp(b));
    let p = |x: f64| lat[((lat.len() - 1) as f64 * x) as usize] * 1e3;
    println!(
        "{name:44} p50 {:7.3} ms   p95 {:7.3} ms   {:8.0} q/s   wrong {wrong}/{nq}",
        p(0.5),
        p(0.95),
        nq as f64 / wall
    );
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "data".into());
    let shape = std::fs::read_to_string(format!("{dir}/shape.txt")).unwrap();
    let s: Vec<usize> = shape.split_whitespace().map(|v| v.parse().unwrap()).collect();
    let (n, d, nq) = (s[0], s[1], s[2]);
    MH_RT.store(s.get(3).copied().unwrap_or(128), std::sync::atomic::Ordering::Relaxed);
    let data = Data { n, d, x: read_f32(&format!("{dir}/base.f32")) };
    let qs = read_f32(&format!("{dir}/query.f32"));
    let truth = read_u32(&format!("{dir}/truth.u32"));
    let axes = read_f32(&format!("{dir}/axes.f32"));
    let q = |i: usize| &qs[i * d..(i + 1) * d];
    let tr = |i: usize| &truth[i * K..(i + 1) * K];
    println!("n = {n}, d = {d}, queries = {nq}, cores = {}", rayon::current_num_threads());

    let t0 = Instant::now();
    let int8 = Int8Index::build(&data);
    println!("int8 index built in {:.2} s ({} MB of codes vs {} MB f32)", t0.elapsed().as_secs_f64(),
        int8.codes.len() / 1_000_000, data.x.len() * 4 / 1_000_000);
    let t0 = Instant::now();
    let tree = ShadowTree::build(&data, axes);
    println!("shadow tree built in {:.2} s ({} nodes)\n", t0.elapsed().as_secs_f64(), tree.nodes.len());

    // latency: one query at a time
    for (name, which) in [("flat f32 scan, 1 thread", 0), ("flat f32 scan, all cores (engine today)", 1), ("shadow tree, 1 thread", 2), ("int8 exact scan, 1 thread", 3), ("int8 exact scan, all cores", 4)] {
        let mut lat = Vec::with_capacity(nq);
        let mut wrong = 0;
        let mut stats = [0u64; 4];
        let w0 = Instant::now();
        for i in 0..nq {
            let t = Instant::now();
            let r = match which {
                0 => flat_scan(&data, q(i), 0, n),
                1 => flat_parallel(&data, q(i)),
                2 => tree.search(q(i), &mut stats),
                3 => int8.search(&data, q(i), false, &mut stats[2]),
                _ => int8.search(&data, q(i), true, &mut stats[2]),
            };
            lat.push(t.elapsed().as_secs_f64());
            wrong += !correct(&r, tr(i)) as usize;
        }
        report(name, &mut lat, wrong, nq, w0.elapsed().as_secs_f64());
        if which >= 3 {
            println!("{:44} exact re-scores {:.4}% of rows", "", stats[2] as f64 / nq as f64 / n as f64 * 100.0);
        }
        if which == 2 {
            let f = |c: u64| c as f64 / nq as f64 / n as f64 * 100.0;
            println!(
                "{:44} touched {:.1}%  24-axis survivors {:.2}%  exact scores {:.3}%  nodes {:.0}/query",
                "", f(stats[0]), f(stats[1]), f(stats[2]), stats[3] as f64 / nq as f64
            );
        }
    }

    // throughput: many queries at once, every core busy
    let w0 = Instant::now();
    let wrong: usize = (0..nq)
        .into_par_iter()
        .map(|i| {
            let mut st = [0u64; 4];
            !correct(&tree.search(q(i), &mut st), tr(i)) as usize
        })
        .sum();
    let wall = w0.elapsed().as_secs_f64();
    println!("\nthroughput, all cores busy:");
    println!("{:44} {:8.0} q/s   wrong {wrong}/{nq}", "shadow tree, queries in parallel", nq as f64 / wall);
    let w0 = Instant::now();
    let wrong: usize = (0..nq).into_par_iter().map(|i| !correct(&flat_scan(&data, q(i), 0, n), tr(i)) as usize).sum();
    let wall = w0.elapsed().as_secs_f64();
    println!("{:44} {:8.0} q/s   wrong {wrong}/{nq}", "flat f32 scan, queries in parallel", nq as f64 / wall);
    let w0 = Instant::now();
    let wrong: usize = (0..nq)
        .into_par_iter()
        .map(|i| {
            let mut c = 0u64;
            !correct(&int8.search(&data, q(i), false, &mut c), tr(i)) as usize
        })
        .sum();
    let wall = w0.elapsed().as_secs_f64();
    println!("{:44} {:8.0} q/s   wrong {wrong}/{nq}", "int8 exact scan, queries in parallel", nq as f64 / wall);
}
