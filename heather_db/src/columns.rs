//! Columns: describe a database by what its rows hold, write rows by name,
//! and ask by name.
//!
//! A database declared with columns works out its own layout: the row width,
//! which slots are numbers, which are learned codes, and which a prediction
//! fills in. Callers never see a vector.
//!
//! - **category**: a name (an id, a merchant, a month). Stored as an id; the
//!   memory learns a code for each value ([`crate::symbols`]).
//! - **number**: a quantity, put on a common scale by the memory itself. The
//!   row stores its signed log (`sign(x)·ln(1 + |x|)`: fixed and exactly
//!   invertible, so a column spanning orders of magnitude does not drown the
//!   others). Whenever rows are compared, that is centred and divided by the
//!   column's spread, from running statistics as they are at that moment, so
//!   every row, early or late, is on the same, latest scale. A column the
//!   database predicts is kept as written, so a prediction comes back in the
//!   column's own units.
//! - **vector**: numbers the caller computed (an embedding), stored as given.
//!
//! The layout: numbers and vectors first, in declared order, then one slot
//! per category. Each predicted column is an attention head over its slot.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::attention::AttentionConfig;
use crate::error::{HeatherError, Result};
use crate::symbols::SymbolConfig;

/// What a column holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Category,
    Number,
    Vector,
}

/// One declared column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub kind: Kind,
    /// Numbers in a vector column; 1 for the others.
    #[serde(default = "one")]
    pub size: usize,
}

fn one() -> usize {
    1
}

impl Column {
    pub fn category(name: &str) -> Self {
        Column {
            name: name.into(),
            kind: Kind::Category,
            size: 1,
        }
    }
    pub fn number(name: &str) -> Self {
        Column {
            name: name.into(),
            kind: Kind::Number,
            size: 1,
        }
    }
    pub fn vector(name: &str, size: usize) -> Self {
        Column {
            name: name.into(),
            kind: Kind::Vector,
            size,
        }
    }
}

/// Code size for categories: rich enough to tell thousands of items apart
/// in a sequence; small for categories that describe a row next to others.
const SEQUENCE_CODE_DIMS: usize = 128;
const ROW_CODE_DIMS: usize = 4;
const MAX_VECTOR: usize = 4096;

/// The database a set of columns describes: its dimension and its attention
/// settings. `predict` names number columns the database learns to fill in;
/// `sequence` makes each collection's rows a history the database learns to
/// continue. At least one is needed: a column database learns from its
/// writes, and those name what to learn.
pub fn plan(
    columns: Vec<Column>,
    predict: &[String],
    sequence: bool,
) -> Result<(usize, AttentionConfig)> {
    let bad = |m: String| Err(HeatherError::InvalidConfig(m));
    if columns.is_empty() {
        return bad("columns names no column".into());
    }
    let mut seen = BTreeSet::new();
    for c in &columns {
        let ok = !c.name.is_empty()
            && c.name.len() <= 63
            && c.name
                .chars()
                .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-');
        if !ok {
            return bad(format!(
                "column name {:?} must be 1-63 chars of [a-z0-9_-]",
                c.name
            ));
        }
        if !seen.insert(c.name.as_str()) {
            return bad(format!("column {:?} is named twice", c.name));
        }
        match c.kind {
            Kind::Vector if !(1..=MAX_VECTOR).contains(&c.size) => {
                return bad(format!(
                    "vector column {:?} must hold 1-{MAX_VECTOR} numbers",
                    c.name
                ));
            }
            Kind::Category | Kind::Number if c.size != 1 => {
                return bad(format!("column {:?} holds one value", c.name));
            }
            _ => {}
        }
    }
    if predict.is_empty() && !sequence {
        return bad(
            "say what the database learns: `predict` (columns to fill in) or `sequence: true` \
             (each collection is a history to continue)"
                .into(),
        );
    }
    let offsets = numeric_offsets(&columns);
    let mut heads = BTreeMap::new();
    for p in predict {
        match columns.iter().position(|c| &c.name == p) {
            None => return bad(format!("predict names {p:?}, which is not a column")),
            Some(i) if columns[i].kind != Kind::Number => {
                return bad(format!(
                    "predict names {p:?}, a {:?} column; only number columns can be predicted",
                    columns[i].kind
                ));
            }
            Some(i) => {
                heads.insert(p.clone(), vec![offsets[&i]]);
            }
        }
    }
    let numeric: usize = columns
        .iter()
        .filter(|c| c.kind != Kind::Category)
        .map(|c| c.size)
        .sum();
    let categories: Vec<String> = columns
        .iter()
        .filter(|c| c.kind == Kind::Category)
        .map(|c| c.name.clone())
        .collect();
    let d = numeric + categories.len();
    let symbols = (!categories.is_empty()).then(|| {
        let mut s = SymbolConfig::new(categories);
        s.dims = if sequence {
            SEQUENCE_CODE_DIMS
        } else {
            ROW_CODE_DIMS
        };
        s
    });
    let config = AttentionConfig {
        time: sequence,
        heads,
        symbols,
        columns: Some(columns),
        ..AttentionConfig::default()
    };
    config.validate(d)?;
    Ok((d, config))
}

/// Where each number or vector column starts in the numeric part, by
/// column index.
fn numeric_offsets(columns: &[Column]) -> BTreeMap<usize, usize> {
    let mut at = 0;
    let mut out = BTreeMap::new();
    for (i, c) in columns.iter().enumerate() {
        if c.kind != Kind::Category {
            out.insert(i, at);
            at += c.size;
        }
    }
    out
}

/// One value in a row.
#[derive(Debug, Clone, PartialEq)]
pub enum Cell {
    Text(String),
    Number(f64),
    Vector(Vec<f64>),
}

/// A row by column name.
pub type Row = BTreeMap<String, Cell>;

/// A number column's running statistics, of the signed log of its values.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Running {
    n: u64,
    mean: f64,
    m2: f64,
}

fn signed_log(x: f64) -> f64 {
    x.signum() * x.abs().ln_1p()
}

impl Running {
    pub fn add(&mut self, x: f64) {
        let t = signed_log(x);
        self.n += 1;
        let delta = t - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (t - self.mean);
    }

    /// `x` on the column's common scale: signed log, centred, divided by the
    /// spread (1 until the column has a spread).
    pub fn scale(&self, x: f64) -> f64 {
        self.standardize(signed_log(x))
    }

    /// A stored (signed-log) value, centred and divided by the spread.
    pub fn standardize(&self, t: f64) -> f64 {
        let sd = if self.n > 1 {
            (self.m2 / (self.n - 1) as f64).sqrt()
        } else {
            0.0
        };
        let sd = if sd > 1e-12 { sd } else { 1.0 };
        (t - self.mean) / sd
    }
}

/// Running statistics by number column.
pub type Scales = BTreeMap<String, Running>;

/// A row split for the engine: its numeric part, its category values, and
/// (for a query) the numeric slots of the columns it left out.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    pub numeric: Vec<f64>,
    pub values: BTreeMap<String, String>,
    pub missing: Vec<usize>,
}

fn is_scaled(c: &Column, heads: &BTreeMap<String, Vec<usize>>) -> bool {
    c.kind == Kind::Number && !heads.contains_key(&c.name)
}

/// Checks a row against the columns: no unknown column, each value of its
/// column's kind, and (unless `partial`) every column present.
pub fn check(columns: &[Column], row: &Row, partial: bool) -> Result<()> {
    let bad = |m: String| Err(HeatherError::InvalidInput(m));
    for (name, cell) in row {
        let Some(c) = columns.iter().find(|c| &c.name == name) else {
            return bad(format!("no column named {name:?}"));
        };
        match (c.kind, cell) {
            (Kind::Category, Cell::Text(_) | Cell::Number(_)) => {}
            (Kind::Number, Cell::Number(x)) if x.is_finite() => {}
            (Kind::Vector, Cell::Vector(v))
                if v.len() == c.size && v.iter().all(|x| x.is_finite()) => {}
            (Kind::Category, _) => return bad(format!("{name:?} takes a name or a number")),
            (Kind::Number, _) => return bad(format!("{name:?} takes a finite number")),
            (Kind::Vector, _) => {
                return bad(format!("{name:?} takes {} finite numbers", c.size));
            }
        }
    }
    if !partial && let Some(c) = columns.iter().find(|c| !row.contains_key(&c.name)) {
        return bad(format!("every row needs a value for {:?}", c.name));
    }
    Ok(())
}

/// Adds a written row's numbers to the running statistics.
pub fn observe(
    columns: &[Column],
    heads: &BTreeMap<String, Vec<usize>>,
    scales: &mut Scales,
    row: &Row,
) {
    for c in columns.iter().filter(|c| is_scaled(c, heads)) {
        if let Some(Cell::Number(x)) = row.get(&c.name) {
            scales.entry(c.name.clone()).or_default().add(*x);
        }
    }
}

/// A category value as text: a whole number without its decimal point, so
/// `4471` and `"4471"` are the same value.
fn as_text(cell: &Cell) -> String {
    match cell {
        Cell::Text(s) => s.clone(),
        Cell::Number(x) if x.fract() == 0.0 && x.abs() < 1e15 => format!("{}", *x as i64),
        Cell::Number(x) => x.to_string(),
        Cell::Vector(_) => unreachable!("checked"),
    }
}

/// The numeric slots of the scaled number columns, with their names.
pub fn scaled_slots(
    columns: &[Column],
    heads: &BTreeMap<String, Vec<usize>>,
) -> Vec<(usize, String)> {
    let offsets = numeric_offsets(columns);
    columns
        .iter()
        .enumerate()
        .filter(|(_, c)| is_scaled(c, heads))
        .map(|(i, c)| (offsets[&i], c.name.clone()))
        .collect()
}

/// Puts a stored or encoded row's scaled number slots on the common scale,
/// with the statistics as they are now.
pub fn standardize_row(slots: &[(usize, String)], scales: &Scales, row: &mut [f64]) {
    for (slot, name) in slots {
        let r = scales.get(name).cloned().unwrap_or_default();
        row[*slot] = r.standardize(row[*slot]);
    }
}

/// Splits a checked row into the engine's parts, as stored: a scaled number
/// column as its signed log (see [`standardize_row`] for the rest), the
/// others as written. A column the row leaves out is zero in the numeric part
/// and listed in `missing`.
pub fn encode(columns: &[Column], heads: &BTreeMap<String, Vec<usize>>, row: &Row) -> Encoded {
    let mut numeric = Vec::new();
    let mut values = BTreeMap::new();
    let mut missing = Vec::new();
    for c in columns {
        let cell = row.get(&c.name);
        match (c.kind, cell) {
            (Kind::Category, Some(v)) => {
                values.insert(c.name.clone(), as_text(v));
            }
            (Kind::Category, None) => {}
            (_, None) => {
                missing.extend(numeric.len()..numeric.len() + c.size);
                numeric.extend(std::iter::repeat_n(0.0, c.size));
            }
            (Kind::Number, Some(Cell::Number(x))) => numeric.push(if is_scaled(c, heads) {
                signed_log(*x)
            } else {
                *x
            }),
            (Kind::Vector, Some(Cell::Vector(v))) => numeric.extend_from_slice(v),
            _ => unreachable!("checked"),
        }
    }
    Encoded {
        numeric,
        values,
        missing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn access() -> Vec<Column> {
        vec![
            Column::category("role"),
            Column::number("amount"),
            Column::category("resource"),
            Column::number("granted"),
            Column::vector("text", 3),
        ]
    }

    #[test]
    fn plan_lays_out_numbers_then_categories() {
        let (d, cfg) = plan(access(), &["granted".into()], false).unwrap();
        assert_eq!(d, 1 + 1 + 3 + 2);
        assert_eq!(cfg.heads["granted"], vec![1]);
        assert_eq!(
            cfg.symbols.as_ref().unwrap().fields,
            vec!["role", "resource"]
        );
        assert_eq!(cfg.symbols.as_ref().unwrap().dims, ROW_CODE_DIMS);
        assert!(!cfg.time);
        let (_, seq) = plan(vec![Column::category("item")], &[], true).unwrap();
        assert!(seq.time);
        assert_eq!(seq.symbols.unwrap().dims, SEQUENCE_CODE_DIMS);
    }

    #[test]
    fn plan_refuses_what_it_cannot_learn() {
        assert!(plan(access(), &[], false).is_err(), "nothing to learn");
        assert!(
            plan(access(), &["role".into()], false).is_err(),
            "a category"
        );
        assert!(plan(access(), &["nope".into()], false).is_err(), "unknown");
        let twice = vec![Column::number("a"), Column::number("a")];
        assert!(plan(twice, &["a".into()], false).is_err());
        assert!(plan(vec![Column::number("Bad Name")], &[], true).is_err());
    }

    #[test]
    fn rows_encode_by_name() {
        let cols = access();
        let heads = BTreeMap::from([("granted".to_string(), vec![1])]);
        let mut scales = Scales::new();
        let mut row = Row::from([
            ("role".into(), Cell::Number(4471.0)),
            ("amount".into(), Cell::Number(100.0)),
            ("resource".into(), Cell::Text("r-1".into())),
            ("granted".into(), Cell::Number(1.0)),
            ("text".into(), Cell::Vector(vec![0.1, 0.2, 0.3])),
        ]);
        check(&cols, &row, false).unwrap();
        observe(&cols, &heads, &mut scales, &row);
        let mut e = encode(&cols, &heads, &row);
        assert_eq!(
            e.values["role"], "4471",
            "a whole number is the same value as its text"
        );
        assert_eq!(
            e.numeric,
            vec![101f64.ln(), 1.0, 0.1, 0.2, 0.3],
            "stored: the signed log; granted as written"
        );
        let slots = scaled_slots(&cols, &heads);
        assert_eq!(slots, vec![(0, "amount".to_string())]);
        standardize_row(&slots, &scales, &mut e.numeric);
        assert_eq!(e.numeric[0], 0.0, "one amount centres to 0");
        row.remove("amount");
        assert!(
            check(&cols, &row, false).is_err(),
            "a write needs every column"
        );
        check(&cols, &row, true).unwrap();
        assert_eq!(encode(&cols, &heads, &row).missing, vec![0]);
        row.insert("nope".into(), Cell::Number(1.0));
        assert!(check(&cols, &row, true).is_err());
    }

    #[test]
    fn numbers_share_a_scale() {
        let mut r = Running::default();
        for x in [1.0, 10.0, 100.0, 1_000.0, 10_000.0] {
            r.add(x);
        }
        let scaled: Vec<f64> = [1.0, 100.0, 10_000.0].iter().map(|&x| r.scale(x)).collect();
        assert!(
            scaled[1].abs() < 0.2,
            "the middle of the range is near 0: {scaled:?}"
        );
        assert!(scaled[0] < -1.0 && scaled[2] > 1.0, "{scaled:?}");
        assert!(
            scaled[2] < 2.0,
            "a log keeps the largest value close: {scaled:?}"
        );
    }
}
