use std::ops::{Add, Mul, Neg, Sub};

use crate::ops;
use crate::snapshot::EAMSnapshot;

// --- EAMSnapshot + EAMSnapshot ---

impl Add for EAMSnapshot {
    type Output = EAMSnapshot;
    fn add(self, rhs: EAMSnapshot) -> EAMSnapshot {
        ops::add(&self, &rhs).expect("EAM addition failed: dimension mismatch")
    }
}

impl Add<&EAMSnapshot> for &EAMSnapshot {
    type Output = EAMSnapshot;
    fn add(self, rhs: &EAMSnapshot) -> EAMSnapshot {
        ops::add(self, rhs).expect("EAM addition failed: dimension mismatch")
    }
}

// --- EAMSnapshot - EAMSnapshot ---

impl Sub for EAMSnapshot {
    type Output = EAMSnapshot;
    fn sub(self, rhs: EAMSnapshot) -> EAMSnapshot {
        ops::sub(&self, &rhs).expect("EAM subtraction failed: dimension mismatch")
    }
}

impl Sub<&EAMSnapshot> for &EAMSnapshot {
    type Output = EAMSnapshot;
    fn sub(self, rhs: &EAMSnapshot) -> EAMSnapshot {
        ops::sub(self, rhs).expect("EAM subtraction failed: dimension mismatch")
    }
}

// --- -EAMSnapshot ---

impl Neg for EAMSnapshot {
    type Output = EAMSnapshot;
    fn neg(self) -> EAMSnapshot {
        ops::negate(&self).expect("EAM negation failed")
    }
}

impl Neg for &EAMSnapshot {
    type Output = EAMSnapshot;
    fn neg(self) -> EAMSnapshot {
        ops::negate(self).expect("EAM negation failed")
    }
}

// --- EAMSnapshot * f64 ---

impl Mul<f64> for EAMSnapshot {
    type Output = EAMSnapshot;
    fn mul(self, rhs: f64) -> EAMSnapshot {
        ops::scale(&self, rhs).expect("EAM scaling failed")
    }
}

impl Mul<f64> for &EAMSnapshot {
    type Output = EAMSnapshot;
    fn mul(self, rhs: f64) -> EAMSnapshot {
        ops::scale(self, rhs).expect("EAM scaling failed")
    }
}

// --- f64 * EAMSnapshot ---

impl Mul<EAMSnapshot> for f64 {
    type Output = EAMSnapshot;
    fn mul(self, rhs: EAMSnapshot) -> EAMSnapshot {
        ops::scale(&rhs, self).expect("EAM scaling failed")
    }
}

impl Mul<&EAMSnapshot> for f64 {
    type Output = EAMSnapshot;
    fn mul(self, rhs: &EAMSnapshot) -> EAMSnapshot {
        ops::scale(rhs, self).expect("EAM scaling failed")
    }
}
