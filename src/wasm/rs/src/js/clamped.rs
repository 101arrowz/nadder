use core::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Copy)]
pub struct ClampedU8(pub u8);

impl From<u8> for ClampedU8 {
    fn from(val: u8) -> Self {
        ClampedU8(val)
    }
}

impl Add for ClampedU8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.0.saturating_add(rhs.0).into()
    }
}

impl Sub for ClampedU8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.0.saturating_sub(rhs.0).into()
    }
}

impl Mul for ClampedU8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.0.saturating_mul(rhs.0).into()
    }
}

impl Div for ClampedU8 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.0.saturating_div(rhs.0).into()
    }
}
