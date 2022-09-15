use super::ClampedU8;

pub trait Cast<T> {
    fn cast(self) -> T;
}

macro_rules! num_impl {
    ($($t:ty),+) => {
        $(
            impl Cast<$t> for i8 {
                fn cast(self) -> $t {
                    self as $t
                }
            }
            impl Cast<$t> for u8 {
                fn cast(self) -> $t {
                    self as $t
                }
            }
            impl Cast<$t> for ClampedU8 {
                fn cast(self) -> $t {
                    self.0 as $t
                }
            }
            impl Cast<ClampedU8> for $t {
                fn cast(self) -> ClampedU8 {
                    ClampedU8(self as u8)
                }
            }
            impl Cast<$t> for i16 {
                fn cast(self) -> $t {
                    self as $t
                }
            }
            impl Cast<$t> for u16 {
                fn cast(self) -> $t {
                    self as $t
                }
            }
            impl Cast<$t> for i32 {
                fn cast(self) -> $t {
                    self as $t
                }
            }
            impl Cast<$t> for u32 {
                fn cast(self) -> $t {
                    self as $t
                }
            }
            impl Cast<$t> for f32 {
                fn cast(self) -> $t {
                    self as $t
                }
            }
            impl Cast<$t> for f64 {
                fn cast(self) -> $t {
                    self as $t
                }
            }
            impl Cast<$t> for i64 {
                fn cast(self) -> $t {
                    self as $t
                }
            }
            impl Cast<$t> for u64 {
                fn cast(self) -> $t {
                    self as $t
                }
            }
        )+
    }
}

num_impl!(i8, u8, i16, u16, i32, u32, f32, f64, i64, u64);

impl Cast<ClampedU8> for ClampedU8 {
    fn cast(self) -> ClampedU8 {
        self
    }
}