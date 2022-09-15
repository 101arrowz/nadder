use super::array::Array;

pub struct Bitset {
    raw: *mut u32,
    length: usize,
    offset: usize
}

impl Bitset {
    pub unsafe fn new(raw: *mut u32, length: usize, offset: usize) -> Bitset {
        Bitset { raw, length, offset }
    }

    pub fn ptr(&self) -> *mut u32 {
        self.raw
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn offset(&self) -> usize {
        self.offset
    }
}

impl Array for Bitset {
    // u32 for automatic casting
    type Item = u32;

    #[inline]
    fn get(&self, mut idx: usize) -> u32 {
        idx += self.offset;
        if idx > self.length {
            panic!("invalid index access");
        }
        (unsafe { *self.raw.offset(idx as isize >> 5) }) & (1 << (idx & 31)) >> (idx & 31)
    }

    #[inline]
    fn set(&mut self, mut idx: usize, val: u32) {
        idx += self.offset;
        if idx > self.length {
            panic!("invalid index modification");
        }
        let flag = (1 << (idx & 31)) - 1;
        unsafe {
            if val != 0 {
                *self.raw.offset(idx as isize >> 5) |= flag;
            } else {
                *self.raw.offset(idx as isize >> 5) &= !flag;
            }
        }
    }
}
