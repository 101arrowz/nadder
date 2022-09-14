use super::array::Array;

pub struct Bitset {
    raw: *mut u32,
    length: usize,
}

impl Bitset {
    pub unsafe fn new(raw: *mut u32, length: usize) -> Bitset {
        Bitset { raw, length }
    }

    pub fn ptr(&self) -> *mut u32 {
        self.raw
    }

    pub fn len(&self) -> usize {
        self.length
    }
}

impl Array for Bitset {
    type Elem = bool;

    #[inline]
    fn get(&self, idx: usize) -> bool {
        if idx > self.length {
            panic!("invalid index access");
        }
        (unsafe { *self.raw.offset(idx as isize >> 5) }) & ((1 << (idx & 31)) - 1) != 0
    }

    #[inline]
    fn set(&mut self, idx: usize, val: bool) {
        if idx > self.length {
            panic!("invalid index modification");
        }
        let flag = (1 << (idx & 31)) - 1;
        unsafe {
            if val {
                *self.raw.offset(idx as isize >> 5) |= flag;
            } else {
                *self.raw.offset(idx as isize >> 5) &= !flag;
            }
        }
    }
}
