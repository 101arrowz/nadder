use super::array::Array;

pub struct Bitset<'a> {
    raw: &'a mut [u32],
    length: usize
}

impl Bitset<'_> {
    pub fn new(raw: &mut [u32], length: usize) -> Bitset<'_> {
        Bitset {
            raw,
            length
        }
    }
}

impl Array for Bitset<'_> {
    type Elem = bool;

    fn get(&self, idx: usize) -> bool {
        if idx > self.length {
            panic!("invalid index access");
        }
        (unsafe { *self.raw.get_unchecked(idx >> 5) }) & ((1 << (idx & 31)) - 1) != 0
    }

    fn set(&mut self, idx: usize, val: bool) {
        if idx > self.length {
            panic!("invalid index modification");
        }
        let flag = (1 << (idx & 31)) - 1;
        unsafe {
            if val {
                *self.raw.get_unchecked_mut(idx >> 5) |= flag;
            } else {
                *self.raw.get_unchecked_mut(idx >> 5) &= !flag;
            }
        }
        
    }
}