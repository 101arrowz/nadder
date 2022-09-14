pub trait Array {
    type Elem: Copy;

    fn get(&self, idx: usize) -> Self::Elem;
    fn set(&mut self, idx: usize, val: Self::Elem);
}

impl<T: Copy> Array for &mut [T] {
    type Elem = T;

    #[inline]
    fn get(&self, idx: usize) -> T {
        self[idx]
    }

    #[inline]
    fn set(&mut self, idx: usize, val: T) {
        self[idx] = val;
    }
}
