pub trait Array {
    type Item: Copy;

    fn get(&self, idx: usize) -> Self::Item;
    fn set(&mut self, idx: usize, val: Self::Item);
}

impl<T: Copy> Array for &mut [T] {
    type Item = T;

    #[inline]
    fn get(&self, idx: usize) -> T {
        unsafe { *self.get_unchecked(idx) }
    }

    #[inline]
    fn set(&mut self, idx: usize, val: T) {
        unsafe { *self.get_unchecked_mut(idx) = val; }
    }
}
