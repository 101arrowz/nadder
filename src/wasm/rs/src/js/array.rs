pub trait Array {
    type Elem;

    fn get(&self, idx: usize) -> Self::Elem;
    fn set(&mut self, idx: usize, val: Self::Elem);
}

impl<T> Array for &mut [T] {
    type Elem = T;

    fn get(&self, idx: usize) -> T {
        self[idx]
    }

    fn set(&mut self, idx: usize, val: T) {
        self[idx] = val;
    }
}