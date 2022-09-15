extern crate alloc;
use core::ffi::c_void;

use alloc::vec::Vec;

mod clamped;
use clamped::*;

mod cast;
pub use cast::*;

mod bitset;
pub use bitset::*;

mod array;
pub use array::*;

// mod complex;
// use complex::*;

pub struct NDView<A: Array> {
    pub dims: Vec<isize>,
    pub strides: Vec<isize>,
    pub offset: usize,
    pub data: A,
}

#[repr(u32)]
enum DataType {
    Int8 = 1,
    Uint8 = 2,
    Uint8Clamped = 4,
    Int16 = 8,
    Uint16 = 16,
    Int32 = 32,
    Uint32 = 64,
    Float32 = 128,
    Float64 = 256,
    // TODO
    // Complex = 512,
    Bool = 1024,
    // TODO
    // String = 2048,
    Int64 = 4096,
    Uint64 = 8192,
    // TODO
    // Any = 16384,
}

pub enum ForeignNDView {
    Int8(NDView<&'static mut [i8]>),
    Uint8(NDView<&'static mut [u8]>),
    Uint8Clamped(NDView<&'static mut [ClampedU8]>),
    Int16(NDView<&'static mut [i16]>),
    Uint16(NDView<&'static mut [u16]>),
    Int32(NDView<&'static mut [i32]>),
    Uint32(NDView<&'static mut [u32]>),
    Float32(NDView<&'static mut [f32]>),
    Float64(NDView<&'static mut [f64]>),
    // complex todo
    Bool(NDView<Bitset>),
    Int64(NDView<&'static mut [i64]>),
    Uint64(NDView<&'static mut [u64]>),
}

extern "C" {
    fn dtype(id: i32) -> DataType;
    fn ndim(id: i32) -> usize;
    fn dim(id: i32, ind: usize) -> isize;
    fn stride(id: i32, ind: usize) -> isize;
    fn buf(id: i32) -> *mut c_void;
    fn buflen(id: i32) -> usize;
    fn bufoff(id: i32) -> usize;
    fn off(id: i32) -> usize;
    fn register(
        dtype: DataType,
        ndim: usize,
        dims: *const isize,
        strides: *const isize,
        buflen: usize,
        buf: *mut c_void,
        offset: usize,
        boff: usize
    ) -> i32;
}

impl ForeignNDView {
    pub fn import(id: i32) -> ForeignNDView {
        unsafe {
            let datatype = dtype(id);
            let ptr = buf(id);
            let len = buflen(id);
            let offset = off(id);
            let num_dims = ndim(id);
            let mut dims = Vec::with_capacity(num_dims);
            let mut strides = Vec::with_capacity(num_dims);
            for i in 0..num_dims {
                dims.push(dim(id, i));
                strides.push(stride(id, i));
            }
            match datatype {
                DataType::Int8 => ForeignNDView::Int8(NDView {
                    strides,
                    dims,
                    offset,
                    data: core::slice::from_raw_parts_mut(ptr as *mut i8, len),
                }),
                DataType::Uint8 => ForeignNDView::Uint8(NDView {
                    strides,
                    dims,
                    offset,
                    data: core::slice::from_raw_parts_mut(ptr as *mut u8, len),
                }),
                DataType::Uint8Clamped => ForeignNDView::Uint8Clamped(NDView {
                    strides,
                    dims,
                    offset,
                    data: core::slice::from_raw_parts_mut(ptr as *mut ClampedU8, len),
                }),
                DataType::Int16 => ForeignNDView::Int16(NDView {
                    strides,
                    dims,
                    offset,
                    data: core::slice::from_raw_parts_mut(ptr as *mut i16, len),
                }),
                DataType::Uint16 => ForeignNDView::Uint16(NDView {
                    strides,
                    dims,
                    offset,
                    data: core::slice::from_raw_parts_mut(ptr as *mut u16, len),
                }),
                DataType::Int32 => ForeignNDView::Int32(NDView {
                    strides,
                    dims,
                    offset,
                    data: core::slice::from_raw_parts_mut(ptr as *mut i32, len),
                }),
                DataType::Uint32 => ForeignNDView::Uint32(NDView {
                    strides,
                    dims,
                    offset,
                    data: core::slice::from_raw_parts_mut(ptr as *mut u32, len),
                }),
                DataType::Float32 => ForeignNDView::Float32(NDView {
                    strides,
                    dims,
                    offset,
                    data: core::slice::from_raw_parts_mut(ptr as *mut f32, len),
                }),
                DataType::Float64 => ForeignNDView::Float64(NDView {
                    strides,
                    dims,
                    offset,
                    data: core::slice::from_raw_parts_mut(ptr as *mut f64, len),
                }),
                DataType::Uint64 => ForeignNDView::Uint64(NDView {
                    strides,
                    dims,
                    offset,
                    data: core::slice::from_raw_parts_mut(ptr as *mut u64, len),
                }),
                DataType::Int64 => ForeignNDView::Int64(NDView {
                    strides,
                    dims,
                    offset,
                    data: core::slice::from_raw_parts_mut(ptr as *mut i64, len),
                }),
                DataType::Bool => ForeignNDView::Bool(NDView {
                    strides,
                    dims,
                    offset,
                    data: Bitset::new(ptr as *mut u32, len, bufoff(id)),
                }),
                #[allow(unreachable_patterns)]
                _ => todo!(),
            }
        }
    }

    pub fn export(&self) -> i32 {
        let (dtype, dims, strides, offset, ptr, len, boff) = match self {
            ForeignNDView::Int8(ndview) => (
                DataType::Int8,
                &ndview.dims,
                ndview.strides.as_ptr(),
                ndview.offset,
                ndview.data.as_ptr() as *mut c_void,
                ndview.data.len(),
                0
            ),
            ForeignNDView::Uint8(ndview) => (
                DataType::Uint8,
                &ndview.dims,
                ndview.strides.as_ptr(),
                ndview.offset,
                ndview.data.as_ptr() as *mut c_void,
                ndview.data.len(),
                0
            ),
            ForeignNDView::Uint8Clamped(ndview) => (
                DataType::Uint8Clamped,
                &ndview.dims,
                ndview.strides.as_ptr(),
                ndview.offset,
                ndview.data.as_ptr() as *mut c_void,
                ndview.data.len(),
                0
            ),
            ForeignNDView::Int16(ndview) => (
                DataType::Int16,
                &ndview.dims,
                ndview.strides.as_ptr(),
                ndview.offset,
                ndview.data.as_ptr() as *mut c_void,
                ndview.data.len(),
                0
            ),
            ForeignNDView::Uint16(ndview) => (
                DataType::Uint16,
                &ndview.dims,
                ndview.strides.as_ptr(),
                ndview.offset,
                ndview.data.as_ptr() as *mut c_void,
                ndview.data.len(),
                0
            ),
            ForeignNDView::Int32(ndview) => (
                DataType::Int32,
                &ndview.dims,
                ndview.strides.as_ptr(),
                ndview.offset,
                ndview.data.as_ptr() as *mut c_void,
                ndview.data.len(),
                0
            ),
            ForeignNDView::Uint32(ndview) => (
                DataType::Uint32,
                &ndview.dims,
                ndview.strides.as_ptr(),
                ndview.offset,
                ndview.data.as_ptr() as *mut c_void,
                ndview.data.len(),
                0
            ),
            ForeignNDView::Float32(ndview) => (
                DataType::Float32,
                &ndview.dims,
                ndview.strides.as_ptr(),
                ndview.offset,
                ndview.data.as_ptr() as *mut c_void,
                ndview.data.len(),
                0
            ),
            ForeignNDView::Float64(ndview) => (
                DataType::Float64,
                &ndview.dims,
                ndview.strides.as_ptr(),
                ndview.offset,
                ndview.data.as_ptr() as *mut c_void,
                ndview.data.len(),
                0
            ),
            ForeignNDView::Bool(ndview) => (
                DataType::Bool,
                &ndview.dims,
                ndview.strides.as_ptr(),
                ndview.offset,
                ndview.data.ptr() as *mut c_void,
                ndview.data.len(),
                ndview.data.offset()
            ),
            ForeignNDView::Uint64(ndview) => (
                DataType::Uint64,
                &ndview.dims,
                ndview.strides.as_ptr(),
                ndview.offset,
                ndview.data.as_ptr() as *mut c_void,
                ndview.data.len(),
                0
            ),
            ForeignNDView::Int64(ndview) => (
                DataType::Int64,
                &ndview.dims,
                ndview.strides.as_ptr(),
                ndview.offset,
                ndview.data.as_ptr() as *mut c_void,
                ndview.data.len(),
                0
            ),
        };
        unsafe { register(dtype, dims.len(), dims.as_ptr(), strides, len, ptr, offset, boff) }
    }
}
