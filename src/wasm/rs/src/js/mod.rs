extern crate alloc;
use core::ffi::c_void;

use alloc::vec::Vec;

mod clamped;
use clamped::*;

mod bitset;
use bitset::*;

mod array;
use array::*;

mod complex;
use complex::*;

pub struct NDView<A: Array> {
    pub strides: Vec<isize>,
    pub dims: Vec<isize>,
    pub data: A
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
    Complex = 512,
    Bool = 1024,
    Int64 = 4096,
    Uint64 = 8192,
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
    Int64(NDView<&'static mut [i64]>),
    Uint64(NDView<&'static mut [u64]>),
    Bool(NDView<Bitset<'static>>),
}

extern {
    fn dtype(id: i32) -> DataType;
    fn ndim(id: i32) -> usize;
    fn dim(id: i32, ind: usize) -> isize;
    fn stride(id: i32, ind: usize) -> isize;
    fn buf(id: i32) -> *mut c_void;
    fn buflen(id: i32) -> usize;
}

impl ForeignNDView {
    pub fn load(id: i32) -> ForeignNDView {
        unsafe {
            let datatype = dtype(id);
            let ptr = buf(id);
            let len = buflen(id);
            let num_dims = ndim(id);
            let dims = Vec::with_capacity(num_dims);
            let strides = Vec::with_capacity(num_dims);
            for i in 0..num_dims {
                dims.push(dim(id, i));
                strides.push(stride(id, i));
            }
            match datatype {
                DataType::Int8 => ForeignNDView::Int8(NDView {
                    strides,
                    dims,
                    data: core::slice::from_raw_parts_mut(ptr as *mut i8, len)
                }),
                DataType::Uint8 => ForeignNDView::Uint8(NDView {
                    strides,
                    dims,
                    data: core::slice::from_raw_parts_mut(ptr as *mut u8, len)
                }),
                DataType::Uint8Clamped => ForeignNDView::Uint8Clamped(NDView {
                    strides,
                    dims,
                    data: core::slice::from_raw_parts_mut(ptr as *mut ClampedU8, len)
                }),
                DataType::Int16 => ForeignNDView::Int16(NDView {
                    strides,
                    dims,
                    data: core::slice::from_raw_parts_mut(ptr as *mut i16, len)
                }),
                DataType::Uint16 => ForeignNDView::Uint16(NDView {
                    strides,
                    dims,
                    data: core::slice::from_raw_parts_mut(ptr as *mut u16, len)
                }),
                DataType::Int32 => ForeignNDView::Int32(NDView {
                    strides,
                    dims,
                    data: core::slice::from_raw_parts_mut(ptr as *mut i32, len)
                }),
                DataType::Uint32 => ForeignNDView::Uint32(NDView {
                    strides,
                    dims,
                    data: core::slice::from_raw_parts_mut(ptr as *mut u32, len)
                }),
                DataType::Float32 => ForeignNDView::Float32(NDView {
                    strides,
                    dims,
                    data: core::slice::from_raw_parts_mut(ptr as *mut f32, len)
                }),
                DataType::Float64 => ForeignNDView::Float64(NDView {
                    strides,
                    dims,
                    data: core::slice::from_raw_parts_mut(ptr as *mut f64, len)
                }),
                DataType::Uint64 => ForeignNDView::Uint64(NDView {
                    strides,
                    dims,
                    data: core::slice::from_raw_parts_mut(ptr as *mut u64, len)
                }),
                DataType::Int64 => ForeignNDView::Int64(NDView {
                    strides,
                    dims,
                    data: core::slice::from_raw_parts_mut(ptr as *mut i64, len)
                }),
                _ => todo!()
            }
        }
    }
}
