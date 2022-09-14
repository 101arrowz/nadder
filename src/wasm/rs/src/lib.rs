#![no_std]
#![feature(core_intrinsics)]
#![feature(alloc_error_handler)]
mod js;
mod setup;

use core::alloc::{GlobalAlloc, Layout};
extern crate alloc;
use alloc::{vec::Vec, string::String, format};
use js::*;
use setup::*;

#[no_mangle]
pub extern "C" fn malloc(size: usize, align: usize) -> *mut u8 {
    let layout = Layout::from_size_align(size, align).unwrap();
    unsafe { ALLOC.alloc(layout) }
}

#[no_mangle]
pub extern "C" fn free(ptr: *mut u8, size: usize, align: usize) {
    let layout = Layout::from_size_align(size, align).unwrap();
    unsafe { ALLOC.dealloc(ptr, layout) }
}

#[no_mangle]
pub unsafe extern "C" fn add(a: i32, b: i32) -> i32 {
    let (a, b) = (ForeignNDView::import(a), ForeignNDView::import(b));
    
    match (a, b) {
        (ForeignNDView::Int64(mut a), ForeignNDView::Int64(mut b)) => {
            let size = a.dims.iter().copied().fold(1, |a, b| a * b) as usize;
            let dims = a.dims.clone();
            let strides = a
                .dims
                .iter()
                .copied()
                .scan(size as isize, |size, x| {
                    *size /= x;
                    Some(*size)
                })
                .collect::<Vec<_>>();
            let mut c = NDView {
                strides,
                dims,
                offset: a.offset,
                data: core::slice::from_raw_parts_mut(
                    ALLOC.alloc(Layout::from_size_align_unchecked(size * 8, 8)) as *mut i64,
                    size,
                ),
            };
            fn add_inner(
                a: &mut NDView<&mut [i64]>,
                b: &mut NDView<&mut [i64]>,
                c: &mut NDView<&mut [i64]>,
                dim: usize,
                mut ai: isize,
                mut bi: isize,
                mut ci: isize,
            ) {
                if dim == a.dims.len() {
                    c.data.set(
                        ci as usize,
                        a.data.get(ai as usize) + b.data.get(bi as usize),
                    );
                } else {
                    for _ in 0..a.dims[dim] {
                        add_inner(a, b, c, dim + 1, ai, bi, ci);
                        ai += a.strides[dim];
                        bi += b.strides[dim];
                        ci += c.strides[dim];
                    }
                }
            }
            let ao = a.offset as isize;
            let bo = b.offset as isize;
            let co = c.offset as isize;
            add_inner(&mut a, &mut b, &mut c, 0, ao, bo, co);
            ForeignNDView::export(&ForeignNDView::Int64(c))
        }
        _ => todo!(),
    }
}



#[no_mangle]
pub unsafe extern "C" fn mul(a: i32, b: i32) -> i32 {
    let (a, b) = (ForeignNDView::import(a), ForeignNDView::import(b));
    
    match (a, b) {
        (ForeignNDView::Int64(mut a), ForeignNDView::Int64(mut b)) => {
            let size = a.dims.iter().copied().fold(1, |a, b| a * b) as usize;
            let dims = a.dims.clone();
            let strides = a
                .dims
                .iter()
                .copied()
                .scan(size as isize, |size, x| {
                    *size /= x;
                    Some(*size)
                })
                .collect::<Vec<_>>();
            let mut c = NDView {
                strides,
                dims,
                offset: a.offset,
                data: core::slice::from_raw_parts_mut(
                    ALLOC.alloc(Layout::from_size_align_unchecked(size * 8, 8)) as *mut i64,
                    size,
                ),
            };
            fn add_inner(
                a: &mut NDView<&mut [i64]>,
                b: &mut NDView<&mut [i64]>,
                c: &mut NDView<&mut [i64]>,
                dim: usize,
                mut ai: isize,
                mut bi: isize,
                mut ci: isize,
            ) {
                if dim == a.dims.len() {
                    c.data.set(
                        ci as usize,
                        a.data.get(ai as usize) * b.data.get(bi as usize),
                    );
                } else {
                    for _ in 0..a.dims[dim] {
                        add_inner(a, b, c, dim + 1, ai, bi, ci);
                        ai += a.strides[dim];
                        bi += b.strides[dim];
                        ci += c.strides[dim];
                    }
                }
            }
            let ao = a.offset as isize;
            let bo = b.offset as isize;
            let co = c.offset as isize;
            add_inner(&mut a, &mut b, &mut c, 0, ao, bo, co);
            ForeignNDView::export(&ForeignNDView::Int64(c))
        }
        _ => todo!(),
    }
}
