#![no_std]
#![feature(core_intrinsics)]
#![feature(alloc_error_handler)]
mod js;
mod setup;

use core::alloc::{GlobalAlloc, Layout};
extern crate alloc;
use alloc::vec::Vec;
use js::*;
use setup::*;

#[no_mangle]
pub unsafe extern "C" fn add(a: i32, b: i32) -> i32 {
    let (a, b) = (ForeignNDView::import(a), ForeignNDView::import(b));

    match (a, b) {
        (ForeignNDView::Int32(mut a), ForeignNDView::Int32(mut b)) => {
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
                    ALLOC.alloc(Layout::from_size_align_unchecked(size, 4)) as *mut i32,
                    size,
                ),
            };
            fn add_inner(
                a: &mut NDView<&mut [i32]>,
                b: &mut NDView<&mut [i32]>,
                c: &mut NDView<&mut [i32]>,
                dim: usize,
                mut ai: isize,
                mut bi: isize,
                mut ci: isize,
            ) {
                if dim == 0 {
                    c.data.set(
                        ci as usize,
                        a.data.get(ai as usize) + b.data.get(bi as usize),
                    );
                } else {
                    for _ in 0..a.dims[0] {
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
            ForeignNDView::export(&ForeignNDView::Int32(c))
        }
        _ => todo!(),
    }
}
