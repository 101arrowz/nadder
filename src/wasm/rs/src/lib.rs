#![no_std]
#![feature(core_intrinsics)]
#![feature(alloc_error_handler)]
mod js;
mod setup;

use core::{alloc::{GlobalAlloc, Layout}, ops::{Add, Sub, Mul}};
use paste::paste;
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

macro_rules! inner_ndview {
    ($val:expr, $name:ident => $exec:block) => {
        match &mut $val {
            ForeignNDView::Int8($name) => $exec,
            ForeignNDView::Uint8($name) => $exec,
            ForeignNDView::Uint8Clamped($name) => $exec,
            ForeignNDView::Int16($name) => $exec,
            ForeignNDView::Uint16($name) => $exec,
            ForeignNDView::Int32($name) => $exec,
            ForeignNDView::Uint32($name) => $exec,
            ForeignNDView::Float32($name) => $exec,
            ForeignNDView::Float64($name) => $exec,
            ForeignNDView::Bool($name) => $exec,
            ForeignNDView::Int64($name) => $exec,
            ForeignNDView::Uint64($name) => $exec,
            #[allow(unreachable_patterns)]
            _ => unimplemented!()
        }
    }
}

macro_rules! expand_ndview {
    ($name:ident <- $val:expr => $exec:block) => {
        inner_ndview!($val, $name => $exec)
    };
    ($n0:ident <- $v0:expr, $($name:ident <- $val:expr),+ => $exec:block) => {
        inner_ndview!($v0, $n0 => {
            expand_ndview!($($name <- $val),+ => $exec)
        })
    };
}

macro_rules! get_first {
    ($start:ident $(,$next:ident),*) => {
        $start
    };
}

macro_rules! ufunc {
    ($name:ident, ($($gen:ident : $bound:ty),+), ($($arg:ident : $gena:ident),+), ($($out:ident : $geno:ident),+), $core:block) => {
        paste! {
            #[no_mangle]
            pub extern "C" fn $name($([<$arg _id>]: i32),+, $([<$out _id>]: i32),+, where_id: i32) {
                $(let mut [<$arg _view>] = ForeignNDView::import([<$arg _id>]);)+
                $(let mut [<$out _view>] = ForeignNDView::import([<$out _id>]);)+
                let wh = if where_id == 0 {
                    None
                } else {
                    if let ForeignNDView::Bool(val) = ForeignNDView::import(where_id) {
                        Some(val)
                    } else {
                        panic!("invalid where")
                    }
                };
                fn ufunc_impl<$($gen : $bound),+>(
                    $([<$arg _view>]: &NDView<impl Array<Item = impl Cast<$gena>>>),+,
                    $([<$out _view>]: &mut NDView<impl Array<Item = $geno>>),+,
                    wh: Option<&NDView<Bitset>>
                ) {
                    if wh.is_some() { todo!(); }
                    unsafe fn ufunc_impl_inner<$($gen : $bound),+>(
                        $([<$arg _view>]: &NDView<impl Array<Item = impl Cast<$gena>>>),+,
                        $([<$out _view>]: &mut NDView<impl Array<Item = $geno>>),+,
                        dim: usize,
                        dims: &[isize],
                        $(mut [<$arg i>]: isize),+,
                        $(mut [<$out i>]: isize),+
                    ) {
                        if dim == dims.len() {
                            $(let $arg: $gena = [<$arg _view>].data.get([<$arg i>] as usize).cast();)+
                            $(let $out: $geno;)+
                            $core;
                            $([<$out _view>].data.set([<$out i>] as usize, $out);)+
                            //ufunc_impl_inner_fast($([<$arg _view>]),+, $([<$out _view>]),+, dim, dims, $([<$arg i>]),+, $([<$out i>]),+);
                        } else {
                            $(let [<$arg inc>] = *[<$arg _view>].strides.get_unchecked(dim);)+
                            $(let [<$out inc>] = *[<$out _view>].strides.get_unchecked(dim);)+
                            let dimsize = *dims.get_unchecked(dim);
                            for _ in 0..dimsize {
                                ufunc_impl_inner($([<$arg _view>]),+, $([<$out _view>]),+, dim + 1, dims, $([<$arg i>]),+, $([<$out i>]),+);
                                $([<$arg i>] += [<$arg inc>];)+
                                $([<$out i>] += [<$out inc>];)+
                            }
                        }
                    }
                    unsafe fn ufunc_impl_inner_fast<$($gen : $bound),+>(
                        $([<$arg _view>]: &NDView<impl Array<Item = impl Cast<$gena>>>),+,
                        $([<$out _view>]: &mut NDView<impl Array<Item = $geno>>),+,
                        dim: usize,
                        dims: &[isize],
                        $(mut [<$arg i>]: isize),+,
                        $(mut [<$out i>]: isize),+
                    ) {
                        let left = dims.len() - dim;
                        $(let mut [<$arg si>] = *[<$arg _view>].strides.get_unchecked(0);)+
                        $(let mut [<$out si>] = *[<$out _view>].strides.get_unchecked(0);)+
                        let di = *dims.get_unchecked(0);
                        if left > 1 {
                            $(let mut [<$arg sj>] = *[<$arg _view>].strides.get_unchecked(1);)+
                            $(let mut [<$out sj>] = *[<$out _view>].strides.get_unchecked(1);)+
                            let dj = *dims.get_unchecked(1);
                            $([<$arg si>] -= (dj as isize) * [<$arg sj>];)+
                            $([<$out si>] -= (dj as isize) * [<$out sj>];)+
                            if left > 2 {
                                $(let mut [<$arg sk>] = *[<$arg _view>].strides.get_unchecked(2);)+
                                $(let mut [<$out sk>] = *[<$out _view>].strides.get_unchecked(2);)+
                                let dk = *dims.get_unchecked(2);
                                $([<$arg sj>] -= (dk as isize) * [<$arg sk>];)+
                                $([<$out sj>] -= (dk as isize) * [<$out sk>];)+
                                if left > 3 {
                                    $(let [<$arg sl>] = *[<$arg _view>].strides.get_unchecked(3);)+
                                    $(let [<$out sl>] = *[<$out _view>].strides.get_unchecked(3);)+
                                    let dl = *dims.get_unchecked(3);
                                    $([<$arg sk>] -= (dl as isize) * [<$arg sl>];)+
                                    $([<$out sk>] -= (dl as isize) * [<$out sl>];)+
                                    for _ in 0..di {
                                        for _ in 0..dj {
                                            for _ in 0..dk {
                                                for _ in 0..dl {
                                                    $(let $arg: $gena = [<$arg _view>].data.get([<$arg i>] as usize).cast();)+
                                                    $(let $out: $geno;)+
                                                    $core;
                                                    $([<$out _view>].data.set([<$out i>] as usize, $out);)+
                                                    $([<$arg i>] += [<$arg sl>];)+
                                                    $([<$out i>] += [<$out sl>];)+
                                                }
                                                $([<$arg i>] += [<$arg sk>];)+
                                                $([<$out i>] += [<$out sk>];)+
                                            }
                                            $([<$arg i>] += [<$arg sj>];)+
                                            $([<$out i>] += [<$out sj>];)+
                                        }
                                        $([<$arg i>] += [<$arg si>];)+
                                        $([<$out i>] += [<$out si>];)+
                                    }
                                } else {
                                    for _ in 0..di {
                                        for _ in 0..dj {
                                            for _ in 0..dk {
                                                $(let $arg: $gena = [<$arg _view>].data.get([<$arg i>] as usize).cast();)+
                                                $(let $out: $geno;)+
                                                $core;
                                                $([<$out _view>].data.set([<$out i>] as usize, $out);)+
                                                $([<$arg i>] += [<$arg sk>];)+
                                                $([<$out i>] += [<$out sk>];)+
                                            }
                                            $([<$arg i>] += [<$arg sj>];)+
                                            $([<$out i>] += [<$out sj>];)+
                                        }
                                        $([<$arg i>] += [<$arg si>];)+
                                        $([<$out i>] += [<$out si>];)+
                                    }
                                }
                            } else {
                                for _ in 0..di {
                                    for _ in 0..dj {
                                        $(let $arg: $gena = [<$arg _view>].data.get([<$arg i>] as usize).cast();)+
                                        $(let $out: $geno;)+
                                        $core;
                                        $([<$out _view>].data.set([<$out i>] as usize, $out);)+
                                        $([<$arg i>] += [<$arg sj>];)+
                                        $([<$out i>] += [<$out sj>];)+
                                    }
                                    $([<$arg i>] += [<$arg si>];)+
                                    $([<$out i>] += [<$out si>];)+
                                }
                            }
                            panic!();
                        } else {
                            for _ in 0..di {
                                $(let $arg: $gena = [<$arg _view>].data.get([<$arg i>] as usize).cast();)+
                                $(let $out: $geno;)+
                                $core;
                                $([<$out _view>].data.set([<$out i>] as usize, $out);)+
                                $([<$arg i>] += [<$arg si>];)+
                                $([<$out i>] += [<$out si>];)+
                            }
                        }
                    }

                    $(let [<$arg i>] = [<$arg _view>].offset as isize;)+
                    $(let [<$out i>] = [<$out _view>].offset as isize;)+
                    let dims = &get_first!($([<$arg _view>]),+).dims;
                    unsafe {
                        if dims.len() > 4 {
                            ufunc_impl_inner($([<$arg _view>]),+, $([<$out _view>]),+, 0, dims, $([<$arg i>]),+, $([<$out i>]),+);
                        } else {
                            ufunc_impl_inner_fast($([<$arg _view>]),+, $([<$out _view>]),+, 0, dims, $([<$arg i>]),+, $([<$out i>]),+);
                        }
                    }
                }
                expand_ndview!(
                    $([<$arg _view_inner>] <- [<$arg _view>]),+,
                    $([<$out _view_inner>] <- [<$out _view>]),+
                    => {
                    ufunc_impl($([<$arg _view_inner>]),+, $([<$out _view_inner>]),+, wh.as_ref())
                });
            }
        }
    }
}

ufunc!(sub, (T: Sub<Output = T>), (a: T, b: T), (c: T), {
    c = a - b;
});

ufunc!(add, (T: Add<Output = T>), (a: T, b: T), (c: T), {
    c = a + b;
});

ufunc!(mul, (T: Mul<Output = T>), (a: T, b: T), (c: T), {
    c = a * b;
});