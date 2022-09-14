extern crate wee_alloc;

#[panic_handler]
unsafe fn panic(_: &core::panic::PanicInfo) -> ! {
    core::intrinsics::abort()
}

#[alloc_error_handler]
unsafe fn fail(_: core::alloc::Layout) -> ! {
    core::intrinsics::abort()
}

#[global_allocator]
pub static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
