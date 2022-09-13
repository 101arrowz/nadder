extern crate wee_alloc;

#[panic_handler]
unsafe fn panic(_: &core::panic::PanicInfo) -> ! {
    core::intrinsics::abort()
}

#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
