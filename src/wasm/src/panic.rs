#[panic_handler]
unsafe fn panic(_: &core::panic::PanicInfo) -> ! {
    core::intrinsics::abort()
}