fn main() {
    // Compile C Q4 kernel with platform-specific optimizations
    let mut build = cc::Build::new();
    build.file("csrc/q4_dot.c");
    build.opt_level(3);

    // ARM: enable dotprod for vdotq_s32
    #[cfg(target_arch = "aarch64")]
    {
        build.flag("-march=armv8.2-a+dotprod");
    }

    // x86: would add AVX2/AVX-512 flags here
    #[cfg(target_arch = "x86_64")]
    {
        build.flag("-mavx2");
    }

    build.compile("q4_dot");
}
