fn main() {
    // Set RUNPATH so the dynamic linker can find libtorch relative to the binary.
    // This handles tarball installs where libtorch is placed in ../lib/ next to the binary.
    // For binstall/cargo-install, the launcher sets LD_LIBRARY_PATH before exec.
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");

    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../lib");
}
