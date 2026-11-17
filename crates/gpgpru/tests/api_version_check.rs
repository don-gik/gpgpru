use std::process::Command;

#[test]
fn check_public_types_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<gpgpru_core::Device>();
}

#[test]
fn check_crates_io_semver_compatibility() {
    let installed = Command::new("cargo")
        .args(["semver-checks", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !installed {
        eprintln!("[warning] cargo-semver-checks not installed, skipping semver test.");
        return;
    }

    let status = Command::new("cargo")
        .args(["semver-checks", "-p", env!("CARGO_PKG_NAME")])
        .status()
        .expect("Failed to execute cargo semver-checks");

    assert!(
        status.success(),
        "SemVer check failed for {}!",
        env!("CARGO_PKG_NAME")
    );
}
