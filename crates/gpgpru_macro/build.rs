use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

fn get_cache_dir(out_dir: &Path) -> PathBuf {
    let target_dir = env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| out_dir.ancestors().nth(4).map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("target"));

    let cache_dir = target_dir.join("gpgpru_cache");
    let _ = fs::create_dir_all(&cache_dir);
    cache_dir
}

fn find_system_slang_root() -> Option<PathBuf> {
    // 1. Check env variables
    for key in ["SLANG_DIR", "SLANG_ROOT", "VULKAN_SDK"] {
        if let Ok(val) = env::var(key) {
            let p = PathBuf::from(val);
            if p.join("include/slang.h").exists() {
                return Some(p);
            }
        }
    }

    // 2. Search slangc from PATH
    if let Some(path_var) = env::var_os("PATH") {
        let exe_name = if cfg!(windows) { "slangc.exe" } else { "slangc" };
        for dir in env::split_paths(&path_var) {
            let candidate = dir.join(exe_name);
            if candidate.is_file() {
                let real_exe = fs::canonicalize(&candidate).unwrap_or(candidate);
                if let Some(root) = real_exe.parent().and_then(|bin| bin.parent()) {
                    if root.join("include/slang.h").exists() {
                        return Some(root.to_path_buf());
                    }
                }
            }
        }
    }

    // 3. Unix path check
    if cfg!(unix) && !cfg!(target_os = "macos") {
        let default_opt = PathBuf::from("/opt/shader-slang-bin");
        if default_opt.join("include/slang.h").exists() {
            return Some(default_opt);
        }
    }

    None
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(docsrs)");
    println!("cargo:rerun-if-env-changed=SLANG_DIR");
    println!("cargo:rerun-if-env-changed=SLANG_ROOT");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-env-changed=PATH");

    if env::var("DOCS_RS").is_ok() {
        println!("cargo:rustc-cfg=docsrs");
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cpp_source_dir = manifest_dir.join("shadertool-cpp");

    let log_path = get_cache_dir(&out_dir).join("build.log");
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .ok();

    macro_rules! build_log {
        ($($arg:tt)*) => {
            if let Some(ref mut f) = log_file {
                let _ = writeln!(f, $($arg)*);
            }
        };
    }

    build_log!("=== build.rs start ===");
    build_log!("OUT_DIR: {}", out_dir.display());

    let system_slang = find_system_slang_root();
    build_log!("Detected system Slang root: {:?}", system_slang);

    // 1. CMake Build
    let mut config = cmake::Config::new(&cpp_source_dir);
    config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON");

    if let Some(ref slang_root) = system_slang {
        config.define("SLANG_ROOT", slang_root);
    }

    let dst = config.build();
    let dst_lib = dst.join("lib");
    let dst_lib64 = dst.join("lib64");
    build_log!("CMake install dst: {}", dst.display());

    // 2. Link C++ static library & SPIRV-Tools
    println!("cargo:rustc-link-search=native={}", dst_lib.display());
    println!("cargo:rustc-link-search=native={}", dst_lib64.display());
    println!("cargo:rustc-link-lib=static=shadertool");
    println!("cargo:rustc-link-lib=dylib=SPIRV-Tools-opt");
    println!("cargo:rustc-link-lib=dylib=SPIRV-Tools");

    // 3. Platform-specific dependencies
    let target = env::var("TARGET").unwrap();
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=dylib=c++");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=dylib=stdc++");
        println!("cargo:rustc-link-lib=dylib=pthread");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=m");
    } else if target.contains("windows") {
        println!("cargo:rustc-link-lib=dylib=user32");
        println!("cargo:rustc-link-lib=dylib=shell32");
        println!("cargo:rustc-link-lib=dylib=advapi32");
        println!("cargo:rustc-link-lib=dylib=ole32");
    }

    // 4. Resolve Slang library directory (system slang -> then search for releases)
    let slang_root = system_slang.unwrap_or_else(|| out_dir.join("build/_deps/slang_release-src"));
    let slang_lib_dir = if slang_root.join("lib64").exists() && !slang_root.join("lib").exists() {
        slang_root.join("lib64")
    } else {
        slang_root.join("lib")
    };

    build_log!(
        "Resolved slang_lib_dir: {} (exists: {})",
        slang_lib_dir.display(),
        slang_lib_dir.exists()
    );

    if slang_lib_dir.exists() {
        println!("cargo:rustc-link-search=native={}", slang_lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=slang");

        if target.contains("linux") {
            println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dst_lib.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", slang_lib_dir.display());
        } else if target.contains("apple") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dst_lib.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", slang_lib_dir.display());
        }

        if target.contains("windows") {
            let slang_bin_dir = slang_root.join("bin");
            let slang_dll = if slang_lib_dir.join("slang.dll").exists() {
                slang_lib_dir.join("slang.dll")
            } else {
                slang_bin_dir.join("slang.dll")
            };

            build_log!("Windows slang.dll path: {} (exists: {})", slang_dll.display(), slang_dll.exists());

            if slang_dll.exists() {
                if let Some(target_dir) = out_dir.ancestors().nth(3) {
                    let res1 = fs::copy(&slang_dll, target_dir.join("slang.dll"));
                    let res2 = fs::copy(&slang_dll, target_dir.join("deps").join("slang.dll"));
                    build_log!("Copied slang.dll to target: {:?}, deps: {:?}", res1.is_ok(), res2.is_ok());
                }
            }
        }
    } else {
        build_log!("WARNING: slang_lib_dir does not exist! Linking will likely fail.");
    }

    // 5. Trigger rebuild on C++ source changes
    println!("cargo:rerun-if-changed={}/CMakeLists.txt", cpp_source_dir.display());
    println!("cargo:rerun-if-changed={}/src", cpp_source_dir.display());
    println!("cargo:rerun-if-changed={}/include", cpp_source_dir.display());
    build_log!("=== build.rs end ===\n");
}
