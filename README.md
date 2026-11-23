# gpgpru

A lightweight, zero-boilerplate Rust crate for GPGPU compute powered by **Vulkan 1.3+** and **Slang**.

Write your shaders in [Slang](https://shader-slang.com), annotate a Rust function signature with `#[kernel]`, and let compile-time reflection automatically handle push descriptors, push constants, layout validation, and pipeline barrier hazards.

## Features

- **Compile-Time Shader Reflection (`#[kernel]`)**: Automatically compiles `.slang` shaders to SPIR-V and binds `GpuSlice` arguments (`VK_KHR_push_descriptor`) and scalar push constants with compile-time type & offset checks.
- **Automatic Hazard Tracking**: Queue multiple compute kernels and `dma` transfers—`gpgpru` tracks buffer accesses across pages and inserts the required RAW/WAR/WAW pipeline barriers automatically.
- **Safe Async Memory Management**: Dropping a `GpuBuffer` defers VRAM deallocation via an internal timeline-aware garbage collector until the GPU finishes executing all dependent batches.
- **On-Silicon Profiling**: Measure real GPU kernel execution time using hardware query pools (`gpgpru::utils::GpuTimer`).

## Quick Start

```rust
use gpgpru::prelude::*;
use gpgpru::{create_buffer, kernel, read_buffer, Device};

#[kernel("shaders/add.slang", "add")]
fn add(a: &GpuSlice<f32>, b: &GpuSlice<f32>, out: &mut GpuSlice<f32>, len: u32) {}

fn main() -> gpgpru::Result<()> {
    let device = Device::new()?;

    let a = create_buffer(&vec![1.0f32; 1024]).with(&device)?;
    let b = create_buffer(&vec![2.0f32; 1024]).with(&device)?;
    let mut out = create_buffer::<f32>(1024).with(&device)?;

    // Dispatch on the GPU (supports 1D, 2D, or 3D workgroups)
    let ticket = add(&a, &b, &mut out, 1024).with(&device, 4)?;
    device.wait_timeline(ticket)?;

    // Read results back to the CPU
    let mut host_out = vec![0.0f32; 1024];
    read_buffer(&out).into(&mut host_out).with(&device)?;
    Ok(())
}
```

## Benchmark

A basic GEMM benchmark is included in the examples:
```bash
cargo run --example gemm -p gpgpru --release
```
*(Note: Current performance reflects a naive reference kernel rather than CPU dispatch overhead.)*

## Requirements & Troubleshooting

- **Vulkan 1.3+**: Requires a GPU and driver supporting Vulkan 1.3 and `VK_KHR_push_descriptor`.
- **Slang & Vulkan SDK**: The `gpgpru_macro` build script automatically fetches the Slang shader compiler library if `slangc` is not found globally, so the first build may take a little while.
- **Custom SDK Paths**: If CMake cannot locate your Vulkan SDK (including `spirv-opt` when using `high` optimization) or Slang installation, add their root directories to `CMAKE_PREFIX_PATH`:
  ```bash
  export CMAKE_PREFIX_PATH="/path/to/slang:/path/to/vulkan-sdk:\$CMAKE_PREFIX_PATH"
  ```

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.
