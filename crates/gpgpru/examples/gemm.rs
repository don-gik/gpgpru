use gpgpru::config::{DeviceConfig, DeviceSelector};
use gpgpru::{Device, GpuSlice, Result, create_buffer, kernel, read_buffer};
use std::time::Instant;

#[kernel("examples/gemm.slang", "gemm")]
fn gemm(
    A: &mut GpuSlice<f32>,
    B: &mut GpuSlice<f32>,
    C: &mut GpuSlice<f32>,
    M: u32,
    N: u32,
    K: u32,
    alpha: f32,
    beta: f32,
) {
}

fn main() -> Result<()> {
    let config = DeviceConfig::default().with_selector(DeviceSelector::Default);
    let device = Device::with(config)?;
    println!("Vulkan device loaded successfully.");

    // Matrix dimensions (Must be multiples of 4 for float4 vectorization)
    let (m, n, k) = (2048, 2048, 2048);
    let (alpha, beta) = (1.0f32, 0.0f32);

    let host_a = vec![1.0f32; (m * k) as usize];
    let host_b = vec![2.0f32; (k * n) as usize];
    let mut host_c = vec![0.0f32; (m * n) as usize];

    let mut buf_a = create_buffer(&host_a).with(&device)?;
    let mut buf_b = create_buffer(&host_b).with(&device)?;
    let mut buf_c = create_buffer(&host_c).with(&device)?;

    // Block dimensions for 128x64 tiling
    let (bm, bn) = (128, 64);
    let groups = ((n + bn - 1) / bn, (m + bm - 1) / bm, 1);

    // Warm-up pass
    gemm(&mut buf_a, &mut buf_b, &mut buf_c, m, n, k, alpha, beta).queue(&device, groups)?;
    device.wait_timeline(device.dispatch()?)?;

    // Benchmark
    let iterations = 100;
    let start = Instant::now();

    for _ in 0..iterations {
        gemm(&mut buf_a, &mut buf_b, &mut buf_c, m, n, k, alpha, beta).queue(&device, groups)?;
    }
    device.wait_timeline(device.dispatch()?)?;

    // Compute performance
    let avg_dur = start.elapsed() / iterations;
    let ops_per_gemm = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let gflops = (ops_per_gemm / avg_dur.as_secs_f64()) / 1e9;

    println!("Execution time (avg): {:?}", avg_dur);
    println!("Performance: {:.2} GFLOPS", gflops);

    // Verification
    read_buffer(&buf_c).into(&mut host_c).with(&device)?;
    let expected = (k as f32) * 2.0;

    println!("Validation C[0..5]: {:?}", &host_c[0..5]);
    assert_eq!(host_c[0], expected, "GEMM calculation mismatch!");

    Ok(())
}
