use gpgpru::config::{DeviceConfig, DeviceSelector};
use gpgpru::utils::GpuTimer;
use gpgpru::{Device, GpuSlice, Result, create_buffer, kernel, read_buffer};

#[kernel("examples/gemm.slang", "gemm")]
fn gemm(
    A: &GpuSlice<f32>,
    B: &GpuSlice<f32>,
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

    let buf_a = create_buffer(&host_a).with(&device)?;
    let buf_b = create_buffer(&host_b).with(&device)?;
    let mut buf_c = create_buffer(&host_c).with(&device)?;

    // Block dimensions for 64x64 tiling
    let (bm, bn) = (64, 64);
    let groups = ((n + bn - 1) / bn, (m + bm - 1) / bm, 1);

    // Warm-up pass
    for _ in 0..10 {
        gemm(&buf_a, &buf_b, &mut buf_c, m, n, k, alpha, beta).queue(&device, groups)?;
        device.wait_timeline(device.dispatch()?)?;
    }

    // Benchmark
    let timer = GpuTimer::new(&device)?;

    let s0 = timer.stamp(&device);
    gemm(&buf_a, &buf_b, &mut buf_c, m, n, k, alpha, beta).queue(&device, groups)?;
    let s1 = timer.stamp(&device);
    device.wait_timeline(device.dispatch()?)?;

    // Compute performance
    let t0 = timer.fetch(&device, s0)?;
    let t1 = timer.fetch(&device, s1)?;
    let avg_dur = t1 - t0;
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
