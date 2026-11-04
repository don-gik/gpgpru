// examples/hazard.rs

use gpgpru::prelude::*;
use gpgpru::{Device, GpuSlice, Result, create_buffer, kernel, read_buffer};

#[kernel("examples/hazard.slang", "fill")]
pub fn fill(data: &mut GpuSlice<f32>, data_len: u32, val: f32) {}

#[kernel("examples/hazard2.slang", "multiply_copy")]
pub fn multiply_copy(
    src: &GpuSlice<f32>,
    src_len: u32,
    dst: &mut GpuSlice<f32>,
    dst_len: u32,
    multiplier: f32,
) {
}

fn main() -> Result<()> {
    let device = Device::new()?;
    const SIZE: usize = 4096;
    let group_count = SIZE.div_ceil(256) as u32;

    let mut buf_a = create_buffer(&vec![0.0f32; SIZE])
        .on(Shared)
        .with(&device)?;
    let mut buf_b = create_buffer(&vec![0.0f32; SIZE])
        .on(Shared)
        .with(&device)?;
    let mut buf_c = create_buffer(&vec![0.0f32; SIZE])
        .on(Shared)
        .with(&device)?;

    // ------------------------------------------------------------------------
    // Dispatch Pipeline
    // ------------------------------------------------------------------------

    let buf_a_size = buf_a.size() as u32;
    let buf_b_size = buf_b.size() as u32;
    let buf_c_size = buf_c.size() as u32;

    // Step 1: BufA Fill 3.0
    fill(&mut buf_a, buf_a_size, 3.0).queue(&device, (group_count, 1, 1))?;

    // Step 2: BufA -> BufB (Read BufA, Write BufB = 12.0)
    multiply_copy(&buf_a, buf_a_size, &mut buf_b, buf_b_size, 4.0)
        .queue(&device, (group_count, 1, 1))?;

    // Step 3: BufA Fill 100.0 (Write BufA = 100.0)
    fill(&mut buf_a, buf_a_size, 100.0).queue(&device, (group_count, 1, 1))?;

    // Step 4: BufB -> BufC (Read BufB, Write BufC = 24.0)
    multiply_copy(&buf_b, buf_b_size, &mut buf_c, buf_c_size, 2.0)
        .queue(&device, (group_count, 1, 1))?;

    // Step 5: BufB -> 500.0 (Write BufB = 500.0)
    fill(&mut buf_b, buf_b_size as u32, 500.0).queue(&device, (group_count, 1, 1))?;

    device.dispatch()?;

    // ------------------------------------------------------------------------
    // Verification
    // ------------------------------------------------------------------------

    let mut res_a = vec![0.0f32; SIZE];
    let mut res_b = vec![0.0f32; SIZE];
    let mut res_c = vec![0.0f32; SIZE];

    read_buffer(&buf_a).into(&mut res_a).with(&device)?;
    read_buffer(&buf_b).into(&mut res_b).with(&device)?;
    read_buffer(&buf_c).into(&mut res_c).with(&device)?;

    assert_eq!(res_a[0], 100.0, "WAW/WAR on BufA failed");
    assert_eq!(res_b[0], 500.0, "WAW/WAR on BufB failed");
    assert_eq!(res_c[0], 24.0, "RAW/WAR Chain on BufC failed");

    Ok(())
}
