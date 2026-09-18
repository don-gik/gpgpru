// examples/hazard.rs

use gpgpru::prelude::*;
use gpgpru::{Device, Result, compute, create_buffer, read_buffer};

#[compute(threads(256, 1, 1))]
fn fill(data: &mut [f32], val: f32, global_id_x: GlobalIdX) {
    let idx = global_id_x;
    if idx < data.len() {
        data[idx] = val;
    }
}

#[compute(threads(256, 1, 1))]
fn multiply_copy(src: &[f32], dst: &mut [f32], multiplier: f32, global_id_x: GlobalIdX) {
    let idx = global_id_x;
    if idx < src.len() && idx < dst.len() {
        dst[idx] = src[idx] * multiplier;
    }
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

    // Step 1: BufA Fill 3.0
    fill(&mut buf_a, 3.0).queue(&device, (group_count, 1, 1))?;

    // Step 2: BufA -> BufB (Read BufA, Write BufB = 12.0)
    multiply_copy(&buf_a, &mut buf_b, 4.0).queue(&device, (group_count, 1, 1))?;

    // Step 3: BufA Fill 100.0 (Write BufA = 100.0)
    fill(&mut buf_a, 100.0).queue(&device, (group_count, 1, 1))?;

    // Step 4: BufB -> BufC (Read BufB, Write BufC = 24.0)
    multiply_copy(&buf_b, &mut buf_c, 2.0).queue(&device, (group_count, 1, 1))?;

    // Step 5: BufB -> 500.0 (Write BufB = 500.0)
    fill(&mut buf_b, 500.0).queue(&device, (group_count, 1, 1))?;

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
