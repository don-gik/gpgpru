// examples/basic_init.rs

use gpgpru::prelude::*;
use gpgpru::{Device, Result, compute, create_buffer, read_buffer};

#[compute(threads(256, 1, 1))]
fn scale(data: &mut [f32], factor: f32, global_id_x: GlobalIdX) {
    let idx = global_id_x;
    if idx < data.len() {
        data[idx] *= factor;
    }
}

fn main() -> Result<()> {
    const DATA_SIZE: usize = 1_000_000;
    const THREADS_PER_GROUP: usize = 256;
    const FACTOR: f32 = 2.0;

    println!("Initializing Device");
    let device = Device::new()?;

    let host_data = vec![1.0f32; DATA_SIZE];
    let mut buffer = create_buffer(&host_data).on(Shared).with(&device)?;

    let group_count = (DATA_SIZE.div_ceil(THREADS_PER_GROUP)) as u32;

    println!("Running Kernel");
    scale(&mut buffer, FACTOR).with(&device, (group_count, 1, 1))?;

    let mut result = vec![0.0f32; DATA_SIZE];
    read_buffer(&buffer).into(&mut result).with(&device)?;

    println!("Sample Output [0..3]: {:?}", &result[..3]);
    // assert_eq!(result[0], 2.0);

    Ok(())
}
