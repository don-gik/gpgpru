// examples/basic_init.rs

use gpgpru::prelude::*;
use gpgpru::{Device, GpuSlice, Result, create_buffer, kernel, read_buffer};

#[kernel("examples/clean.slang", "scale")]
pub fn scale(data: &mut GpuSlice<f32>, data_len: u32, factor: f32) {}

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
    let buffer_size = buffer.size() as u32;
    scale(&mut buffer, buffer_size, FACTOR).with(&device, (group_count, 1, 1))?;

    let mut result = vec![0.0f32; DATA_SIZE];
    read_buffer(&buffer).into(&mut result).with(&device)?;

    println!("Sample Output [0..3]: {:?}", &result[..3]);
    // assert_eq!(result[0], 2.0);

    Ok(())
}
