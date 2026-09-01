// examples/basic_init.rs

use gpgpru::prelude::*;
use gpgpru::{Device, Result, compute, create_buffer, read_buffer};

#[compute(threads(64, 1, 1))]
fn scale(data: &mut [f32], factor: f32, global_id_x: GlobalIdX) {
    let idx = global_id_x;
    if idx < data.len() {
        data[idx] *= factor;
    }
}

fn main() -> Result<()> {
    println!("Initializing gpgpru device");

    let device = Device::new()?;

    println!("Success, loaded vulkan device.");

    let mut host_data = vec![1.0f32; 1024];
    let mut buffer = create_buffer(&host_data).with(&device)?;

    println!("Buffer created.");

    let groups = (1024 / 64, 1, 1);
    scale(&mut buffer, 2.0f32).with(&device, groups)?;

    println!("Kernel run succeeded.");

    read_buffer(&buffer).into(&mut host_data).with(&device)?;

    println!("{:?}", host_data);

    println!("Drop device testing");

    Ok(())
}
