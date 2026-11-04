// examples/basic_init.rs

// use gpgpru::prelude::*;
use gpgpru::config::{DeviceConfig, DeviceSelector};
use gpgpru::{Device, GpuSlice, Result, create_buffer, kernel, read_buffer};

use std::time::Instant;

#[kernel("examples/basic_init.slang", "scale")]
fn scale(_gpu_data_ptr: &mut GpuSlice<f32>, _gpu_data_len: u32, val: f32) {}

fn main() -> Result<()> {
    println!("Initializing gpgpru device");

    let config = DeviceConfig::default().with_selector(DeviceSelector::Default);
    let device = Device::with(config)?;

    println!("Success, loaded vulkan device.");

    let size = 524288;

    let mut host_data = vec![1.0f32; size];
    let mut buffer = create_buffer(&host_data).with(&device)?;

    // dbg!(&buffer.bda());

    println!("Buffer created.");

    let groups = (size / 128, 1, 1);
    let buffer_len = buffer.size() as u32;

    let start = Instant::now();
    for _i in 0..5 {
        scale(&mut buffer, buffer_len, 1.0f32).queue(&device, groups)?;
    }
    device.wait_timeline(device.dispatch()?)?;
    let dur = (Instant::now() - start) / 5;

    let flops = (size as f64 * 100000.0 * 3.0) / dur.as_secs_f64();

    println!("Kernel run succeeded in {:?}", dur);
    println!("{:.2} GFLOPS", flops / 1e9);

    read_buffer(&buffer).into(&mut host_data).with(&device)?;

    println!("{:?}", &host_data[0..3]);

    println!("Drop device testing");

    Ok(())
}
