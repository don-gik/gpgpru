// examples/basic_init.rs

use anyhow::Result;
use gpgpru::Device;

fn main() -> Result<()> {
    println!("Initializing gpgpru device");

    let _device = Device::new()?;

    println!("Success, loaded vulkan device.");

    println!("Drop device testing");

    Ok(())
}
