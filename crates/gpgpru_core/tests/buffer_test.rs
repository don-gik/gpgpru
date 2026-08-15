// tests/buffer_test.rs

mod common;

use gpgpru_core::create_buffer;

#[test]
fn test_public_buffer_create() {
    let device = common::setup_device();
    let data = vec![1.0f32, 2.0, 3.0, 4.0];

    let gpu_buf = create_buffer(&data)
        .with(&device)
        .expect("Failed Creating Buffer");

    assert_eq!(gpu_buf.size, 16);
}
