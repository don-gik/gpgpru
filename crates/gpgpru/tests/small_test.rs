// // tests/small_test.rs

// use gpgpru::ffi::compute_hlsl;
// use gpgpru::{Device, Result, create_buffer, read_buffer};

// #[compute_hlsl]
// pub fn scale_array(data: &mut GpuBuffer, factor: f32) {
//     r#"
//     RWStructuredBuffer<float> data : register(u0);

//     struct PushConstants {
//         float factor;
//     };

//     [[vk::push_constant]]
//     PushConstants push;

//     [numthreads(64, 1, 1)]
//     void main(uint3 id : SV_DispatchThreadID) {
//         data[id.x] *= push.factor;
//     }
//     "#
// }

// #[test]
// fn test_gpgpu_pipeline_end_to_end() -> Result<()> {
//     // 1. Device Initialization
//     let device = Device::new()?;

//     // 2. Test Data & Buffer
//     let input_data = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];

//     let mut gpu_buf = create_buffer(&input_data).with(&device)?;

//     // 3. HLSL Kernel (scale_array)
//     // 4 Data -> (1, 1, 1) Workgroup
//     let _exec = scale_array(&mut gpu_buf, 3.5f32).with(&device, (1, 1, 1))?;

//     // 4-A. read_buffer().into(&mut host_data).with(&device)
//     let mut host_out = vec![0.0f32; 4];
//     read_buffer(&gpu_buf).into(&mut host_out).with(&device)?;

//     assert_eq!(host_out, vec![3.5f32, 7.0f32, 10.5f32, 14.0f32]);

//     // 4-B. read_buffer().with::<T>(&device) -> Vec
//     let auto_vec: Vec<f32> = read_buffer(&gpu_buf).with(&device)?;
//     assert_eq!(auto_vec, vec![3.5f32, 7.0f32, 10.5f32, 14.0f32]);

//     Ok(())
// }
