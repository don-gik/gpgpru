# Gpgpru

A simple rust crate for gpgpu programming.
It is lightweight and uses vulkan as backend.

## Benchmark

A basic benchmark tuned to Intel Iris Xe is available through : 
```bash
cargo run --example gemm -p gpgpru --release
```
It outputs 850GFLOPS on average, for Intel Iris Xe, which is 42.6% of theoretical performance. May work better on dGPUs.

## Troubleshooting

**Disclaimer : gpgpru_macro crate downloads slang shader library and etc. file for compiling and optimization. This may take some time.**

## License
This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.
