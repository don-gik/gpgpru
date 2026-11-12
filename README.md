# Gpgpru

A simple rust crate for gpgpu programming.
It is lightweight and uses vulkan as backend.

## Benchmark

A basic benchmark, now naively implemented is below:
```bash
cargo run --example gemm -p gpgpru --release
```
It is currently very slow due to not CPU overhead but GPU kernel optimization.

## Troubleshooting

**Disclaimer : gpgpru_macro crate downloads slang shader library and etc. file for compiling and optimization. This may take some time.**

## License
This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.
