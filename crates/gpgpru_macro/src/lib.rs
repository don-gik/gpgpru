mod ffi;
mod kernel;

use ffi::*;
use kernel::*;

use gpgpru_core::SlangProfile;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;

/// A macro for linking slang shader kernel.
///
/// It reads shader, and use reflection data to correctly validate + link the values.
/// In default, automatically uses Push Descriptor and Push Constants for speed. (Currently not configurable.)
///
/// # Examples
///
/// ```no_run
/// use gpgpru::{kernel, GpuSlice};
///
/// #[kernel("examples/gemm.slang", "gemm", profiles = [Spirv1_6], high)]
/// fn gemm(
///     A: &GpuSlice<f32>,
///     B: &GpuSlice<f32>,
///     C: &mut GpuSlice<f32>,
///     M: u32,
///     N: u32,
///     K: u32,
///     alpha: f32,
///     beta: f32,
/// ) {
/// }
/// ```
///
/// # Params
///
/// - file path: Can be used without any notation - `#[kernel(add.slang)]`.
/// - entry point: Defaults to `main`. Macro will report when specified entry point cannot be found.
/// - profiles: Profiles that slangc must fetch. Following is supported - `Spirv1_5`, `Spirv1_6`, `SpvGroupNonUniform`, `SpvGroupNonUniformVote`, `SpvGroupNonUniformArithmetic`, `SpvGroupNonUniformBallot`, `SpvGroupNonUniformShuffle`, `SpvGroupNonUniformShuffleRelative`, `SpvCooperativeMatrixKHR`, `SpvFloat16`, `SpvInt64`.
///   slangc automatically processes profiles even when not specified, so don't worry.
/// - high precision: Option for spirv-opt. Default is none, to specify just enter `high` or `high_precision`. **This macro requires user to install Spirv-Opt package otherwise build.rs will panic.**
///
/// # Notes (Must Read)
///
/// The macro will panic if the Vulkan SDK (including `spirv-opt`) is not available.
///
/// It will automatically fetch `slang.so` if `slangc` is not globally specified.
///
/// **If the macro cannot find the Vulkan SDK or `slangc`, please add their paths to `CMAKE_PREFIX_PATH`.**
///
/// For internal implementation details on shader compilation, see the [shadertool-cpp](https://github.com/don-gik/shadertool-cpp) repository.
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, body: TokenStream) -> TokenStream {
    match kernel_impl(attr.into(), body.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// A macro for ignoring `kernel`'s validation pass.
///
/// **This may result in error that is impossible to fix.**
/// Macro will generate code that copies byte, instead of casting.
/// For most cases, please do not use this macro to avoid unstable crash.
///
/// # Example
///
/// ```no_run
/// use gpgpru::{kernel, GpuSlice};
///
/// #[kernel("examples/gemm.slang", "gemm", profiles = [Spirv1_6], high)]
/// fn gemm(
///     A: &GpuSlice<f32>,
///     B: &GpuSlice<f32>,
///     C: &mut GpuSlice<f32>,
///     #[raw]
///     M: f32,
///     N: u32,
///     K: u32,
///     alpha: f32,
///     beta: f32,
/// ) {
/// }
/// ```
#[proc_macro_attribute]
pub fn raw(_attr: TokenStream, body: TokenStream) -> TokenStream {
    body
}
