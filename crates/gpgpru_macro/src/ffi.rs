pub mod raw {
    use serde::{Deserialize, Serialize};
    use std::ffi::c_char;

    #[repr(u8)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
    pub enum ScalarKind {
        None = 0,
        Void = 1,
        Bool = 2,
        Int32 = 3,
        UInt32 = 4,
        Int64 = 5,
        UInt64 = 6,
        Float16 = 7,
        Float32 = 8,
        Float64 = 9,
        Int8 = 10,
        UInt8 = 11,
        Int16 = 12,
        UInt16 = 13,
        IntPtr = 14,
        UIntPtr = 15,
    }

    #[repr(u8)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
    pub enum MemberKind {
        Primitive = 0,
        Struct = 1,
        Pointer = 2, // BDA
    }

    #[repr(u8)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
    pub enum ResourceKind {
        PushConstant = 0,
        Uniform = 1,
        Texture = 2,
        Sampler = 3,
        StorageTexture = 4,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct CompactMember {
        pub name: *const c_char,
        pub offset: u32,
        pub size: u32,
        pub kind: MemberKind,
        pub scalar_type: ScalarKind,
        pub row_count: u8,
        pub col_count: u8,
        pub array_count: u32,
        pub parent_index: i32,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct CompactResource {
        pub name: *const c_char,
        pub kind: ResourceKind,
        pub set: u32,
        pub binding: u32,
        pub total_size: u32,
        pub members: *const CompactMember,
        pub member_count: u32,
    }

    #[repr(C)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[allow(unused)]
    pub enum ErrorCode {
        Ok = 0,
        FileNotFound,
        SyntaxError,
        LinkError,
        OptimizationFailed,
        InternalError,
        UnknownError = 9999,
    }

    #[repr(C)]
    #[derive(Debug, Clone)]
    pub struct ErrorInfo {
        pub code: ErrorCode,
        pub message: *const c_char,
        pub file_path: *const c_char,
        pub line: i32,
        pub column: i32,
    }

    #[repr(C)]
    #[derive(Debug, Clone)]
    pub struct CompileResult {
        pub success: bool,
        pub error_info: ErrorInfo,
        pub optim_info: ErrorInfo,

        pub code_bytes: *const u8,
        pub code_size: usize,

        pub dep_file_path: *const *const c_char,
        pub dep_file_count: usize,

        pub resources: *const CompactResource,
    }

    #[repr(C)]
    pub struct Context {
        _private: [u8; 0],
    }

    unsafe extern "C" {
        pub fn shadertool_create_context() -> *mut Context;
        pub fn shadertool_destroy_context(ctx: *mut Context);
        pub fn shadertool_compile_shader(
            ctx: *const Context,
            module_name: *const c_char,
            module_path: *const c_char,
            entry_point: *const c_char,
            should_optimize: bool,
            high_prevision: bool,
        ) -> CompileResult;
        pub fn shadertool_free_compile_result(result: *mut CompileResult);
    }
}

use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::sync::LazyLock;

#[derive(Debug, thiserror::Error)]
pub enum SlangError {
    #[error("Nul bytes included in input string: {0}")]
    InvalidString(#[from] std::ffi::NulError),

    #[error("Slang shader compilation failed [{code:?}]: {message} (Line {line}:{column})")]
    CompileFailed {
        code: raw::ErrorCode,
        message: String,
        file_path: String,
        line: i32,
        column: i32,
    },

    #[error("Slang shader optimization failed [{code:?}]: {message} (Line {line}:{column})")]
    OptimizationFailed {
        code: raw::ErrorCode,
        message: String,
        file_path: String,
        line: i32,
        column: i32,
    },

    #[error("Shader file not found at path: '{file_path}'")]
    FileNotFound { file_path: String },
}

pub struct ShaderContext(*mut raw::Context);

unsafe impl Send for ShaderContext {}
unsafe impl Sync for ShaderContext {}

pub static GLOBAL_CONTEXT: LazyLock<ShaderContext> = LazyLock::new(ShaderContext::new);

impl ShaderContext {
    pub fn new() -> Self {
        let context_ptr = unsafe { raw::shadertool_create_context() };
        Self(context_ptr)
    }

    pub fn compile(
        &self,
        module_name: &str,
        path: &str,
        entry: &str,
        high_precision: bool,
    ) -> Result<SafeCompileResult, SlangError> {
        let raw_module_name = CString::new(module_name)?;
        let raw_path = CString::new(path)?;
        let raw_entry = CString::new(entry)?;

        let raw_result = unsafe {
            raw::shadertool_compile_shader(
                self.0,
                raw_module_name.as_ptr(),
                raw_path.as_ptr(),
                raw_entry.as_ptr(),
                !Self::is_opt_level_zero(),
                high_precision,
            )
        };

        let safe_result = SafeCompileResult(raw_result);

        if let Some(err) = safe_result.to_error() {
            return Err(err);
        }

        Ok(safe_result)
    }

    fn is_opt_level_zero() -> bool {
        let args: Vec<String> = std::env::args().collect();

        let explicitly_zero = args.iter().any(|arg| arg.contains("opt-level=0"));

        let has_any_optimization = args.iter().any(|arg| {
            (arg.contains("opt-level=") || arg.starts_with("opt-level"))
                && !arg.contains("opt-level=0")
        });

        explicitly_zero || !has_any_optimization
    }
}

impl Drop for ShaderContext {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { raw::shadertool_destroy_context(self.0) };
        }
    }
}

#[derive(Debug, Clone)]
pub struct SafeCompileResult(pub(crate) raw::CompileResult);

impl SafeCompileResult {
    pub fn spirv_code(&self) -> &[u8] {
        if self.0.code_bytes.is_null() || self.0.code_size == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.0.code_bytes, self.0.code_size) }
        }
    }

    pub fn spirv_u32(&self) -> &[u32] {
        let bytes = self.spirv_code();
        if bytes.len() % 4 != 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, bytes.len() / 4) }
    }

    pub fn dependencies(&self) -> impl Iterator<Item = &str> + '_ {
        let raw_ptrs = if self.0.dep_file_path.is_null() || self.0.dep_file_count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.0.dep_file_path, self.0.dep_file_count) }
        };

        raw_ptrs.iter().filter_map(|&ptr| {
            if ptr.is_null() {
                None
            } else {
                unsafe { CStr::from_ptr(ptr) }.to_str().ok()
            }
        })
    }

    pub fn reflections(&self) -> ResourceIterator<'_> {
        ResourceIterator {
            ptr: self.0.resources,
            _marker: PhantomData,
        }
    }

    pub fn to_error(&self) -> Option<SlangError> {
        let parse_info = |info: &raw::ErrorInfo| -> (String, String) {
            let msg = if info.message.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(info.message) }
                    .to_string_lossy()
                    .into_owned()
            };

            let file = if info.file_path.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(info.file_path) }
                    .to_string_lossy()
                    .into_owned()
            };

            (msg, file)
        };

        if self.0.error_info.code != raw::ErrorCode::Ok {
            let (message, file_path) = parse_info(&self.0.error_info);
            return Some(SlangError::CompileFailed {
                code: self.0.error_info.code,
                message: if message.is_empty() {
                    "Compilation failed".into()
                } else {
                    message
                },
                file_path,
                line: self.0.error_info.line,
                column: self.0.error_info.column,
            });
        }

        if self.0.optim_info.code != raw::ErrorCode::Ok || !self.0.success {
            let (message, file_path) = parse_info(&self.0.optim_info);
            return Some(SlangError::OptimizationFailed {
                code: self.0.optim_info.code,
                message: if message.is_empty() {
                    "Optimization failed".into()
                } else {
                    message
                },
                file_path,
                line: self.0.optim_info.line,
                column: self.0.optim_info.column,
            });
        }

        None
    }
}

impl Drop for SafeCompileResult {
    fn drop(&mut self) {
        unsafe { raw::shadertool_free_compile_result(&mut self.0) };
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SafeMember<'a> {
    raw: &'a raw::CompactMember,
}

impl<'a> SafeMember<'a> {
    /// *const c_char -> &'a str
    pub fn name(&self) -> &'a str {
        if self.raw.name.is_null() {
            ""
        } else {
            unsafe { CStr::from_ptr(self.raw.name) }
                .to_str()
                .unwrap_or("")
        }
    }

    pub fn offset(&self) -> u32 {
        self.raw.offset
    }
    pub fn size(&self) -> u32 {
        self.raw.size
    }
    pub fn kind(&self) -> raw::MemberKind {
        self.raw.kind
    }
    pub fn scalar_type(&self) -> raw::ScalarKind {
        self.raw.scalar_type
    }
    pub fn row_count(&self) -> u8 {
        self.raw.row_count
    }
    pub fn col_count(&self) -> u8 {
        self.raw.col_count
    }
    pub fn array_count(&self) -> u32 {
        self.raw.array_count
    }
    pub fn parent_index(&self) -> i32 {
        self.raw.parent_index
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SafeResource<'a> {
    raw: &'a raw::CompactResource,
}

impl<'a> SafeResource<'a> {
    /// *const c_char -> &'a str
    pub fn name(&self) -> &'a str {
        if self.raw.name.is_null() {
            ""
        } else {
            unsafe { CStr::from_ptr(self.raw.name) }
                .to_str()
                .unwrap_or("")
        }
    }

    pub fn kind(&self) -> raw::ResourceKind {
        self.raw.kind
    }
    pub fn set(&self) -> u32 {
        self.raw.set
    }
    pub fn binding(&self) -> u32 {
        self.raw.binding
    }
    pub fn total_size(&self) -> u32 {
        self.raw.total_size
    }

    pub fn members(&self) -> impl Iterator<Item = SafeMember<'a>> {
        let raw_slice = if self.raw.members.is_null() || self.raw.member_count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.raw.members, self.raw.member_count as usize) }
        };

        raw_slice.iter().map(|m| SafeMember { raw: m })
    }
}

pub struct ResourceIterator<'a> {
    ptr: *const raw::CompactResource,
    _marker: PhantomData<&'a raw::CompactResource>,
}

impl<'a> Iterator for ResourceIterator<'a> {
    type Item = SafeResource<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr.is_null() {
            return None;
        }

        unsafe {
            let current = &*self.ptr;
            // Sentinel
            if current.name.is_null() {
                return None;
            }

            self.ptr = self.ptr.add(1);

            Some(SafeResource { raw: current })
        }
    }
}
