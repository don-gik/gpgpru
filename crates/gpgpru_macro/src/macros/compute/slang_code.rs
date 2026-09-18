// src/rust_emit/slang_code.rs

use std::ffi::CString;

use shader_slang::{
    Blob, CompileTarget, CompilerOptions, Downcast, GlobalSession, SessionDesc, TargetDesc,
};

fn blob_to_u32_slice(blob: &Blob) -> Vec<u32> {
    let byte_slice: &[u8] = blob.as_slice();
    let byte_len = byte_slice.len();

    assert!(
        byte_len % 4 == 0,
        "SPIR-V blob size must be a multiple of 4 bytes"
    );
    let u32_len = byte_len / 4;

    let raw_ptr = byte_slice.as_ptr() as *const u32;
    let spirv_slice: &[u32] = unsafe { std::slice::from_raw_parts(raw_ptr, u32_len) };

    spirv_slice.to_vec()
}

pub fn compile_slang(
    slang_code: &str,
    entry_point_name: &str,
    unique_hash: &str,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let metadata = cargo_metadata::MetadataCommand::new().exec()?;
    let target_dir = metadata.target_directory;
    let target_cache_dir = target_dir.join("gpgpru/shaders");

    std::fs::create_dir_all(&target_cache_dir)?;

    let file_name = format!("{}_{}.slang", entry_point_name, unique_hash);
    let full_path = target_cache_dir.join(&file_name);

    std::fs::write(&full_path, slang_code)?;

    let dir_path = full_path
        .parent()
        .ok_or("Parent directory cannot be found.")?;
    let c_dir_path = CString::new(dir_path.to_string().clone().into_bytes())?;
    let dir_ptr: *const std::os::raw::c_char = c_dir_path.as_ptr();

    let file_str = full_path.file_name().ok_or("Failed to get file name")?;

    // Slang Compiler
    let global_session = match GlobalSession::new() {
        Some(session) => Ok(session),
        None => Err("Slang Shader Compiler Session could not be loaded."),
    }?;

    let session_options = CompilerOptions::default();

    let target_desc = TargetDesc::default()
        .format(CompileTarget::Spirv)
        .profile(global_session.find_profile("glsl_450"));

    let targets = [target_desc];
    let search_paths = [dir_ptr];

    let session_desc = SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&session_options);

    let session = match global_session.create_session(&session_desc) {
        Some(session) => Ok(session),
        None => Err("Session could not be created."),
    }?;
    let module = session.load_module(file_str)?;
    let entry_point = match module.find_entry_point_by_name(entry_point_name) {
        Some(entry_point) => Ok(entry_point),
        None => Err("Entry Point not found"),
    }?;

    let program = session.create_composite_component_type(&[
        module.downcast().clone(),
        entry_point.downcast().clone(),
    ])?;

    let linked_program = program.link()?;

    let _reflection = linked_program.layout(0)?;
    let shader_bytecode = linked_program.entry_point_code(0, 0)?;

    Ok(blob_to_u32_slice(&shader_bytecode))
}
