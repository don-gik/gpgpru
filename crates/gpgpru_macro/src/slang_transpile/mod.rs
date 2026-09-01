// src/slang_transpile/mod.rs

mod emitter;

pub(crate) use emitter::SlangEmitter;

use crate::slang_fields::SlangField;

use std::collections::HashMap;

pub fn build_slang_shader(
    struct_name: &str,
    threads: (u32, u32, u32),
    entry_name: &str,
    unpacked_fields: &[SlangField],
    system_inputs: &HashMap<String, String>,
    push_field_names: Vec<String>,
    block: &syn::Block,
) -> syn::Result<String> {
    let mut code = String::new();

    // Push Constants
    code.push_str(&format!("struct {}\n{{\n", struct_name));
    for field in unpacked_fields {
        code.push_str(&format!("    {} {};\n", field.slang_type, field.name));
    }
    code.push_str("};\n\n");

    // Global VK Push Constant
    code.push_str(&format!(
        "[[vk::push_constant]]\n{} g_args;\n\n",
        struct_name
    ));

    // Compute Entry & Header
    let (tx, ty, tz) = threads;
    code.push_str("[shader(\"compute\")]\n");
    code.push_str(&format!("[numthreads({}, {}, {})]\n", tx, ty, tz));
    code.push_str(&format!(
        "void {}(uint3 thread_id : SV_DispatchThreadID) {{",
        entry_name
    ));

    let mut emitter = SlangEmitter::new(system_inputs, push_field_names);
    emitter.emit_block(block);

    if let Some(combined_err) = emitter.check_errors() {
        return Err(combined_err);
    }

    code.push_str(&emitter.output);
    Ok(code)
}
