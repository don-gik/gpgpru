use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::{env, fs, path::Path};

#[derive(Deserialize, Default)]
struct Config {
    #[serde(default)]
    swizzles: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    structs: BTreeMap<String, BTreeMap<String, String>>,
    #[serde(default)]
    functions: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    math: MathConfig,
    #[serde(default)]
    methods: MethodsConfig,
}

#[derive(Deserialize, Default)]
struct MathConfig {
    #[serde(default)]
    returns_scalar: BTreeMap<String, String>,
    #[serde(default)]
    returns_same: BTreeMap<String, String>,
}

#[derive(Deserialize, Default)]
struct MethodsConfig {
    #[serde(default)]
    to_function: ToFunctionConfig,
    #[serde(default)]
    keep_method: KeepMethodConfig,
}

#[derive(Deserialize, Default)]
struct ToFunctionConfig {
    #[serde(default)]
    returns_scalar: BTreeMap<String, String>,
    #[serde(default)]
    returns_same: BTreeMap<String, String>,
}

#[derive(Deserialize, Default)]
struct KeepMethodConfig {
    #[serde(default)]
    returns_explicit: BTreeMap<String, Vec<String>>,
}

// --- Main Builder ---

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=prelude.toml");

    let toml_str = fs::read_to_string("prelude.toml")?;
    let config: Config = toml::from_str(&toml_str)?;

    let out_dir = env::var("OUT_DIR")?;
    let out_path = Path::new(&out_dir);

    generate_field_resolver(&config, out_path)?;
    generate_call_resolver(&config, out_path)?;
    generate_method_resolver(&config, out_path)?;

    Ok(())
}

// --- Code Generators ---

fn generate_field_resolver(
    config: &Config,
    out_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut code = String::new();

    writeln!(
        code,
        "pub fn resolve_field(base: &GpuType, field: &str) -> Option<GpuType> {{"
    )?;
    writeln!(code, "    match base {{")?;

    // 1. Vector Swizzles
    writeln!(
        code,
        "        GpuType::Vector {{ element, len }} => match field {{"
    )?;
    for chars_vec in config.swizzles.values() {
        let chars: Vec<char> = chars_vec
            .iter()
            .map(|c| c.chars().next().unwrap())
            .collect();
        for target_len in 1..=4 {
            generate_swizzles(&chars, "", target_len, &mut code)?;
        }
    }
    writeln!(code, "            _ => None,")?;
    writeln!(code, "        }},")?;

    // 2. Matrix Fields
    writeln!(
        code,
        "        GpuType::Matrix {{ element, rows, cols }} => match field {{"
    )?;
    for (i, axis) in ["x", "y", "z", "w"].iter().enumerate() {
        writeln!(
            code,
            "            \"{axis}_axis\" if *cols > {i} => Some(GpuType::Vector {{ element: *element, len: *rows }}),"
        )?;
    }
    for r in 0..4 {
        for c in 0..4 {
            writeln!(
                code,
                "            \"_m{r}{c}\" if *rows > {r} && *cols > {c} => Some(GpuType::Scalar(*element)),"
            )?;
        }
    }
    for r in 1..=4 {
        for c in 1..=4 {
            writeln!(
                code,
                "            \"_{r}{c}\" if *rows >= {r} && *cols >= {c} => Some(GpuType::Scalar(*element)),"
            )?;
        }
    }
    writeln!(code, "            _ => None,")?;
    writeln!(code, "        }},")?;

    // 3. Struct Fields
    writeln!(
        code,
        "        GpuType::Struct {{ name, .. }} => match name.as_ref() {{"
    )?;
    for (struct_name, fields) in &config.structs {
        writeln!(code, "            \"{struct_name}\" => match field {{")?;
        for (field_name, return_type) in fields {
            let rs_type = rust_type_to_gpu_type(return_type);
            writeln!(code, "                \"{field_name}\" => Some({rs_type}),")?;
        }
        writeln!(code, "                _ => None,")?;
        writeln!(code, "            }},")?;
    }
    writeln!(code, "            _ => None,")?;
    writeln!(code, "        }},")?;

    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}")?;
    writeln!(code, "}}")?;

    fs::write(out_dir.join("field_resolver_generated.rs"), code)?;
    Ok(())
}

fn generate_call_resolver(
    config: &Config,
    out_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut code = String::new();

    writeln!(code, "use crate::types::{{GpuType, LiteralKind}};")?;
    writeln!(
        code,
        "\npub fn resolve_call(segments: &[String], arg_types: &[GpuType]) -> Option<(GpuType, &'static str)> {{"
    )?;
    writeln!(
        code,
        "    let seg_refs: Vec<&str> = segments.iter().map(|s| s.as_str()).collect();"
    )?;
    writeln!(code, "    match seg_refs.as_slice() {{")?;

    // Functions
    for (func_name, data) in &config.functions {
        let pattern = rust_path_to_slice_pattern(func_name);
        let rs_type = rust_type_to_gpu_type(&data[0]);
        let slang_name = &data[1];
        writeln!(
            code,
            "        {pattern} => Some(({rs_type}, \"{slang_name}\")),"
        )?;
    }

    // Math: returns_scalar
    for (rust_name, slang_name) in &config.math.returns_scalar {
        let pattern = rust_path_to_slice_pattern(rust_name);
        writeln!(
            code,
            "        {pattern} => arg_types.first().and_then(|arg| match arg {{"
        )?;
        writeln!(
            code,
            "            GpuType::Vector {{ element, .. }} | GpuType::Scalar(element) => Some((GpuType::Scalar(*element), \"{slang_name}\")),"
        )?;
        writeln!(code, "            _ => None,")?;
        writeln!(code, "        }}),")?;
    }

    // Math: returns_same
    for (rust_name, slang_name) in &config.math.returns_same {
        let pattern = rust_path_to_slice_pattern(rust_name);
        writeln!(
            code,
            "        {pattern} => arg_types.first().cloned().map(|ty| (ty, \"{slang_name}\")),"
        )?;
    }

    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}")?;
    writeln!(code, "}}")?;

    fs::write(out_dir.join("call_resolver_generated.rs"), code)?;
    Ok(())
}

fn generate_method_resolver(
    config: &Config,
    out_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut code = String::new();

    writeln!(code, "use crate::types::MethodAction;\n")?;
    writeln!(
        code,
        "pub fn resolve_method(name: &str, arg_types: &[GpuType]) -> Option<MethodAction> {{"
    )?;
    writeln!(code, "    match name {{")?;

    // Methods -> Functions (Scalar)
    for (rust_name, slang_name) in &config.methods.to_function.returns_scalar {
        writeln!(
            code,
            "        \"{rust_name}\" => arg_types.first().and_then(|arg| match arg {{"
        )?;
        writeln!(
            code,
            "            GpuType::Vector {{ element, .. }} | GpuType::Scalar(element) => Some(MethodAction::AsFunction(GpuType::Scalar(*element), \"{slang_name}\")),"
        )?;
        writeln!(code, "            _ => None,")?;
        writeln!(code, "        }}),")?;
    }

    // Methods -> Functions (Same)
    for (rust_name, slang_name) in &config.methods.to_function.returns_same {
        writeln!(
            code,
            "        \"{rust_name}\" => arg_types.first().cloned().map(|ty| MethodAction::AsFunction(ty, \"{slang_name}\")),"
        )?;
    }

    // Methods -> Kept Methods (Explicit)
    for (rust_name, data) in &config.methods.keep_method.returns_explicit {
        let rs_type = rust_type_to_gpu_type(&data[0]);
        let slang_name = &data[1];
        writeln!(
            code,
            "        \"{rust_name}\" => Some(MethodAction::AsMethod({rs_type}, \"{slang_name}\")),"
        )?;
    }

    writeln!(code, "        _ => None,")?;
    writeln!(code, "    }}")?;
    writeln!(code, "}}")?;

    fs::write(out_dir.join("method_resolver_generated.rs"), code)?;
    Ok(())
}

// --- Helpers ---

fn generate_swizzles(
    chars: &[char],
    current: &str,
    target_len: usize,
    code: &mut String,
) -> std::fmt::Result {
    if current.len() == target_len {
        let min_len = current
            .chars()
            .map(|c| match c {
                'x' | 'r' => 1,
                'y' | 'g' => 2,
                'z' | 'b' => 3,
                'w' | 'a' => 4,
                _ => 1,
            })
            .max()
            .unwrap_or(1);

        let rs_type = if target_len == 1 {
            "GpuType::Scalar(*element)".to_string()
        } else {
            format!("GpuType::Vector {{ element: *element, len: {target_len} }}")
        };

        writeln!(
            code,
            "            \"{current}\" if *len >= {min_len} => Some({rs_type}),"
        )?;
        return Ok(());
    }

    for &c in chars {
        let mut next = String::from(current);
        next.push(c);
        generate_swizzles(chars, &next, target_len, code)?;
    }

    Ok(())
}

/// Converts a path like "Vec2::new" into a slice pattern like `["Vec2", "new"]`
fn rust_path_to_slice_pattern(path: &str) -> String {
    let parts: Vec<String> = path.split("::").map(|p| format!("\"{p}\"")).collect();
    format!("[{}]", parts.join(", "))
}

/// Converts shorthand types like "Vec2" to the actual `GpuType` enum representation.
fn rust_type_to_gpu_type(ty: &str) -> String {
    match ty {
        "f32" => "GpuType::Scalar(LiteralKind::Float32)",
        "i32" => "GpuType::Scalar(LiteralKind::Int32)",
        "u32" => "GpuType::Scalar(LiteralKind::UInt32)",
        "Vec2" => "GpuType::Vector { element: LiteralKind::Float32, len: 2 }",
        "Vec3" => "GpuType::Vector { element: LiteralKind::Float32, len: 3 }",
        "Vec4" => "GpuType::Vector { element: LiteralKind::Float32, len: 4 }",
        "Mat4" => "GpuType::Matrix { element: LiteralKind::Float32, rows: 4, cols: 4 }",
        _ => panic!("Unsupported field/return type: {ty}"),
    }
    .to_string()
}
