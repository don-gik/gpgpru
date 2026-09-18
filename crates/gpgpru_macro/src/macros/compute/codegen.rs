// src/lowering/codegen.rs

use super::{ComputeAttr, FieldInfo};

use crate::StmtId;
use crate::TypeArena;

use crate::{CodeWriter, LoweringBuilder};

impl LoweringBuilder {
    pub fn emit_slang_fn(
        &self,
        fn_name: &str,
        arena: &TypeArena,
        attr: &ComputeAttr,
        pushed_fields: &Vec<FieldInfo>,
        block_id: &StmtId,
    ) -> String {
        let mut code = String::new();

        code.push_str("struct PushConstants {\n");

        for info in pushed_fields {
            code.push_str(&format!(
                "    {} {};\n",
                arena.to_slang_string(info.ty),
                &info.name
            ));
        }

        code.push_str("};\n\n");

        code.push_str("[vk::push_constant]\n");
        code.push_str("PushConstants g_args;\n\n");

        let (x, y, z) = attr.threads;
        code.push_str("[shader(\"compute\")]\n");
        code.push_str(&format!("[numthreads({}, {}, {})]\n", x, y, z));

        let mut writer = CodeWriter::new();
        self.emit_stmt(block_id, arena, &mut writer);

        code.push_str(&format!(
            "void {}(uint3 _sv_thread_id : SV_DispatchThreadID) {}",
            fn_name, writer.buf,
        ));

        code
    }
}
