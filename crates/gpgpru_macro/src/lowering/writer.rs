// src/lowering/writer.rs

pub struct CodeWriter {
    pub buf: String,
    indent_level: usize,
    at_line_start: bool,
}

impl CodeWriter {
    pub fn new() -> Self {
        Self {
            buf: String::with_capacity(4096),
            indent_level: 0,
            at_line_start: true,
        }
    }

    pub fn write(&mut self, s: &str) {
        if self.at_line_start && !s.is_empty() {
            self.buf.push_str(&"    ".repeat(self.indent_level));
            self.at_line_start = false;
        }
        self.buf.push_str(s);
    }

    pub fn writeln(&mut self, s: &str) {
        self.write(s);
        self.buf.push('\n');
        self.at_line_start = true;
    }

    pub fn block<F>(&mut self, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.writeln("{");
        self.indent_level += 1;
        f(self);
        self.indent_level -= 1;
        self.write("}");
    }

    pub fn paren<F>(&mut self, f: F)
    where
        F: FnOnce(&mut Self),
    {
        self.write("(");
        f(self);
        self.write(")");
    }
}
