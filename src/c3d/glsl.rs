// SPDX-License-Identifier: GPL-3.0-or-later

//! GLSL ES 1.00 -> naga, for the `CAP_PROGRAMMABLE` tier's `SHADER_COMPILE`
//! (`docs/internals/c3d.md`, `c3d-cmd-shaders-and-programs`).
//!
//! The wire protocol mandates GLSL ES 1.00 source and deliberately
//! standardises no backend-native shader binary (see the specification's
//! "Programmable tier" chapter and `programmable-gpu-proposal.md`'s "Mapping
//! onto wgpu"). `naga`'s GLSL front end, however, only parses core-profile
//! GLSL (`in`/`out`, no `gl_FragColor`) -- GLSL ES 1.00's `attribute`/
//! `varying`/`gl_FragColor` surface does not parse there at all. This module
//! is the small, deliberately bounded rewrite that bridges the two: it is
//! NOT a general GLSL-ES-to-core transpiler, just the handful of syntactic
//! substitutions GLSL ES 1.00 vertex/fragment shaders actually use.
//!
//! This module (and its `naga` dependency) is gated behind the `c3d` cargo
//! feature exactly like [`super::render`]: compiling a shader is GPU-backend
//! work, not pure protocol logic, so it has no business in the
//! always-built-everywhere `state`/`dispatch`/`ring` modules.

use naga::ShaderStage;

/// Rewrites GLSL ES 1.00 source into the core-profile subset `naga::front::
/// glsl` accepts, for one shader stage.
///
/// Known, bounded rewrites:
/// - `precision <qualifier> <type>;` statements are dropped (naga's parser
///   already defaults to a working precision; GLSL ES's qualifier has no
///   core-profile equivalent naga understands).
/// - `attribute` -> `in` (vertex stage only; GLSL ES only allows `attribute`
///   in vertex shaders, so no stage ambiguity exists here).
/// - `varying` -> `out` in the vertex stage, `in` in the fragment stage.
/// - `gl_FragColor` (fragment stage only) -> a synthesised `out vec4`
///   declaration, with every occurrence of `gl_FragColor` rewritten to
///   reference it. `gl_FragData[n]` (multiple render targets) is out of
///   scope for this draft's single-colour-attachment framebuffer model and
///   is left unrewritten, so a shader using it fails to parse rather than
///   silently misbehaving.
/// - `texture2D(`/`textureCube(` -> `texture(`, GLSL ES 1.00's sampling
///   functions onto core GLSL's single overloaded name.
///
/// Rewriting is done with whole-word substitution (never inside identifiers
/// like `attribute_foo`), which is sufficient because every substituted
/// token is a GLSL ES reserved word or built-in name a shader cannot
/// otherwise use as an identifier.
pub fn rewrite_to_core_profile(source: &str, stage: ShaderStage) -> String {
    let mut out = strip_precision_statements(source);
    out = replace_word(&out, "attribute", "in");
    out = replace_word(
        &out,
        "varying",
        match stage {
            ShaderStage::Vertex => "out",
            // Every other stage is unreachable: this tier only ever
            // compiles a vertex or fragment shader (`c3d-cmd-shaders-and-
            // programs`'s `SHADER_CREATE` only accepts `VERTEX_SHADER`/
            // `FRAGMENT_SHADER`). Fall back to "in" rather than panic on
            // guest-adjacent input that can't actually reach this arm.
            _ => "in",
        },
    );
    out = replace_word(&out, "texture2D", "texture");
    out = replace_word(&out, "textureCube", "texture");
    if stage == ShaderStage::Fragment && contains_word(&out, "gl_FragColor") {
        out = replace_word(&out, "gl_FragColor", "c3d_frag_color");
        out = format!("out vec4 c3d_frag_color;\n{out}");
    }
    out
}

fn is_ident_char(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// Whole-word substitution: `from` is replaced only where it is not
/// adjacent to another identifier character on either side, so (for
/// example) rewriting `in` never touches `int` or `attribute_in`.
fn replace_word(source: &str, from: &str, to: &str) -> String {
    let bytes = source.as_bytes();
    let from_bytes = from.as_bytes();
    let mut result = String::with_capacity(source.len());
    let mut i = 0;
    while i < bytes.len() {
        let at_boundary_start = i == 0 || !is_ident_char(bytes[i - 1]);
        let matches = at_boundary_start
            && bytes[i..].starts_with(from_bytes)
            && bytes
                .get(i + from_bytes.len())
                .is_none_or(|&c| !is_ident_char(c));
        if matches {
            result.push_str(to);
            i += from_bytes.len();
        } else {
            // Safe: we only ever skip whole UTF-8 characters here because
            // every `from` we search for is pure ASCII, so a non-matching
            // position always starts a valid char boundary.
            let ch_len = source[i..].chars().next().map_or(1, |c| c.len_utf8());
            result.push_str(&source[i..i + ch_len]);
            i += ch_len;
        }
    }
    result
}

fn contains_word(source: &str, word: &str) -> bool {
    replace_word(source, word, "").len() != source.len()
}

/// Drops every `precision <qualifier> <type>;` statement. A small
/// hand-written scan rather than a regex dependency: the grammar is just
/// "precision", whitespace, an identifier, whitespace, an identifier,
/// optional whitespace, `;`.
fn strip_precision_statements(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    loop {
        let Some(at) = find_word(rest, "precision") else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..at]);
        let after = &rest[at..];
        match after.find(';') {
            Some(semi) => rest = &after[semi + 1..],
            None => {
                // Malformed; leave the rest for naga to reject with a real
                // parse error rather than silently eating the whole file.
                out.push_str(after);
                break;
            }
        }
    }
    out
}

fn find_word(source: &str, word: &str) -> Option<usize> {
    let bytes = source.as_bytes();
    let word_bytes = word.as_bytes();
    let mut i = 0;
    while i + word_bytes.len() <= bytes.len() {
        let at_boundary_start = i == 0 || !is_ident_char(bytes[i - 1]);
        if at_boundary_start
            && bytes[i..].starts_with(word_bytes)
            && bytes
                .get(i + word_bytes.len())
                .is_none_or(|&c| !is_ident_char(c))
        {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Result of compiling one shader's GLSL ES 1.00 source
/// ([`rewrite_to_core_profile`] then `naga::front::glsl::Frontend::parse`).
#[derive(Debug, Clone)]
pub enum CompileResult {
    Ok(Box<naga::Module>),
    Err(String),
}

impl CompileResult {
    pub fn is_ok(&self) -> bool {
        matches!(self, CompileResult::Ok(_))
    }
}

/// Compiles one shader stage's GLSL ES 1.00 source. Never panics on
/// malformed guest-supplied source: a parse failure is `CompileResult::Err`
/// with naga's diagnostic text as the info log, exactly as `SHADER_STATUS`
/// needs it (`c3d-cmd-programmable-queries`).
pub fn compile(source: &str, stage: ShaderStage) -> CompileResult {
    let rewritten = rewrite_to_core_profile(source, stage);
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options::from(stage);
    match frontend.parse(&options, &rewritten) {
        Ok(module) => CompileResult::Ok(Box::new(module)),
        Err(errors) => CompileResult::Err(errors.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VERTEX_SRC: &str = "attribute vec4 position;\nvoid main() {\n    gl_Position = position;\n}\n";
    const FRAGMENT_SRC_OK: &str = "precision mediump float;\nvoid main() {\n    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);\n}\n";
    const FRAGMENT_SRC_BAD: &str =
        "precision mediump float;\nvoid main() {\n    gl_FragColor = vec4(1.0, 0.0, 0.0\n}\n";

    #[test]
    fn rewrites_attribute_and_varying_by_stage() {
        let v = rewrite_to_core_profile("attribute vec4 a; varying vec4 v;", ShaderStage::Vertex);
        assert_eq!(v, "in vec4 a; out vec4 v;");
        let f = rewrite_to_core_profile("varying vec4 v;", ShaderStage::Fragment);
        assert_eq!(f, "in vec4 v;");
    }

    #[test]
    fn rewrites_gl_frag_color_to_a_synthesised_out_var() {
        let f = rewrite_to_core_profile("void main() { gl_FragColor = vec4(0); }", ShaderStage::Fragment);
        assert!(f.starts_with("out vec4 c3d_frag_color;\n"));
        assert!(f.contains("c3d_frag_color = vec4(0);"));
        assert!(!f.contains("gl_FragColor"));
    }

    #[test]
    fn strips_precision_statements() {
        let out = strip_precision_statements("precision mediump float;\nvoid main() {}\n");
        assert_eq!(out, "\nvoid main() {}\n");
    }

    #[test]
    fn does_not_rewrite_inside_other_identifiers() {
        // "in" must not match inside "int"; "attribute" must not match
        // inside a hypothetical "attribute_count" identifier.
        let out = replace_word("int attribute_count = 1;", "int", "IN");
        assert_eq!(out, "IN attribute_count = 1;");
        let out = replace_word("int attribute_count = 1;", "attribute", "ATTR");
        assert_eq!(out, "int attribute_count = 1;");
    }

    #[test]
    fn compiles_the_minimal_vertical_slice_vertex_shader() {
        let result = compile(VERTEX_SRC, ShaderStage::Vertex);
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn compiles_the_minimal_vertical_slice_fragment_shader() {
        let result = compile(FRAGMENT_SRC_OK, ShaderStage::Fragment);
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn reports_a_real_parse_failure_for_malformed_source() {
        let result = compile(FRAGMENT_SRC_BAD, ShaderStage::Fragment);
        assert!(!result.is_ok());
    }
}
