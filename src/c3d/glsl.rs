// SPDX-License-Identifier: GPL-3.0-or-later

//! GLSL ES 1.00 -> naga, for the `CAP_PROGRAMMABLE` tier's `SHADER_COMPILE`
//! (`docs/internals/c3d.md`, `c3d-cmd-shaders-and-programs`).
//!
//! The wire protocol mandates GLSL ES 1.00 source and deliberately
//! standardises no backend-native shader binary (see the specification's
//! "Programmable tier" chapter and `programmable-gpu-proposal.md`'s "Mapping
//! onto wgpu"). Getting from that source to something `wgpu` can execute is
//! a two-stage pipeline, because no single crate does both halves:
//!
//! 1. **Validate** with `glsl_lang`, a real GLSL/GLSL ES grammar-aware
//!    parser (unlike `naga`'s own GLSL front end, which only understands
//!    core-profile syntax -- no `attribute`/`varying`/`gl_FragColor` at
//!    all). This is the authoritative compile-success/failure answer and
//!    the source of `SHADER_STATUS`'s diagnostic text: a real parser
//!    catches real GLSL ES mistakes with a real message, rather than this
//!    module's own rewrite silently mistranslating them into something
//!    core-profile-legal-but-wrong, or `naga` rejecting valid-but-ES-shaped
//!    input with a confusing core-profile-flavoured error.
//! 2. **Translate**, only once step 1 has already accepted the source: a
//!    small, deliberately bounded textual rewrite (`attribute`->`in`,
//!    `varying`->`in`/`out` by stage, `gl_FragColor`->a synthesised `out`
//!    variable, `texture2D`/`textureCube`->`texture`, `precision`
//!    statements dropped) ahead of `naga::front::glsl::Frontend::parse`,
//!    which is what actually produces the `naga::Module` `wgpu` executes.
//!    This is NOT a general GLSL-ES-to-core transpiler -- just the handful
//!    of substitutions GLSL ES 1.00 vertex/fragment shaders actually use --
//!    and it is not the thing guest-visible compile diagnostics come from
//!    any more: if it has a gap `naga` doesn't tolerate even though step 1
//!    already accepted the source, that is logged as this module's own
//!    bug, not reported to the guest as an invalid shader (see
//!    [`compile`]'s doc comment).
//!
//! This module (and its `naga`/`glsl_lang` dependencies) is gated behind
//! the `c3d` cargo feature exactly like [`super::render`]: compiling a
//! shader is GPU-backend work, not pure protocol logic, so it has no
//! business in the always-built-everywhere `state`/`dispatch`/`ring`
//! modules.

use naga::ShaderStage;

/// Rewrites GLSL ES 1.00 source into the core-profile subset `naga::front::
/// glsl` accepts, for one shader stage.
///
/// Known, bounded rewrites:
/// - `precision <qualifier> <type>;` statements are dropped (naga's parser
///   already defaults to a working precision; GLSL ES's qualifier has no
///   core-profile equivalent naga understands).
/// - `attribute` -> `layout(location = N) in`, `N` counting each
///   occurrence in source order starting at `0` (vertex stage only; GLSL
///   ES only allows `attribute` in vertex shaders, so no stage ambiguity
///   exists here). An explicit location is required for every attribute
///   past the first: naga's GLSL front end does not auto-increment
///   locations for unbound `attribute`/`in` declarations the way a real
///   GLES2 link step would -- confirmed directly, the hard way, against
///   a two-attribute shader whose naga module tagged *both* inputs
///   `location: 0`, which a pipeline can't bind unambiguously. Numbering
///   by declaration order is what a guest library that never calls
///   `PROGRAM_BIND_ATTRIB_LOCATION` has to assume anyway (GLES2 permits
///   either), so this is also the right default, not just a workaround.
///   `PROGRAM_BIND_ATTRIB_LOCATION` itself is not honoured by this
///   rewrite -- it operates per shader, independent of any program --
///   and remains a known gap; see `render.rs`'s `ProgramSlot`.
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
    out = inject_attribute_locations(&out);
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
        // An explicit location is required here: unlike vertex attribute
        // inputs (which naga assigns a location to automatically, by
        // declaration order, confirmed in this module's tests), a
        // fragment output naga's GLSL front end leaves un-bound
        // (`FunctionResult::binding: None`) unless one is given --
        // confirmed directly, the hard way, against a real draw that
        // silently produced no colour output until this was added.
        out = format!("layout(location = 0) out vec4 c3d_frag_color;\n{out}");
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
/// Replaces each `attribute` with `layout(location = N) in`, `N`
/// counting occurrences in source order starting at `0`. See
/// [`rewrite_to_core_profile`]'s doc comment for why this numbering
/// (rather than trusting naga to assign one) is necessary.
fn inject_attribute_locations(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    let mut next_location = 0u32;
    loop {
        let Some(at) = find_word(rest, "attribute") else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..at]);
        out.push_str(&format!("layout(location = {next_location}) in"));
        next_location += 1;
        rest = &rest[at + "attribute".len()..];
    }
    out
}

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
/// malformed guest-supplied source.
///
/// Two stages, per this module's doc comment: `glsl_lang` validates the
/// source as real GLSL ES 1.00 first, and its diagnostic is what
/// `CompileResult::Err` carries on failure -- the actual text
/// `SHADER_STATUS` (`c3d-cmd-programmable-queries`) reports. Only source
/// `glsl_lang` has already accepted is handed to the
/// rewrite-then-`naga::front::glsl` translation step that produces the
/// executable `naga::Module`. If that *second* step fails despite the
/// first having accepted the source, the source itself was valid GLSL ES
/// and the fault is this module's own rewrite -- that is logged, not
/// reported to the guest as an invalid shader, since blaming the guest
/// for this module's bug would be actively misleading during
/// development (and is also just wrong).
pub fn compile(source: &str, stage: ShaderStage) -> CompileResult {
    use glsl_lang::ast::TranslationUnit;
    use glsl_lang::parse::{DefaultParse, ParseOptions};

    let parse_options = ParseOptions {
        default_version: 100,
        ..Default::default()
    };
    if let Err(e) = TranslationUnit::parse_with_options(source, &parse_options) {
        return CompileResult::Err(e.to_string());
    }

    let rewritten = rewrite_to_core_profile(source, stage);
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options::from(stage);
    match frontend.parse(&options, &rewritten) {
        Ok(module) => CompileResult::Ok(Box::new(module)),
        Err(errors) => {
            log::error!(
                "c3d: glsl_lang accepted this GLSL ES source but naga's \
                 post-rewrite translation rejected it -- this is a gap in \
                 glsl.rs's rewrite, not a guest shader error: {errors}\n\
                 --- original source ---\n{source}\n\
                 --- rewritten source ---\n{rewritten}"
            );
            CompileResult::Err(format!(
                "internal: valid GLSL ES source failed Copperline's own \
                 translation step ({errors})"
            ))
        }
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
    fn an_unbound_attribute_gets_a_declaration_order_location() {
        // render.rs's programmable draw depends on this exact behaviour
        // to map VERTEX_ATTRIB_POINTER's index onto the shader's actual
        // input when the guest never calls PROGRAM_BIND_ATTRIB_LOCATION:
        // naga assigns locations to unbound GLSL ES attributes by
        // declaration order, starting at 0 -- confirmed here rather than
        // assumed.
        let result = compile(VERTEX_SRC, ShaderStage::Vertex);
        let CompileResult::Ok(module) = result else {
            panic!("{result:?}");
        };
        let ep = module
            .entry_points
            .iter()
            .find(|e| e.stage == naga::ShaderStage::Vertex)
            .unwrap();
        let arg = &ep.function.arguments[0];
        assert_eq!(arg.name, Some("position".to_string()));
        assert_eq!(
            arg.binding,
            Some(naga::Binding::Location {
                location: 0,
                interpolation: Some(naga::Interpolation::Perspective),
                sampling: None,
                blend_src: None,
                per_primitive: false,
            })
        );
    }

    #[test]
    fn rewrites_attribute_and_varying_by_stage() {
        let v = rewrite_to_core_profile("attribute vec4 a; varying vec4 v;", ShaderStage::Vertex);
        assert_eq!(v, "layout(location = 0) in vec4 a; out vec4 v;");
        let f = rewrite_to_core_profile("varying vec4 v;", ShaderStage::Fragment);
        assert_eq!(f, "in vec4 v;");
    }

    #[test]
    fn injects_sequential_locations_for_multiple_attributes() {
        let v = rewrite_to_core_profile(
            "attribute vec4 position;\nattribute vec3 normal;\nattribute vec2 uv;\n",
            ShaderStage::Vertex,
        );
        assert_eq!(
            v,
            "layout(location = 0) in vec4 position;\n\
             layout(location = 1) in vec3 normal;\n\
             layout(location = 2) in vec2 uv;\n"
        );
    }

    #[test]
    fn rewrites_gl_frag_color_to_a_synthesised_out_var() {
        let f = rewrite_to_core_profile("void main() { gl_FragColor = vec4(0); }", ShaderStage::Fragment);
        assert!(f.starts_with("layout(location = 0) out vec4 c3d_frag_color;\n"));
        assert!(f.contains("c3d_frag_color = vec4(0);"));
        assert!(!f.contains("gl_FragColor"));
    }

    #[test]
    fn the_synthesised_fragment_output_gets_location_zero() {
        // naga wraps a single fragment output in a one-member struct
        // rather than binding FunctionResult directly (the top-level
        // `result.binding` is always None here; the location lives on
        // the struct member instead) -- found the hard way, against a
        // real draw that rendered nothing until `rewrite_to_core_profile`
        // started emitting an explicit `layout(location = 0)` on the
        // synthesised `out` variable. This pins both facts down.
        let result = compile(FRAGMENT_SRC_OK, ShaderStage::Fragment);
        let CompileResult::Ok(module) = result else {
            panic!("{result:?}");
        };
        let ep = module
            .entry_points
            .iter()
            .find(|e| e.stage == naga::ShaderStage::Fragment)
            .unwrap();
        let result = ep.function.result.as_ref().unwrap();
        assert_eq!(result.binding, None);
        let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
            panic!("expected the fragment result to be struct-typed");
        };
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].name.as_deref(), Some("c3d_frag_color"));
        assert_eq!(
            members[0].binding,
            Some(naga::Binding::Location {
                location: 0,
                interpolation: Some(naga::Interpolation::Perspective),
                sampling: None,
                blend_src: None,
                per_primitive: false,
            })
        );
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

    #[test]
    fn the_compile_error_comes_from_glsl_lang_not_naga() {
        // glsl_lang's diagnostics are its own distinctive shape
        // ("0:4:1: unexpected `}`, ..."); pinning this ensures the real
        // GLSL ES validator is actually being consulted, not bypassed in
        // favour of naga's own (core-profile-flavoured) error text.
        let CompileResult::Err(log) = compile(FRAGMENT_SRC_BAD, ShaderStage::Fragment) else {
            panic!("expected a compile failure");
        };
        assert!(log.contains("unexpected"), "{log}");
    }
}
