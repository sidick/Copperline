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
/// - `uniform sampler2D <name>;`/`uniform samplerCube <name>;` -> a pair of
///   declarations, `uniform texture2D <name>_tex;` and `uniform sampler
///   <name>_samp;` (each getting its own binding number), and every call
///   `texture2D(<name>, ...)`/`textureCube(<name>, ...)` -> `texture(
///   sampler2D(<name>_tex, <name>_samp), ...)` (likewise for `samplerCube`).
///   Confirmed directly: naga's GLSL front end has no representation for
///   GLSL ES's combined sampler type at all -- parsing a bare `uniform
///   sampler2D tex;` fails with `NotImplemented("variable qualifier")`,
///   because naga's GLSL front end only understands the Vulkan-GLSL-style
///   split between a `textureND` (a `TypeInner::Image`, no attached
///   sampler) and a separate `sampler`/`samplerShadow` object, recombined
///   at the call site via a constructor call naga recognises by name
///   (`sampler2D(someTexture, someSampler)`, dispatched to its own
///   `MacroCall::Sampler`) -- not by declaring a combined type up front.
///   This rewrite produces exactly that split/recombine shape. GLES2 has no
///   bare (sampler-less) texture object, so every `textureND(` call site
///   necessarily names one of these rewritten uniforms -- there is nothing
///   else it could be naming. Known gap: an array of samplers (`uniform
///   sampler2D tex[4];`) is not handled -- the `[4]` suffix lands only on
///   the synthesised `sampler` declaration, not the paired `textureND`
///   one, which would miscompile. No GLES2 workload exercising this has
///   come up yet; revisit if one does.
/// - `uniform` -> `layout(binding = N) uniform`, `N` counting each
///   occurrence in source order, starting at [`VERTEX_BINDING_BASE`] or
///   [`FRAGMENT_BINDING_BASE`] depending on stage. Unlike attributes,
///   naga's GLSL front end does not accept a plain `uniform` declaration
///   *at all* without one -- confirmed directly: "uniform/buffer blocks
///   require layout(binding=X)" is naga's own parse error for the
///   unrewritten form. The two stages' ranges are kept disjoint
///   (deliberately far apart, not just adjacent) because a vertex and a
///   fragment shader compile independently, with no knowledge of which
///   program they'll eventually be linked into -- by the time `PROGRAM_
///   LINK` pairs them, their binding numbers are already baked into
///   each compiled `naga::Module` and can't be renumbered, so same-stage
///   numbering starting at `0` on both sides would collide the moment
///   two shaders using any uniforms were linked into one program sharing
///   one bind group.
///
/// Rewriting is done with whole-word substitution (never inside identifiers
/// like `attribute_foo`), which is sufficient because every substituted
/// token is a GLSL ES reserved word or built-in name a shader cannot
/// otherwise use as an identifier.
pub fn rewrite_to_core_profile(source: &str, stage: ShaderStage) -> String {
    let mut out = strip_precision_statements(source);
    out = inject_attribute_locations(&out);
    out = rewrite_sampler_calls(&out, "texture2D", "sampler2D");
    out = rewrite_sampler_calls(&out, "textureCube", "samplerCube");
    out = inject_uniform_bindings(&out, stage);
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

/// Rewrites every call `<fn_name>(<ident>, ...)` to `texture(<sampler_ctor>(
/// <ident>_tex, <ident>_samp), ...)`. See [`rewrite_to_core_profile`]'s doc
/// comment for why: GLES2's combined sampler uniforms are split into a
/// bare `textureND` and a `sampler` by [`inject_uniform_bindings`], and
/// naga only recombines them at the call site via this constructor-call
/// form. The first argument is assumed to be a bare identifier (the
/// uniform's name) -- GLES2 has no way to produce a sampler value other
/// than naming a `uniform sampler2D`/`samplerCube` variable directly, so
/// this always holds for valid input.
fn rewrite_sampler_calls(source: &str, fn_name: &str, sampler_ctor: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    loop {
        let Some(at) = find_word(rest, fn_name) else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..at]);
        let after_name = &rest[at + fn_name.len()..];
        let after_name_trimmed = after_name.trim_start();
        if !after_name_trimmed.starts_with('(') {
            // Not actually a call (shouldn't happen for a GLES2 reserved
            // function name); copy the name through unchanged and keep
            // scanning past it so the loop can't spin forever.
            out.push_str(fn_name);
            rest = after_name;
            continue;
        }
        let after_paren = after_name_trimmed[1..].trim_start();
        let ident_len = after_paren
            .as_bytes()
            .iter()
            .take_while(|&&c| is_ident_char(c))
            .count();
        let ident = &after_paren[..ident_len];
        let remainder = &after_paren[ident_len..];
        out.push_str("texture(");
        out.push_str(sampler_ctor);
        out.push('(');
        out.push_str(ident);
        out.push_str("_tex, ");
        out.push_str(ident);
        out.push_str("_samp)");
        rest = remainder;
    }
    out
}

/// Binding numbers assigned to a vertex shader's `uniform` declarations
/// start here. See [`rewrite_to_core_profile`]'s doc comment.
const VERTEX_BINDING_BASE: u32 = 0;
/// Binding numbers assigned to a fragment shader's `uniform` declarations
/// start here -- far enough past any plausible vertex-stage count that
/// the two stages' ranges never collide once linked into one program's
/// shared bind group, while staying safely under `wgpu`'s own
/// `max_bindings_per_bind_group` limit (confirmed directly: `wgpu`
/// rejects a bind group layout entry at binding index `1000` outright,
/// "greater than the maximum number 1000" -- so this is not just "a
/// generous-looking round number", it is close to the actual ceiling).
/// `256` stays comfortably below both that ceiling and any plausible
/// GLES2 vertex-stage uniform-variable count.
const FRAGMENT_BINDING_BASE: u32 = 256;

/// Replaces each `uniform` with `layout(binding = N) uniform` (keeping
/// the word itself, unlike [`inject_attribute_locations`]'s `attribute`
/// -> `in` substitution), `N` counting occurrences in source order from
/// the stage's base. See [`rewrite_to_core_profile`]'s doc comment.
fn inject_uniform_bindings(source: &str, stage: ShaderStage) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    let mut next_binding = match stage {
        ShaderStage::Vertex => VERTEX_BINDING_BASE,
        _ => FRAGMENT_BINDING_BASE,
    };
    loop {
        let Some(at) = find_word(rest, "uniform") else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..at]);
        let after_uniform = &rest[at + "uniform".len()..];
        let after_ws = after_uniform.trim_start();
        let type_len = after_ws
            .as_bytes()
            .iter()
            .take_while(|&&c| is_ident_char(c))
            .count();
        let type_name = &after_ws[..type_len];

        let texture_type = match type_name {
            "sampler2D" => Some("texture2D"),
            "samplerCube" => Some("textureCube"),
            _ => None,
        };
        if let Some(texture_type) = texture_type {
            let after_type_ws = after_ws[type_len..].trim_start();
            let name_len = after_type_ws
                .as_bytes()
                .iter()
                .take_while(|&&c| is_ident_char(c))
                .count();
            let name = &after_type_ws[..name_len];
            out.push_str(&format!(
                "layout(binding = {next_binding}) uniform {texture_type} {name}_tex;\n\
                 layout(binding = {}) uniform sampler {name}_samp",
                next_binding + 1
            ));
            next_binding += 2;
            rest = &after_type_ws[name_len..];
        } else {
            out.push_str(&format!("layout(binding = {next_binding}) uniform"));
            next_binding += 1;
            rest = after_uniform;
        }
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
    fn compiles_a_sampler2d_uniform_through_the_full_pipeline() {
        let src = "precision mediump float;\n\
                   uniform sampler2D tex;\n\
                   varying vec2 uv;\n\
                   void main() {\n    gl_FragColor = texture2D(tex, uv);\n}\n";
        let result = compile(src, ShaderStage::Fragment);
        let CompileResult::Ok(module) = result else {
            panic!("{result:?}");
        };
        let names: Vec<_> = module
            .global_variables
            .iter()
            .map(|(_, gv)| gv.name.clone().unwrap_or_default())
            .collect();
        assert!(names.contains(&"tex_tex".to_string()));
        assert!(names.contains(&"tex_samp".to_string()));
    }

    #[test]
    fn compiles_a_sampler_cube_uniform_through_the_full_pipeline() {
        let src = "precision mediump float;\n\
                   uniform samplerCube cube;\n\
                   varying vec3 dir;\n\
                   void main() {\n    gl_FragColor = textureCube(cube, dir);\n}\n";
        let result = compile(src, ShaderStage::Fragment);
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn a_sampler_uniform_and_a_plain_uniform_get_disjoint_bindings() {
        // The sampler splits into two bindings; a following plain uniform
        // must not collide with either half.
        let src = "precision mediump float;\n\
                   uniform sampler2D tex;\n\
                   uniform vec4 tint;\n\
                   varying vec2 uv;\n\
                   void main() {\n    gl_FragColor = texture2D(tex, uv) * tint;\n}\n";
        let result = compile(src, ShaderStage::Fragment);
        let CompileResult::Ok(module) = result else {
            panic!("{result:?}");
        };
        let bindings: Vec<_> = module
            .global_variables
            .iter()
            .filter_map(|(_, gv)| gv.binding.map(|b| b.binding))
            .collect();
        let unique: std::collections::BTreeSet<_> = bindings.iter().copied().collect();
        assert_eq!(bindings.len(), unique.len(), "binding collision: {bindings:?}");
    }

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
    fn uniforms_get_sequential_bindings_in_group_zero() {
        let src = "uniform float scale;\nuniform vec4 tint;\nattribute vec4 position;\nvoid main() { gl_Position = position * scale * tint; }\n";
        let result = compile(src, ShaderStage::Vertex);
        let CompileResult::Ok(module) = result else {
            panic!("{result:?}");
        };
        let uniforms: Vec<_> = module
            .global_variables
            .iter()
            .filter(|(_, gv)| gv.space == naga::AddressSpace::Uniform)
            .map(|(_, gv)| (gv.name.clone(), gv.binding))
            .collect();
        assert_eq!(
            uniforms,
            vec![
                (
                    Some("scale".to_string()),
                    Some(naga::ResourceBinding { group: 0, binding: 0 })
                ),
                (
                    Some("tint".to_string()),
                    Some(naga::ResourceBinding { group: 0, binding: 1 })
                ),
            ]
        );
    }

    #[test]
    fn fragment_stage_uniforms_use_the_disjoint_binding_range() {
        let out = inject_uniform_bindings("uniform vec4 a; uniform vec4 b;", ShaderStage::Fragment);
        assert_eq!(
            out,
            format!(
                "layout(binding = {FRAGMENT_BINDING_BASE}) uniform vec4 a; layout(binding = {}) uniform vec4 b;",
                FRAGMENT_BINDING_BASE + 1
            )
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
