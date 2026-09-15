//! Regression tests for wgsl-rs#164: vector `==` must lower to
//! `all(lhs == rhs)` and `!=` to `!(all(lhs == rhs))`.
//!
//! Rust's `PartialEq` on vectors yields `bool` (elementwise-all), but WGSL
//! `==` on vectors yields `vecN<bool>` — naga rejects the generated shader
//! wherever a `bool` is required (bindings, returns, conditions).
//!
//! These drive the real `#[wgsl]` macro, render to WGSL, and assert on both
//! specific substrings (to pin the exact lowering) and naga validation.
//! Like `unary_complement.rs`, `skip_validation` keeps the macro's
//! auto-emitted `__validate_wgsl` test off the `--no-default-features`
//! path; the explicit `#[cfg(feature = "validation")]` tests below cover
//! validation instead.

#![allow(dead_code)]
#![allow(unused_variables)]

use wgsl_rs::wgsl;

/// The #164 repro shape: a bool-typed let binding initialized from a
/// vector comparison. Today this renders `vecN<bool>` into a `bool`
/// binding, which naga rejects.
#[wgsl(skip_validation)]
mod eq_repro {
    use wgsl_rs::std::*;

    pub fn compute() -> bool {
        let x: bool = vec2f(0.0, 0.0) == vec2f(0.0, 0.0);
        x
    }
}

#[wgsl(skip_validation)]
mod eq_lowering {
    use wgsl_rs::std::*;

    /// Constructors on both sides.
    pub fn constructors_eq() -> bool {
        vec3f(1.0, 2.0, 3.0) == vec3f(1.0, 2.0, 3.0)
    }

    /// `!=` lowers to `!(all(lhs == rhs))` — matching `PartialEq::ne`,
    /// which is `!(self == other)`, not `all(lhs != rhs)`.
    pub fn constructors_ne() -> bool {
        vec4f(1.0, 2.0, 3.0, 4.0) != vec4f(4.0, 3.0, 2.0, 1.0)
    }

    /// Locals typed by their initializer.
    pub fn locals_eq(a: Vec2f, b: Vec2f) -> bool {
        let c = a;
        let d = b;
        c == d
    }

    /// Locals with explicit type annotations.
    pub fn annotated_eq(a: Vec3f, b: Vec3f) -> bool {
        let e: Vec3f = a;
        let f: Vec3f = b;
        e == f
    }

    /// Multi-component swizzles are vectors.
    pub fn swizzle_eq(a: Vec3f, b: Vec3f) -> bool {
        a.xzy() == b.zyx()
    }

    // NOTE: bool-vector params (`Vec2b` / `Vec2<bool>`) currently render as
    // the unknown WGSL identifier `vec2b`, so a `Vec2b == Vec2b` case cannot
    // validate yet. That type-render gap is separate from this fix.

    /// Monomorphized generic: the wrap must also apply inside the
    /// instantiated copy of `generic_eq`, where `T` has become `Vec2f`.
    pub fn generic_eq<T: PartialEq>(a: T, b: T) -> bool {
        a == b
    }

    pub fn mono_eq(a: Vec2f, b: Vec2f) -> bool {
        generic_eq::<Vec2f>(a, b)
    }

    /// `cmp_eq` is the componentwise escape hatch: it renders as the raw
    /// WGSL `==` operator, yielding a vecN<bool> mask. Feeding it to
    /// `all` exercises the whole path without needing a bool-vector
    /// return type in the signature (the `Vec2b` alias currently renders
    /// as an unknown identifier — a separate, pre-existing gap).
    pub fn mask_all_eq(a: Vec2f, b: Vec2f) -> bool {
        all(cmp_eq(a, b))
    }

    /// `cmp_ne` renders as the raw WGSL `!=` operator.
    pub fn mask_any_ne(a: Vec3f, b: Vec3f) -> bool {
        any(cmp_ne(a, b))
    }

    /// Scalars must NOT be wrapped.
    pub fn scalar_eq_stays(a: f32, b: f32) -> bool {
        a == b
    }

    /// Single-component swizzles are scalars and must NOT be wrapped.
    pub fn swizzle_scalar_ne_stays(a: Vec2f, b: Vec2f) -> bool {
        a.x != b.y
    }
}

// ===== WGSL source assertions =====

#[test]
fn repro_wraps_constructors_in_all() {
    let src = eq_repro::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    assert!(
        src.contains("let x: bool = all(vec2f(0.0, 0.0) == vec2f(0.0, 0.0));"),
        "the #164 repro must emit `all(...)`, got: {src}"
    );
}

#[test]
fn vector_eq_ne_lower_to_all() {
    let src = eq_lowering::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    for (needle, why) in [
        (
            "return all(vec3f(1.0, 2.0, 3.0) == vec3f(1.0, 2.0, 3.0));",
            "constructor == wraps in all()",
        ),
        (
            "return !all(vec4f(1.0, 2.0, 3.0, 4.0) == vec4f(4.0, 3.0, 2.0, 1.0));",
            "vector != lowers to !(all(lhs == rhs)); renders without parens around the call",
        ),
        ("return all(c == d);", "locals compared via =="),
        ("return all(e == f);", "annotated locals compared via =="),
        (
            "return all(a.xzy == b.zyx);",
            "multi-component swizzles are vectors",
        ),
        (
            "return all(a == b);",
            "monomorphized generic_eq wraps its body",
        ),
        (
            "return _1generic_eq_vec2f(a, b);",
            "mono call site is rewritten",
        ),
    ] {
        assert!(
            src.contains(needle),
            "expected `{needle}` ({why}) in: {src}"
        );
    }
    // The mono instance `_1generic_eq_vec2f` must wrap its body.
    assert!(
        src.contains("generic_eq") && src.contains("return all(a == b);"),
        "monomorphized generic_eq must wrap `a == b` in all(), got: {src}"
    );
    // No raw vector == survived unwrapped at a return position.
    assert!(
        !src.contains("return c == d;")
            && !src.contains("return e == f;")
            && !src.contains("return a.xzy() == b.zyx();"),
        "no vector ==/!= should survive lowering, got: {src}"
    );
}

#[test]
fn scalar_comparisons_stay_raw() {
    let src = eq_lowering::WGSL_SOURCE
        .wgsl_source()
        .expect("render should succeed");
    assert!(
        src.contains("return a == b;"),
        "scalar == must stay raw, got: {src}"
    );
    assert!(
        src.contains("return a.x != b.y;"),
        "single-component swizzle != must stay raw, got: {src}"
    );
}

// ===== naga validation =====

#[cfg(feature = "validation")]
#[test]
fn repro_validates() {
    eq_repro::WGSL_SOURCE.validate().expect("naga validation");
}

#[cfg(feature = "validation")]
#[test]
fn lowering_validates() {
    eq_lowering::WGSL_SOURCE
        .validate()
        .expect("naga validation");
}

// ===== CPU-side parity =====
//
// The `#[wgsl]` module stays valid Rust; these lock the two-worlds
// agreement: GPU `all(lhs == rhs)` must produce the same values as
// Rust's vector `PartialEq`.

#[test]
fn cpu_vector_eq_values() {
    use eq_lowering as m;
    use wgsl_rs::std::*;

    let v = vec2f(1.0, 2.0);
    let w = vec2f(3.0, 4.0);
    assert!(m::locals_eq(v, v));
    assert!(!m::locals_eq(v, w));
    assert!(m::annotated_eq(vec3f(1.0, 2.0, 3.0), vec3f(1.0, 2.0, 3.0)));
    assert!(!m::annotated_eq(vec3f(1.0, 2.0, 3.0), vec3f(3.0, 2.0, 1.0)));
    assert!(m::constructors_eq());
    assert!(m::constructors_ne());
    assert!(m::swizzle_eq(vec3f(1.0, 2.0, 3.0), vec3f(2.0, 3.0, 1.0)));
    assert!(!m::swizzle_eq(vec3f(1.0, 2.0, 3.0), vec3f(1.0, 2.0, 3.0)));
    assert!(m::mono_eq(v, v));
    assert!(!m::mono_eq(v, w));
    assert!(m::scalar_eq_stays(2.0, 2.0));
    assert!(!m::scalar_eq_stays(1.0, 2.0));
    assert!(m::swizzle_scalar_ne_stays(vec2f(1.0, 2.0), vec2f(3.0, 4.0)));
    assert!(m::mask_all_eq(v, v));
    assert!(!m::mask_all_eq(v, w));
    assert!(m::mask_any_ne(vec3f(1.0, 2.0, 3.0), vec3f(3.0, 2.0, 1.0)));
    assert!(!m::mask_any_ne(vec3f(1.0, 2.0, 3.0), vec3f(1.0, 2.0, 3.0)));
}
