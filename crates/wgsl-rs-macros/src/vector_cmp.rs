//! Vector comparison lowering for wgsl-rs#164.
//!
//! Rust's `==` on vectors produces a `bool` (`PartialEq`), but WGSL's `==`
//! on vectors produces `vecN<bool>` — so wherever a vector comparison
//! feeds a `bool`-typed position (let bindings, returns, conditions) the
//! generated WGSL is invalid. The IR does not track expression types
//! (#145), so this pass walks the parse AST *after* monomorphization —
//! when generic types are concrete — and rewrites comparisons whose
//! operands both infer as vector-typed:
//!
//! * `a == b`  ->  `all(a == b)`
//! * `a != b`  ->  `!(all(a == b))`   (matching `PartialEq::ne`)
//!
//! Inference is deliberately conservative: operands whose types cannot be
//! determined are left untouched (fail-open), so CPU-only code keeps
//! compiling and un-inferable comparisons behave exactly as before. The
//! componentwise escape hatch is `cmp_eq(a, b)`, which lowers at render
//! time via the binary-operator entries in `builtin_lookup` and is never
//! touched by this pass (`FnCall` nodes are not `Binary` nodes).
//!
//! Known limitation, shared with the #160 `!` -> `~` lowering: generic
//! entry-point templates still carry `TypeParam` types when their IR is
//! built lazily at runtime, so a vector `==` inside a runtime-instantiated
//! generic template is not wrapped. Monomorphized functions — the common
//! case — are covered, because `monomorphize` appends their concrete
//! copies to the module content before this pass runs.

use crate::parse::{
    self, BinOp, Block, CaseSelector, ElseBody, Expr, FnPath, ImplItem, Item, Lit, ReturnType,
    ScalarType, Stmt, Type, UnOp,
};
use proc_macro2::{Ident, Span};
use std::collections::HashMap;
use syn::punctuated::Punctuated;

/// Placeholder used to temporarily empty an expression slot while its
/// contents are being rebuilt. Never survives the pass.
const TMP_IDENT: &str = "__wgsl_vector_cmp_tmp";

/// Module-wide symbol information used to infer expression types.
struct Symbols {
    /// Free functions by name -> return type (`None` for `->`-less fns).
    /// Monomorphized instances are included: mono appends their concrete
    /// copies to module content before this pass runs. Impl methods are
    /// excluded — calls to them go through `FnPath::TypeMethod`, which
    /// inference does not resolve.
    fns: HashMap<String, Option<Type>>,
    /// Struct fields by struct name: `(field name, type)`.
    structs: HashMap<String, Vec<(String, Type)>>,
    /// Module-level consts by name.
    consts: HashMap<String, Type>,
}

/// Scoped local-variable types, innermost scope last.
struct Locals {
    scopes: Vec<HashMap<String, Type>>,
}

impl Locals {
    fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        self.scopes.pop();
    }

    fn insert(&mut self, name: &str, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    fn get(&self, name: &str) -> Option<&Type> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }
}

fn scalar_ty(t: ScalarType, span: Span) -> Type {
    Type::Scalar {
        ty: t,
        ident: Ident::new(t.wgsl_name(), span),
    }
}

/// Build a vector type for inference purposes. The ident is a plausible
/// alias (`vec2f`); it is only ever fed back into inference, never
/// rendered.
fn vector_ty(elements: u8, t: ScalarType, span: Span) -> Type {
    let alias = format!("vec{elements}{}", t.short_name());
    Type::Vector {
        elements,
        scalar_ty: t,
        ident: Ident::new(&alias, span),
        scalar: None,
    }
}

/// Parse a WGSL vector constructor alias: `vec2f` -> `(2, F32)`, etc.
fn vector_ctor_alias(name: &str) -> Option<(u8, ScalarType)> {
    let rest = name.strip_prefix("vec")?;
    let mut chars = rest.chars();
    let elements = match chars.next()? {
        '2' => 2,
        '3' => 3,
        '4' => 4,
        _ => return None,
    };
    let scalar = match chars.next()? {
        'f' => ScalarType::F32,
        'i' => ScalarType::I32,
        'u' => ScalarType::U32,
        'b' => ScalarType::Bool,
        _ => return None,
    };
    if chars.next().is_some() {
        return None;
    }
    Some((elements, scalar))
}

/// Combine inferred operand types of an arithmetic/bitwise binary
/// operation: a vector operand dominates an unknown one, two scalars
/// keep the left type.
fn combine_arith(lt: Option<Type>, rt: Option<Type>) -> Option<Type> {
    match (lt, rt) {
        (Some(l @ Type::Vector { .. }), _) => Some(l),
        (_, Some(r @ Type::Vector { .. })) => Some(r),
        (Some(l @ Type::Scalar { .. }), Some(_)) => Some(l),
        _ => None,
    }
}

impl Symbols {
    /// Collect module-wide symbol information (functions, structs, consts),
    /// recursing into nested modules.
    fn build(content: &[Item]) -> Self {
        let mut fns = HashMap::new();
        let mut structs = HashMap::new();
        let mut consts = HashMap::new();
        Self::collect(content, &mut fns, &mut structs, &mut consts);
        Self {
            fns,
            structs,
            consts,
        }
    }

    fn collect(
        content: &[Item],
        fns: &mut HashMap<String, Option<Type>>,
        structs: &mut HashMap<String, Vec<(String, Type)>>,
        consts: &mut HashMap<String, Type>,
    ) {
        for item in content {
            match item {
                Item::Fn(f) => {
                    let ret = match &f.return_type {
                        ReturnType::Type { ty, .. } => Some((**ty).clone()),
                        ReturnType::Default => None,
                    };
                    fns.insert(f.ident.to_string(), ret);
                }
                Item::Struct(s) => {
                    structs.insert(
                        s.ident.to_string(),
                        s.fields
                            .named
                            .iter()
                            .map(|field| (field.ident.to_string(), field.ty.clone()))
                            .collect(),
                    );
                }
                Item::Const(c) => {
                    consts.insert(c.ident.to_string(), c.ty.clone());
                }
                Item::Mod(m) => Self::collect(&m.content, fns, structs, consts),
                _ => {}
            }
        }
    }

    /// Infer the type of an expression, or `None` when it cannot be
    /// determined (the caller must then fail open).
    fn infer(&self, e: &Expr, locals: &Locals) -> Option<Type> {
        match e {
            Expr::Lit(Lit::Bool(_)) => Some(scalar_ty(ScalarType::Bool, e.span())),
            Expr::Lit(Lit::Float(_)) => Some(scalar_ty(ScalarType::F32, e.span())),
            Expr::Lit(Lit::Int(i)) => {
                let scalar = match i.suffix() {
                    "u32" | "usize" => ScalarType::U32,
                    _ => ScalarType::I32,
                };
                Some(scalar_ty(scalar, e.span()))
            }
            Expr::Ident(id) => locals
                .get(&id.to_string())
                .cloned()
                .or_else(|| self.consts.get(&id.to_string()).cloned()),
            Expr::Paren { inner, .. } => self.infer(inner, locals),
            Expr::Reference { expr, .. } => self.infer(expr, locals),
            Expr::Unary { op, expr } => match op {
                // `!vecN<bool>` is componentwise in WGSL and stays a bool
                // vector; any other operand makes `!x` a plain bool.
                UnOp::Not(_) => match self.infer(expr, locals) {
                    Some(
                        t @ Type::Vector {
                            scalar_ty: ScalarType::Bool,
                            ..
                        },
                    ) => Some(t),
                    _ => Some(scalar_ty(ScalarType::Bool, e.span())),
                },
                UnOp::Neg(_) | UnOp::Deref(_) => self.infer(expr, locals),
            },
            Expr::Binary { lhs, op, rhs } => match op {
                // Comparisons and logical ops are bool-typed in Rust
                // regardless of operand types.
                BinOp::Eq(_)
                | BinOp::Ne(_)
                | BinOp::Lt(_)
                | BinOp::Le(_)
                | BinOp::Gt(_)
                | BinOp::Ge(_)
                | BinOp::And(_)
                | BinOp::Or(_) => Some(scalar_ty(ScalarType::Bool, e.span())),
                // Shifts take the lhs's type.
                BinOp::Shl(_) | BinOp::Shr(_) => self.infer(lhs, locals),
                _ => combine_arith(self.infer(lhs, locals), self.infer(rhs, locals)),
            },
            Expr::Cast { ty, .. } => Some((**ty).clone()),
            Expr::Swizzle {
                lhs,
                swizzle,
                params,
                ..
            } => {
                // A swizzle with call arguments (e.g. matrix access via
                // `m.x(i)`) is an indexing operation, not a value.
                if params.as_ref().is_some_and(|p| !p.is_empty()) {
                    return None;
                }
                let n = swizzle.to_string().chars().count();
                let base = self.infer(lhs, locals);
                let elem = match &base {
                    Some(Type::Vector { scalar_ty: s, .. }) => Some(*s),
                    // Matrices are f32-only, so their components are f32.
                    Some(Type::Matrix { .. }) => Some(ScalarType::F32),
                    _ => None,
                };
                match n {
                    // Single component: the base's element scalar.
                    1 => match (&base, elem) {
                        (_, Some(s)) => Some(scalar_ty(s, e.span())),
                        (Some(Type::Scalar { ty, .. }), _) => Some(scalar_ty(*ty, e.span())),
                        _ => None,
                    },
                    // Multi-component swizzles are vectors.
                    2..=4 => Some(vector_ty(
                        n as u8,
                        elem.unwrap_or(ScalarType::F32),
                        e.span(),
                    )),
                    _ => None,
                }
            }
            Expr::ArrayIndexing { lhs, .. } => match self.infer(lhs, locals)? {
                Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } => Some((*elem).clone()),
                // `m[i]` is a column vector; matrices are f32-only.
                Type::Matrix { rows, .. } => Some(vector_ty(rows, ScalarType::F32, e.span())),
                Type::Vector { scalar_ty: s, .. } => Some(scalar_ty(s, e.span())),
                _ => None,
            },
            Expr::FnCall {
                path: FnPath::Ident(f),
                type_args,
                ..
            } => {
                let name = f.to_string();
                // A user-defined fn of the same name wins over builtins and
                // constructor aliases.
                if let Some(ret) = self.fns.get(&name) {
                    return ret.clone();
                }
                // `all`/`any` reduce a vecN<bool> to bool.
                if name == "all" || name == "any" {
                    return Some(scalar_ty(ScalarType::Bool, e.span()));
                }
                if let Some((elements, scalar)) = vector_ctor_alias(&name) {
                    return Some(vector_ty(elements, scalar, e.span()));
                }
                // `vec2::<f32>(...)` style: the turbofish supplies the
                // scalar type.
                if matches!(name.as_str(), "vec2" | "vec3" | "vec4")
                    && type_args.len() == 1
                    && let Type::Scalar { ty, .. } = &type_args[0]
                    && let Some(elements) = name.chars().nth(3).and_then(|c| c.to_digit(10))
                {
                    return Some(vector_ty(elements as u8, *ty, e.span()));
                }
                None
            }
            Expr::FnCall {
                path: FnPath::TypeMethod { .. },
                ..
            } => None,
            Expr::Struct { ident, .. } => Some(Type::Struct {
                ident: ident.clone(),
                type_args: Vec::new(),
                const_args: Vec::new(),
            }),
            Expr::FieldAccess { base, field, .. } => match self.infer(base, locals)? {
                Type::Struct { ident, .. } => self
                    .structs
                    .get(&ident.to_string())?
                    .iter()
                    .find(|(name, _)| name == &field.to_string())
                    .map(|(_, ty)| ty.clone()),
                _ => None,
            },
            // Array literals, type paths, zero-value arrays and linkage
            // accesses: not inferable here.
            _ => None,
        }
    }

    /// Transform `e` in place (wrapping any vector `==`/`!=` found inside)
    /// and return the inferred type of the result, if known.
    fn tx(&self, e: &mut Expr, locals: &mut Locals) -> Option<Type> {
        if !matches!(e, Expr::Binary { .. }) {
            self.tx_children(e, locals);
            return self.infer(e, locals);
        }

        // Take the whole Binary node out of `e` so the children can be
        // transformed and the node reassembled without borrow conflicts.
        let span = e.span();
        let inner = std::mem::replace(e, Expr::Ident(Ident::new(TMP_IDENT, span)));
        let (mut lhs, op, mut rhs) = match inner {
            Expr::Binary { lhs, op, rhs } => (lhs, op, rhs),
            other => {
                *e = other;
                return None; // Unreachable: guarded above.
            }
        };
        let lt = self.tx(&mut lhs, locals);
        let rt = self.tx(&mut rhs, locals);

        /// Both comparisons are lowered the same way when both operands are
        /// vector-typed; scalar/unknown comparisons are left raw.
        #[derive(PartialEq)]
        enum OpKind {
            Eq,
            Ne,
            Shift,
            Arith,
        }
        let kind = match &op {
            BinOp::Eq(_) => OpKind::Eq,
            BinOp::Ne(_) => OpKind::Ne,
            BinOp::Shl(_) | BinOp::Shr(_) => OpKind::Shift,
            _ => OpKind::Arith,
        };

        let both_vec = matches!(
            (&lt, &rt),
            (Some(Type::Vector { .. }), Some(Type::Vector { .. }))
        );

        if (kind == OpKind::Eq || kind == OpKind::Ne) && both_vec {
            // `a == b` -> `all(a == b)`; `a != b` -> `!(all(a == b))`,
            // matching `PartialEq::ne == !(self == other)`. Note the inner
            // operator is always `==`: `all(a != b)` means "every component
            // differs", which is NOT `PartialEq::ne`.
            let mut params = Punctuated::new();
            let inner_op = match kind {
                OpKind::Eq => op,
                OpKind::Ne => BinOp::Eq(Default::default()),
                _ => unreachable!("guarded by both_vec + kind check"),
            };
            params.push(Expr::Binary {
                lhs,
                op: inner_op,
                rhs,
            });
            let all = Expr::FnCall {
                path: FnPath::Ident(Ident::new("all", span)),
                type_args: Vec::new(),
                const_args: Vec::new(),
                paren_token: Default::default(),
                params,
            };
            *e = if kind == OpKind::Eq {
                all
            } else {
                Expr::Unary {
                    op: UnOp::Not(Default::default()),
                    expr: Box::new(all),
                }
            };
            return Some(scalar_ty(ScalarType::Bool, span));
        }

        let combined = match kind {
            OpKind::Shift => lt,
            OpKind::Arith => combine_arith(lt, rt),
            // Comparisons and logical ops are bool in Rust.
            _ => Some(scalar_ty(ScalarType::Bool, span)),
        };
        *e = Expr::Binary { lhs, op, rhs };
        combined
    }

    /// Recurse into an expression's children without inferring this node.
    fn tx_children(&self, e: &mut Expr, locals: &mut Locals) {
        match e {
            Expr::Paren { inner, .. } => {
                self.tx(inner, locals);
            }
            Expr::Reference { expr, .. } => {
                self.tx(expr, locals);
            }
            Expr::Unary { expr, .. } => {
                self.tx(expr, locals);
            }
            Expr::FnCall { params, .. } => {
                for p in params.iter_mut() {
                    self.tx(p, locals);
                }
            }
            Expr::Array { elems, .. } => {
                for el in elems.iter_mut() {
                    self.tx(el, locals);
                }
            }
            Expr::ArrayIndexing { lhs, index, .. } => {
                self.tx(lhs, locals);
                self.tx(index, locals);
            }
            Expr::Swizzle { lhs, params, .. } => {
                self.tx(lhs, locals);
                if let Some(args) = params {
                    for p in args.iter_mut() {
                        self.tx(p, locals);
                    }
                }
            }
            Expr::Cast { lhs, .. } => {
                self.tx(lhs, locals);
            }
            Expr::Struct { fields, .. } => {
                for field in fields.iter_mut() {
                    self.tx(&mut field.expr, locals);
                }
            }
            Expr::FieldAccess { base, .. } => {
                self.tx(base, locals);
            }
            Expr::ZeroValueArray { len, .. } => {
                self.tx(len, locals);
            }
            _ => {}
        }
    }

    /// Transform a block's statements in a fresh scope.
    fn tx_block(&self, b: &mut Block, locals: &mut Locals) {
        locals.push();
        for stmt in b.stmt.iter_mut() {
            self.tx_stmt(stmt, locals);
        }
        locals.pop();
    }

    fn tx_if(&self, i: &mut parse::StmtIf, locals: &mut Locals) {
        self.tx(&mut i.condition, locals);
        self.tx_block(&mut i.then_block, locals);
        if let Some(else_branch) = &mut i.else_branch {
            match &mut else_branch.body {
                ElseBody::Block(block) => self.tx_block(block, locals),
                ElseBody::If(nested) => self.tx_if(nested, locals),
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn tx_stmt(&self, s: &mut Stmt, locals: &mut Locals) {
        match s {
            Stmt::Local(l) => {
                // Transform the initializer first so a `let` bound to a
                // vector comparison records the *wrapped* (bool) type.
                let inferred = l
                    .init
                    .as_mut()
                    .and_then(|init| self.tx(&mut init.expr, locals));
                let bound = match &l.ty {
                    Some((_, ty)) => Some(ty.clone()),
                    None => inferred,
                };
                if let Some(ty) = bound {
                    locals.insert(&l.ident.to_string(), ty);
                }
            }
            Stmt::Const(c) => {
                self.tx(&mut c.expr, locals);
                locals.insert(&c.ident.to_string(), c.ty.clone());
            }
            Stmt::Assignment { lhs, rhs, .. } => {
                self.tx(lhs, locals);
                self.tx(rhs, locals);
            }
            Stmt::CompoundAssignment { lhs, rhs, .. } => {
                self.tx(lhs, locals);
                self.tx(rhs, locals);
            }
            Stmt::While {
                condition, body, ..
            } => {
                self.tx(condition, locals);
                self.tx_block(body, locals);
            }
            Stmt::Loop { body, .. } => self.tx_block(body, locals),
            Stmt::Expr { expr, .. } => {
                self.tx(expr, locals);
            }
            Stmt::If(i) => self.tx_if(i, locals),
            Stmt::Break { .. } | Stmt::Continue { .. } | Stmt::Discard { .. } => {}
            Stmt::Return { expr, .. } => {
                if let Some(e) = expr {
                    self.tx(e, locals);
                }
            }
            Stmt::For(f) => {
                self.tx(&mut f.from, locals);
                self.tx(&mut f.to, locals);
                let var_ty = self.infer(&f.from, locals);
                locals.push();
                if let Some(ty) = var_ty {
                    locals.insert(&f.ident.to_string(), ty);
                }
                self.tx_block(&mut f.body, locals);
                locals.pop();
            }
            Stmt::Switch(sw) => {
                self.tx(&mut sw.selector, locals);
                for arm in sw.arms.iter_mut() {
                    for selector in arm.selectors.iter_mut() {
                        if let CaseSelector::Expr(e) = selector {
                            self.tx(e, locals);
                        }
                    }
                    self.tx_block(&mut arm.body, locals);
                }
            }
            Stmt::Block(b) => self.tx_block(b, locals),
            Stmt::SlabCopy {
                src,
                src_offset,
                dest,
                dest_offset,
                size,
                ..
            } => {
                self.tx(src, locals);
                self.tx(src_offset, locals);
                self.tx(dest, locals);
                self.tx(dest_offset, locals);
                self.tx(size, locals);
            }
            Stmt::Macro { .. } => {}
        }
    }

    /// Seed a locals scope from a function's parameters and walk its body.
    fn tx_fn(&self, f: &mut parse::ItemFn) {
        let mut locals = Locals::new();
        for arg in f.inputs.iter() {
            locals.insert(&arg.ident.to_string(), arg.ty.clone());
        }
        self.tx_block(&mut f.block, &mut locals);
    }
}

/// Rewrite vector `==`/`!=` comparisons in every function body of the
/// module (including monomorphized instances and impl methods).
pub(crate) fn rewrite(content: &mut [Item]) {
    let symbols = Symbols::build(content);
    rewrite_with(content, &symbols);
}

fn rewrite_with(content: &mut [Item], symbols: &Symbols) {
    for item in content.iter_mut() {
        match item {
            Item::Fn(f) => symbols.tx_fn(f),
            Item::Impl(i) => {
                for impl_item in i.items.iter_mut() {
                    match impl_item {
                        ImplItem::Fn(f) => symbols.tx_fn(f),
                        ImplItem::Const(c) => {
                            let mut locals = Locals::new();
                            symbols.tx(&mut c.expr, &mut locals);
                        }
                        ImplItem::Type(_) => {}
                    }
                }
            }
            Item::Const(c) => {
                let mut locals = Locals::new();
                symbols.tx(&mut c.expr, &mut locals);
            }
            // Nested modules build their own symbol table: names inside do
            // not resolve against the enclosing module's items.
            Item::Mod(m) => rewrite(&mut m.content),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_vector_ctor_aliases() {
        assert_eq!(vector_ctor_alias("vec2f"), Some((2, ScalarType::F32)));
        assert_eq!(vector_ctor_alias("vec3i"), Some((3, ScalarType::I32)));
        assert_eq!(vector_ctor_alias("vec4u"), Some((4, ScalarType::U32)));
        assert_eq!(vector_ctor_alias("vec2b"), Some((2, ScalarType::Bool)));
    }

    #[test]
    fn rejects_non_aliases() {
        assert_eq!(vector_ctor_alias("vec2"), None);
        assert_eq!(vector_ctor_alias("vec5f"), None);
        assert_eq!(vector_ctor_alias("vector"), None);
        assert_eq!(vector_ctor_alias("vec2ff"), None);
        assert_eq!(vector_ctor_alias("dot"), None);
    }
}
