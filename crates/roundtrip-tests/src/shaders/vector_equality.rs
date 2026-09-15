//! Roundtrip tests for vector comparison lowering (wgsl-rs#164).
//!
//! Tests:
//! - `==` on vectors, lowered to `all(lhs == rhs)` (bool, matching Rust's
//!   elementwise-all `PartialEq`)
//! - `!=` on vectors, lowered to `!(all(lhs == rhs))` (matching
//!   `PartialEq::ne`)
//! - `cmp_eq` / `cmp_ne`, the componentwise escape hatches that render as raw
//!   WGSL operators and yield a `vecN<bool>` mask
//!
//! Covers Vec4f pairs with both equal and unequal inputs so every test
//! exercises both branches.

use wgsl_rs::wgsl;

use crate::harness::{self, ComparisonResult, RoundtripTest};

const N: usize = 64;

/// vec_eq: `==` on Vec4f pairs, lowered to `all(a == b)`.
#[wgsl]
pub mod vec_eq {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 512]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);
        let base = idx * 8;
        let a = vec4f(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let b = vec4f(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );

        if a == b {
            get_mut!(OUTPUT)[idx] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx] = 0u32;
        }
    }
}

/// vec_ne: `!=` on Vec4f pairs, lowered to `!(all(a == b))`.
#[wgsl]
pub mod vec_ne {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 512]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);
        let base = idx * 8;
        let a = vec4f(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let b = vec4f(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );

        if a != b {
            get_mut!(OUTPUT)[idx] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx] = 0u32;
        }
    }
}

/// cmp_eq_mask: `any(cmp_eq(a, b))` — `cmp_eq` renders as the raw `==`
/// operator, yielding a vec4<bool> mask reduced by `any`.
#[wgsl]
pub mod cmp_eq_mask {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 512]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);
        let base = idx * 8;
        let a = vec4f(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let b = vec4f(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );

        if any(cmp_eq(a, b)) {
            get_mut!(OUTPUT)[idx] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx] = 0u32;
        }
    }
}

/// cmp_ne_mask: `any(cmp_ne(a, b))` — `cmp_ne` renders as the raw `!=`
/// operator, yielding a vec4<bool> mask reduced by `any`.
#[wgsl]
pub mod cmp_ne_mask {
    use wgsl_rs::std::*;

    storage!(group(0), binding(0), INPUT: [f32; 512]);
    storage!(group(0), binding(1), read_write, OUTPUT: [u32; 64]);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let idx = global_id.x as usize;
        let input = get!(INPUT);
        let base = idx * 8;
        let a = vec4f(
            input[base],
            input[base + 1],
            input[base + 2],
            input[base + 3],
        );
        let b = vec4f(
            input[base + 4],
            input[base + 5],
            input[base + 6],
            input[base + 7],
        );

        if any(cmp_ne(a, b)) {
            get_mut!(OUTPUT)[idx] = 1u32;
        } else {
            get_mut!(OUTPUT)[idx] = 0u32;
        }
    }
}

/// Deterministic input: 64 pairs of vec4s. Every 4th pair is identical (so
/// the equality paths take their true branch); the other pairs differ in
/// every component.
fn vector_equality_inputs() -> [f32; 512] {
    let mut v = [0.0f32; 512];
    for i in 0..N {
        let a = [i as f32, i as f32 + 1.0, i as f32 + 2.0, i as f32 + 3.0];
        let b = if i % 4 == 0 {
            a
        } else {
            [a[0] + 1.0, a[1] + 2.0, a[2] + 3.0, a[3] + 4.0]
        };
        let base = i * 8;
        v[base..base + 4].copy_from_slice(&a);
        v[base + 4..base + 8].copy_from_slice(&b);
    }
    v
}

// ============================================================================
// Test Implementation
// ============================================================================

pub struct VectorEqualityTest;

impl RoundtripTest for VectorEqualityTest {
    fn name(&self) -> &str {
        "vector_equality"
    }

    fn description(&self) -> &str {
        "vector ==, !=, cmp_eq, cmp_ne"
    }

    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<ComparisonResult> {
        use wgsl_rs::std::*;

        let mut results = Vec::new();

        macro_rules! drive {
            ($mod_name:ident, $label:expr) => {{
                let inputs = vector_equality_inputs();
                let input_bytes = bytemuck::cast_slice::<f32, u8>(&inputs);

                let mut linkage =
                    wgsl_rs::linkage::wgpu::analyze_wgsl_module(&$mod_name::WGSL_SOURCE).unwrap();
                let gpu_output =
                    harness::run_gpu_compute_linked(&mut harness::GpuComputeParamsLinked {
                        device,
                        queue,
                        linkage: &mut linkage,
                        entry: "main",
                        input_data: input_bytes,
                        output_size: (N * std::mem::size_of::<u32>()) as u64,
                        workgroup_count: (1, 1, 1),
                    });

                let gpu_results = bytemuck::cast_slice::<u8, u32>(&gpu_output);

                $mod_name::INPUT.set(inputs);
                $mod_name::OUTPUT.set([0u32; N]);
                dispatch_workgroups(
                    (1, 1, 1),
                    linkage
                        .compute_entry("main")
                        .expect("main entry present")
                        .workgroup_size,
                    |builtins| {
                        $mod_name::main(builtins.global_invocation_id);
                    },
                );
                let cpu_results: Vec<u32> = $mod_name::OUTPUT.get().to_vec();

                let labels: Vec<String> = (0..N).map(|i| format!("{}[{}]", $label, i)).collect();
                let label_refs: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
                results.push(harness::compare_u32_results(
                    $label,
                    gpu_results,
                    &cpu_results,
                    &label_refs,
                ));
            }};
        }

        drive!(vec_eq, "vec_eq");
        drive!(vec_ne, "vec_ne");
        drive!(cmp_eq_mask, "cmp_eq_mask");
        drive!(cmp_ne_mask, "cmp_ne_mask");

        results
    }
}
