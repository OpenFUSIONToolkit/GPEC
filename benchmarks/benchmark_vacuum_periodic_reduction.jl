# Benchmark: field-periodic vacuum reduction (layer-2) vs. full-torus dense path.
#
# Compares runtime and peak-memory between the two 3D boundary-integral paths for the
# nfp-periodic case:
#
#   full  — expand_field_periods(inputs) forces nfp=1, so compute_vacuum_response runs
#            the original O(N×N) dense double-layer kernel + two N×N LU factorizations.
#   reduced — nfp > 1 dispatches _compute_vacuum_response_3d_periodic, which builds only
#            one period's block-row (M×N, M = N/nfp) and solves per-residue-class M×M
#            systems via the block-circulant DFT decomposition.
#
# Usage (from JPEC root):
#   julia --project=. -t auto benchmarks/benchmark_vacuum_periodic_reduction.jl
#
# Reported quantities per case:
#   runtime        — wall-clock seconds averaged over N_REPS warm runs
#   theory peak    — size of the dominant matrices that must be simultaneously resident:
#                      full:    3 × N×N Float64 matrices (grad_green, grad_green_interior,
#                               green_temp) ≈ 3N²×8 B, pooled by AdaptiveArrayPools
#                      reduced: 2 × M×N Float64 (block-row) + 2 × M×M Complex128 (D̂, Ŝ)
#                               ≈ (2MN×8 + 4M²×8) B
#                    This is the figure that matters for OOM: how much RAM the
#                    algorithm requires, regardless of allocator pooling.
#   @allocated     — bytes newly allocated from the GC during a warm call.
#                    NOTE: the full path reuses N×N pools from AdaptiveArrayPools after
#                    warmup, so @allocated is near zero for the full path even though the
#                    pool itself holds ~3N²×8 B of live memory.  Use "theory peak" for
#                    the honest memory comparison; @allocated is shown as a secondary check.
#   max rel err    — max|wv_red - wv_full| / max|wv_full|  (should be ≲ 1e-14)

using GeneralizedPerturbedEquilibrium.Vacuum
using GeneralizedPerturbedEquilibrium.Vacuum: expand_field_periods
using Printf
using Statistics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of warm (timed) repetitions per case after a single untimed warmup pass.
const N_REPS = 2

using AdaptiveArrayPools: MAYBE_POOLING
MAYBE_POOLING[] = false


# Benchmark cases: increasing full-torus point count N = mtheta * nzeta_p * nfp.
# Each row: (label, mtheta, nzeta_p, nfp, m_modes, n_modes)
# mtheta * nzeta_p must be ≥ PATCH_DIM (23 for default KernelParams3D(11, 20, 5)).
const CASES = [
    (label="small  (N=1152)", mtheta=24, nzeta_p=48, nfp=3, m_modes=collect(-2:2), n_modes=[1, 2, 3]),
    (label="medium (N=2304)", mtheta=48, nzeta_p=48, nfp=3, m_modes=collect(-3:3), n_modes=[1, 2, 3, 4, 5, 6]),
    (label="large  (N=4608)", mtheta=48, nzeta_p=48, nfp=4, m_modes=collect(-3:3), n_modes=[1, 2, 3, 4, 5, 6]),
    (label="stride (N=4608)", mtheta=48, nzeta_p=48, nfp=3, m_modes=collect(-3:3), n_modes=[1, 4])
]

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

"""
    make_one_period(mtheta, nzeta_p, nfp; R0, a, b, ε) -> (X, Y, Z)

Build Cartesian (X, Y, Z) coordinates for one field period of a slightly
non-axisymmetric tokamak boundary. The corrugation amplitude ε breaks axisymmetry while
preserving exact nfp-periodicity under rotation by 2π/nfp about Z.

The output arrays are flat vectors of length mtheta * nzeta_p for direct use in VacuumInput.
Toroidal grid: ζⱼ = (j-1) · 2π / (nzeta_p · nfp), i.e. period-0 of the full-torus grid.
"""
function make_one_period(mtheta, nzeta_p, nfp; R0=1.7, a=0.3, b=0.3, ε=0.05)
    θ_grid = range(; start=0, length=mtheta, step=2π/mtheta)
    X = zeros(mtheta, nzeta_p)
    Y = zeros(mtheta, nzeta_p)
    Z = zeros(mtheta, nzeta_p)
    for j in 1:nzeta_p
        ζ = (j - 1) * 2π / (nzeta_p * nfp)
        for (i, θ) in enumerate(θ_grid)
            # Circular cross-section with a corrugation at the field-period frequency
            R = R0 + a * cos(θ) + ε * cos(θ + nfp * ζ)
            X[i, j] = R * cos(ζ)
            Y[i, j] = R * sin(ζ)
            Z[i, j] = b * sin(θ) + ε * sin(θ + nfp * ζ)
        end
    end
    return vec(X), vec(Y), vec(Z)
end

"""
    make_inputs(mtheta, nzeta_p, nfp, m_modes, n_modes; symmetry) -> (inputs_red, inputs_full)

Return a pair of VacuumInput structs for the same physical boundary:

  - `inputs_red`: one field period + nfp > 1 → dispatches to the reduced path
  - `inputs_full`: full torus (nfp = 1) → dispatches to the dense path
"""
function make_inputs(mtheta, nzeta_p, nfp, m_modes, n_modes; symmetry=true)
    Xp, Yp, Zp = make_one_period(mtheta, nzeta_p, nfp)
    base = VacuumInput(;
        x=Xp, y=Yp, z=Zp,
        mtheta_in=mtheta, nzeta_in=nzeta_p,
        m_modes=m_modes, n_modes=n_modes,
        mtheta=mtheta, nzeta=nzeta_p,
        force_wv_symmetry=symmetry,
        nfp=nfp
    )
    full = expand_field_periods(base)   # nfp set to 1; same geometry tiled to full torus
    return base, full
end

# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

"""
Run `f()` once discarding the result; use to trigger JIT compilation.
"""
warmup(f) = (f(); nothing)

"""
Measure memory used by a single call to `f()`.

Returns `(result, alloc_MiB, rss_delta_MiB)` where:

  - `alloc_MiB`     – bytes allocated by Julia GC (from `@allocated`), converted to MiB.
    This captures all heap traffic including short-lived temporaries.
  - `rss_delta_MiB` – change in process RSS (from `Sys.maxrss()`), converted to MiB.
    Reflects actual VM committed; can be 0 at small N if the OS
    already committed pages during warmup.

The GC is flushed before the call so the RSS baseline is as clean as possible.
"""
function measure_mem(f)
    GC.gc(true)
    base_rss = Sys.maxrss()
    alloc_bytes = @allocated result = f()
    rss_delta = Sys.maxrss() - base_rss
    return result, alloc_bytes / 2^20, rss_delta / 2^20
end

"""
Time `N_REPS` calls to `f()` and return (mean_seconds, last_result).
"""
function measure_time(f)
    times = Vector{Float64}(undef, N_REPS)
    local last
    for k in 1:N_REPS
        t0 = time()
        last = f()
        times[k] = time() - t0
    end
    return mean(times), last
end

# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

println("=" ^ 78)
println("3D Vacuum: Field-Periodic Reduction (layer-2) vs. Full-Torus Dense Path")
println("Threads: $(Threads.nthreads())")
println("=" ^ 78)

wall_settings = WallShapeSettings(; shape="nowall")

for c in CASES
    (; label, mtheta, nzeta_p, nfp, m_modes, n_modes) = c
    inputs_red, inputs_full = make_inputs(mtheta, nzeta_p, nfp, m_modes, n_modes)

    N = mtheta * nzeta_p * nfp
    M = mtheta * nzeta_p
    num_modes = length(m_modes) * length(n_modes)
    n_classes = length(unique(mod.(n_modes, nfp)))

    println("\n$(label)  N=$(N) ($(mtheta)×$(nzeta_p)/period × nfp=$(nfp))  " *
            "M=$(M)  modes=$(num_modes)  residue-classes=$(n_classes)")
    println("-" ^ 78)

    # JIT warmup: small untimed call to avoid measuring compilation
    warmup(() -> compute_vacuum_response(inputs_red, wall_settings))
    warmup(() -> compute_vacuum_response(inputs_full, wall_settings))

    # Runtime
    t_full, wv_full_t = measure_time(() -> compute_vacuum_response(inputs_full, wall_settings))
    t_red, wv_red_t = measure_time(() -> compute_vacuum_response(inputs_red, wall_settings))
    wv_full = wv_full_t[1]
    wv_red = wv_red_t[1]

    # Theoretical peak memory: size of dominant matrices that must be simultaneously resident.
    # Full path: 3 dense N×N Float64 (grad_green, grad_green_interior, green_temp) pooled by AdaptiveArrayPools.
    # Reduced path: 2 M×N Float64 block-rows + 2 M×M Complex128 per-sector buffers (D̂, Ŝ).
    theory_full_mib = 3 * N^2 * 8 / 2^20
    theory_red_mib = (2 * M * N * 8 + 4 * M^2 * 8) / 2^20

    # @allocated for a warm call (see header note: full path shows near-zero due to pooling).
    _, alloc_full, _ = measure_mem(() -> compute_vacuum_response(inputs_full, wall_settings))
    _, alloc_red, _ = measure_mem(() -> compute_vacuum_response(inputs_red, wall_settings))

    # Correctness
    scale = max(maximum(abs, wv_full), eps())
    max_relerr = maximum(abs, wv_red .- wv_full) / scale

    @printf("  %-22s  time=%6.3f s  theory_peak=%6.0f MiB  @alloc=%5.0f MiB\n",
        "full (dense N×N)", t_full, theory_full_mib, alloc_full)
    @printf("  %-22s  time=%6.3f s  theory_peak=%6.0f MiB  @alloc=%5.0f MiB\n",
        "reduced (block-circ.)", t_red, theory_red_mib, alloc_red)
    @printf("  speedup: %.2f×   theory mem reduction: %.1f×   max rel err: %.2e\n",
        t_full / t_red,
        theory_full_mib / max(theory_red_mib, 0.1),
        max_relerr)

    if max_relerr > 1e-8
        @warn "  max_relerr=$(max_relerr) exceeds 1e-8 — check correctness"
    end
end

println()
println("=" ^ 78)
println("Scaling notes")
println("  full: 3 × N×N Float64 ≈ 3N²×8 B  — grows as O(N²)")
println("  reduced: 2 × M×N + 2 × M×M Complex ≈ (2MN + 4M²)×8 B  — grows as O(MN) = O(N²/nfp)")
println("  Memory ratio at fixed N approaches nfp/2 for large N (dominated by MN vs N²).")
println("  Runtime ratio approaches nfp² for large N (dominant cost shifts from kernel→LU).")
println("=" ^ 78)
