# CoupledFull.jl
#
# Full Pletzer-Dewar 1991 / GWP 2016 coupled tearing + interchange
# dispersion: the 2m×2m eigenvalue problem
#
#     det( D' − D(γ) ) = 0
#
# with
#
#     D' = [ A'  B' ]      — from outer-region STRIDE-BVP matching
#          [ Γ'  Δ' ]        (parity-rotated via `pest3_decompose`)
#
#     D(γ) = diag(Δ_interchange_1, …, Δ_interchange_m,
#                 Δ_tearing_1,      …, Δ_tearing_m)
#
# where each `Δ_k` comes from the inner-layer model at surface k. In the
# pressureless limit (SLAYER), `Δ_interchange_k = 0` for all k, so the
# determinant reduces to
#
#     det(A') · det(Δ' − Δ_tearing(γ))                     (C.1)
#
# which agrees with the m×m `MultiSurfaceCoupling` result up to the
# constant prefactor det(A') — handy for regression testing the reduction.
#
# Ordering convention: **parity-major**, matching `dprime_outer_matrix`:
# rows/cols [interchange_s1, …, interchange_sm, tearing_s1, …, tearing_sm].
# This is the natural block structure for the 2×2-block D(γ) diagonal.
#
# This path is NEEDED for GGJ, where the interchange channel carries
# Glasser stabilization. It collapses to the existing `MultiSurfaceCoupling`
# scalar form for pure-tearing (SLAYER) studies.

"""
    MultiSurfaceCouplingFull{V<:AbstractVector{<:SurfaceCoupling}}

Full 2m×2m Pletzer-Dewar dispersion data: a vector of `SurfaceCoupling`
(one per singular surface), the 2m×2m outer-region matrix `D'` in
parity-major ordering, the reference-surface index (defines the Q
normalization via `tauk_ref / tauk_k`), and a truncation `msing_max`.

Calling `mc(Q)` returns `det( D' − D(γ) )` with `D(γ)` the 2m×2m
block-diagonal matrix of per-surface inner-layer responses:

```
upper-left  m×m diagonal:  (Δ_interchange_1, …, Δ_interchange_m)
lower-right m×m diagonal:  (Δ_tearing_1,      …, Δ_tearing_m)
```

Each `Δ_k` is computed as `solve_inner(model, params, Q·tauk_ref/tauk_k)`
and multiplied by `sc.scale` (inner→outer units; 1.0 for GGJ, S^(1/3)
for SLAYER). The `sc.dc` critical offset is subtracted from the
tearing-channel diagonal only (following Fortran SLAYER convention —
χ_parallel-matched dc only applies to the reconnecting channel).

A root in the complex `Q` plane is a coupled tearing+interchange
eigenvalue including Glasser stabilization.
"""
struct MultiSurfaceCouplingFull{V<:AbstractVector{<:SurfaceCoupling}}
    surfaces::V
    dp_full::Matrix{ComplexF64}   # 2m × 2m, parity-major
    ref_idx::Int
    msing_max::Int
end

"""
    multi_surface_coupling_full(surfaces, dp_full;
                                 ref_idx=1,
                                 msing_max=length(surfaces))
        -> MultiSurfaceCouplingFull

Construct a full-dispersion multi-surface coupling from a vector of
`SurfaceCoupling` and a 2m×2m parity-major `dp_full` matrix.

# Arguments

  - `surfaces`: vector of `SurfaceCoupling` (one per singular surface).
  - `dp_full`:  2m × 2m complex matrix in parity-major ordering
    `[A' B'; Γ' Δ']`. Typically obtained from
    `ForceFreeStates.dprime_outer_matrix(intr.delta_prime_raw)`.

# Keyword arguments

  - `ref_idx`   -- index of the reference surface (1 ≤ ref_idx ≤ m).
    Defaults to `1` (Fortran convention).
  - `msing_max` -- number of surfaces to include, counted from the front
    of `surfaces`. Truncates the determinant to the 2·msing_max ×
    2·msing_max upper-left parity-symmetric submatrix. Defaults to
    `length(surfaces)` (use all).
"""
function multi_surface_coupling_full(surfaces::AbstractVector{<:SurfaceCoupling},
                                     dp_full::AbstractMatrix;
                                     ref_idx::Integer=1,
                                     msing_max::Integer=length(surfaces))
    m = length(surfaces)
    size(dp_full) == (2m, 2m) ||
        throw(ArgumentError("multi_surface_coupling_full: dp_full size " *
                            "$(size(dp_full)) ≠ ($(2m), $(2m))"))
    1 <= ref_idx <= m ||
        throw(ArgumentError("multi_surface_coupling_full: ref_idx=$ref_idx " *
                            "out of range 1:$m"))
    1 <= msing_max <= m ||
        throw(ArgumentError("multi_surface_coupling_full: msing_max=$msing_max " *
                            "out of range 1:$m"))
    return MultiSurfaceCouplingFull(surfaces,
                                    Matrix{ComplexF64}(dp_full),
                                    Int(ref_idx), Int(msing_max))
end

# Extract the 2n×2n parity-symmetric sub-matrix for truncation
# msing_max = n ≤ m. Upper-left and lower-right m×m blocks get their
# upper-left n×n corners; cross-parity blocks get their upper-left n×n
# corners too.
function _extract_parity_block(dp_full::AbstractMatrix, m::Int, n::Int)
    n == m && return dp_full
    out = Matrix{ComplexF64}(undef, 2n, 2n)
    # A' block (upper-left m×m of dp_full) → upper-left n×n of out
    @views out[1:n,     1:n    ] .= dp_full[1:n,     1:n    ]
    # B' block (upper-right m×m of dp_full) → upper-right n×n of out
    @views out[1:n,     n+1:2n ] .= dp_full[1:n,     m+1:m+n]
    # Γ' block (lower-left m×m of dp_full) → lower-left n×n of out
    @views out[n+1:2n,  1:n    ] .= dp_full[m+1:m+n, 1:n    ]
    # Δ' block (lower-right m×m of dp_full) → lower-right n×n of out
    @views out[n+1:2n,  n+1:2n ] .= dp_full[m+1:m+n, m+1:m+n]
    return out
end

function (mc::MultiSurfaceCouplingFull)(Q::Number)
    m = length(mc.surfaces)
    n = mc.msing_max
    Qc = ComplexF64(Q)
    ref_tauk = mc.surfaces[mc.ref_idx].tauk

    # Start from a copy of the parity-major outer matrix (truncated to
    # 2n × 2n when msing_max < length(surfaces)).
    M = _extract_parity_block(mc.dp_full, m, n)

    # Subtract block-diagonal D(γ): interchange channel on rows 1..n,
    # tearing channel on rows n+1..2n.
    @inbounds for k in 1:n
        sc   = mc.surfaces[k]
        Q_k  = Qc * (ref_tauk / sc.tauk)
        resp = solve_inner(sc.model, sc.params, Q_k)
        M[k,     k    ] -= resp.interchange * sc.scale
        M[n + k, n + k] -= resp.tearing     * sc.scale + sc.dc
    end
    return det(M)
end
