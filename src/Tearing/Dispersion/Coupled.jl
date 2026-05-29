# Coupled.jl
#
# Multi-surface coupled tearing dispersion residual `det(M(Q))` for the
# Fortran SLAYER `coupling_flag = .TRUE.` path (`dispersion_det`,
# growthrates.f:190-279). Brought together with the per-surface
# `SurfaceCoupling` (PR 3) so a brute-force or AMR scan in PRs 5-6 can
# evaluate either residual through the same Q-callable interface.
#
# Construction:
#
#   mc = multi_surface_coupling(surfaces, dp_matrix; ref_idx=1, msing_max=...)
#
# Evaluation:
#
#   det = mc(Q::ComplexF64)
#
# At each evaluation, for k = 1 .. msing_max, the inner-layer Δ is computed
# at a Q rescaled by `tauk_ref / tauk_k` (mirrors growthrates.f:246), then
# subtracted (with the dc offset) from the diagonal of an `msing_max ×
# msing_max` upper-left submatrix of `dp_matrix`. The off-diagonal Δ'
# couplings are passed through unchanged.

"""
    MultiSurfaceCoupling{V<:AbstractVector{<:SurfaceCoupling}}

Multi-surface dispersion data: a vector of `SurfaceCoupling`, the full Δ'
matrix, the index of the reference surface (whose `tauk` defines the Q
normalization), and the truncation `msing_max` (number of surfaces actually
participating in the determinant). Calling `mc(Q)` returns `det(M(Q))` where

```
M[k,k] = dp_matrix[k,k] - scale_k · Δ_inner_k(Q · tauk_ref / tauk_k) - dc_k
M[i,j] = dp_matrix[i,j]      for i ≠ j        (off-diagonal Δ' couplings)
```

A root of `mc` in the complex `Q` plane is a coupled tearing eigenvalue.
"""
struct MultiSurfaceCoupling{V<:AbstractVector{<:SurfaceCoupling}}
    surfaces::V
    dp_matrix::Matrix{ComplexF64}
    ref_idx::Int
    msing_max::Int
end

"""
    multi_surface_coupling(surfaces, dp_matrix;
                            ref_idx=1,
                            msing_max=min(3, length(surfaces)))
        -> MultiSurfaceCoupling

Construct a multi-surface coupling from a vector of `SurfaceCoupling` and
the full outer-region Δ' matrix. `dp_matrix` must be square with side
length `length(surfaces)` (it is the same matrix returned by
`PerturbedEquilibrium.SingularCoupling`'s STRIDE-style Δ' BVP).

# Keyword arguments

  - `ref_idx`   -- index of the reference surface whose `tauk` defines the
    Q normalization. Defaults to `1` (Fortran convention,
    growthrates.f:246).
  - `msing_max` -- number of surfaces from the front of `surfaces` to
    include in the determinant. Defaults to `min(3, length(surfaces))`:
    Δ' off-diagonal couplings beyond the third surface tend to be erratic
    in practice, so the determinant is conservatively truncated to the
    upper-left `msing_max × msing_max` submatrix of `dp_matrix`. Set
    explicitly (up to `length(surfaces)`) to override.
"""
function multi_surface_coupling(surfaces::AbstractVector{<:SurfaceCoupling},
                                dp_matrix::AbstractMatrix;
                                ref_idx::Integer=1,
                                msing_max::Integer=min(3, length(surfaces)))
    n = length(surfaces)
    size(dp_matrix) == (n, n) ||
        throw(ArgumentError("multi_surface_coupling: dp_matrix size " *
                            "$(size(dp_matrix)) ≠ ($n, $n)"))
    1 <= ref_idx <= n ||
        throw(ArgumentError("multi_surface_coupling: ref_idx=$ref_idx out " *
                            "of range 1:$n"))
    1 <= msing_max <= n ||
        throw(ArgumentError("multi_surface_coupling: msing_max=$msing_max " *
                            "out of range 1:$n"))
    return MultiSurfaceCoupling(surfaces,
                                Matrix{ComplexF64}(dp_matrix),
                                Int(ref_idx), Int(msing_max))
end

function (mc::MultiSurfaceCoupling)(Q::Number)
    n = mc.msing_max
    Qc = ComplexF64(Q)
    ref_tauk = mc.surfaces[mc.ref_idx].tauk

    M = mc.dp_matrix[1:n, 1:n]
    @inbounds for k in 1:n
        sc   = mc.surfaces[k]
        Q_k  = Qc * (ref_tauk / sc.tauk)
        # m×m scalar coupling: use only the tearing channel. The
        # interchange (Glasser-stabilization) channel is carried in the
        # full 4m×4m dispersion in `CoupledFortranMatch.jl`; this reduced
        # form is equivalent for pressureless SLAYER surfaces
        # (Δ_interchange=0) and approximate for GGJ surfaces (drops
        # Glasser stabilization).
        Δ_k  = solve_inner(sc.model, sc.params, Q_k).tearing * sc.scale
        M[k,k] -= Δ_k + sc.dc
    end
    return det(M)
end
