# CoupledFortranMatch.jl
#
# Literal Julia port of Fortran `rmatch/match.f::match_delta` — the full
# Pletzer-Dewar 4m × 4m tearing+interchange dispersion matrix, with the
# m inner-layer resonances decoupled via the matching-identity rows
#
#     C^j_L = d^j_+ − d^j_-
#     C^j_R = -(d^j_+ + d^j_-)
#
# (see Wang-Glasser-Brennan-Liu-Park 2020, Phys. Plasmas **27**, 122503,
# Eq. (11a)-(11d) and Glasser-Wang-Park 2016, Phys. Plasmas **23**, 112506,
# Eq. (36)-(40)).
#
# Why 4m × 4m and not 2m × 2m?
#
#   The outer-region matching matrix D' (Julia `intr.delta_prime_raw`) is
#   expressed in the side-major basis `[L_s1, R_s1, L_s2, R_s2, …]` of
#   large-solution driving amplitudes. The inner-layer Galerkin solver
#   (`solve_inner(GGJModel, …)`) returns Δ_tearing and Δ_interchange in
#   the even/odd parity (+/−) basis instead. The naive relation
#   `det(D' − diag(Δ_+, Δ_-)) = 0` cannot be written directly because
#   the two quantities live in different bases. The Fortran fix is to
#   introduce both sets of amplitudes (`C^j_{L,R}` for outer, `d^j_±` for
#   inner) as explicit unknowns and use the ±1 matching identity as two
#   extra rows per surface, yielding the 4m × 4m linear system. `CoupledFull`
#   in this module tries the naive 2m × 2m form and produces a determinant
#   with structurally-wrong magnitude and topology; this module (Fortran-
#   faithful) reproduces the Pletzer-Dewar result.
#
# Per surface `k` (1-indexed), the 4 block indices are
#
#     idx1 = 2k − 1                      (row/col for C^k_L)
#     idx2 = 2k                          (row/col for C^k_R)
#     idx3 = idx1 + 2m                   (row/col for d^k_+)
#     idx4 = idx2 + 2m                   (row/col for d^k_-)
#
# The global 4m × 4m matrix has:
#
#   - lower-left 2m × 2m block = transpose(dp_raw)
#   - upper-left 2m × 2m block: per-surface 2 × 2 identity
#   - upper-right 2m × 2m block: per-surface 2 × 2 matching identity
#   - lower-right 2m × 2m block: per-surface 2 × 2 inner Δ block
#
# See the per-surface fill table in the body of `(::MultiSurfaceCouplingFortran)`.

"""
    MultiSurfaceCouplingFortran{V<:AbstractVector{<:SurfaceCoupling}}

Fortran-faithful 4m × 4m tearing+interchange dispersion matrix
(`rmatch/match.f::match_delta`, fulldomain=0 branch).

Given the raw 2m × 2m outer-region matrix `dp_raw` (side-major ordering
`[L_s1, R_s1, L_s2, R_s2, …]`, from `intr.delta_prime_raw`) and a vector
of `SurfaceCoupling` (each containing the inner-layer model and
parameters), calling `mc(Q)` assembles the 4m × 4m Pletzer-Dewar
matching matrix and returns `det(mat)`.

Use this instead of `MultiSurfaceCouplingFull` for tearing+interchange
dispersion: `CoupledFull` was a (structurally-incorrect) 2m × 2m
`det(D' − D(γ))` form whose determinant topology does not match Fortran;
`MultiSurfaceCouplingFortran` is the correct Pletzer-Dewar dispersion
relation.

# Fields

  - `surfaces::V`               — per-surface `SurfaceCoupling`.
  - `dp_raw::Matrix{ComplexF64}` — 2m × 2m outer-region matrix (side-major).
  - `ref_idx::Int`              — reference surface for Q rescaling (1-based).
  - `msing_max::Int`            — number of surfaces to include (truncates).
  - `rotation::Vector{Float64}` — per-surface rotation frequencies (s⁻¹).
  - `ntor::Int`                 — toroidal mode number `n` (default 1).
"""
struct MultiSurfaceCouplingFortran{V<:AbstractVector{<:SurfaceCoupling},K<:NamedTuple}
    surfaces::V
    dp_raw::Matrix{ComplexF64}
    ref_idx::Int
    msing_max::Int
    rotation::Vector{Float64}
    ntor::Int
    inner_kwargs::K    # kwargs forwarded to solve_inner; e.g. (pfac=0.1, nx=128, nq=5)
end

"""
    multi_surface_coupling_fortran(surfaces, dp_raw;
                                    ref_idx=1,
                                    msing_max=length(surfaces),
                                    rotation=zeros(length(surfaces)),
                                    ntor=1) -> MultiSurfaceCouplingFortran

Construct the 4m × 4m dispersion matrix driver. `dp_raw` must be the
2m × 2m matrix in side-major ordering (the `intr.delta_prime_raw`
field populated by `ForceFreeStates.compute_delta_prime_matrix!` on the
`use_parallel=true` path). `rotation[k]` is the per-surface rotation
frequency (Fortran `rotation(ising)` in `rmatch.in`); it shifts the
per-surface inner Q argument by `i·ntor·rotation[k]`. Default zero
rotation matches the static-equilibrium case.

# Keyword arguments

  - `ref_idx`   — index of the reference surface whose `tauk` defines the
    Q normalization (1 ≤ ref_idx ≤ m). Defaults to 1.
  - `msing_max` — truncate to the leading `msing_max` surfaces; the
    matching matrix becomes 4·msing_max × 4·msing_max, built from the
    corresponding 2·msing_max × 2·msing_max submatrix of `dp_raw`.
    Defaults to `length(surfaces)`.
  - `rotation`  — per-surface rotation frequencies in s⁻¹ (length m).
    Defaults to all zero.
  - `ntor`      — toroidal mode number n. Defaults to 1.
  - `inner_kwargs` — NamedTuple of kwargs forwarded to `solve_inner` at
    every Q evaluation, e.g. `(pfac=0.1, xfac=10.0, nx=128, nq=5)` to
    match the Fortran `rmatch/DELTAC_LIST` defaults for Galerkin grid
    tuning. Defaults to `NamedTuple()`.
"""
function multi_surface_coupling_fortran(surfaces::AbstractVector{<:SurfaceCoupling},
                                        dp_raw::AbstractMatrix;
                                        ref_idx::Integer=1,
                                        msing_max::Integer=length(surfaces),
                                        rotation::AbstractVector{<:Real}=zeros(length(surfaces)),
                                        ntor::Integer=1,
                                        inner_kwargs::NamedTuple=NamedTuple())
    m = length(surfaces)
    size(dp_raw) == (2m, 2m) ||
        throw(ArgumentError("multi_surface_coupling_fortran: dp_raw size " *
                            "$(size(dp_raw)) ≠ ($(2m), $(2m))"))
    1 <= ref_idx <= m ||
        throw(ArgumentError("multi_surface_coupling_fortran: ref_idx=$ref_idx " *
                            "out of range 1:$m"))
    1 <= msing_max <= m ||
        throw(ArgumentError("multi_surface_coupling_fortran: msing_max=$msing_max " *
                            "out of range 1:$m"))
    length(rotation) == m ||
        throw(ArgumentError("multi_surface_coupling_fortran: rotation length " *
                            "$(length(rotation)) ≠ $m"))
    return MultiSurfaceCouplingFortran(surfaces,
                                       Matrix{ComplexF64}(dp_raw),
                                       Int(ref_idx), Int(msing_max),
                                       Float64.(collect(rotation)),
                                       Int(ntor),
                                       inner_kwargs)
end

# Assemble and return det(mat) where mat is the 4·msing_max × 4·msing_max
# Pletzer-Dewar matching matrix. Direct port of match.f:460-520 (fulldomain=0).
function (mc::MultiSurfaceCouplingFortran)(Q::Number)
    m = mc.msing_max
    s2 = 2m
    s4 = 4m
    Qc = ComplexF64(Q)
    ref_tauk = mc.surfaces[mc.ref_idx].tauk

    # Allocate the matching matrix and fill the lower-left 2m × 2m block
    # with transpose(dp_raw[1:s2, 1:s2]) — exact port of match.f:461.
    mat = zeros(ComplexF64, s4, s4)
    @views mat[s2+1:s4, 1:s2] .= transpose(mc.dp_raw[1:s2, 1:s2])

    # Per-surface inner-layer assembly
    @inbounds for k in 1:m
        sc   = mc.surfaces[k]
        idx1 = 2k - 1          # C^k_L
        idx2 = 2k              # C^k_R
        idx3 = idx1 + s2       # d^k_+
        idx4 = idx2 + s2       # d^k_-

        # Per-surface Q shift — match.f:472: guess_modify = Q + i·n·rotation[k].
        # Also apply ref_tauk / sc.tauk rescaling (we keep the SurfaceCoupling
        # tauk normalization that SLAYER needs; GGJ has tauk=1 so it's a no-op).
        Q_k = Qc * (ref_tauk / sc.tauk) + 1im * mc.ntor * mc.rotation[k]
        resp = solve_inner(sc.model, sc.params, Q_k; mc.inner_kwargs...)

        # Fortran delta(1) = Julia .interchange (post-swap in deltac.f;
        # Julia removes the swap and exposes named fields instead).
        # Fortran delta(2) = Julia .tearing.
        #
        # sc.scale converts inner-basis Δ to outer units (1.0 for GGJ since
        # rescale_delta is applied inside solve_inner; S^(1/3) for SLAYER).
        # NOTE: match.f::match_delta (fulldomain=0, lines 508-519) does
        # NOT add any Δ_crit offset here — delta1,delta2 are the raw
        # inner-layer outputs. The full 4m×4m Pletzer-Dewar residual
        # includes the interchange channel, which provides Glasser
        # (Mercier) stabilization natively; Δ_crit is a slab-layer proxy
        # only relevant to SLAYER's tearing-only model. Earlier versions
        # of this file added `+ sc.dc` to both channels — that was a port
        # error (no corresponding term in Fortran) and is removed here.
        delta1 = resp.interchange * sc.scale
        delta2 = resp.tearing     * sc.scale

        # --- Upper-left 2×2 block: per-surface identity on C_{L,R} ---
        mat[idx1, idx1] = 1
        mat[idx2, idx2] = 1

        # --- Upper-right 2×2 block: matching identity ---
        #   C^k_L = d^k_+ − d^k_-         ⇒ mat[idx1,idx3]=-1, mat[idx1,idx4]=+1
        #   C^k_R = -(d^k_+ + d^k_-)      ⇒ mat[idx2,idx3]=-1, mat[idx2,idx4]=-1
        mat[idx1, idx3] = -1
        mat[idx1, idx4] =  1
        mat[idx2, idx3] = -1
        mat[idx2, idx4] = -1

        # --- Lower-right 2×2 block: inner Δ matching ---
        #   d^k_+ eqn: -Δ_int·d^k_+ + Δ_tear·d^k_- + (outer D' terms) = 0
        #   d^k_- eqn: -Δ_int·d^k_+ - Δ_tear·d^k_- + (outer D' terms) = 0
        # (match.f:504-507)
        mat[idx3, idx3] = -delta1
        mat[idx3, idx4] =  delta2
        mat[idx4, idx3] = -delta1
        mat[idx4, idx4] = -delta2
    end

    return det(mat)
end
