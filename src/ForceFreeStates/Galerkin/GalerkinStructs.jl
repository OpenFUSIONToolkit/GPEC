# GalerkinStructs.jl
#
# Data structures for the RDCON outer-region singular Galerkin solver — the Julia port of
# `gal.f` (Dewar's Galerkin method for singular modes). These mirror the Fortran derived types
# `cell_type`, `interval_type`, and `gal_type` (gal.f).
#
# Fortran uses 0-based dimensions `0:np` and `0:nx`/`0:msing`; here those become Julia 1-based with a
# `+1` index shift (Hermite DOF `ip` ↔ array index `ip+1`; interval `ising` ↔ `intvl[ising+1]`).
#
# Reference: A. H. Glasser, Z. R. Wang & J.-K. Park, Phys. Plasmas 23, 112506 (2016).

# Hermite-cubic order: 4 DOFs per node-pair (value + slope at each end), indexed 0:np in Fortran.
const GAL_NP = 3

"""
Cell type flanking a singular surface (Fortran `cell%etype`, gal.f).
"""
@enum GalCellType GCT_NONE GCT_RES GCT_EXT GCT_EXT1 GCT_EXT2

"""
Which side of the adjacent singular surface a cell lies on (Fortran `cell%extra`).
"""
@enum GalSide GAL_SIDE_NONE GAL_SIDE_LEFT GAL_SIDE_RIGHT

"""
    GalCell

One grid cell of the Galerkin discretization. Port of Fortran `cell_type` (gal.f).

## Fields

  - `etype::GalCellType` — none / resonant / extension(1,2)
  - `extra::GalSide` — none / left / right (side of the neighbouring singular surface)
  - `emap::Int` — global index of the extra (small resonant) DOF; 0 if none
  - `map::Matrix{Int}` — `(mpert, np+1)` local→global DOF map (Fortran `cell%map(mpert, 0:np)`)
  - `x::NTuple{2,Float64}` — `[x_left, x_right]` of the cell
  - `x_lsode::Float64` — resonant-integration truncation point (res cells only)
  - `erhs::ComplexF64`, `ediag::ComplexF64` — resonant RHS and diagonal contributions
  - `rhs::Matrix{ComplexF64}` — `(mpert, np+1)` cell RHS (driving term from the big solution)
  - `emat::Matrix{ComplexF64}` — `(mpert, np+1)` coupling of the extra DOF to the Hermite DOFs
  - `mat::Array{ComplexF64,4}` — `(mpert, mpert, np+1, np+1)` cell stiffness block
"""
mutable struct GalCell
    etype::GalCellType
    extra::GalSide
    emap::Int
    map::Matrix{Int}
    x::NTuple{2,Float64}
    x_lsode::Float64
    erhs::ComplexF64
    ediag::ComplexF64
    rhs::Matrix{ComplexF64}
    emat::Matrix{ComplexF64}
    mat::Array{ComplexF64,4}
end

"""
Construct an empty `GalCell` for `mpert` poloidal modes and Hermite order `np`.
"""
function GalCell(mpert::Int, np::Int=GAL_NP)
    return GalCell(GCT_NONE, GAL_SIDE_NONE, 0,
        zeros(Int, mpert, np + 1), (0.0, 0.0), 0.0,
        zero(ComplexF64), zero(ComplexF64),
        zeros(ComplexF64, mpert, np + 1),
        zeros(ComplexF64, mpert, np + 1),
        zeros(ComplexF64, mpert, mpert, np + 1, np + 1))
end

"""
    GalInterval

One interval between consecutive singular surfaces (or between a surface and a domain edge).
Port of Fortran `interval_type` (gal.f). `x`/`dx` are length `nx+1` (Fortran `0:nx`).
"""
struct GalInterval
    x::Vector{Float64}
    dx::Vector{Float64}
    cells::Vector{GalCell}
end

"""
    GalWorkspace

Global workspace for the Galerkin solve. Port of Fortran `gal_type` (gal.f).

`intvl` has length `msing+1` (Fortran `intvl(0:msing)`, accessed as `intvl[ising+1]`).
`kl = ku = mpert*(np+1)`. The banded `mat` storage layout depends on `solver` (gal.f):

  - `"LU"`       → `ldab = 2*kl + ku + 1`, full band (LAPACK `gbtrf!`/`gbtrs!`)
  - `"cholesky"` → `ldab = kl + 1`, lower band only (LAPACK `pbtrf!`/`pbtrs!`, `'L'`)
"""
mutable struct GalWorkspace
    solver::String
    nx::Int
    nq::Int
    np::Int
    ndim::Int
    kl::Int
    ku::Int
    ldab::Int
    nsol::Int
    intvl::Vector{GalInterval}
    mat::Matrix{ComplexF64}
    rhs::Matrix{ComplexF64}
    sol::Matrix{ComplexF64}
end

"""
    GalSingAsymp

Two-sided singular asymptotics for the Galerkin solver: separate right (sig=+1) and left (sig=−1)
power series, mirroring Fortran sing.f `vmatr`/`vmatl`. Each is evaluated at a positive distance
`|ψ−ψ_s|` (real √, real pfac, no z<0 renormalization), so the directly-extracted Δ′ matches gal.f.
"""
struct GalSingAsymp
    right::SingAsymptotics
    left::SingAsymptotics
end

"""
    GalerkinSolution

Reconstructed outer-region radial solution on the gal-native grid (ξ and analytic ξ′ consumed by
PerturbedEquilibrium). Populated by `gal_output_solution` (GalerkinSolution.jl).

  - `psi::Vector{Float64}`, `q::Vector{Float64}` — radial grid (inner→edge) and its safety factor.
  - `issing::Vector{Bool}` — grid points sitting on a singular surface (skipped; left zero).
  - `xi::Array{ComplexF64,3}`, `xi_deriv::Array{ComplexF64,3}` — `(mpert, ngrid, nsol)` ξ and dξ/dψ.
  - `xi_cut::Array{ComplexF64,3}` — `(mpert, ngrid, nsol)` cut solution: ξ with the leading-order
    resonant content removed. This is the smooth background the resistive inner-layer solution is
    added to when forming the composite solution at a rational surface, so the inner and outer
    solutions overlap in the matching region. Sized `(0,0,0)` when the cut path was not requested.
  - `cut_range::Matrix{Float64}` — `(msing, 2)` ψ bounds of the resonant + extension cells flanking
    each rational surface. Outside these bounds no resonant content is subtracted, so `xi_cut`
    equals `xi` there and the composite solution is undefined; the inner-region solution is only
    meaningful inside this window. Sized `(0,0)` when the cut path was not requested.
"""
struct GalerkinSolution
    psi::Vector{Float64}
    q::Vector{Float64}
    issing::Vector{Bool}
    xi::Array{ComplexF64,3}
    xi_deriv::Array{ComplexF64,3}
    xi_cut::Array{ComplexF64,3}
    cut_range::Matrix{Float64}
end

"""
    GalMatchResult

Coil-driven RPEC matched solution from the outer↔inner asymptotic matching (Fortran rmatch
`match_rpec`). Populated by `gal_match_rpec` (GalerkinMatch.jl) when `gal_match_flag`.

  - `cout::Matrix{ComplexF64}` — `(2·msing, mcoil)` outer-region plasma-solution coefficients.
  - `cin::Matrix{ComplexF64}` — `(2·msing, mcoil)` inner-region coefficients.
  - `xi::Array{ComplexF64,3}`, `xi_deriv::Array{ComplexF64,3}` — `(mpert, ngrid, mcoil)` matched ξ(ψ) and
    analytic ξ′(ψ) on the gal grid, one column per coil drive (identity-at-edge basis).
  - `deltar::Matrix{ComplexF64}` — `(msing, 2)` inner-layer matching data `(Δ₁, Δ₂)` per surface.
  - `bpen::Matrix{ComplexF64}` — `(msing, mcoil)` inner-layer penetrated (reconnected) resonant field at
    each rational surface, one column per coil drive. Read off the GGJ inner solution at the layer center
    (X=0) exactly as Fortran `match_output_solution` builds `intotsol_b` (match.f) — cusp-free, fit-free.
    Zero for the ideal branch (`gal_ideal_flag`), where the inner layer is skipped.
  - `inner_psi::Vector{Vector{Float64}}` — per surface, the inner-layer ψ grid `ψ_s ± X·x0/v1` (left wing
    reversed then right wing, so ψ ascends through `ψ_s`). Empty in the ideal branch.
  - `inner_xi::Vector{Matrix{ComplexF64}}` — per surface, the inner-layer displacement `ξ_ψ(ψ)` on
    `inner_psi`, `(length(inner_psi[s]), mcoil)`, one column per coil drive. This is Fortran `match_solution`'s
    `intotsol` (deltac component 2): `resc·(±Ξ₁·cin[2s] + Ξ₂·cin[2s-1])`, odd parity `Ξ₁` antisymmetric across
    `ψ_s`, even parity `Ξ₂` symmetric, `resc=(v1/x0)^(1/2+p1)`. The raw inner-layer contribution only (no
    regular outer background added), so it shares the singular asymptote with the outer near `ψ_s`.
  - `rpec_eig::Vector{ComplexF64}` — forced eigenvalues `γ_s = 2πi·n·f_s` per surface.
  - `residual::Float64` — relative linear-solve residual `‖mat·cof − rmat‖/‖rmat‖`.
"""
struct GalMatchResult
    cout::Matrix{ComplexF64}
    cin::Matrix{ComplexF64}
    xi::Array{ComplexF64,3}
    xi_deriv::Array{ComplexF64,3}
    deltar::Matrix{ComplexF64}
    bpen::Matrix{ComplexF64}
    inner_psi::Vector{Vector{Float64}}
    inner_xi::Vector{Matrix{ComplexF64}}
    inner_b::Vector{Matrix{ComplexF64}}   # per surface, matched inner-layer b^ψ(ψ) on inner_psi (overlap validation)
    inner_params::Vector{InnerLayer.GGJParameters}  # per-surface layer coefficients from resist_eval (empty in the ideal limit)
    rpec_eig::Vector{ComplexF64}
    residual::Float64
end

"""
    GalerkinResult

Outputs of the outer-region Galerkin solve.

## Fields

  - `delta::Matrix{ComplexF64}` — Δ′ matrix, `(nsol, 2*msing)` with `nsol = 2*msing`; the small
    resonant coefficients (Fortran `delta`, gal.f).
  - `Ap, Bp, Gammap, Deltap::Matrix{ComplexF64}` — PEST-3 matching blocks, each `(msing, msing)`
    (Fortran `gal_write_pest3_data`, gal.f).
  - `msing::Int` — number of resonant singular surfaces included.
  - `sing_psi, sing_q::Vector{Float64}`, `sing_m, sing_n::Vector{Int}` — per-surface identifiers.
  - `di::Vector{Float64}`, `alpha::Vector{ComplexF64}` — Mercier index and exponent per surface.
  - `solution::Union{Nothing,GalerkinSolution}` — reconstructed radial ξ(ψ) and analytic ξ′(ψ) on the
    gal-native grid; `nothing` if no resonant surfaces.
"""
struct GalerkinResult
    delta::Matrix{ComplexF64}
    Ap::Matrix{ComplexF64}
    Bp::Matrix{ComplexF64}
    Gammap::Matrix{ComplexF64}
    Deltap::Matrix{ComplexF64}
    msing::Int
    sing_psi::Vector{Float64}
    sing_q::Vector{Float64}
    sing_m::Vector{Int}
    sing_n::Vector{Int}
    di::Vector{Float64}
    alpha::Vector{ComplexF64}
    delta_coil::Matrix{ComplexF64}   # (mpert, 2*msing) coil-response block (rpec_flag); empty if not computed
    solution::Union{Nothing,GalerkinSolution}
    match::Union{Nothing,GalMatchResult}   # RPEC matched solution (gal_match_flag); nothing otherwise
end
