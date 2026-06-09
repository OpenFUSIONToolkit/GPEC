# GalerkinStructs.jl
#
# Data structures for the RDCON outer-region singular Galerkin solver — the Julia port of
# `gal.f` (Dewar's Galerkin method for singular modes). These mirror the Fortran derived types
# `cell_type`, `interval_type`, and `gal_type` (gal.f:48-76).
#
# Fortran uses 0-based dimensions `0:np` and `0:nx`/`0:msing`; here those become Julia 1-based with a
# `+1` index shift (Hermite DOF `ip` ↔ array index `ip+1`; interval `ising` ↔ `intvl[ising+1]`).
#
# Reference: A. H. Glasser, Z. R. Wang & J.-K. Park, Phys. Plasmas 23, 112506 (2016).

# Hermite-cubic order: 4 DOFs per node-pair (value + slope at each end), indexed 0:np in Fortran.
const GAL_NP = 3

"""Cell type flanking a singular surface (Fortran `cell%etype`, gal.f)."""
@enum GalCellType GCT_NONE GCT_RES GCT_EXT GCT_EXT1 GCT_EXT2

"""Which side of the adjacent singular surface a cell lies on (Fortran `cell%extra`)."""
@enum GalSide GAL_SIDE_NONE GAL_SIDE_LEFT GAL_SIDE_RIGHT

"""
    GalCell

One grid cell of the Galerkin discretization. Port of Fortran `cell_type` (gal.f:52-61).

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

"""Construct an empty `GalCell` for `mpert` poloidal modes and Hermite order `np`."""
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
Port of Fortran `interval_type` (gal.f:63-66). `x`/`dx` are length `nx+1` (Fortran `0:nx`).
"""
struct GalInterval
    x::Vector{Float64}
    dx::Vector{Float64}
    cells::Vector{GalCell}
end

"""
    GalWorkspace

Global workspace for the Galerkin solve. Port of Fortran `gal_type` (gal.f:68-76).

`intvl` has length `msing+1` (Fortran `intvl(0:msing)`, accessed as `intvl[ising+1]`).
`kl = ku = mpert*(np+1)`. The banded `mat` storage layout depends on `solver` (gal.f:164-168):
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
    GalerkinResult

Outputs of the outer-region Galerkin solve.

## Fields

  - `delta::Matrix{ComplexF64}` — Δ′ matrix, `(nsol, 2*msing)` with `nsol = 2*msing`; the small
    resonant coefficients (Fortran `delta`, gal.f:1405-1417).
  - `Ap, Bp, Gammap, Deltap::Matrix{ComplexF64}` — PEST-3 matching blocks, each `(msing, msing)`
    (Fortran `gal_write_pest3_data`, gal.f:1726-1745).
  - `msing::Int` — number of resonant singular surfaces included.
  - `sing_psi, sing_q::Vector{Float64}`, `sing_m, sing_n::Vector{Int}` — per-surface identifiers.
  - `di::Vector{Float64}`, `alpha::Vector{ComplexF64}` — Mercier index and exponent per surface.
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
end
