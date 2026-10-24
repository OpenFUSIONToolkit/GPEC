"""
    DirectIngest

The serializable raw arrays and scalars captured by a direct-equilibrium reader
(`read_efit`, `read_imas`, `sol_run` is analytic and does not use this) — everything
needed to rebuild a `DirectRunInput`'s splines without re-reading the original g-file.
Stored on `DirectRunInput.ingest` / `PlasmaEquilibrium.ingest` and dumped to
`gpec.h5` so a run can be replayed from any directory. The interpolants themselves are
not serializable; they are reconstructed from these nodes by `build_direct_from_ingest`.

## Fields

  - `sq_xs::Vector{Float64}` — normalized-ψ knots for the 1D profile spline
  - `sq_fs::Matrix{Float64}` — 1D profile node values (F, μ₀P, q, √ψ_norm)
  - `psi_xs::Vector{Float64}` — R grid for the 2D flux map [m]
  - `psi_ys::Vector{Float64}` — Z grid for the 2D flux map [m]
  - `psi_rz::Matrix{Float64}` — processed poloidal flux on the (R, Z) grid [Wb/rad]
  - `rmin/rmax/zmin/zmax::Float64` — computational-grid bounds [m]
  - `psio::Float64` — total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
  - `bt_sign::Int` — sign of the toroidal field (+1 or -1)
  - `ip_sign::Int` — sign of the plasma current (+1 or -1) as the source file states it
"""
struct DirectIngest
    sq_xs::Vector{Float64}
    sq_fs::Matrix{Float64}
    psi_xs::Vector{Float64}
    psi_ys::Vector{Float64}
    psi_rz::Matrix{Float64}
    rmin::Float64
    rmax::Float64
    zmin::Float64
    zmax::Float64
    psio::Float64
    bt_sign::Int
    ip_sign::Int
end

"""
    InverseIngest

The serializable raw arrays and scalars captured by an inverse-equilibrium reader
(`read_chease_ascii`, `read_chease_binary`, `read_ifile`) — everything needed to rebuild an
`InverseRunInput`'s splines without re-reading the original CHEASE file or i-file. Stored on
`InverseRunInput.ingest` / `PlasmaEquilibrium.ingest` and reconstructed by
`build_inverse_from_ingest`. See [`DirectIngest`](@ref) for the role this plays in
the `gpec.h5` rerun snapshot.

## Fields

  - `sq_xs::Vector{Float64}` — normalized-ψ knots for the 1D profile spline
  - `sq_fs::Matrix{Float64}` — 1D profile node values
  - `rz_xs::Vector{Float64}` — ψ grid for the R, Z maps
  - `rz_ys::Vector{Float64}` — θ grid for the R, Z maps
  - `R_nodes::Matrix{Float64}` — R node values on the (ψ, θ) grid [m]
  - `Z_nodes::Matrix{Float64}` — Z node values on the (ψ, θ) grid [m]
  - `ro::Float64` — R of magnetic axis [m]
  - `zo::Float64` — Z of magnetic axis [m]
  - `psio::Float64` — total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
"""
struct InverseIngest
    sq_xs::Vector{Float64}
    sq_fs::Matrix{Float64}
    rz_xs::Vector{Float64}
    rz_ys::Vector{Float64}
    R_nodes::Matrix{Float64}
    Z_nodes::Matrix{Float64}
    ro::Float64
    zo::Float64
    psio::Float64
end

# Equilibria captured for replay carry one of these; analytic equilibria carry `nothing`
# and are regenerated from their TOML section instead.
const EquilibriumIngest = Union{Nothing,DirectIngest,InverseIngest}

"""
    DirectRunInput(...)

A container struct that bundles all necessary inputs for the `direct_run` function.
It is created by one of the equilibrium file read-in functions after processing the
raw equilibrium data and preparing the initial splines.

## Fields

  - `config::EquilibriumConfig`
    The equilibrium configuration object.

  - `sq_in`
    1D spline data versus normalized poloidal flux `psin`.
    Quantities:

     1. `F = R * B_t` — toroidal flux function [m·T]
     2. `μ₀ * Pressure` — plasma pressure (non-negative) [T²]
     3. `q` — safety factor profile
     4. `√ψ_norm` — square root of normalized flux
  - `psi_in`
    2D cubic interpolant on the (R, Z) grid [m].
    The values correspond to the **poloidal flux** adjusted to be zero at the boundary [Wb/rad].
    Definitions:

     1. `ψ(R, Z) = ψ_boundary - ψ(R, Z)`

     2. `ψ = ψ * sign(ψ(centerR, centerZ))`

          * 1D profiles are represented by `CubicInterpolant` or `CubicSeriesInterpolant`
          * 2D flux surfaces by `CubicInterpolantND`
  - `psi_in_xs::Vector{Float64}` — R coordinate grid for psi_in [m]
  - `psi_in_ys::Vector{Float64}` — Z coordinate grid for psi_in [m]
  - `rmin::Float64` — Minimum R-coordinate of the computational grid [m]
  - `rmax::Float64` — Maximum R-coordinate of the computational grid [m]
  - `zmin::Float64` — Minimum Z-coordinate of the computational grid [m]
  - `zmax::Float64` — Maximum Z-coordinate of the computational grid [m]
  - `psio::Float64` — Total flux difference `|ψ_axis - ψ_boundary|` [Wb/rad]
  - `bt_sign::Int` — Sign of the toroidal field (+1 or -1); read from fpol sign in EFIT g-files
  - `ip_sign::Int` — Sign of the plasma current (+1 or -1); read from the `current` header value
    in EFIT g-files and `global_quantities.ip` in IMAS. The internal flux is always made
    positive at the axis, so the sign is kept here (never recovered from the computed current)
    and fixes the SFL→machine toroidal-angle handedness `helicity = bt_sign × ip_sign`.
  - `ingest::EquilibriumIngest` — captured raw arrays for the `gpec.h5` rerun snapshot
    (a [`DirectIngest`](@ref) for file-based reads, or `nothing` for analytic equilibria)
  - `psihigh_resolved::Float64` — outer flux limit the equilibrium is formed on: `config.psihigh`
    clamped to the outermost closed flux surface by [`resolve_psihigh!`](@ref). Defaults to
    `config.psihigh` and only differs for efit-family equilibria whose requested limit falls
    outside the closed-flux region. The solvers build their ψ grid from this field. IMAS
    equilibria are read into this struct but are not in `EFIT_KINDS`, so they are never
    clamped and always keep the request.
"""
mutable struct DirectRunInput{S<:FastInterpolations.CubicSeriesInterpolant,I2D<:FastInterpolations.CubicInterpolantND}
    config::EquilibriumConfig
    sq_in::S       # 1D profile spline for F, P, q
    psi_in::I2D    # 2D flux interpolant (CubicInterpolantND)
    psi_in_xs::Vector{Float64}  # R coordinates
    psi_in_ys::Vector{Float64}  # Z coordinates
    rmin::Float64    # Minimum R-coordinate of the computational grid [m].
    rmax::Float64    # Maximum R-coordinate of the computational grid [m].
    zmin::Float64    # Minimum Z-coordinate of the computational grid [m].
    zmax::Float64    # Maximum Z-coordinate of the computational grid [m].
    psio::Float64    # The total flux difference |ψ_axis - ψ_boundary| [Weber / radian].
    bt_sign::Int     # Sign of the toroidal field: +1 or -1 (from fpol sign in g-file)
    ip_sign::Int     # Sign of the plasma current: +1 or -1 (from the g-file current / IMAS ip)
    ingest::EquilibriumIngest
    psihigh_resolved::Float64
end

# Readers construct without a resolved psihigh; it starts at the request and `resolve_psihigh!`
# clamps it for efit-family equilibria.
DirectRunInput(config::EquilibriumConfig, sq_in, psi_in, psi_in_xs, psi_in_ys,
    rmin, rmax, zmin, zmax, psio, bt_sign, ip_sign, ingest) =
    DirectRunInput(config, sq_in, psi_in, psi_in_xs, psi_in_ys,
        rmin, rmax, zmin, zmax, psio, bt_sign, ip_sign, ingest, config.psihigh)

"""
    InverseRunInput(...)

A container struct for inputs to the `inverse_run` function.

## Fields

  - `config::EquilibriumConfig` - The equilibrium configuration object
  - `sq_in::CubicSeriesInterpolant` - 1D profile spline for F, P, q
  - `rz_in_xs::Vector{Float64}` - ψ coordinate grid for rz_in
  - `rz_in_ys::Vector{Float64}` - θ coordinate grid for rz_in
  - `rz_in_R::CubicInterpolantND` - R coordinate interpolant [m]
  - `rz_in_Z::CubicInterpolantND` - Z coordinate interpolant [m]
  - `ro::Float64` - R-coordinate of magnetic axis [m]
  - `zo::Float64` - Z-coordinate of magnetic axis [m]
  - `psio::Float64` - Total flux difference |ψ_axis - ψ_boundary| [Wb/rad]
  - `ingest::EquilibriumIngest` - captured raw arrays for the `gpec.h5` rerun snapshot
    (an [`InverseIngest`](@ref) for file-based reads, or `nothing` for analytic equilibria)
  - `psihigh_resolved::Float64` - outer flux limit the equilibrium is formed on; see
    [`DirectRunInput`](@ref). `config.psihigh`, clamped to the outermost input surface for i-files and
    reruns (`resolve_psihigh!`); `efit_by_inversion` forwards the clamped value from its `DirectRunInput`.
"""
mutable struct InverseRunInput{S<:FastInterpolations.CubicSeriesInterpolant,I2D<:FastInterpolations.CubicInterpolantND}
    config::EquilibriumConfig
    sq_in::S   # 1D profile spline for F, P, q
    rz_in_xs::Vector{Float64}   # ψ coordinates
    rz_in_ys::Vector{Float64}   # θ coordinates
    rz_in_R::I2D                # R coordinate interpolant
    rz_in_Z::I2D                # Z coordinate interpolant
    ro::Float64                 # R axis location
    zo::Float64                 # Z axis location
    psio::Float64               # Total flux difference |psi_axis - psi_boundary|
    ingest::EquilibriumIngest
    psihigh_resolved::Float64
end

InverseRunInput(config::EquilibriumConfig, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z,
    ro, zo, psio, ingest) =
    InverseRunInput(config, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z,
        ro, zo, psio, ingest, config.psihigh)

"""
    build_direct_from_ingest(config::EquilibriumConfig, ingest::DirectIngest) -> DirectRunInput

Rebuild a `DirectRunInput` from a [`DirectIngest`](@ref) captured by `read_efit`/`read_imas`
(or restored from `Input/RawInputs/Equilibrium/` inside `gpec.h5`). Inverse of that capture:
reconstructs the splines so the rerun path skips the g-file/IMAS parse, reusing the existing
solver dispatch.
"""
function build_direct_from_ingest(config::EquilibriumConfig, ingest::DirectIngest)
    sq_in = cubic_interp(ingest.sq_xs, Series(ingest.sq_fs); extrap=ExtendExtrap())
    psi_in = cubic_interp((ingest.psi_xs, ingest.psi_ys), ingest.psi_rz; extrap=ExtendExtrap())
    return DirectRunInput(config, sq_in, psi_in, ingest.psi_xs, ingest.psi_ys,
        ingest.rmin, ingest.rmax, ingest.zmin, ingest.zmax, ingest.psio, ingest.bt_sign, ingest.ip_sign, ingest)
end

"""
    build_inverse_from_ingest(config::EquilibriumConfig, ingest::InverseIngest) -> InverseRunInput

Rebuild an `InverseRunInput` from an [`InverseIngest`](@ref) captured by
`read_chease_ascii`/`read_chease_binary`. Inverse of that capture; used by the rerun path to
skip the CHEASE parse.
"""
function build_inverse_from_ingest(config::EquilibriumConfig, ingest::InverseIngest)
    sq_in = cubic_interp(ingest.sq_xs, Series(ingest.sq_fs); extrap=ExtendExtrap())
    opts2d = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    rz_in_R = cubic_interp((ingest.rz_xs, ingest.rz_ys), ingest.R_nodes; opts2d...)
    rz_in_Z = cubic_interp((ingest.rz_xs, ingest.rz_ys), ingest.Z_nodes; opts2d...)
    return InverseRunInput(config, sq_in, ingest.rz_xs, ingest.rz_ys, rz_in_R, rz_in_Z,
        ingest.ro, ingest.zo, ingest.psio, ingest)
end
