"""
    DirectIngest

Raw arrays and scalars captured by a direct-equilibrium reader (`read_efit`, `read_imas`), enough to
rebuild a `DirectRunInput`'s splines via `build_direct_from_ingest`. Written to `gpec.h5` so a run can be
replayed without the original source.

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

Raw arrays and scalars captured by an inverse-equilibrium reader (`read_chease_ascii`, `read_chease_binary`,
`read_ifile`), enough to rebuild an `InverseRunInput`'s splines via `build_inverse_from_ingest`. The inverse
counterpart of [`DirectIngest`](@ref).

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

Inputs to `direct_run`, built by a direct-equilibrium reader from the raw data.

## Fields

  - `config::EquilibriumConfig` — the equilibrium configuration
  - `sq_in` — 1D spline in normalized ψ of `|F| = R·|B_t|` (m·T), `μ₀P` clipped at zero (T²), `q`, `√ψ_norm`
  - `psi_in` — 2D interpolant of the poloidal flux on (R, Z), `ψ_boundary - ψ` signed positive at the axis (Wb/rad)
  - `psi_in_xs::Vector{Float64}` — R coordinate grid for psi_in [m]
  - `psi_in_ys::Vector{Float64}` — Z coordinate grid for psi_in [m]
  - `rmin::Float64` — Minimum R-coordinate of the computational grid [m]
  - `rmax::Float64` — Maximum R-coordinate of the computational grid [m]
  - `zmin::Float64` — Minimum Z-coordinate of the computational grid [m]
  - `zmax::Float64` — Maximum Z-coordinate of the computational grid [m]
  - `psio::Float64` — Total flux difference `|ψ_axis - ψ_boundary|` [Wb/rad]
  - `bt_sign::Int` — Sign of the toroidal field (+1 or -1), from the sign of F
  - `ip_sign::Int` — Sign of the plasma current (+1 or -1) as the source states it (g-file `current`, IMAS
    `global_quantities.ip`); the internal flux is always positive at the axis, so this sign, not the computed
    current, sets the SFL→machine handedness `helicity = bt_sign × ip_sign`
  - `ingest::EquilibriumIngest` — captured raw arrays for the `gpec.h5` rerun snapshot
    (a [`DirectIngest`](@ref) for file-based reads, or `nothing` for analytic equilibria)
  - `psihigh_resolved::Float64` — outer flux limit the solvers build their ψ grid on: `config.psihigh`, clamped
    to the outermost closed flux surface by [`resolve_psihigh!`](@ref) for efit-family equilibria (not IMAS)
"""
mutable struct DirectRunInput{S<:FastInterpolations.CubicSeriesInterpolant,I2D<:FastInterpolations.CubicInterpolantND}
    config::EquilibriumConfig
    sq_in::S
    psi_in::I2D
    psi_in_xs::Vector{Float64}
    psi_in_ys::Vector{Float64}
    rmin::Float64
    rmax::Float64
    zmin::Float64
    zmax::Float64
    psio::Float64
    bt_sign::Int
    ip_sign::Int
    ingest::EquilibriumIngest
    psihigh_resolved::Float64
end

# Readers construct without a resolved psihigh; it starts at the request until `resolve_psihigh!` clamps it
DirectRunInput(config::EquilibriumConfig, sq_in, psi_in, psi_in_xs, psi_in_ys,
    rmin, rmax, zmin, zmax, psio, bt_sign, ip_sign, ingest) =
    DirectRunInput(config, sq_in, psi_in, psi_in_xs, psi_in_ys,
        rmin, rmax, zmin, zmax, psio, bt_sign, ip_sign, ingest, config.psihigh)

"""
    InverseRunInput(...)

Inputs to `inverse_run`, built by an inverse-equilibrium reader.

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
  - `psihigh_resolved::Float64` - outer flux limit the solvers build their ψ grid on: `config.psihigh`, clamped
    to the outermost input surface for i-files and reruns (`resolve_psihigh!`); `efit_by_inversion` forwards
    the value clamped on its `DirectRunInput`
"""
mutable struct InverseRunInput{S<:FastInterpolations.CubicSeriesInterpolant,I2D<:FastInterpolations.CubicInterpolantND}
    config::EquilibriumConfig
    sq_in::S
    rz_in_xs::Vector{Float64}
    rz_in_ys::Vector{Float64}
    rz_in_R::I2D
    rz_in_Z::I2D
    ro::Float64
    zo::Float64
    psio::Float64
    ingest::EquilibriumIngest
    psihigh_resolved::Float64
end

InverseRunInput(config::EquilibriumConfig, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z,
    ro, zo, psio, ingest) =
    InverseRunInput(config, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z,
        ro, zo, psio, ingest, config.psihigh)

"""
    build_direct_from_ingest(config::EquilibriumConfig, ingest::DirectIngest) -> DirectRunInput

Rebuild a `DirectRunInput`'s splines from a [`DirectIngest`](@ref) (from `read_efit`/`read_imas`, or restored
from `Input/RawInputs/Equilibrium/` in `gpec.h5`), so a rerun skips the g-file/IMAS parse.
"""
function build_direct_from_ingest(config::EquilibriumConfig, ingest::DirectIngest)
    sq_in = cubic_interp(ingest.sq_xs, Series(ingest.sq_fs); extrap=ExtendExtrap())
    psi_in = cubic_interp((ingest.psi_xs, ingest.psi_ys), ingest.psi_rz; extrap=ExtendExtrap())
    return DirectRunInput(config, sq_in, psi_in, ingest.psi_xs, ingest.psi_ys,
        ingest.rmin, ingest.rmax, ingest.zmin, ingest.zmax, ingest.psio, ingest.bt_sign, ingest.ip_sign, ingest)
end

"""
    build_inverse_from_ingest(config::EquilibriumConfig, ingest::InverseIngest) -> InverseRunInput

Rebuild an `InverseRunInput`'s splines from an [`InverseIngest`](@ref), so a rerun skips the CHEASE or
i-file parse.
"""
function build_inverse_from_ingest(config::EquilibriumConfig, ingest::InverseIngest)
    sq_in = cubic_interp(ingest.sq_xs, Series(ingest.sq_fs); extrap=ExtendExtrap())
    opts2d = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    rz_in_R = cubic_interp((ingest.rz_xs, ingest.rz_ys), ingest.R_nodes; opts2d...)
    rz_in_Z = cubic_interp((ingest.rz_xs, ingest.rz_ys), ingest.Z_nodes; opts2d...)
    return InverseRunInput(config, sq_in, ingest.rz_xs, ingest.rz_ys, rz_in_R, rz_in_Z,
        ingest.ro, ingest.zo, ingest.psio, ingest)
end
