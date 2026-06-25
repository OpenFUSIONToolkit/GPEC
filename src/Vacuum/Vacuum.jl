module Vacuum

using TOML, SpecialFunctions, LinearAlgebra, Printf
using FastInterpolations
using FastGaussQuadrature: gausslegendre
using StaticArrays: SVector
using SparseArrays
using AdaptiveArrayPools

# Import parent modules
import ..Equilibrium
using ..Utilities.FourierTransforms: compute_fourier_coefficients, fourier_transform!, fourier_inverse_transform!

include("Utilities.jl")
include("DataTypes.jl")
include("PnQuadCache.jl")
include("Kernel2D.jl")
include("Kernel3D.jl")
include("Field.jl")

export VacuumInput, WallShapeSettings
export compute_vacuum_response, compute_vacuum_response!, compute_vacuum_field
export extract_plasma_surface_at_psi
export PlasmaGeometry

"""
    compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings)

Compute the vacuum response matrix and both Green's functions using provided vacuum inputs.

Single entry point for vacuum calculations.

  - For **3D** (`inputs.nzeta > 1`), computes the full coupled response across all (m,n) modes

  - For **2D geometry** (`inputs.nzeta == 1`), supports either:

      + **single-n** (`length(inputs.n_modes) == 1`): computes (m,n) response for `n = inputs.n_modes[1]`
      + **multi-n** (`length(inputs.n_modes) > 1`): loops over `inputs.n_modes` and returns
        **blocks** of the full response matrices with one block per toroidal mode number.

This is the pure Julia implementation that replaces the Fortran `mscvac` function.
It computes both interior (grri) and exterior (grre) Green's functions for GPEC response calculations.

# Arguments

  - `inputs`: `VacuumInput` struct with mode numbers, grid resolution, and boundary info.
  - `wall_settings::WallShapeSettings`: Wall geometry configuration.

# Returns

  - `wv`: Complex vacuum response matrix.
  - `grri`: Interior Green's function matrix.
  - `grre`: Exterior Green's function matrix.
  - `xzpts`: Coordinate array (mtheta×4 for 2D, mtheta*nzeta×4 for 3D) [R_plasma, Z_plasma, R_wall, Z_wall].
"""
@with_pool pool function _compute_vacuum_response_single!(
    wv::AbstractMatrix{ComplexF64},
    grri_in::AbstractMatrix{Float64},
    grre_in::AbstractMatrix{Float64},
    plasma_pts::AbstractMatrix{Float64},
    wall_pts::AbstractMatrix{Float64},
    inputs::VacuumInput,
    wall_settings::WallShapeSettings;
    n_override::Union{Nothing,Int}=nothing
)

    # Initialize surface geometries
    plasma_surf = inputs.nzeta > 1 ? PlasmaGeometry3D(inputs) : PlasmaGeometry(inputs)
    wall = inputs.nzeta > 1 ? WallGeometry3D(inputs, wall_settings) : WallGeometry(inputs, plasma_surf, wall_settings)

    # Compute Fourier basis coefficients
    ν = hasproperty(plasma_surf, :ν) ? plasma_surf.ν : nothing
    cos_mn_basis, sin_mn_basis = compute_fourier_coefficients(inputs.mtheta, inputs.m_modes, inputs.nzeta, inputs.n_modes; n_2D=n_override, ν=ν)
    num_points_surf, num_modes = size(cos_mn_basis)

    # Create kernel parameters structs used to dispatch to the correct kernel
    if inputs.nzeta > 1
        # Hardcode these values for now - can expose to the user in the future
        kparams = KernelParams3D(11, 20, 5)
    else
        kparams = KernelParams2D(n_override)
    end

    # Active rows for computation (plasma only if no wall, plasma+wall if wall present)
    num_points_total = wall.nowall ? num_points_surf : 2 * num_points_surf

    # Local work arrays
    grad_green = zeros!(pool, num_points_total, num_points_total)
    green_temp = zeros!(pool, num_points_surf, num_points_surf)

    # Views into output Green's function matrices for the active rows/columns
    grre = @view grre_in[1:num_points_total, :]
    grri = @view grri_in[1:num_points_total, :]

    # Plasma–Plasma block
    kernel!(grad_green, green_temp, plasma_surf, plasma_surf, kparams)

    # Fourier transform obs=plasma, src=plasma block
    fourier_transform!(grre, green_temp, cos_mn_basis)
    fourier_transform!(grre, green_temp, sin_mn_basis; col_offset=num_modes)

    if !wall.nowall
        # Plasma–Wall block
        kernel!(grad_green, green_temp, plasma_surf, wall, kparams)
        # Wall–Wall block
        kernel!(grad_green, green_temp, wall, wall, kparams)
        # Wall–Plasma block
        kernel!(grad_green, green_temp, wall, plasma_surf, kparams)
        # Fourier transform obs=wall, src=plasma block
        fourier_transform!(grre, green_temp, cos_mn_basis; row_offset=num_points_surf)
        fourier_transform!(grre, green_temp, sin_mn_basis; row_offset=num_points_surf, col_offset=num_modes)
    end

    # Compute both Green's functions: exterior (kernelsign=+1) then interior (kernelsign=-1)
    grri .= grre # start from same as exterior
    grad_green_interior = similar!(pool, grad_green)
    grad_green_interior .= grad_green

    # Solve exterior first, overwriting grad_green to save memory since we already have the interior kernel
    F_ext = lu!(grad_green)
    ldiv!(F_ext, grre)

    # Interior flips the sign of the normal, but not the diagonal terms, so we multiply by -1 and add 2I to the diagonal
    grad_green_interior .*= -1
    for i in 1:num_points_total
        grad_green_interior[i, i] += 2.0
    end
    F_int = lu!(grad_green_interior)
    ldiv!(F_int, grri)

    # Perform inverse Fourier transforms to get response matrix components [Chance Phys. Plasmas 2007 052506 eq. 115-118]
    arr, aii, ari, air = ntuple(_ -> zeros(num_modes, num_modes), 4)
    fourier_inverse_transform!(arr, grre, cos_mn_basis)
    fourier_inverse_transform!(aii, grre, sin_mn_basis; col_offset=num_modes)
    fourier_inverse_transform!(ari, grre, sin_mn_basis)
    fourier_inverse_transform!(air, grre, cos_mn_basis; col_offset=num_modes)

    # fourier_inverse_transform! uses 1/N normalization; restore the 4π² physics factor
    # from the vacuum Green's function integral [Chance 2007 eq. 114-118]
    arr .*= 4π^2
    aii .*= 4π^2
    ari .*= 4π^2
    air .*= 4π^2

    # Final form of vacuum response matrix [Chance Phys. Plasmas 2007 052506 eq. 114]
    wv .= complex.(arr .+ aii, air .- ari)
    inputs.force_wv_symmetry && hermitianpart!(wv)

    # Fill coordinate arrays
    if inputs.nzeta > 1 # 3D
        plasma_pts .= plasma_surf.r
        wall_pts .= wall.r
    else # 2D
        @views begin
            plasma_pts[:, 1] .= plasma_surf.x
            plasma_pts[:, 2] .= 0.0
            plasma_pts[:, 3] .= plasma_surf.z
            wall_pts[:, 1] .= wall.x
            wall_pts[:, 2] .= 0.0
            wall_pts[:, 3] .= wall.z
        end
    end
end

"""
    compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings)

Allocate and return the vacuum response matrix and Green's functions for the given
vacuum inputs.

This is a thin allocating wrapper around the in‑place [`compute_vacuum_response!`]
implementation. For performance‑critical paths that already own preallocated storage
(e.g. `ForceFreeStates.VacuumData`), prefer the in‑place method to avoid extra
heap allocations.
"""
@with_pool pool function compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings)

    # Field-periodic 3D reduction: exploit the block-circulant structure of an nfp-periodic
    # boundary to solve only reduced per-residue-class systems instead of the full torus.
    if inputs.nzeta > 1 && inputs.nfp > 1
        return _compute_vacuum_response_3d_periodic(inputs, wall_settings)
    end

    # Reconstruct the full torus from a single field period (no-op unless nfp > 1)
    inputs = expand_field_periods(inputs)

    # Allocate storage for the vacuum response matrix and Green's functions
    numpoints = inputs.mtheta * inputs.nzeta
    num_modes = length(inputs.m_modes) * length(inputs.n_modes)

    vac = (
        wv=zeros(ComplexF64, num_modes, num_modes),
        grri=zeros(2 * numpoints, 2 * num_modes),
        grre=zeros(2 * numpoints, 2 * num_modes),
        plasma_pts=zeros(numpoints, 3),
        wall_pts=zeros(numpoints, 3)
    )
    compute_vacuum_response!(vac, inputs, wall_settings)

    return vac.wv, vac.grri, vac.grre, vac.plasma_pts, vac.wall_pts
end

"""
    _compute_vacuum_response_3d_periodic(inputs::VacuumInput, wall_settings::WallShapeSettings)

Field-periodic ("layer-2") vacuum response for a 3D boundary with `inputs.nfp > 1`.

The boundary is `nfp`-periodic, so the single-/double-layer boundary-integral operators `S`
and `D` are block-circulant in the field-period index. Writing the full response as
`wv = (4π²/N)·Eᴴ·D⁻¹S·E` (`E` the complex Fourier basis, `N = mtheta·nzeta_full`), the
block-circulant structure block-diagonalizes the problem by toroidal residue class
`k = mod(n, nfp)`: modes with different `k` do not couple, and within a class

    D̂ₖ = Σ_d D_d ω^{k d},   Ŝₖ = Σ_d S_d ω^{k d},   ω = exp(-2πi/nfp),

where `D_d`, `S_d` are the `M×M` (`M = N/nfp`) blocks coupling observers in field period 0 to
sources in field period `d`. Each class then needs a single `M×M` solve

    wv[class k] = (4π²/M)·E_localᴴ·(D̂ₖ \\ Ŝₖ)·E_local,

with `E_local` the basis restricted to one field period. Only the first block-row of the
operators is built (observers in period 0), so the kernel cost drops by `nfp` and the dense
`O(N³)` factorization is replaced by per-class `O(M³)` solves.

Only the response matrix `wv` is produced (the only quantity the 3D bridge consumes); the
interior/exterior Green's-function matrices are returned zeroed. Supports `nowall` only.
"""
function _compute_vacuum_response_3d_periodic(inputs::VacuumInput, wall_settings::WallShapeSettings)

    wall_settings.shape == "nowall" || error("Field-periodic 3D vacuum reduction supports nowall only (got shape=\"$(wall_settings.shape)\")")

    nfp = inputs.nfp

    # Full-torus geometry (O(N) memory) used as the source surface; observers are restricted
    # to field period 0 below, so the dense N×N operators are never formed.
    full = expand_field_periods(inputs)
    plasma_surf = PlasmaGeometry3D(full)

    mtheta = full.mtheta
    nzeta_full = full.nzeta
    nzeta_full % nfp == 0 || error("nzeta_full=$nzeta_full must be divisible by nfp=$nfp for field-periodic reduction")
    N = mtheta * nzeta_full          # full-torus point count
    M = N ÷ nfp                      # points per field period (= mtheta * nzeta_per_period)

    m_modes = inputs.m_modes
    n_modes = inputs.n_modes
    mpert = length(m_modes)
    num_modes = mpert * length(n_modes)

    # First block-row of the operators: observers in field period 0 (M rows) × all sources (N cols)
    kparams = KernelParams3D(11, 20, 5)
    grad_row = zeros(M, N)   # double-layer D (with +I on the period-0 diagonal block)
    green_row = zeros(M, N)  # single-layer S
    compute_3D_kernel_matrices!(grad_row, green_row, plasma_surf, plasma_surf, kparams.PATCH_RAD, kparams.RAD_DIM, kparams.INTERP_ORDER; n_obs=M)

    # Complex Fourier basis on a single field period: E_local[idx, col] = exp(i(mθ - nζ_local)).
    # Columns are ordered m-fast, n-slow to match the wv layout used by the 3D bridge.
    θ_grid = range(; start=0, length=mtheta, step=2π/mtheta)
    ζ_grid = range(; start=0, length=nzeta_full, step=2π/nzeta_full)
    E_local = zeros(ComplexF64, M, num_modes)
    nzeta_p = nzeta_full ÷ nfp
    for (idx_n, n) in enumerate(n_modes), (idx_m, m) in enumerate(m_modes)
        col = idx_m + (idx_n - 1) * mpert
        for jl in 1:nzeta_p, i in 1:mtheta
            idx = i + (jl - 1) * mtheta
            E_local[idx, col] = cis(m * θ_grid[i] - n * ζ_grid[jl])
        end
    end

    # Solve one reduced M×M system per toroidal residue class k = mod(n, nfp). Reusing the D̂/Ŝ
    # buffers and solving in place (lu!/ldiv!) keeps the peak footprint at a few M×M matrices.
    wv = zeros(ComplexF64, num_modes, num_modes)
    D̂ = Matrix{ComplexF64}(undef, M, M)
    Ŝ = Matrix{ComplexF64}(undef, M, M)
    for k in unique(mod.(n_modes, nfp))
        # Reduced operators D̂ₖ = Σ_d D_d ω^{k d}, Ŝₖ = Σ_d S_d ω^{k d} (fused, allocation-free)
        fill!(D̂, 0)
        fill!(Ŝ, 0)
        for d in 0:(nfp-1)
            phase = cis(-2π * (k * d) / nfp)
            cols = (d*M+1):((d+1)*M)
            @views @. D̂ += phase * grad_row[:, cols]
            @views @. Ŝ += phase * green_row[:, cols]
        end

        # Exterior response operator for this sector: solve D̂ₖ G = Ŝₖ in place (Ŝ ← G = D̂ₖ⁻¹Ŝₖ)
        ldiv!(lu!(D̂), Ŝ)
        mode_cols = [(idx_m + (idx_n-1)*mpert) for (idx_n, n) in enumerate(n_modes) if mod(n, nfp) == k for idx_m in 1:mpert]
        Ek = @view E_local[:, mode_cols]
        wv[mode_cols, mode_cols] .= (4π^2 / M) .* (Ek' * (Ŝ * Ek))
    end

    inputs.force_wv_symmetry && hermitianpart!(wv)

    # Only wv is consumed by the 3D bridge; return zeroed Green's functions and the plasma points.
    grri = zeros(2 * N, 2 * num_modes)
    grre = zeros(2 * N, 2 * num_modes)
    wall_pts = zeros(N, 3)
    return wv, grri, grre, copy(plasma_surf.r), wall_pts
end

"""
    compute_vacuum_response!(vac_data, inputs::VacuumInput, wall_settings::WallShapeSettings)

In-place variant that computes the vacuum response and directly populates the arrays
stored in `vac_data`.

The `vac_data` argument is expected to provide the following writable fields with
compatible sizes:

  - `wv::AbstractMatrix{ComplexF64}`             – vacuum response matrix
  - `grri::AbstractMatrix{Float64}`              – interior Green's functions
  - `grre::AbstractMatrix{Float64}`              – exterior Green's functions
  - `plasma_pts::AbstractMatrix{Float64}`        – plasma surface coordinates
  - `wall_pts::AbstractMatrix{Float64}`          – wall surface coordinates

This is designed to work with `ForceFreeStates.VacuumData` but does not depend on
its concrete type (duck-typed on field names only).
"""
function compute_vacuum_response!(vac_data, inputs::VacuumInput, wall_settings::WallShapeSettings)

    mpert = length(inputs.m_modes)
    numpert_total = mpert * length(inputs.n_modes)

    # 3D vacuum: full coupled response across (m,n) from a single kernel call
    if inputs.nzeta > 1
        _compute_vacuum_response_single!(
            vac_data.wv,
            vac_data.grri,
            vac_data.grre,
            vac_data.plasma_pts,
            vac_data.wall_pts,
            inputs,
            wall_settings
        )
    else
        # 2D vacuum: fill diagonal blocks of the response matrix
        vac_data.wv .= 0

        # Each n is independent in 2D geometry → fill diagonal blocks inside preallocated arrays
        for (idx_n, n) in enumerate(inputs.n_modes)
            block_idx = ((idx_n-1)*mpert+1):(idx_n*mpert)
            cols = vcat(block_idx, numpert_total .+ block_idx)

            wv_block = @view vac_data.wv[block_idx, block_idx]
            grri_block = @view vac_data.grri[:, cols]
            grre_block = @view vac_data.grre[:, cols]

            _compute_vacuum_response_single!(
                wv_block,
                grri_block,
                grre_block,
                vac_data.plasma_pts,
                vac_data.wall_pts,
                inputs,
                wall_settings;
                n_override=n
            )
        end
    end
end

"""
    compute_vacuum_field(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall::WallGeometry,
           Bn::Vector{<:Number}, R_grid::AbstractVector, Z_grid::AbstractVector)

Calculate the perturbed magnetic field in the vacuum region resulting from a normal
magnetic field perturbation (`Bn`) at the plasma surface. Replaces `mscfld` from Fortran.

This function orchestrates the vacuum field calculation by:

 1. Calling `vaccal!` to compute the vacuum response kernel (`grri`)
 2. Defining a grid of points (`R_grid`, `Z_grid`) where the field is to be calculated
 3. Calling `_pickup_field` to compute the magnetic field components on that grid using the kernel
    and the source perturbation `Bn`

# Arguments

  - `inputs::VacuumInput`: Struct containing vacuum calculation parameters (n, mpert, mtheta, etc.)
  - `plasma_surf::PlasmaGeometry`: Struct with plasma surface geometry and basis functions
  - `wall::WallGeometry`: Struct with wall geometry
  - `Bn::Vector{<:Number}`: Complex vector of Fourier harmonics of the normal magnetic field
    perturbation at the plasma surface, `B_n = B_n_real + i*B_n_imag`. Length must be `mpert`.
  - `R_grid::AbstractVector`: Vector of R coordinates for the output field grid
  - `Z_grid::AbstractVector`: Vector of Z coordinates for the output field grid

# Returns

  - `B_R::Matrix{ComplexF64}`: The R-component of the magnetic field on the grid
  - `B_Z::Matrix{ComplexF64}`: The Z-component of the magnetic field on the grid
  - `B_phi::Matrix{ComplexF64}`: The toroidal component of the magnetic field on the grid
  - `grid_info::Matrix{Int}`: Information about the grid points (1=inside plasma, 0=outside)
"""
function compute_vacuum_field(inputs::VacuumInput, plasma_surf::PlasmaGeometry, wall::WallGeometry,
    Bn::Vector{<:Number}, R_grid::AbstractVector, Z_grid::AbstractVector)

    # 1. Call vaccal! to get the inverted Green's function matrix
    # The Fortran version calls the whole chain (ent33 -> vaccal),
    # here we assume vaccal! provides what we need.
    wv, grri = vaccal!(inputs, plasma_surf, wall)

    # Separate real and imaginary parts of the source perturbation
    Bn_real = real.(Bn)
    Bn_imag = imag.(Bn)

    # 2. Define grid and parameters for pickup routine
    nx = length(R_grid)
    nz = length(Z_grid)

    # 3. Call the field pickup routine
    B_R, B_Z, B_phi, grid_info = _pickup_field(
        inputs, plasma_surf, grri, Bn_real, Bn_imag, R_grid, Z_grid
    )

    return B_R, B_Z, B_phi, grid_info
end
end
