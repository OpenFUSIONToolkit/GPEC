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
    _assemble_vacuum_response!(wv, grri_in, grre_in, plasma_surf, wall, cos_mn_basis, sin_mn_basis, kparams; force_wv_symmetry=true)

Shared boundary-integral assembly for a single dense vacuum system.

Given constructed surface geometries, the Fourier bases, and the kernel dispatch parameters,
this builds the double-/single-layer operators, solves the exterior and interior systems, and
inverse-Fourier-transforms the result into the vacuum response matrix `wv` and the Green's
functions `grri`/`grre`. It is the geometry-agnostic kernel shared by the 2D (per-n) and dense
3D (wall) paths; all dimensionality/symmetry decisions are made by the callers, which pass in
the appropriate `plasma_surf`/`wall` types, bases, and `kparams`.

# Arguments

  - `wv`: vacuum response matrix block to fill (`num_modes × num_modes`).
  - `grri_in`, `grre_in`: interior/exterior Green's-function matrices; the active
    `1:num_points_total` rows are written.
  - `plasma_surf`, `wall`: constructed surface geometries (2D or 3D variants; `wall` must
    expose a `nowall::Bool` field).
  - `cos_mn_basis`, `sin_mn_basis`: Fourier basis coefficients (`num_points_surf × num_modes`).
  - `kparams`: kernel dispatch parameters (`KernelParams2D` or `KernelParams3D`).
  - `force_wv_symmetry`: enforce Hermitian symmetry on `wv`.

# Reference

[Chance Phys. Plasmas 2007 052506 eq. 114-118]
"""
@maybe_with_pool pool function _assemble_vacuum_response!(
    wv::AbstractMatrix{ComplexF64},
    grri_in::AbstractMatrix{Float64},
    grre_in::AbstractMatrix{Float64},
    plasma_surf,
    wall,
    cos_mn_basis::AbstractMatrix{Float64},
    sin_mn_basis::AbstractMatrix{Float64},
    kparams;
    force_wv_symmetry::Bool=true
)
    num_points_surf, num_modes = size(cos_mn_basis)

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

    # Final form of vacuum response matrix [Chance Phys. Plasmas 2007 052506 eq. 114]
    wv .= 4π^2 * complex.(arr .+ aii, air .- ari)
    force_wv_symmetry && hermitianpart!(wv)
end

"""
    _compute_vacuum_response_2d!(vac_data, inputs::VacuumInput, wall_settings::WallShapeSettings)

2D (axisymmetric, `inputs.nzeta == 1`) vacuum response calculation.

Each toroidal mode `n` decouples in 2D geometry, so the routine loops over `inputs.n_modes`,
filling the corresponding diagonal block of the response matrix and the matching column block
of the Green's functions via [`_assemble_vacuum_response!`]. Geometry is rebuilt per `n` (it is
`n`-independent, but the Fourier basis is not), and the coordinate arrays are filled identically
on each pass.
"""
function _compute_vacuum_response_2d!(vac_data, inputs::VacuumInput, wall_settings::WallShapeSettings)

    mpert = length(inputs.m_modes)
    numpert_total = mpert * length(inputs.n_modes)

    vac_data.wv .= 0

    # Geometry and Fourier basis for this toroidal mode (n_2D = n, ν from the plasma surface)
    plasma_surf = PlasmaGeometry(inputs)
    wall = WallGeometry(inputs, plasma_surf, wall_settings)

    for (idx_n, n) in enumerate(inputs.n_modes)
        cos_mn_basis, sin_mn_basis = compute_fourier_coefficients(inputs.mtheta, inputs.m_modes, inputs.nzeta, inputs.n_modes; n_2D=n, ν=plasma_surf.ν)
        kparams = KernelParams2D(n)

        # Diagonal block of wv and matching column block of the Green's functions
        block_idx = ((idx_n-1)*mpert+1):(idx_n*mpert)
        cols = vcat(block_idx, numpert_total .+ block_idx)
        wv_block = @view vac_data.wv[block_idx, block_idx]
        grri_block = @view vac_data.grri[:, cols]
        grre_block = @view vac_data.grre[:, cols]

        _assemble_vacuum_response!(wv_block, grri_block, grre_block, plasma_surf, wall, cos_mn_basis, sin_mn_basis, kparams; force_wv_symmetry=inputs.force_wv_symmetry)
    end

    # Populate coordinate arrays
    @views begin
        vac_data.plasma_pts[:, 1] .= plasma_surf.x
        vac_data.plasma_pts[:, 2] .= 0.0
        vac_data.plasma_pts[:, 3] .= plasma_surf.z
        vac_data.wall_pts[:, 1] .= wall.x
        vac_data.wall_pts[:, 2] .= 0.0
        vac_data.wall_pts[:, 3] .= wall.z
    end
end

"""
    _compute_vacuum_response_3d!(vac_data, inputs::VacuumInput, wall_settings::WallShapeSettings)

3D (`inputs.nzeta > 1`) vacuum response calculation, split into two internally-coherent branches.

**nowall** (`wall_settings.shape == "nowall"`) — block-circulant field-period reduction.
The `nfp`-periodic boundary makes the single-/double-layer operators `S`, `D` block-circulant
in the field-period index. Writing the full response as `wv = (4π²/N)·Eᴴ·D⁻¹S·E` (`E` the
complex Fourier basis, `N = mtheta·nzeta_full`), the structure block-diagonalizes the problem by
toroidal residue class `k = mod(n, nfp)`: modes with different `k` do not couple, and within a
class

    D̂ₖ = Σ_d D_d ω^{k d},   Ŝₖ = Σ_d S_d ω^{k d},   ω = exp(-2πi/nfp),

with `D_d`, `S_d` the `M×M` (`M = N/nfp`) blocks coupling observers in field period 0 to sources
in period `d`. Each class needs one `M×M` solve `wv[class k] = (4π²/M)·E_localᴴ·(D̂ₖ \\ Ŝₖ)·E_local`.
Only the first block-row of the operators is built, so the kernel cost drops by `nfp` and the
dense `O(N³)` factorization is replaced by per-class `O(M³)` solves. For `nfp == 1` this
degenerates to a **single residue class with `M = N`**, reproducing the dense solve to round-off.
Only `wv` is produced; `grri`/`grre` are returned zeroed.

**wall** (any other shape) — dense, wall-capable path. Block-circulant does not yet support
walls, so this builds `PlasmaGeometry3D`/`WallGeometry3D` and calls [`_assemble_vacuum_response!`],
producing `wv`, `grri`, and `grre`. This branch is `nfp=1`-oriented; `nfp>1 + wall` has no
consumer and is rejected.
"""
@maybe_with_pool pool function _compute_vacuum_response_3d!(vac_data, inputs::VacuumInput, wall_settings::WallShapeSettings)

    # TODO(block-circulant + wall): an nfp-periodic wall is block-circulant under the same
    # field-period rotation as the plasma, so the coupled [plasma; wall] × [plasma; wall]
    # boundary-integral operator is itself block-circulant in the field-period index. To retire
    # the dense wall branch below: build the first block-row of the 2×2 (plasma/wall) block
    # operator (observers in period 0, M_p + M_w rows × full-torus sources), form the per-residue
    # -class reduced operators D̂ₖ, Ŝₖ over the combined surface exactly as in the nowall case,
    # solve the per-class systems, and apply the per-period basis to the plasma-observer rows. The
    # wall enters only as extra rows/cols of the circulant blocks; the residue-class
    # decomposition is unchanged.
    if wall_settings.shape != "nowall"
        # ── Dense, wall-capable 3D path ──────────────────────────────────────────────────────
        inputs.nfp > 1 && error("3D vacuum with a wall supports nfp=1 only (got nfp=$(inputs.nfp)); use nowall for the field-periodic block-circulant path")

        plasma_surf = PlasmaGeometry3D(inputs)
        wall = WallGeometry3D(inputs, wall_settings)
        cos_mn_basis, sin_mn_basis = compute_fourier_coefficients(inputs.mtheta, inputs.m_modes, inputs.nzeta, inputs.n_modes)
        kparams = KernelParams3D(11, 20, 5)

        _assemble_vacuum_response!(vac_data.wv, vac_data.grri, vac_data.grre, plasma_surf, wall, cos_mn_basis, sin_mn_basis, kparams; force_wv_symmetry=inputs.force_wv_symmetry)

        vac_data.plasma_pts .= plasma_surf.r
        vac_data.wall_pts .= wall.r
        return nothing
    end

    # ── nowall: block-circulant field-period reduction (all nfp; nfp=1 ⇒ single class, M=N) ──
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
    grad_row = zeros!(pool, M, N)   # double-layer D (with +I on the period-0 diagonal block)
    green_row = zeros!(pool, M, N)  # single-layer S
    compute_3D_kernel_matrices!(grad_row, green_row, plasma_surf, plasma_surf, kparams.PATCH_RAD, kparams.RAD_DIM, kparams.INTERP_ORDER; n_obs=M)

    # Complex Fourier basis on a single field period: E_local[idx, col] = exp(i(mθ - nζ_local)).
    # Columns are ordered m-fast, n-slow to match the wv layout used by the 3D bridge.
    θ_grid = range(; start=0, length=mtheta, step=2π/mtheta)
    ζ_grid = range(; start=0, length=nzeta_full, step=2π/nzeta_full)
    E_local = zeros!(pool, ComplexF64, M, num_modes)
    nzeta_p = nzeta_full ÷ nfp
    for (idx_n, n) in enumerate(n_modes), (idx_m, m) in enumerate(m_modes)
        col = idx_m + (idx_n - 1) * mpert
        for jl in 1:nzeta_p, i in 1:mtheta
            idx = i + (jl - 1) * mtheta
            E_local[idx, col] = cis(m * θ_grid[i] - n * ζ_grid[jl])
        end
    end

    # TODO(3D grri/grre): per residue class k, after ldiv!(lu!(D̂ₖ), Ŝₖ) (the exterior operator
    # D̂ₖ⁻¹Ŝₖ), assemble the point→mode Green's-function columns by applying the per-period basis
    # E_local and the interior variant (-D + 2I ⇒ block-circulant interior operator with the
    # period-0 diagonal sign/identity handled per class), mirroring the dense
    # _assemble_vacuum_response! grri/grre construction. Scatter each class's columns back into
    # the full [2N × 2·num_modes] arrays by residue class. Until then grri/grre are returned zeroed.

    # Solve one reduced M×M system per toroidal residue class k = mod(n, nfp). Reusing the D̂/Ŝ
    # buffers and solving in place (lu!/ldiv!) keeps the peak footprint at a few M×M matrices.
    fill!(vac_data.wv, 0)
    D̂ = zeros!(pool, ComplexF64, M, M)
    Ŝ = zeros!(pool, ComplexF64, M, M)
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
        vac_data.wv[mode_cols, mode_cols] .= (4π^2 / M) .* (Ek' * (Ŝ * Ek))
    end

    inputs.force_wv_symmetry && hermitianpart!(vac_data.wv)

    # Only wv is produced on the nowall path; zero the Green's functions and fill plasma points.
    fill!(vac_data.grri, 0)
    fill!(vac_data.grre, 0)
    vac_data.plasma_pts .= plasma_surf.r
    fill!(vac_data.wall_pts, 0)
end

"""
    compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings)

Allocate and return the vacuum response matrix and Green's functions for the given vacuum
inputs. Thin allocating wrapper around the in-place [`compute_vacuum_response!`]: it sizes the
output arrays for the full torus and forwards to the same 2D/3D workers. For performance-critical
paths that already own preallocated storage (e.g. `ForceFreeStates.VacuumData`), prefer the
in-place method to avoid extra heap allocations.

# Returns

  - `wv`: complex vacuum response matrix (`num_modes × num_modes`).
  - `grri`, `grre`: interior/exterior Green's functions (zeroed on the 3D nowall path).
  - `plasma_pts`, `wall_pts`: surface coordinate arrays.
"""
function compute_vacuum_response(inputs::VacuumInput, wall_settings::WallShapeSettings)

    num_points = inputs.mtheta * inputs.nzeta * inputs.nfp # mtheta for 2D
    num_modes = length(inputs.m_modes) * length(inputs.n_modes)

    vac = (
        wv=zeros(ComplexF64, num_modes, num_modes),
        grri=zeros(2 * num_points, 2 * num_modes),
        grre=zeros(2 * num_points, 2 * num_modes),
        plasma_pts=zeros(num_points, 3),
        wall_pts=zeros(num_points, 3)
    )
    compute_vacuum_response!(vac, inputs, wall_settings)

    return vac.wv, vac.grri, vac.grre, vac.plasma_pts, vac.wall_pts
end

"""
    compute_vacuum_response!(vac_data, inputs::VacuumInput, wall_settings::WallShapeSettings)

In-place variant that computes the vacuum response and directly populates the arrays stored in
`vac_data`. Dispatches on dimensionality only: 2D (`inputs.nzeta == 1`) routes to
[`_compute_vacuum_response_2d!`], 3D to [`_compute_vacuum_response_3d!`].

The `vac_data` argument is expected to provide the following writable fields with compatible
sizes:

  - `wv::AbstractMatrix{ComplexF64}`             – vacuum response matrix
  - `grri::AbstractMatrix{Float64}`              – interior Green's functions
  - `grre::AbstractMatrix{Float64}`              – exterior Green's functions
  - `plasma_pts::AbstractMatrix{Float64}`        – plasma surface coordinates
  - `wall_pts::AbstractMatrix{Float64}`          – wall surface coordinates

This is designed to work with `ForceFreeStates.VacuumData` but does not depend on its concrete
type (duck-typed on field names only).
"""
function compute_vacuum_response!(vac_data, inputs::VacuumInput, wall_settings::WallShapeSettings)
    if inputs.nzeta == 1
        _compute_vacuum_response_2d!(vac_data, inputs, wall_settings)
    else
        _compute_vacuum_response_3d!(vac_data, inputs, wall_settings)
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
