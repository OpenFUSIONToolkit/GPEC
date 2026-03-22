"""
    CoilFourier

Converts coil Biot-Savart fields on the plasma boundary into Fourier mode amplitudes
suitable for the perturbed equilibrium pipeline.

## Pipeline
1. `sample_boundary_grid` — evaluate (R, Z, φ) and metric on the plasma boundary
2. `compute_biot_savart_boundary!` (BiotSavart.jl) — compute B at all grid points
3. `project_normal_field!` — extract the normal (∇ψ) component
4. `fourier_decompose_bn` — 2D Fourier decompose to get bmn amplitudes
5. `compute_coil_forcing_modes!` — top-level entry point combining all steps
"""

using FastInterpolations: cubic_interp, PeriodicBC, LinearBinary

"""
    BoundaryGrid

Pre-computed plasma boundary grid for evaluating coil fields.

## Fields
- `mtheta`, `nzeta`: grid dimensions
- `R`, `Z`: cylindrical coordinates `[mtheta]` (same for all ζ, axisymmetric)
- `phi_grid`: toroidal angle grid `[nzeta]` = 0, 2π/nzeta, ..., 2π(nzeta-1)/nzeta
- `dR_dtheta`, `dZ_dtheta`: poloidal derivatives `[mtheta]`
"""
struct BoundaryGrid
    mtheta::Int
    nzeta::Int
    R::Vector{Float64}
    Z::Vector{Float64}
    phi_grid::Vector{Float64}
    dR_dtheta::Vector{Float64}
    dZ_dtheta::Vector{Float64}
end

"""
    sample_boundary_grid(equil, mtheta, nzeta) -> BoundaryGrid

Evaluate plasma boundary geometry at a uniform (mtheta × nzeta) grid.

Uses `equil.rzphi_rsquared` and `equil.rzphi_offset` splines at ψ=1.
Poloidal derivatives dR/dθ and dZ/dθ are computed via periodic cubic splines
on the resulting R(θ), Z(θ) data, following the pattern in `Vacuum/Utilities.jl`.
"""
function sample_boundary_grid(equil::Equilibrium.PlasmaEquilibrium, mtheta::Int, nzeta::Int)
    # Build uniform theta grid (same convention as equil.rzphi_ys, but potentially finer)
    theta_grid = range(0; length=mtheta, step=1.0/mtheta)

    R_arr = zeros(mtheta)
    Z_arr = zeros(mtheta)
    hint2d = (Ref(1), Ref(1))

    for (i, θ_sfl) in enumerate(theta_grid)
        r_minor = sqrt(equil.rzphi_rsquared((1.0, θ_sfl); hint=hint2d))
        θ_cyl   = 2π * (θ_sfl + equil.rzphi_offset((1.0, θ_sfl); hint=hint2d))
        R_arr[i] = equil.ro + r_minor * cos(θ_cyl)
        Z_arr[i] = equil.zo + r_minor * sin(θ_cyl)
    end

    # Compute dR/dθ and dZ/dθ via periodic cubic splines on the boundary contour.
    # LinearBinary search is optimal here: evaluation points are monotonically increasing
    # (the same uniform grid used to construct the spline).
    θ_phys = range(0; length=mtheta, step=2π/mtheta)
    spline_R = cubic_interp(θ_phys, R_arr; bc=PeriodicBC(; endpoint=:exclusive, period=2π), search=LinearBinary())
    spline_Z = cubic_interp(θ_phys, Z_arr; bc=PeriodicBC(; endpoint=:exclusive, period=2π), search=LinearBinary())

    dR_dθ = zeros(mtheta)
    dZ_dθ = zeros(mtheta)
    hint_R = Ref(1)
    hint_Z = Ref(1)
    for i in 1:mtheta
        dR_dθ[i] = spline_R(θ_phys[i]; deriv=1, hint=hint_R)
        dZ_dθ[i] = spline_Z(θ_phys[i]; deriv=1, hint=hint_Z)
    end

    phi_grid = collect(range(0; length=nzeta, step=2π/nzeta))

    return BoundaryGrid(mtheta, nzeta, R_arr, Z_arr, phi_grid, dR_dθ, dZ_dθ)
end

"""
    project_normal_field!(bn, B_R, B_Z, grid)

Project the cylindrical magnetic field (B_R, B_Z) onto the plasma boundary normal
direction ∇ψ and store in `bn[mtheta, nzeta]`.

The unnormalized normal flux density is:
  bn(θ, ζ) = B_R(θ,ζ) × (∂Z/∂θ) - B_Z(θ,ζ) × (∂R/∂θ)

This is proportional to B·∇ψ × R (the flux per unit solid angle), which is the
natural quantity for Fourier expansion in GPEC flux coordinates.
[See: Chance Phys. Plasmas 1997, Eq. for normal field projection]

## Arguments
- `bn`: output array `[mtheta, nzeta]`; overwritten
- `B_R`, `B_Z`: cylindrical field components, length `mtheta × nzeta` (flat, θ-major ordering)
- `grid`: pre-computed boundary geometry from `sample_boundary_grid`
"""
function project_normal_field!(
    bn::Matrix{Float64},
    B_R::AbstractVector{Float64},
    B_Z::AbstractVector{Float64},
    grid::BoundaryGrid
)
    mtheta = grid.mtheta
    nzeta  = grid.nzeta
    @assert size(bn) == (mtheta, nzeta)
    @assert length(B_R) == mtheta * nzeta
    @assert length(B_Z) == mtheta * nzeta

    @inbounds for j in 1:nzeta
        for i in 1:mtheta
            idx = i + (j - 1) * mtheta
            bn[i, j] = B_R[idx] * grid.dZ_dtheta[i] - B_Z[idx] * grid.dR_dtheta[i]
        end
    end
end

"""
    fourier_decompose_bn(bn, grid, n, m_low, m_high) -> Vector{ForcingMode}

2D Fourier decompose `bn[mtheta, nzeta]` to extract mode amplitudes for toroidal
mode number `n` and poloidal range `m_low:m_high`.

Coefficients are computed as:
  bmn = (2 / (mtheta × nzeta)) × Σ_{i,j} bn[i,j] × exp(-i(m×θᵢ - n×ζⱼ))

The factor 2 matches the GPEC/DCON convention for real signals where positive
and negative m modes are related by conjugation.

Uses `compute_fourier_coefficients` from `Utilities.FourierTransforms` with the
3D (mtheta×nzeta, mpert) basis matrix (npert=1, nlow=n).
"""
function fourier_decompose_bn(
    bn::Matrix{Float64},
    grid::BoundaryGrid,
    n::Int,
    m_low::Int,
    m_high::Int
)
    mtheta = grid.mtheta
    nzeta  = grid.nzeta
    mpert  = m_high - m_low + 1

    # Build 2D basis: cos(m*θ - n*ζ) and sin(m*θ - n*ζ)
    # Using 3D call with npert=1, nlow=n gives shape (mtheta*nzeta, mpert)
    cos_basis, sin_basis = compute_fourier_coefficients(mtheta, mpert, m_low, nzeta, 1, n)

    bn_flat = vec(bn)  # column-major: bn_flat[i + (j-1)*mtheta] = bn[i,j] ✓
    scale = 2.0 / (mtheta * nzeta)

    bmn_real = scale .* (cos_basis'  * bn_flat)
    bmn_imag = scale .* (-sin_basis' * bn_flat)

    modes = ForcingMode[]
    for (idx, m) in enumerate(m_low:m_high)
        push!(modes, ForcingMode(; n=n, m=m,
            amplitude=complex(bmn_real[idx], bmn_imag[idx])))
    end
    return modes
end

"""
    compute_coil_forcing_modes!(forcing_modes, coil_sets, equil, cfg, n, m_low, m_high; verbose)

Top-level entry point: compute Fourier mode amplitudes of the normal magnetic
field from all coil sets on the plasma boundary.

## Pipeline
1. Build boundary grid at (mtheta × nzeta) resolution
2. Lay out observation points in (R, φ, Z) for all (θ, ζ) combinations
3. Run threaded Biot-Savart summation over all coil sets
4. Project B field onto plasma boundary normal (∇ψ direction)
5. 2D Fourier decompose to get bmn amplitudes for mode range

Result is appended to `forcing_modes` (existing content is cleared first).
"""
function compute_coil_forcing_modes!(
    forcing_modes::Vector{ForcingMode},
    coil_sets::Vector{CoilSet},
    equil::Equilibrium.PlasmaEquilibrium,
    cfg::CoilConfig,
    n::Int,
    m_low::Int,
    m_high::Int;
    verbose::Bool = false
)
    nzeta = cfg.nzeta_coil > 0 ? cfg.nzeta_coil : 32 * max(1, abs(n))
    mtheta = cfg.mtheta_coil

    verbose && @info "Computing coil forcing modes: mtheta=$mtheta, nzeta=$nzeta, n=$n, m=$m_low:$m_high"

    grid = sample_boundary_grid(equil, mtheta, nzeta)

    # Lay out observation points: (theta_i, zeta_j) → cylindrical (R, phi, Z)
    nobs  = mtheta * nzeta
    obs_R   = zeros(nobs)
    obs_phi = zeros(nobs)
    obs_Z   = zeros(nobs)

    for j in 1:nzeta
        for i in 1:mtheta
            idx = i + (j - 1) * mtheta
            obs_R[idx]   = grid.R[i]
            obs_phi[idx] = grid.phi_grid[j]
            obs_Z[idx]   = grid.Z[i]
        end
    end

    B_R   = zeros(nobs)
    B_phi = zeros(nobs)
    B_Z   = zeros(nobs)
    compute_biot_savart_boundary!(B_R, B_phi, B_Z, obs_R, obs_phi, obs_Z, coil_sets)

    verbose && @info "  Max |B_R| = $(maximum(abs, B_R)) T, Max |B_Z| = $(maximum(abs, B_Z)) T"

    bn = zeros(mtheta, nzeta)
    project_normal_field!(bn, B_R, B_Z, grid)

    verbose && @info "  Max |bn| = $(maximum(abs, bn)) T"

    modes = fourier_decompose_bn(bn, grid, n, m_low, m_high)

    empty!(forcing_modes)
    append!(forcing_modes, modes)

    verbose && @info "  Computed $(length(modes)) forcing modes for n=$n"
end

export BoundaryGrid, sample_boundary_grid
export project_normal_field!, fourier_decompose_bn
export compute_coil_forcing_modes!
