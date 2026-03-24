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
    sample_boundary_grid(equil, mtheta, nzeta; psi=1.0) -> BoundaryGrid

Evaluate plasma geometry at a uniform (mtheta × nzeta) grid on the flux surface `psi`.

Uses `equil.rzphi_rsquared` and `equil.rzphi_offset` splines at the given `psi`.
Defaults to `psi=1.0` (true plasma boundary). Use a smaller value (e.g. the
Fortran `psilim`) to evaluate on an interior truncation surface.
Poloidal derivatives dR/dθ and dZ/dθ are computed via periodic cubic splines
on the resulting R(θ), Z(θ) data, following the pattern in `Vacuum/Utilities.jl`.

The toroidal grid direction follows the Fortran GPEC convention:
  `phi_j = -helicity × 2π × j/nzeta`   where `helicity = sign(Bt) × sign(Ip)`.
This is derived from `equil.params.bt_sign` and `equil.params.crnt`.
For DIII-D (Bt < 0, Ip > 0 → helicity = -1): phi increases with j (standard direction).
For positive-helicity machines (Bt > 0, Ip > 0 → helicity = +1): phi decreases with j.
"""
function sample_boundary_grid(equil::Equilibrium.PlasmaEquilibrium, mtheta::Int, nzeta::Int;
                               psi::Float64=1.0)
    # Build uniform theta grid (same convention as equil.rzphi_ys, but potentially finer)
    theta_grid = range(0; length=mtheta, step=1.0/mtheta)

    R_arr = zeros(mtheta)
    Z_arr = zeros(mtheta)
    hint2d = (Ref(1), Ref(1))

    for (i, θ_sfl) in enumerate(theta_grid)
        r_minor = sqrt(equil.rzphi_rsquared((psi, θ_sfl); hint=hint2d))
        θ_cyl   = 2π * (θ_sfl + equil.rzphi_offset((psi, θ_sfl); hint=hint2d))
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

    # Helicity sets the direction of the toroidal angle grid to match Fortran convention:
    #   phi_j = -helicity × 2π × j/nzeta,  helicity = sign(Bt) × sign(Ip)
    bt_sign = !isnothing(equil.params.bt_sign) ? equil.params.bt_sign : 1
    ip_sign = !isnothing(equil.params.crnt) ? Int(sign(equil.params.crnt)) : 1
    helicity = bt_sign * ip_sign
    phi_grid = collect(range(0; length=nzeta, step=-helicity * 2π/nzeta))

    return BoundaryGrid(mtheta, nzeta, R_arr, Z_arr, phi_grid, dR_dθ, dZ_dθ)
end

"""
    project_normal_flux!(bn, B_R, B_Z, grid)

Project the cylindrical magnetic field (B_R, B_Z) onto the plasma boundary normal
direction ∇ψ and store in `bn[mtheta, nzeta]`.

The computed quantity is the physical flux element per (θ, ζ) cell [T·m²]:
  bn(θ, ζ) = R(θ) × (B_R(θ,ζ) × ∂Z/∂θ - B_Z(θ,ζ) × ∂R/∂θ)

This is proportional to `dΦ/(dθ·dζ)` and matches the SFL-flux convention (`sfl_flux_Wb`)
used by the permeability matrix. The R factor arises from the Jacobian of the flux-surface
coordinate system (Chance 1997 Eq. for normal field projection).

Julia uses 2π-angle convention throughout (θ ∈ [0, 2π], ζ ∈ [0, 2π]).
The Fortran GPEC uses unit-normalized angles (θ_norm ∈ [0, 1]) and therefore includes
an extra (2π)² factor: Fortran `Phi_x ≈ (2π)² × Julia sfl_flux_Wb`.

## Arguments
- `bn`: output array `[mtheta, nzeta]`; overwritten
- `B_R`, `B_Z`: cylindrical field components, length `mtheta × nzeta` (flat, θ-major ordering)
- `grid`: pre-computed boundary geometry from `sample_boundary_grid`
"""
function project_normal_flux!(
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
            bn[i, j] = grid.R[i] * (B_R[idx] * grid.dZ_dtheta[i] - B_Z[idx] * grid.dR_dtheta[i])
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

When called after `project_normal_flux!`, the returned amplitudes are in `sfl_flux_Wb`
convention (T·m²) with Julia's 2π-angle convention (θ ∈ [0, 2π]).
To compare with Fortran `Phi_x`: Julia sfl_flux ≈ Fortran Phi_x / (2π)².

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
flux from all coil sets on the plasma boundary.

## Pipeline
1. Build boundary grid at (mtheta × nzeta) resolution, with helicity from `equil.params`
2. Lay out observation points in (R, φ, Z) for all (θ, ζ) combinations
3. Run threaded Biot-Savart summation over all coil sets
4. Project B field onto plasma boundary normal flux (`project_normal_flux!`)
5. 2D Fourier decompose to get bmn amplitudes for mode range

Output amplitudes are in `sfl_flux_Wb` convention (T·m², Julia 2π-angle convention).
No normalization conversion is needed when using these modes with `compute_plasma_response!`.

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
    project_normal_flux!(bn, B_R, B_Z, grid)

    verbose && @info "  Max |bn| = $(maximum(abs, bn)) T·m²"

    modes = fourier_decompose_bn(bn, grid, n, m_low, m_high)

    empty!(forcing_modes)
    append!(forcing_modes, modes)

    verbose && @info "  Computed $(length(modes)) forcing modes for n=$n"
end

"""
    convert_forcing_normalization!(modes, from_tag, equil, n, m_low, m_high; psi, mtheta, nzeta)

Convert `ForcingMode` amplitudes from `from_tag` normalization to `"sfl_flux_Wb"`,
which is the internal convention expected by `compute_plasma_response!`.

If `from_tag == "sfl_flux_Wb"`, modes are already in the correct convention and
this function is a no-op.

If `from_tag == "normal_field_T"`, the amplitudes represent Fourier modes of B·n̂
in Tesla with Julia's 2π-angle convention (θ ∈ [0, 2π]).  Conversion to sfl_flux_Wb:
  1. Inverse-Fourier reconstruct B·n̂(θ, ζ) from the input modes
  2. Multiply pointwise by R(θ) × |dr/dθ(θ)| (from `sample_boundary_grid`)
  3. Re-Fourier transform to get sfl_flux_Wb mode amplitudes

This conversion is a mode-mixing operation: since R and |dr/dθ| vary around
the boundary, multiplying in real space couples different m modes.

## Arguments
- `modes`: `Vector{ForcingMode}` with amplitudes to convert (modified in place)
- `from_tag`: source normalization — `"normal_field_T"` or `"sfl_flux_Wb"`
- `equil`: `PlasmaEquilibrium` providing boundary geometry
- `n`: toroidal mode number
- `m_low`, `m_high`: poloidal mode range (must cover all modes in `modes`)
- `psi`: flux surface for boundary geometry (default 1.0)
- `mtheta`, `nzeta`: grid resolution for the conversion (defaults: 256, 64)
"""
function convert_forcing_normalization!(
    modes::Vector{ForcingMode},
    from_tag::String,
    equil::Equilibrium.PlasmaEquilibrium,
    n::Int,
    m_low::Int,
    m_high::Int;
    psi::Float64 = 1.0,
    mtheta::Int = 256,
    nzeta::Int = 64
)
    from_tag == "sfl_flux_Wb" && return  # already correct, nothing to do

    if from_tag != "normal_field_T"
        error("Unknown forcing normalization: \"$from_tag\". Supported: \"normal_field_T\", \"sfl_flux_Wb\".")
    end

    mpert = m_high - m_low + 1
    grid = sample_boundary_grid(equil, mtheta, nzeta; psi=psi)

    # Reconstruct real-space B·n̂(θ, ζ) from input Fourier modes
    cos_basis, sin_basis = compute_fourier_coefficients(mtheta, mpert, m_low, nzeta, 1, n)

    # Build amplitude vectors (real, imag) ordered m_low:m_high
    amp_real = zeros(mpert)
    amp_imag = zeros(mpert)
    for mode in modes
        idx = mode.m - m_low + 1
        1 <= idx <= mpert || continue
        amp_real[idx] = real(mode.amplitude)
        amp_imag[idx] = imag(mode.amplitude)
    end

    # Inverse DFT: field_flat[i + (j-1)*mtheta] = Σ_m (amp_r*cos + amp_i*sin)
    bn_hat = (cos_basis * amp_real .- sin_basis * amp_imag)  # length mtheta*nzeta
    bn_field = reshape(bn_hat, mtheta, nzeta)

    # Multiply by R × |dr/dθ| to convert normal field → SFL flux
    arc = sqrt.(grid.dR_dtheta .^ 2 .+ grid.dZ_dtheta .^ 2)  # |dr/dθ|, length mtheta
    for j in 1:nzeta
        for i in 1:mtheta
            bn_field[i, j] *= grid.R[i] * arc[i]
        end
    end

    # Re-Fourier transform bn_field → sfl_flux_Wb mode amplitudes
    bn_flat = vec(bn_field)
    scale = 2.0 / (mtheta * nzeta)
    new_real = scale .* (cos_basis'  * bn_flat)
    new_imag = scale .* (-sin_basis' * bn_flat)

    # Write back into modes vector (in place, same ordering)
    for mode in modes
        idx = mode.m - m_low + 1
        1 <= idx <= mpert || continue
        mode.amplitude = complex(new_real[idx], new_imag[idx])
    end
end

export BoundaryGrid, sample_boundary_grid
export project_normal_flux!, fourier_decompose_bn
export compute_coil_forcing_modes!
export convert_forcing_normalization!
