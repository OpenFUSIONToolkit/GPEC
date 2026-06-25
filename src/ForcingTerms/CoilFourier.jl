"""
    CoilFourier

Converts coil Biot-Savart fields on the plasma boundary into Fourier mode amplitudes
suitable for the perturbed equilibrium pipeline.

## Pipeline
- `sample_boundary_grid` — evaluate (R, Z) and unit-norm metric on the plasma boundary
- `compute_biot_savart_boundary!` (BiotSavart.jl) — compute B at all grid points
- `project_normal_flux!` — compute flux Φ_x = 2π×R×(B_R ∂Z/∂θ − B_Z ∂R/∂θ)
- `fourier_decompose_bn` — 2D Fourier decompose to get bmn amplitudes
- `compute_coil_forcing_modes!` — top-level entry point combining all steps
"""

using FastInterpolations: cubic_interp, PeriodicBC, DerivOp

# Default toroidal points per period when nzeta_coil is not specified
const NZETA_POINTS_PER_PERIOD = 32

"""
    BoundaryGrid

Pre-computed plasma boundary grid for evaluating coil fields.

## Fields

  - `mtheta`, `nzeta`: grid dimensions
  - `R`, `Z`: cylindrical coordinates `[mtheta]` (same for all ζ, axisymmetric)
  - `phi_grid`: base toroidal angle grid `[nzeta]` in radians = `-helicity × 2π × j/nzeta`
  - `phi_offset`: per-θ toroidal angle correction `[mtheta]` from SFL coordinates:
    `ν(ψ, θ)` scaled by `-helicity`, so the physical toroidal angle at `(i, j)` is
    `phi_grid[j] + phi_offset[i]`. Matches Fortran's `phi = -helicity*(2π*ζ + dphi(ψ,θ))`.
    Zero for axisymmetric equilibria on-axis; non-zero off-axis due to SFL coordinate
    transform (Hamada/SFL θ ≠ geometric θ introduces a toroidal offset).
  - `dR_dtheta`, `dZ_dtheta`: poloidal derivatives w.r.t. unit-norm angle θ_norm ∈ [0,1]
    `dR_dtheta[i] = dR/dθ_norm = 2π × dR/dθ_phys`
"""
struct BoundaryGrid
    mtheta::Int
    nzeta::Int
    R::Vector{Float64}
    Z::Vector{Float64}
    phi_grid::Vector{Float64}
    phi_offset::Vector{Float64}
    dR_dtheta::Vector{Float64}
    dZ_dtheta::Vector{Float64}
end

"""
    sample_boundary_grid(equil, mtheta, nzeta; psi=1.0) -> BoundaryGrid

Evaluate plasma geometry at a uniform (mtheta × nzeta) grid on the flux surface `psi`.

Uses `equil.rzphi_rsquared` and `equil.rzphi_offset` splines at the given `psi`.
Defaults to `psi=1.0` (true plasma boundary). Use a smaller value (e.g. the
Fortran `psilim`) to evaluate on an interior truncation surface.
Poloidal derivatives dR/dθ_norm and dZ/dθ_norm (unit-norm θ_norm ∈ [0,1]) are computed
via periodic cubic splines on the resulting R(θ_norm), Z(θ_norm) data.

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
        θ_cyl = 2π * (θ_sfl + equil.rzphi_offset((psi, θ_sfl); hint=hint2d))
        R_arr[i] = equil.ro + r_minor * cos(θ_cyl)
        Z_arr[i] = equil.zo + r_minor * sin(θ_cyl)
    end

    # Compute dR/dθ_norm and dZ/dθ_norm via periodic cubic splines on the boundary contour.
    # Use unit-norm θ_norm ∈ [0,1] as the spline x-axis (matches equilibrium convention).
    spline_R = cubic_interp(theta_grid, R_arr; bc=PeriodicBC(; endpoint=:exclusive, period=1.0))
    spline_Z = cubic_interp(theta_grid, Z_arr; bc=PeriodicBC(; endpoint=:exclusive, period=1.0))

    dR_dθ = zeros(mtheta)
    dZ_dθ = zeros(mtheta)
    hint_R = Ref(1)
    hint_Z = Ref(1)
    for i in 1:mtheta
        dR_dθ[i] = spline_R(theta_grid[i]; deriv=DerivOp(1), hint=hint_R)   # dR/dθ_norm
        dZ_dθ[i] = spline_Z(theta_grid[i]; deriv=DerivOp(1), hint=hint_Z)   # dZ/dθ_norm
    end

    # Helicity sets the direction of the toroidal angle grid to match Fortran convention:
    #   phi_j = -helicity × 2π × j/nzeta,  helicity = sign(Bt) × sign(Ip)
    bt_sign = !isnothing(equil.params.bt_sign) ? equil.params.bt_sign : 1
    ip_sign = !isnothing(equil.params.crnt) ? Int(sign(equil.params.crnt)) : 1
    helicity = bt_sign * ip_sign
    phi_grid = collect(range(0; length=nzeta, step=(-helicity * 2π/nzeta)))

    # Toroidal angle offset ν(ψ, θ_SFL): in SFL coordinates the physical toroidal angle at
    # grid point (θ_SFL, ζ_SFL) is  φ_phys = -helicity*(2π*ζ_SFL + ν(ψ,θ_SFL)).
    # This matches Fortran's  phi = -helicity*(twopi*czeta + crzphi_f(3)).
    # For a circular boundary ν≈0, but for D-shaped DIII-D geometry it can be several radians.
    phi_offset = zeros(mtheta)
    hint_nu = (Ref(1), Ref(1))
    for (i, θ_sfl) in enumerate(theta_grid)
        phi_offset[i] = -helicity * equil.rzphi_nu((psi, θ_sfl); hint=hint_nu)
    end

    return BoundaryGrid(mtheta, nzeta, R_arr, Z_arr, phi_grid, phi_offset, dR_dθ, dZ_dθ)
end

"""
    project_normal_flux!(bn, B_R, B_Z, grid)

Project the cylindrical magnetic field (B_R, B_Z) onto the plasma boundary normal
direction ∇ψ and store in `bn[mtheta, nzeta]`.

Computes the unit-norm flux element Phi_x per (θ_norm, ζ_norm) cell [T·m²]:
bn(θ_norm, ζ_norm) = 2π × R(θ_norm) × (B_R × ∂Z/∂θ_norm - B_Z × ∂R/∂θ_norm)

The `2π` factor comes from the toroidal Jacobian ∂r/∂ζ_norm = 2π·R·ê_φ in the
cross-product ∂r/∂θ_norm × ∂r/∂ζ_norm. The derivatives dR/dθ_norm and dZ/dθ_norm
are stored in `grid.dR_dtheta` and `grid.dZ_dtheta` (unit-norm convention from
`sample_boundary_grid`).

The output matches Fortran GPEC's `Phi_x` convention directly (no extra factor needed).

## Arguments

  - `bn`: output array `[mtheta, nzeta]`; overwritten in-place
  - `B_R`, `B_Z`: cylindrical field components, length `mtheta × nzeta` (flat, θ-major)
  - `grid`: pre-computed boundary geometry from `sample_boundary_grid`
"""
function project_normal_flux!(
    bn::Matrix{Float64},
    B_R::AbstractVector{Float64},
    B_Z::AbstractVector{Float64},
    grid::BoundaryGrid
)
    mtheta = grid.mtheta
    nzeta = grid.nzeta
    @assert size(bn) == (mtheta, nzeta)
    @assert length(B_R) == mtheta * nzeta
    @assert length(B_Z) == mtheta * nzeta

    @inbounds for j in 1:nzeta
        for i in 1:mtheta
            idx = i + (j - 1) * mtheta
            bn[i, j] = 2π * grid.R[i] * (B_R[idx] * grid.dZ_dtheta[i] - B_Z[idx] * grid.dR_dtheta[i])
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

When called after `project_normal_flux!`, the returned amplitudes are in unit-norm
convention equal to Fortran `Phi_x` (T·m² per unit-norm cell).

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
    nzeta = grid.nzeta

    # Build 2D basis: cos(m*θ - n*ζ) and sin(m*θ - n*ζ)
    # Using 3D call with npert=1, nlow=n gives shape (mtheta*nzeta, mpert)
    cos_basis, sin_basis = compute_fourier_coefficients(mtheta, m_low:m_high, nzeta, [n])

    bn_flat = vec(bn)  # column-major: bn_flat[i + (j-1)*mtheta] = bn[i,j] ✓
    scale = 2.0 / (mtheta * nzeta)

    bmn_real = scale .* (cos_basis' * bn_flat)
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

  - Build boundary grid at (mtheta × nzeta) resolution, with helicity from `equil.params`
  - Lay out observation points in (R, φ, Z) for all (θ, ζ) combinations
  - Run threaded Biot-Savart summation over all coil sets (matches Fortran `field_bs_psi`)
  - Project B field onto plasma boundary normal flux (`project_normal_flux!`)
  - 2D Fourier decompose to get bmn amplitudes for mode range

Output amplitudes are in unit-norm convention (= Fortran `Phi_x`).
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
    psi::Float64=1.0,
    verbose::Bool=false
)
    nzeta = cfg.nzeta_coil > 0 ? cfg.nzeta_coil : NZETA_POINTS_PER_PERIOD * max(1, abs(n))
    mtheta = cfg.mtheta_coil

    verbose && @info "Computing coil forcing modes: mtheta=$mtheta, nzeta=$nzeta, n=$n, m=$m_low:$m_high, psi=$psi"

    grid = sample_boundary_grid(equil, mtheta, nzeta; psi)

    # Lay out observation points: (theta_i, zeta_j) → cylindrical (R, phi, Z)
    nobs = mtheta * nzeta
    obs_R = zeros(nobs)
    obs_phi = zeros(nobs)
    obs_Z = zeros(nobs)

    for j in 1:nzeta
        for i in 1:mtheta
            idx = i + (j - 1) * mtheta
            obs_R[idx] = grid.R[i]
            obs_phi[idx] = grid.phi_grid[j] + grid.phi_offset[i]
            obs_Z[idx] = grid.Z[i]
        end
    end

    B_R = zeros(nobs)
    B_phi = zeros(nobs)
    B_Z = zeros(nobs)
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

Convert `ForcingMode` amplitudes from `from_tag` normalization to the internal
unit-norm convention (= Fortran `Phi_x`), expected by `compute_plasma_response!`.

If `from_tag == "sfl_flux_Wb"`, the user has provided amplitudes in the 2π-angle
SFL-flux convention. These are scaled by (2π)² to reach unit-norm.

If `from_tag == "normal_field_T"`, the amplitudes represent Fourier modes of B·n̂
in Tesla (2π-angle convention). Conversion to unit-norm Phi_x:

  - Inverse-Fourier reconstruct B·n̂(θ, ζ) from the input modes
  - Multiply pointwise by 2π × R(θ) × |dr/dθ_norm(θ)|
  - Re-Fourier transform to get unit-norm mode amplitudes

The 2π factor comes from the toroidal Jacobian ∂r/∂ζ_norm = 2π·R·ê_φ.
This conversion is a mode-mixing operation because R and |dr/dθ| vary poloidally.

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
    psi::Float64=1.0,
    mtheta::Int=256,
    nzeta::Int=64
)
    if from_tag == "sfl_flux_Wb"
        # User provided 2π-angle SFL flux; scale by (2π)² to reach unit-norm (= Phi_x)
        for mode in modes
            mode.amplitude *= (2π)^2
        end
        return
    end

    if from_tag != "normal_field_T"
        error("Unknown forcing normalization: \"$from_tag\". Supported: \"normal_field_T\", \"sfl_flux_Wb\".")
    end

    mpert = m_high - m_low + 1
    grid = sample_boundary_grid(equil, mtheta, nzeta; psi=psi)

    # Reconstruct real-space B·n̂(θ, ζ) from input Fourier modes
    cos_basis, sin_basis = compute_fourier_coefficients(mtheta, m_low:m_high, nzeta, [n])

    # Build amplitude vectors (real, imag) ordered m_low:m_high
    amp_real = zeros(mpert)
    amp_imag = zeros(mpert)
    for mode in modes
        idx = mode.m - m_low + 1
        1 <= idx <= mpert || continue
        amp_real[idx] = real(mode.amplitude)
        amp_imag[idx] = imag(mode.amplitude)
    end

    # Inverse DFT: reconstruct B·n̂(θ, ζ) at grid points
    bn_hat = (cos_basis * amp_real .- sin_basis * amp_imag)  # length mtheta*nzeta
    bn_field = reshape(bn_hat, mtheta, nzeta)

    # Multiply by 2π × R × |dr/dθ_norm| to convert B·n̂ → unit-norm Phi_x integrand.
    # The 2π is the toroidal Jacobian (∂r/∂ζ_norm = 2π·R·ê_φ); arc is the unit-norm
    # arc length |dr/dθ_norm| from sample_boundary_grid.
    arc = sqrt.(grid.dR_dtheta .^ 2 .+ grid.dZ_dtheta .^ 2)  # |dr/dθ_norm|
    for j in 1:nzeta
        for i in 1:mtheta
            bn_field[i, j] *= 2π * grid.R[i] * arc[i]
        end
    end

    # Re-Fourier transform bn_field → unit-norm (Phi_x) mode amplitudes
    bn_flat = vec(bn_field)
    scale = 2.0 / (mtheta * nzeta)
    new_real = scale .* (cos_basis' * bn_flat)
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
