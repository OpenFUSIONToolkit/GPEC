"""
Response matrix construction for perturbed equilibrium calculations.

Based on gpresp.f from GPEC, implementing resp_index=0 (energy-based inductance).
Uses ForceFreeStates eigenmode solutions and vacuum response data.

Reference: [Park Phys. Plasmas 2009 056115]
"""

# Use FourierTransform utility instead of FFTW for theta ↔ mode transforms
using ..Utilities.FourierTransforms

"""
    build_flux_matrix(equil, ffs_results, intr) -> Matrix{ComplexF64}

Calculate the vacuum poloidal flux matrix at the plasma boundary:
bwp_mn[i,j] = 1im * (dΨ/dρ) * (m - n * q) * ξ_ψ(i,j)

Arguments:

  - `equil`: Plasma equilibrium
  - `ffs_results`: ForceFreeStates eigenmode ODE results
  - `intr`: ForceFreeStates internal state with mode information

Returns:

  - `bwp_mn`: Complex flux matrix [mode_i, eigenmode_j]
"""
function build_flux_matrix(equil::Equilibrium.PlasmaEquilibrium, ffs_results::OdeState, intr::ForceFreeStatesInternal)::Matrix{ComplexF64}

    # Extract boundary displacements and equilibrium quantities
    ξ_psi_boundary = ffs_results.u_store[:, :, 1, ffs_results.step]
    q_boundary = ffs_results.q_store[ffs_results.step]
    dPsi_drho = (2π)^2 * equil.psio

    # Compute singular factor for each Fourier mode
    singfac = vec((intr.mlow:intr.mhigh) .- q_boundary .* (intr.nlow:intr.nhigh)')

    # Compute normal magnetic field at plasma boundary [Park Phys. Plasmas 2009 056115 eq. 4]
    bwp_mn = 1im * dPsi_drho .* singfac .* ξ_psi_boundary

    return bwp_mn
end

"""
    calc_plasma_inductance(ffs_intr, wt0, psio) -> Matrix{ComplexF64}

Compute the plasma inductance Λ from the displacement-space energy matrix ``W_0 = W_p + W_v``
(Fortran `gpresp.f`, `resp_induct_flag=TRUE`):

 1. Normalize `wt0` to SI units (`ψ₀²/(μ₀·2)`, as in `idcon.f`).
 2. Convert ξ → Φ via ``Λ⁻¹ = 2·T₁·W₀·T₂`` with ``T_{ii} ∝ 1/(χ₁ s_i 2π)`` and
    ``s_i = m_i - n·q_lim``; the ``s_i s_j`` factor undoes the `wv` scaling in `free_run!`.
 3. Invert to obtain Λ

## Arguments

  - `ffs_intr`: ForceFreeStates internal state (`mlow`, `mhigh`, `nlow`, `nhigh`, `qlim`)
  - `wt0`: Total energy matrix `wp + wv` [numpert_total × numpert_total]
  - `psio`: Toroidal flux [Wb/rad]

## Returns

  - Plasma inductance Λ matrix [numpert_total × numpert_total]
"""
function calc_plasma_inductance(ffs_intr::ForceFreeStatesInternal, wt0::Matrix{ComplexF64}, psio::Float64)::Matrix{ComplexF64}

    # Singular factors s_i = m_i - n*qlim  (same as Fortran: mfac(i) - nn*qlim)
    singfac = vec((ffs_intr.mlow:ffs_intr.mhigh) .- ffs_intr.qlim .* (ffs_intr.nlow:ffs_intr.nhigh)')

    # Convert to metric units
    mu0 = 4π * 1e-7
    wt0_norm = wt0 .* (psio^2 / (mu0 * 2))

    # Convert from displacement to flux space using 1 / (singfac * chi1 * 2π) factor
    chi1 = 2π * psio
    wt0_norm .*= 2 ./ (2π * chi1)^2 ./ (singfac' .* singfac)

    return inv(wt0_norm)
end

"""
    build_control_surface_rootarea_to_area_weight(
        equil::Equilibrium.PlasmaEquilibrium,
        ffs_intr::ForceFreeStatesInternal
    )::Tuple{Matrix{ComplexF64}, Float64}

Build the numpert_total × numpert_total root-area-weighted → area-weighted field operator
`S = Σ/√A` at the control surface (psilim), and return it together with the scalar surface area
`A = jarea`. The mpert × mpert single-n block (Equilibrium.rootarea_to_area_weight) is repeated
block-diagonally over the `npert` toroidal harmonics, matching the numpert_total mode ordering used
by the response matrices.

`S` maps a root-area-weighted control-surface field `b̃` to the area-weighted field `b̄`
(`b̄ = S·b̃`); poloidal flux is the scalar product `Φ = A·b̄` (so the b̃→flux conform operator is
`R = S·A`, used only internally). `S` and `A` are the recovery aids users need to express the stored
coordinate-invariant (b̃) matrices in the area-weighted field or recover flux — see
`field_space_response_matrices`. [Pharr 2026]
"""
function build_control_surface_rootarea_to_area_weight(
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal
)::Tuple{Matrix{ComplexF64},Float64}
    mpert = ffs_intr.mpert
    npert = ffs_intr.npert
    Npert = ffs_intr.numpert_total

    mtheta_eq = length(equil.rzphi_ys)
    ft = Utilities.FourierTransforms.FourierTransform(mtheta_eq, mpert, ffs_intr.mlow)
    S_block = Equilibrium.rootarea_to_area_weight(equil, ffs_intr.psilim, ft)
    jarea = Equilibrium.flux_surface_area(equil, ffs_intr.psilim, mtheta_eq)

    npert == 1 && return (Matrix{ComplexF64}(S_block), jarea)

    S_full = zeros(ComplexF64, Npert, Npert)
    for in in 1:npert
        r = ((in-1)*mpert+1):(in*mpert)
        S_full[r, r] .= S_block
    end
    return (S_full, jarea)
end

"""
    field_space_response_matrices(
        plasma_inductance, surface_inductance, permeability, reluctance,
        rootarea_to_area_weight, jarea
    )::NamedTuple

Express the control-surface response matrices in the coordinate-invariant root-area-weighted
field (b̃) space, given the flux-space matrices, the b̃→b̄ operator `S = rootarea_to_area_weight`,
and the scalar surface area `jarea`. The brief internal flux-conform operator is `R = S·jarea`
(`Φ = R·b̃`); poloidal flux never leaves this function.

The matrices fall into two algebraic classes:

  - **Operators** (map flux → flux): permeability `P` (Φ_tot = P·Φ_x) transforms by similarity
    `P̃ = R⁻¹·P·R`. Its singular values are coordinate-invariant.
  - **Quadratic generators** (energy = Φ†·G⁻¹·Φ): inductances `Λ`, `L` transform by congruence
    `G̃ = R⁻¹·G·R⁻†`; the inverse-inductance-like reluctance `ϱ` (energy = Φ†·ϱ·Φ) transforms as
    `ϱ̃ = R†·ϱ·R`. Their spectra are coordinate-invariant.

These rules are mutually consistent: `P̃ = Λ̃·L̃⁻¹ = R⁻¹·Λ·L⁻¹·R = R⁻¹·P·R`, and
`ϱ̃ = L̃⁻¹·(Λ̃−L̃)·L̃⁻¹`. To recover the area-weighted (`b̄`) forms, conform with `S` instead of `R`
(e.g. `L_b̄ = S·L̃·S†`); recover flux with the scalar `A`: `Φ = A·b̄`. [Pharr 2026]
"""
function field_space_response_matrices(
    plasma_inductance::Matrix{ComplexF64},
    surface_inductance::Matrix{ComplexF64},
    permeability::Matrix{ComplexF64},
    reluctance::Matrix{ComplexF64},
    rootarea_to_area_weight::Matrix{ComplexF64},
    jarea::Float64
)::NamedTuple
    R = rootarea_to_area_weight .* jarea   # b̃→flux conform operator Σ·√A = (Σ/√A)·A
    R_inv = inv(R)
    return (
        plasma_inductance=R_inv * plasma_inductance * R_inv',
        surface_inductance=R_inv * surface_inductance * R_inv',
        permeability=R_inv * permeability * R,
        reluctance=R' * reluctance * R
    )
end

"""
    map_forcing_to_eigenmodes(
        forcing_modes::Vector{ForcingMode},
        intr::ForceFreeStatesInternal
    )::Vector{ComplexF64}

Map external forcing modes to eigenmode basis.

Matches forcing mode numbers (n,m) to the eigenmode basis used in ForceFreeStates
and creates a forcing vector in that basis.

**Unit convention**: `ForcingMode.amplitude` is in unit-norm convention (= Fortran Phi_x,
T·m² per unit-norm cell). Files loaded in `normal_field_T` or `sfl_flux_Wb` convention
are automatically converted to unit-norm on load (see `ForcingMode` docstring).

## Arguments

  - `forcing_modes`: External forcing modes (amplitudes in unit-norm / Phi_x convention)
  - `intr`: ForceFreeStates internal state with mode arrays

## Returns

  - Forcing vector in eigenmode basis [mpert]
"""
function map_forcing_to_eigenmodes(
    forcing_modes::Vector{ForcingMode},
    intr::ForceFreeStatesInternal
)::Vector{ComplexF64}

    numpert_total = intr.mpert * intr.npert
    forcing_vector = zeros(ComplexF64, numpert_total)

    # Create mode index map: (m,n) -> linear index
    for forcing_mode in forcing_modes
        # Find matching mode in eigenmode basis
        for i in 1:numpert_total
            # Calculate m and n for this index
            # Using 0-based indexing converted to 1-based:
            # m = (i-1) % mpert + mlow
            # n = (i-1) ÷ mpert + nlow
            m_mode = (i - 1) % intr.mpert + intr.mlow
            n_mode = (i - 1) ÷ intr.mpert + intr.nlow

            if m_mode == forcing_mode.m && n_mode == forcing_mode.n
                forcing_vector[i] = forcing_mode.amplitude
                break
            end
        end
    end

    return forcing_vector
end
