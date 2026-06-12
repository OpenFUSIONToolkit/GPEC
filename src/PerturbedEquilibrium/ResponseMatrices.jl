"""
Response matrix construction for perturbed equilibrium calculations.

Based on gpresp.f from GPEC, implementing resp_index=0 (energy-based inductance).
Uses ForceFreeStates eigenmode solutions and vacuum response data.

Reference: [Park Phys. Plasmas 2009 056115]
"""

# Use FourierTransform utility instead of FFTW for theta ↔ mode transforms
using ..Utilities.FourierTransforms

"""
    extract_boundary_displacements(
        equil::Equilibrium.PlasmaEquilibrium,
        ForceFreeStates_results::OdeState,
        intr::ForceFreeStatesInternal
    )::NamedTuple

Extract eigenmode displacements and equilibrium quantities at the plasma boundary.

This function extracts the data needed to compute the normal magnetic field at the
plasma surface from ForceFreeStates eigenmode solutions.

## What's extracted:

1. **Boundary displacement**: ξ_ψ from `u_store[:, :, 1, end]`
   - This is the radial (normal) component of the eigenmode displacement
   - At the last radial integration point (plasma edge)
   - Dimensions: [numpert_total, numpert_total]

2. **Flux surface spacing**: dΨ/dρ at boundary
   - From equilibrium bicubic spline evaluation
   - Needed to convert displacement to magnetic field

3. **Safety factor**: q at boundary
   - Used to compute singular factors (m - n*q)
   - Identifies resonant surfaces

## Arguments
- `equil`: Equilibrium solution containing flux surfaces and q-profile
- `ForceFreeStates_results`: ODE integration results containing u_store with eigenmodes
- `intr`: ForceFreeStates internal state with boundary location (psilim)

## Returns
Named tuple with:
- `ξ_psi_boundary`: Boundary displacement [numpert_total, numpert_total]
- `dPsi_drho`: Flux surface spacing at boundary (scalar)
- `q_boundary`: Safety factor at boundary (scalar)
- `psi_boundary`: Normalized flux at boundary (scalar)
"""
function extract_boundary_displacements(
    equil::Equilibrium.PlasmaEquilibrium,
    ForceFreeStates_results::OdeState,
    intr::ForceFreeStatesInternal
)
    # Extract boundary displacement (normal component)
    # u_store dimensions: [numpert_total, numpert_total, 2, numsteps]
    # Index 1 in 3rd dimension is ξ_ψ (radial displacement)
    # Last index in 4th dimension is the boundary
    ξ_psi_boundary = ForceFreeStates_results.u_store[:, :, 1, ForceFreeStates_results.step]

    # Get boundary location in normalized flux coordinates
    psi_boundary = ForceFreeStates_results.psi_store[ForceFreeStates_results.step]

    # Evaluate equilibrium quantities at boundary
    # Safety factor at boundary
    q_boundary = ForceFreeStates_results.q_store[ForceFreeStates_results.step]

    # FFS ODE integrates in ψ (normalized flux), so bwp_mn = chi1·singfac·2πi·ξ_ψ
    # where chi1 = 2π·psio  (Fortran idcon.f: chi1 = twopi*psio)
    # Combined flux factor = chi1·2π = (2π)²·psio  (gpeq.f: bwp_mn = chi1·singfac·twopi·ifac·xsp)
    dPsi_drho = (2π)^2 * equil.psio

    return (
        ξ_psi_boundary = ξ_psi_boundary,
        dPsi_drho = dPsi_drho,
        q_boundary = q_boundary,
        psi_boundary = psi_boundary
    )
end

"""
    compute_normal_magnetic_field(
        boundary_data::NamedTuple,
        intr::ForceFreeStatesInternal
    )::Matrix{ComplexF64}

Compute normal magnetic field at plasma boundary from eigenmode displacements.

This is the key step that converts eigenmode displacements to magnetic field perturbations
at the plasma surface. From the ideal MHD constraint [Park Phys. Plasmas 2009 056115 eq. 4]:

    B_n = i * (dΨ/dρ) * (m - n*q) * ξ_ψ

where ξ_ψ is the radial displacement eigenfunction.

## Physical Interpretation [Park Phys. Plasmas 2007 052110 Section II]:
- ξ_ψ[i,j]: Displacement of mode i due to eigenmode j
- singfac[i] = m[i] - n*q: Singular factor measuring distance from rational surface
- dΨ/dρ: Converts displacement to flux perturbation (poloidal flux gradient)
- Factor of i: Phase relationship for oscillating fields in complex representation

## Arguments
- `boundary_data`: Output from extract_boundary_displacements()
  - ξ_psi_boundary: Boundary displacement [numpert_total, numpert_total]
  - dPsi_drho: Flux surface spacing at boundary (scalar)
  - q_boundary: Safety factor at boundary (scalar)
  - psi_boundary: Normalized flux at boundary (scalar)
- `intr`: ForceFreeStates internal state with mode arrays (mlow, mhigh, nlow, etc.)

## Returns
- `bwp_mn[numpert_total, numpert_total]`: Normal magnetic field matrix where bwp_mn[i,j]
  is the normal field of Fourier mode i in response to eigenmode j
"""
function compute_normal_magnetic_field(
    boundary_data::NamedTuple,
    intr::ForceFreeStatesInternal
)::Matrix{ComplexF64}

    numpert_total = intr.numpert_total
    bwp_mn = zeros(ComplexF64, numpert_total, numpert_total)

    # Extract boundary data
    ξ_psi = boundary_data.ξ_psi_boundary
    dPsi_drho = boundary_data.dPsi_drho
    q_boundary = boundary_data.q_boundary

    # Compute singular factor for each Fourier mode [Park Phys. Plasmas 2009 056115 eq. 4]
    # singfac[i] = m[i] - n*q_boundary measures distance from rational surface
    # Mode indexing: modes are ordered as (m, n) pairs
    # Linear index i corresponds to: m = (i-1) % mpert + mlow, n = (i-1) ÷ mpert + nlow
    singfac = zeros(Float64, numpert_total)
    for i in 1:numpert_total
        m_mode = (i - 1) % intr.mpert + intr.mlow
        n_mode = (i - 1) ÷ intr.mpert + intr.nlow
        singfac[i] = m_mode - n_mode * q_boundary
    end

    # Compute normal magnetic field [Park Phys. Plasmas 2009 056115 eq. 4]
    # bwp_mn[i,j] = i * (dΨ/dρ) * singfac[i] * ξ_ψ[i,j]
    for i in 1:numpert_total
        for j in 1:numpert_total
            bwp_mn[i, j] = 1im * dPsi_drho * singfac[i] * ξ_psi[i, j]
        end
    end

    return bwp_mn
end

"""
    build_flux_matrix(
        equil::Equilibrium.PlasmaEquilibrium,
        ForceFreeStates_results::OdeState,
        vac_data::VacuumData,
        intr::ForceFreeStatesInternal
    )::Matrix{ComplexF64}

Build vacuum poloidal flux matrix from ForceFreeStates eigenmode solutions.

This extracts the vacuum flux response for each eigenmode at the plasma boundary.
In GPEC, this comes from `bwp_mn` (boundary normal field) computed from eigenmode
displacements.

The flux matrix relates eigenmode displacements to vacuum poloidal flux:
1. Extract eigenmode displacement at plasma boundary from u_store
2. Compute normal magnetic field: B_ψ = i×(dΨ/dρ)×(m - n×q)×ξ_ψ
3. Result is flux[mode_i, eigenmode_j] = bwp_mn[i,j]

## Arguments
- `equil`: Equilibrium solution containing flux surfaces and q-profile
- `ForceFreeStates_results`: ForceFreeStates ODE integration results containing eigenmodes
- `vac_data`: Vacuum response data from free boundary calculation
- `intr`: ForceFreeStates internal state with mode information

## Returns
- `flxmats[numpert_total, numpert_total]`: Complex flux matrix where flxmats[i,j] is the
  vacuum flux of mode i in response to eigenmode j
"""
function build_flux_matrix(
    equil::Equilibrium.PlasmaEquilibrium,
    ForceFreeStates_results::OdeState,
    vac_data::VacuumData,
    intr::ForceFreeStatesInternal
)::Matrix{ComplexF64}

    # Step 1: Extract boundary displacements and equilibrium quantities
    boundary_data = extract_boundary_displacements(equil, ForceFreeStates_results, intr)

    # Step 2: Compute normal magnetic field at plasma boundary
    # This is the actual implementation of GPEC's bwp_mn calculation
    # bwp_mn[i,j] = i * (dΨ/dρ) * (m[i] - n*q_boundary) * ξ_ψ[i,j]
    flxmats = compute_normal_magnetic_field(boundary_data, intr)

    return flxmats
end

"""
    calc_plasma_inductance(
        vac_data::VacuumData,
        ffs_intr::ForceFreeStatesInternal,
        psio::Float64
    )::Matrix{ComplexF64}

Calculate plasma inductance matrix Λ using the wt0-based energy formula
(matches Fortran `gpresp_induct` with `resp_induct_flag=TRUE`).

    Λ = inv(2·t₁·wt0·t₂)

where t₁ = im/(χ₁·s_i·2π), t₂ = -im/(χ₁·s_j·2π), s_i = m_i - n·q_lim.

Note: `vac_data.wt0` already contains singfac² factors (s_i·s_j) baked into the
vacuum term via the scaling in `free_run!`. The t₁/t₂ factors divide by s_i·s_j,
correctly recovering the properly-normalized inductance.

## Arguments
- `vac_data`: Vacuum data containing wt0 (total energy matrix before eigenvector sorting)
- `ffs_intr`: ForceFreeStates internal state with mode info (mlow, mpert, nlow, qlim)
- `psio`: Total toroidal flux [Wb/rad] from equilibrium (equil.psio)

## Returns
- Plasma inductance matrix Lambda [numpert_total × numpert_total]
"""
function calc_plasma_inductance(
    vac_data::VacuumData,
    ffs_intr::ForceFreeStatesInternal,
    psio::Float64
)::Matrix{ComplexF64}

    mpert = ffs_intr.numpert_total
    chi1  = 2π * psio          # = Fortran's chi1 = twopi*psio
    n     = ffs_intr.nlow
    qlim  = ffs_intr.qlim      # q at psilim

    # Singular factors s_i = m_i - n*qlim  (same as Fortran: mfac(i) - nn*qlim)
    s = [((i-1) % ffs_intr.mpert + ffs_intr.mlow) - n * qlim for i in 1:mpert]

    # Fortran idcon_norm: wt0 = wt0/(mu0*2)*psio^2
    # Julia's vac_data.wt0 is raw wp+wv; Fortran additionally scales by psio^2/(mu0*2)
    mu0 = 4π * 1e-7
    wt0_norm = vac_data.wt0 .* (psio^2 / (mu0 * 2))

    # Build temp2[i,j] = 2·t1_i·wt0[i,j]·t2_j (matches Fortran gpresp_induct)
    temp2 = Matrix{ComplexF64}(undef, mpert, mpert)
    for i in 1:mpert, j in 1:mpert
        t1 = im / (chi1 * s[i] * 2π)
        t2 = -im / (chi1 * s[j] * 2π)
        temp2[i,j] = 2 * t1 * wt0_norm[i,j] * t2
    end

    return inv(temp2)
end

"""
    pack_complex_grouped!(packed, modes)

Pack complex mode coefficients into grouped real/imaginary format for Green's function application.

Converts [a+bi, c+di, ...] to [a, c, ..., b, d, ...] (real block first, then imaginary block).

This matches the column layout of Julia's Vacuum grri/grre matrices, which store
cos-response columns (1:mpert) followed by sin-response columns (mpert+1:2mpert).
[Vacuum.jl: fourier_transform!(grre, cos_mn_basis) then fourier_transform!(grre, sin_mn_basis; col_offset=mpert)]

## Arguments
- `packed`: Output array [2*mpert]
- `modes`: Complex Fourier mode coefficients [mpert]
"""
function pack_complex_grouped!(packed::AbstractVector{Float64}, modes::AbstractVector{ComplexF64})::AbstractVector{Float64}
    mpert = length(modes)
    for i in 1:mpert
        packed[i]       = real(modes[i])
        packed[i+mpert] = imag(modes[i])
    end
    return packed
end

"""
    calc_permeability(
        plasma_inductance::Matrix{ComplexF64},
        surface_inductance::Matrix{ComplexF64}
    )::Matrix{ComplexF64}

Calculate permeability matrix P = Λ·L⁻¹ (matches Fortran `gpresp_permeab`).

## Arguments
- `plasma_inductance`: Plasma inductance matrix Lambda
- `surface_inductance`: Surface inductance matrix L

## Returns
- Permeability matrix P = Lambda * L^{-1} [mpert, mpert]
"""
function calc_permeability(
    plasma_inductance::Matrix{ComplexF64},
    surface_inductance::Matrix{ComplexF64}
)::Matrix{ComplexF64}
    # P = Lambda * L^{-1}  (right-division solves for P s.t. P*L = Lambda)
    return plasma_inductance / surface_inductance
end

"""
    build_control_surface_ptof(
        equil::Equilibrium.PlasmaEquilibrium,
        ffs_intr::ForceFreeStatesInternal
    )::Matrix{ComplexF64}

Build the numpert_total × numpert_total root-area-weighted field → flux operator `ptof` at the
control surface (psilim). The mpert × mpert single-n block (Equilibrium.control_surface_ptof)
is repeated block-diagonally over the `npert` toroidal harmonics, matching the
numpert_total mode ordering used by the response matrices.

`ptof` maps a root-area-weighted control-surface field `b̃` to the coordinate flux harmonics
`Φ`: `Φ = ptof · b̃`. It is the operator users need to recover the flux-space matrices from
the stored coordinate-invariant (field-space) ones — see `field_space_response_matrices`.
[Pharr 2026]
"""
function build_control_surface_ptof(
    equil::Equilibrium.PlasmaEquilibrium,
    ffs_intr::ForceFreeStatesInternal
)::Matrix{ComplexF64}
    mpert = ffs_intr.mpert
    npert = ffs_intr.npert
    Npert = ffs_intr.numpert_total

    mtheta_eq = length(equil.rzphi_ys)
    ft = Utilities.FourierTransforms.FourierTransform(mtheta_eq, mpert, ffs_intr.mlow)
    ptof_block = Equilibrium.control_surface_ptof(equil, ffs_intr.psilim, ft)

    npert == 1 && return Matrix{ComplexF64}(ptof_block)

    ptof_full = zeros(ComplexF64, Npert, Npert)
    for in in 1:npert
        r = ((in - 1) * mpert + 1):(in * mpert)
        ptof_full[r, r] .= ptof_block
    end
    return ptof_full
end

"""
    field_space_response_matrices(
        plasma_inductance, surface_inductance, permeability, reluctance, ptof
    )::NamedTuple

Express the control-surface response matrices in the coordinate-invariant root-area-weighted
field (b̃) space, given the flux-space matrices and the `ptof` operator (`Φ = ptof·b̃`).

The matrices fall into two algebraic classes:
  - **Operators** (map flux → flux): permeability `P` (Φ_tot = P·Φ_x) transforms by similarity
    `P̃ = ptof⁻¹·P·ptof`. Its singular values are coordinate-invariant.
  - **Quadratic generators** (energy = Φ†·G⁻¹·Φ): inductances `Λ`, `L` transform by congruence
    `G̃ = ptof⁻¹·G·ptof⁻†`; the inverse-inductance-like reluctance `ϱ` (energy = Φ†·ϱ·Φ)
    transforms as `ϱ̃ = ptof†·ϱ·ptof`. Their spectra are coordinate-invariant.

These rules are mutually consistent: `P̃ = Λ̃·L̃⁻¹ = ptof⁻¹·Λ·L⁻¹·ptof = ptof⁻¹·P·ptof`, and
`ϱ̃ = L̃⁻¹·(Λ̃−L̃)·L̃⁻¹`. To recover the flux-space forms, invert each map with the stored
`ptof` (e.g. `P_flux = ptof·P̃·ptof⁻¹`, `L_flux = ptof·L̃·ptof†`). [Pharr 2026]
"""
function field_space_response_matrices(
    plasma_inductance::Matrix{ComplexF64},
    surface_inductance::Matrix{ComplexF64},
    permeability::Matrix{ComplexF64},
    reluctance::Matrix{ComplexF64},
    ptof::Matrix{ComplexF64}
)::NamedTuple
    ptof_inv = inv(ptof)
    return (
        plasma_inductance=ptof_inv * plasma_inductance * ptof_inv',
        surface_inductance=ptof_inv * surface_inductance * ptof_inv',
        permeability=ptof_inv * permeability * ptof,
        reluctance=ptof' * reluctance * ptof
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

"""
    compute_plasma_response_vector(
        permeability::Matrix{ComplexF64},
        forcing_vector::Vector{ComplexF64}
    )::Vector{ComplexF64}

Compute plasma response to external forcing.

Response = Permeability * Forcing

## Arguments
- `permeability`: Permeability matrix
- `forcing_vector`: External forcing in eigenmode basis

## Returns
- Plasma response vector in eigenmode basis
"""
function compute_plasma_response_vector(
    permeability::Matrix{ComplexF64},
    forcing_vector::Vector{ComplexF64}
)::Vector{ComplexF64}

    return permeability * forcing_vector
end
