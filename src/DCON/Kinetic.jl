"""
    compute_tpsi_matrices(psifac, n, ell, equil, ctrl, intr; is_electron, particle_type)

Placeholder for kinetic matrix calculation via drift-kinetic theory.
Returns (mpert, mpert, 6) arrays for energy (kwmat) and torque (ktmat).

# Arguments
- `psifac::Float64`: Normalized poloidal flux ψ
- `n::Int`: Toroidal mode number
- `ell::Int`: Parallel wave number index
- `equil::Equilibrium.PlasmaEquilibrium`: Equilibrium data
- `ctrl::DconControl`: Control parameters with kinetic flags
- `intr::DconInternal`: Mode numbers (mpert, etc.)
- `is_electron::Bool`: True for electrons, false for ions
- `particle_type::String`: "wmm" for energy, "tmm" for torque

# Returns
- `(kwmat, ktmat)`: Tuple of (mpert, mpert, 6) complex arrays

# Matrix Components (3rd dimension)
Component i adds to ideal MHD matrix: 1=A, 2=B, 3=C, 4=D, 5=E, 6=H

# Future Implementation
Will call PENTRC routines for velocity-space integrals over trapped/passing particles.
Currently returns zeros for infrastructure testing.
"""
function compute_tpsi_matrices(
    psifac::Float64,
    n::Int,
    ell::Int,
    equil::Equilibrium.PlasmaEquilibrium,
    ctrl::DconControl,
    intr::DconInternal;
    is_electron::Bool,
    particle_type::String
)
    # Placeholder: return zeros of correct shape
    kwmat = zeros(ComplexF64, intr.mpert, intr.mpert, 6)
    ktmat = zeros(ComplexF64, intr.mpert, intr.mpert, 6)

    # TODO: Call PENTRC interface here for real drift-kinetic physics
    # This will involve velocity-space integrals over distribution function

    return kwmat, ktmat
end
