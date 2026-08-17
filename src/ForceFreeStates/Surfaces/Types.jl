# Singular-surface data types shared across subsystems.

"""
    SingType

A mutable struct holding data related to the singular surfaces in the equilibrium.

## Fields

  - `psifac::Float64` - Normalized flux coordinate at the singular surface
  - `rho::Float64` - Radial coordinate (√ψ)
  - `m::Vector{Int}` - Poloidal mode number(s)
  - `n::Vector{Int}` - Toroidal mode number(s)
  - `q::Float64` - Safety factor (= m/n)
  - `q1::Float64` - Derivative of safety factor with respect to ψ
  - `delta_prime::Vector{ComplexF64}` - **STUB (not physically valid)**. Per-surface ca-based Δ' estimate retained for future work / debugging only. The physically valid Δ' is `ForceFreeStatesInternal.delta_prime_matrix`, computed via the STRIDE global BVP (Glasser 2018 PoP 25, 032501). Do not use this field for tearing-stability analysis; do not expect agreement with `delta_prime_matrix`.
  - `delta_prime_col::Matrix{ComplexF64}` - **STUB (not physically valid)**. Per-surface ca-based Δ' column retained for future work / debugging only. Shape (numpert_total × n_res_modes); `delta_prime_col[j, i] = (ca_r[j,ipert_res_i,2] - ca_l[j,ipert_res_i,2]) / (4π²·psio)`. The diagonal element matches the (also stubbed) `delta_prime[i]`. Only populated for the Riccati/parallel FM paths. The physically valid Δ' is `ForceFreeStatesInternal.delta_prime_matrix`; this field exists for future development on intra-surface coupling diagnostics, not for production use.
"""
@kwdef mutable struct SingType
    psifac::Float64 = 0.0
    rho::Float64 = 0.0
    m::Vector{Int} = Int[]
    n::Vector{Int} = Int[]
    q::Float64 = 0.0
    q1::Float64 = 0.0
    delta_prime::Vector{ComplexF64} = ComplexF64[]
    delta_prime_col::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, 0, 0)
    ua_left::Array{ComplexF64,3} = Array{ComplexF64}(undef, 0, 0, 0)   # asymptotic basis at left inner-layer boundary
    ua_right::Array{ComplexF64,3} = Array{ComplexF64}(undef, 0, 0, 0)  # asymptotic basis at right inner-layer boundary
    psi_ua_left::Float64 = 0.0   # ψ where ua_left was evaluated (left inner-layer boundary)
    psi_ua_right::Float64 = 0.0  # ψ where ua_right was evaluated (right inner-layer boundary)
    restype::Any = nothing       # ResistGeometry from ResistEval.jl (populated by resist_eval_all!); typed `Any` to avoid a cross-file type reference
end

"""
    SingAsymptotics

A struct containing asymptotic expansion data for ideal ForceFreeStates calculations at a singular surface.
This data is computed on-demand during singular surface crossings in `cross_ideal_singular_surf!`.

## Fields

  - `alpha::Vector{ComplexF64}` - Resonant matrix eigenvalues
  - `r1::Vector{Int}` - Resonant indices along first index
  - `r2::Vector{Int}` - Resonant indices along second index
  - `n1::Vector{Int}` - Nonresonant indices along first index
  - `n2::Vector{Int}` - Nonresonant indices along second index
  - `power::Vector{ComplexF64}` - Power series coefficients
  - `vmat::Array{ComplexF64,4}` - Power series of V matrix for asymptotic analysis
  - `mmat::Array{ComplexF64,4}` - Power series of M matrix for asymptotic analysis
  - `m0mat::Matrix{ComplexF64}` - Zeroth order M matrix projected onto resonant subspace
"""
struct SingAsymptotics
    sing_order::Int
    alpha::Vector{ComplexF64}
    r1::Vector{Int}
    r2::Vector{Int}
    n1::Vector{Int}
    n2::Vector{Int}
    power::Vector{ComplexF64}
    vmat::Array{ComplexF64,4}
    mmat::Array{ComplexF64,4}
    m0mat::Matrix{ComplexF64}
end
