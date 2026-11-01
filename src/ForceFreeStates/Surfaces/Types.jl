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
  - `delta_prime::Vector{ComplexF64}` - **Debug stub, not physically valid.** Per-surface ca-based Δ' estimate;
    use `ForceFreeStatesInternal.delta_prime_matrix` (STRIDE global BVP, Glasser 2018 PoP 25, 032501) for tearing stability.
  - `delta_prime_col::Matrix{ComplexF64}` - **Debug stub, not physically valid.** Per-surface ca-based Δ' columns, `(numpert_total, n_res_modes)`,
    `delta_prime_col[j, i] = (ca_r[j,ipert_res_i,2] - ca_l[j,ipert_res_i,2]) / (4π²·psio)`; diagonal matches `delta_prime`. Riccati/parallel paths only.
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

Frobenius asymptotic expansion at one side of a singular surface, built by `compute_sing_asymptotics`.

## Fields

  - `sing_order::Int` - Frobenius expansion order
  - `alpha::Vector{ComplexF64}` - Mercier exponent α = √(tr²/4 − det) of each resonant block.
  - `r1::Vector{Int}` - Resonant indices along first index
  - `r2::Vector{Int}` - Resonant indices along second index
  - `power::Vector{ComplexF64}` - Frobenius power of each column in |ψ − ψ_s| (0 off resonance). Resonant powers are sig·λ of that side's M₀ block, big solution (more negative real part) first.
  - `vmat::Array{ComplexF64,4}` - Power series of V matrix for asymptotic analysis
"""
struct SingAsymptotics
    sing_order::Int
    alpha::Vector{ComplexF64}
    r1::Vector{Int}
    r2::Vector{Int}
    power::Vector{ComplexF64}
    vmat::Array{ComplexF64,4}
end
