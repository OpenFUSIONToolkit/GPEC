# GGJParameters.jl
#
# Physical parameters for the Glasser–Greene–Johnson inner-layer model and
# the derived scale factors that map between physical and inner-layer
# (Wasow-normalized) variables. Mirrors the Fortran `resist_type` defined
# in rmatch/deltar_mod and rmatch/deltac_mod.

"""
    GGJParameters

Dimensionless parameters of the Glasser–Greene–Johnson inner-layer model
at a single rational surface, plus the local Alfvén/resistive timescales
needed to scale the matching data back to physical Δ.

Fields are the same as the Fortran `resist_type`:

| field   | meaning                                                       |
|---------|---------------------------------------------------------------|
| `E`     | Glasser interchange parameter (enters Mercier `D_I = E+F+H−¼`) |
| `F`     | Glasser interchange parameter                                 |
| `G`     | Coupling coefficient (curvature × pressure gradient)          |
| `H`     | Pfirsch–Schlüter coefficient                                  |
| `K`     | Glasser parameter                                             |
| `M`     | Mercier-related auxiliary parameter (held but not used here)  |
| `taua`  | Local Alfvén time at the rational surface                      |
| `taur`  | Local resistive time at the rational surface                   |
| `v1`    | Linear scale factor used in the V₁ rescaling                   |
| `ising` | Index of the singular surface (traceability only)              |

The complex growth rate `γ` is **not** stored here; it is passed as a
separate argument to `solve_inner`.
"""
Base.@kwdef struct GGJParameters
    E::Float64
    F::Float64
    G::Float64
    H::Float64
    K::Float64
    M::Float64 = 0.0
    taua::Float64
    taur::Float64
    v1::Float64 = 1.0
    ising::Int = 0
end

"""
    mercier_di(p::GGJParameters) -> Float64

Mercier interchange index `D_I = E + F + H − 1/4` (deltac_run line 143).
"""
mercier_di(p::GGJParameters) = p.E + p.F + p.H - 0.25

"""
    mercier_dr(p::GGJParameters) -> Float64

Resistive interchange index `D_R = E + F + H²` (deltar_run derivation,
deltac_run line 142).
"""
mercier_dr(p::GGJParameters) = p.E + p.F + p.H * p.H

"""
    p1(p::GGJParameters) -> Float64

`p₁ = √(−D_I)`. The Mercier-stable branch requires `D_I < 0`; this
function asserts it.
"""
function p1(p::GGJParameters)
    di = mercier_di(p)
    di < 0 || throw(ArgumentError("GGJParameters: D_I = $di must be negative (Mercier-stable required for inner-layer model)"))
    return sqrt(-di)
end

"""
    sfac(p::GGJParameters) -> Float64

Lundquist-like ratio `S = τ_R / τ_A` used to define the inner-layer
length and time scales.
"""
sfac(p::GGJParameters) = p.taur / p.taua

"""
    x0(p::GGJParameters) -> Float64

Inner-layer length scale `X₀ = S^(−1/3)` (deltac_run line 149).
"""
x0(p::GGJParameters) = sfac(p)^(-1.0 / 3.0)

"""
    q0(p::GGJParameters) -> Float64

Inner-layer growth-rate scale `Q₀ = X₀ / τ_A` (deltac_run line 150).
"""
q0(p::GGJParameters) = x0(p) / p.taua

"""
    inner_Q(p::GGJParameters, γ::Number) -> ComplexF64

Dimensionless inner-layer growth rate `Q = γ / Q₀` used by the inps Wasow
basis (deltac_run line 151). The argument `γ` may be real or complex; the
result is always complex.
"""
inner_Q(p::GGJParameters, γ::Number) = ComplexF64(γ) / q0(p)

"""
    rescale_delta(Δ, p::GGJParameters) -> SVector{2,ComplexF64}

Apply the Wang 2020 `X₀^(2√(−D_I))` rescaling that maps the inner-layer
matching data back to physical Δ at the rational surface (deltac_run line
192). Operates element-wise on a 2-vector of `(Δ_odd, Δ_even)`.
"""
function rescale_delta(Δ::AbstractVector, p::GGJParameters)
    s = sfac(p)
    pp = p1(p)
    fac = s^(2.0 * pp / 3.0) * p.v1^(2.0 * pp)
    return SVector{2,ComplexF64}(Δ[1] * fac, Δ[2] * fac)
end
