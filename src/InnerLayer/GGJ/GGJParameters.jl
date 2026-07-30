# GGJParameters.jl
#
# Physical parameters for the Glasser–Greene–Johnson inner-layer model and
# the derived scale factors that map between physical and inner-layer
# (Wasow-normalized) variables. Mirrors the Fortran `resist_type` defined
# in rmatch/deltar_mod and rmatch/deltac_mod.

"""
    GGJParameters

Glasser–Greene–Johnson inner-layer parameters at one rational surface: the
flux-surface-averaged equilibrium coefficients of GWP2016 Eq. (A8) plus the
local timescales that scale the matching data back to physical Δ. Same
fields as the Fortran `resist_type`:

| field   | meaning                                                        |
|:------- |:-------------------------------------------------------------- |
| `E`     | Glasser interchange parameter (enters Mercier `D_I = E+F+H−¼`) |
| `F`     | Glasser interchange parameter                                  |
| `G`     | Coupling coefficient (curvature × pressure gradient)           |
| `H`     | Pfirsch–Schlüter coefficient                                   |
| `K`     | Glasser parameter                                              |
| `M`     | Mercier-related auxiliary parameter (held but not used here)   |
| `taua`  | Local Alfvén time at the rational surface                      |
| `taur`  | Local resistive time at the rational surface                   |
| `v1`    | Linear scale factor used in the V₁ rescaling                   |
| `ising` | Index of the singular surface (traceability only)              |

The growth rate `γ` is not stored here; it is a separate argument to `solve_inner`.
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

Mercier interchange index `D_I = E + F + H − 1/4` (GWP2016 Eq. A9); `D_I > 0` means local ideal interchange instability.
"""
mercier_di(p::GGJParameters) = p.E + p.F + p.H - 0.25

"""
    mercier_dr(p::GGJParameters) -> Float64

Resistive interchange index `D_R = E + F + H²` (GWP2016 Eq. A10); `D_R > 0` means local resistive interchange instability.
"""
mercier_dr(p::GGJParameters) = p.E + p.F + p.H * p.H

"""
    p1(p::GGJParameters) -> Float64

`p₁ = √(−D_I)`, setting the large-x Frobenius exponents `r± = 3/2 ± p₁`
(GW2020 Eq. 49). Throws unless `D_I < 0` (Mercier-stable required).
"""
function p1(p::GGJParameters)
    di = mercier_di(p)
    di < 0 || throw(ArgumentError("GGJParameters: D_I = $di must be negative (Mercier-stable required for inner-layer model)"))
    return sqrt(-di)
end

"""
    sfac(p::GGJParameters) -> Float64

Lundquist number `S = τ_R / τ_A`.
"""
sfac(p::GGJParameters) = p.taur / p.taua

"""
    x0(p::GGJParameters) -> Float64

Inner-layer length scale `X₀ = S^(−1/3)` (GWP2016 Eq. A14); physical `x = X₀ X`.
"""
x0(p::GGJParameters) = sfac(p)^(-1.0 / 3.0)

"""
    q0(p::GGJParameters) -> Float64

Growth-rate scale `Q₀ = X₀ / τ_A` (GWP2016 Eq. A15); physical rate `= Q₀ Q`.
"""
q0(p::GGJParameters) = x0(p) / p.taua

"""
    inner_Q(p::GGJParameters, γ::Number) -> ComplexF64

Scaled inner-layer growth rate `Q = γ / Q₀` (GWP2016 Eq. A15).
"""
inner_Q(p::GGJParameters, γ::Number) = ComplexF64(γ) / q0(p)

"""
    rescale_delta(Δ, p::GGJParameters) -> SVector{2,ComplexF64}

Rescale the matching data to physical Δ by `S^(2√(−D_I)/3) · v₁^(2√(−D_I))`
(GWP2016 Sec. IV), element-wise on `(Δ_odd, Δ_even)`.
"""
function rescale_delta(Δ::AbstractVector, p::GGJParameters)
    s = sfac(p)
    pp = p1(p)
    fac = s^(2.0 * pp / 3.0) * p.v1^(2.0 * pp)
    return SVector{2,ComplexF64}(Δ[1] * fac, Δ[2] * fac)
end

# Profile conversions shared by the solve_inner_profile backends: δψ per unit inner
# coordinate X = v₁·δψ/X₀, and the big-branch (μ₋ = −1/2−p₁) amplitude rescale to the
# outer δψ-normalization, resc = (v₁/X₀)^(−μ₋) — the companion of rescale_delta's
# (v₁/X₀)^(2p₁) = (v₁/X₀)^(μ₊−μ₋).
_profile_conversions(p::GGJParameters) = (; dψdx=x0(p) / p.v1, rescale=(p.v1 / x0(p))^(0.5 + p1(p)))
