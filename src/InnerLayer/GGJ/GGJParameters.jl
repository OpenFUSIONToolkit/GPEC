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
needed to scale the matching data back to physical Δ. The equilibrium
coefficients `E, F, G, H, K, M` are the flux-surface averages defined in
GWP2016 Eq. (A8); they enter the inner-region equations (Eq. 11).

Fields are the same as the Fortran `resist_type`:

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

Mercier interchange index `D_I = E + F + H − 1/4` (GWP2016 Eq. A9; also the
quantity under the root in the GW2020 Eq. 49 power exponents). `D_I > 0`
indicates local ideal interchange instability.
"""
mercier_di(p::GGJParameters) = p.E + p.F + p.H - 0.25

"""
    mercier_dr(p::GGJParameters) -> Float64

Resistive interchange index `D_R = E + F + H² = D_I + (H − 1/2)²`
(GWP2016 Eq. A10). `D_R > 0` indicates local resistive interchange instability.
"""
mercier_dr(p::GGJParameters) = p.E + p.F + p.H * p.H

"""
    p1(p::GGJParameters) -> Float64

`p₁ = √(−D_I)`, the Mercier power that sets the large-x Frobenius exponents
`r± = 3/2 ± √(−D_I)` (GW2020 Eq. 49; μ = −1/2 ± √(−D_I) in GWP2016 Eq. 26).
The Mercier-stable branch requires `D_I < 0`; this function asserts it.
"""
function p1(p::GGJParameters)
    di = mercier_di(p)
    di < 0 || throw(ArgumentError("GGJParameters: D_I = $di must be negative (Mercier-stable required for inner-layer model)"))
    return sqrt(-di)
end

"""
    sfac(p::GGJParameters) -> Float64

Lundquist number `S = τ_R / τ_A` (GWP2016 Sec. I) that sets the inner-layer
length and time scales `X₀ ∝ S^(−1/3)`, `Q₀ ∝ S^(−1/3)/τ_A`.
"""
sfac(p::GGJParameters) = p.taur / p.taua

"""
    x0(p::GGJParameters) -> Float64

Inner-layer displacement scale factor `X₀ = S^(−1/3)` (GWP2016 Eq. A14),
relating scaled `X` to physical `x = ψ − ψ₀` via `x = X₀ X`.
"""
x0(p::GGJParameters) = sfac(p)^(-1.0 / 3.0)

"""
    q0(p::GGJParameters) -> Float64

Inner-layer growth-rate scale factor `Q₀ = X₀ / τ_A` (GWP2016 Eq. A15),
relating the scaled growth rate `Q` to the physical rate `s` via `s = Q₀ Q`.
"""
q0(p::GGJParameters) = x0(p) / p.taua

"""
    inner_Q(p::GGJParameters, γ::Number) -> ComplexF64

Dimensionless scaled inner-layer growth rate `Q = γ / Q₀` (GWP2016 Eq. A15),
the eigenvalue appearing in the inner-region equations (Eq. 11) and the inps
Wasow basis. The argument `γ` may be real or complex; the result is always complex.
"""
inner_Q(p::GGJParameters, γ::Number) = ComplexF64(γ) / q0(p)

"""
    rescale_delta(Δ, p::GGJParameters) -> SVector{2,ComplexF64}

Map the scaled inner-layer matching data back to physical Δ at the rational
surface by the `X₀^(−2√(−D_I)) = S^(2√(−D_I)/3)` rescaling, together with the
`v₁^(2√(−D_I))` linear-scale factor, implied by the power-like matching
(GWP2016 Sec. IV; the inner solution ~ X^{r±} converts to physical x = X₀ X, so
the large/small amplitude ratio Δ scales by X₀^{−(r₊−r₋)} = X₀^(−2√(−D_I))).
Operates element-wise on a 2-vector of `(Δ_odd, Δ_even)`.
"""
function rescale_delta(Δ::AbstractVector, p::GGJParameters)
    s = sfac(p)
    pp = p1(p)
    fac = s^(2.0 * pp / 3.0) * p.v1^(2.0 * pp)
    return SVector{2,ComplexF64}(Δ[1] * fac, Δ[2] * fac)
end
