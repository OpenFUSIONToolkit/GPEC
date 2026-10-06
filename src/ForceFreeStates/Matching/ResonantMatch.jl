# ResonantMatch.jl
#
# The outer<->inner resistive matching currency: the unified result type and the 4·msing
# matching system (Fortran rmatch match_rpec, match.f; equivalently Wang et al. 2020,
# PoP 27, 122509 Eq. 11: C = -(Δ_out - Δ_in(i2πf))^{-1} Δ_coil). The driver that fills a
# `MatchResult` lives in Matching/MatchProblem.jl, loaded after the result machinery.

"""
    MatchResult

Product of the driven (RPEC) outer↔inner asymptotic matching at prescribed per-surface
eigenvalues — one type for every producing formalism. The matching system itself is
basis-free (it needs only the raw Δ′ and coil blocks), so the coefficient and resonant
fields are always populated; the profile fields need the producing solve's retained outer
basis and stay EMPTY when it has none (a Riccati-fed match) or when the inner layer was
skipped (`ideal`).

## Fields

  - `cout::Matrix{ComplexF64}` - `(2·msing, ncoil)` outer-region plasma-solution coefficients.
  - `cin::Matrix{ComplexF64}` - `(2·msing, ncoil)` inner-region coefficients.
  - `deltar::Matrix{ComplexF64}` - `(msing, 2)` inner-layer matching data `(Δ₁, Δ₂)` per surface.
  - `rpec_eig::Vector{ComplexF64}` - Forced eigenvalues `γ_s = 2πi·n·f_s` per surface.
  - `residual::Float64` - Relative linear-solve residual `‖mat·cof − rmat‖/‖rmat‖`.
  - `bpen::Matrix{ComplexF64}` - `(msing, ncoil)` inner-layer penetrated (reconnected) resonant
    field at each rational surface, read off the inner solution at the layer center (X = 0)
    exactly as Fortran `match_output_solution` builds `intotsol_b` — cusp-free, fit-free.
    Zeros in the ideal branch, where the inner layer is skipped.
  - `reconnected_flux::Matrix{ComplexF64}` - `(2·msing, ncoil)` reconnected resonant flux,
    `Δ_coil + Δ_outᵀ·cout`.
  - `xi::Array{ComplexF64,3}`, `xi_deriv::Array{ComplexF64,3}` - `(mpert, ngrid, ncoil)` matched
    outer ξ(ψ) and analytic ξ′(ψ), one column per coil drive (identity-at-edge basis). Empty
    without a retained outer basis.
  - `inner_psi::Vector{Vector{Float64}}` - Per surface, the inner-layer ψ grid `ψ_s ± X·x0/v1`
    (left wing reversed then right, ψ ascending through `ψ_s`). Empty in the ideal branch.
  - `inner_xi::Vector{Matrix{ComplexF64}}` - Per surface, the composite inner-region `ξ_ψ(ψ)`
    on `inner_psi`, one column per coil drive (layer solution plus the cut outer background).
  - `inner_b::Vector{Matrix{ComplexF64}}` - Per surface, the composite inner-region `b^ψ(ψ)`.
  - `inner_params::Vector{InnerLayer.GGJParameters}` - Per-surface layer parameters the inner
    solves ran with. Empty in the ideal branch.
"""
struct MatchResult
    cout::Matrix{ComplexF64}
    cin::Matrix{ComplexF64}
    deltar::Matrix{ComplexF64}
    rpec_eig::Vector{ComplexF64}
    residual::Float64
    bpen::Matrix{ComplexF64}
    reconnected_flux::Matrix{ComplexF64}
    xi::Array{ComplexF64,3}
    xi_deriv::Array{ComplexF64,3}
    inner_psi::Vector{Vector{Float64}}
    inner_xi::Vector{Matrix{ComplexF64}}
    inner_b::Vector{Matrix{ComplexF64}}
    inner_params::Vector{InnerLayer.GGJParameters}
end

"""
    _match_system(dp_raw, dp_coil, deltar) -> (cout, cin, residual)

Assemble and solve the `4·msing` matching system `mat·[cout; cin] = rmat` coupling the
outer Δ′ blocks to the per-surface inner-layer `(Δ₁, Δ₂)` (Fortran rmatch `match_rpec`,
match.f): the outer rows carry `transpose(dp_raw)` against the coil source `−dp_coil`,
and each surface contributes the parity coupling and inner-Δ sign blocks.
"""
function _match_system(dp_raw::AbstractMatrix, dp_coil::AbstractMatrix, deltar::AbstractMatrix)
    msing = size(dp_raw, 1) ÷ 2
    ncoil = size(dp_coil, 2)
    mat = zeros(ComplexF64, 4msing, 4msing)
    rmat = zeros(ComplexF64, 4msing, ncoil)
    @views mat[(2msing+1):4msing, 1:2msing] .= transpose(dp_raw)   # Δ_out
    @views rmat[(2msing+1):4msing, :] .= .-dp_coil                 # −Δ_coil source (already surface-side × edge mode)
    for ising in 1:msing
        idx1 = 2ising - 1
        idx2 = 2ising
        idx3 = idx1 + 2msing
        idx4 = idx2 + 2msing
        delta1 = deltar[ising, 1]
        delta2 = deltar[ising, 2]
        mat[idx1, idx1] = 1
        mat[idx2, idx2] = 1
        mat[idx1, idx3] = -1
        mat[idx1, idx4] = 1
        mat[idx2, idx3] = -1
        mat[idx2, idx4] = -1
        # inner-layer Δ block signs per match.f match_rpec
        mat[idx3, idx3] = -delta1
        mat[idx3, idx4] = delta2
        mat[idx4, idx3] = -delta1
        mat[idx4, idx4] = -delta2
    end

    cof = mat \ rmat
    residual = norm(mat * cof - rmat) / max(norm(rmat), 1e-300)
    return cof[1:2msing, :], cof[(2msing+1):4msing, :], residual
end
