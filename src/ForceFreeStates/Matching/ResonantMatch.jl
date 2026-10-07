# ResonantMatch.jl
#
# Matching result and the 4·msing outer↔inner system (rmatch match_rpec; Wang et al. 2020 PoP 27 122509, Eq. 11).

"""
    MatchResult

Matching coefficients and matched profiles, found in `result.galerkin.match`. The profile
fields are empty when the solve kept no Galerkin basis; the inner fields are empty with `ideal`.

## Fields
  - `cout`, `cin`: `(2msing, ncoil)` outer and inner coefficients.
  - `deltar`: `(msing, 2)` inner-layer Δ per surface and parity.
  - `rpec_eig`: forced eigenvalue γ = 2πi·n·f per surface.
  - `residual`: relative residual of the linear solve.
  - `bpen`: `(msing, ncoil)` penetrated resonant field at each layer center.
  - `reconnected_flux`: `(2msing, ncoil)` `Δ_coil + Δ_outᵀ·cout`.
  - `xi`, `xi_deriv`: `(mpert, ngrid, ncoil)` matched outer ξ and ξ′, one column per coil.
  - `inner_psi`, `inner_xi`, `inner_b`: per surface, the inner ψ grid and the composite ξ_ψ and b^ψ on it.
  - `inner_params`: per-surface `GGJParameters` used.
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

# Solve the 4·msing system mat·[cout; cin] = rmat (rmatch match_rpec).
function _match_system(dp_raw::AbstractMatrix, dp_coil::AbstractMatrix, deltar::AbstractMatrix)
    msing = size(dp_raw, 1) ÷ 2
    ncoil = size(dp_coil, 2)
    mat = zeros(ComplexF64, 4msing, 4msing)
    rmat = zeros(ComplexF64, 4msing, ncoil)
    @views mat[(2msing+1):4msing, 1:2msing] .= transpose(dp_raw)   # Δ_out
    @views rmat[(2msing+1):4msing, :] .= .-dp_coil                 # −Δ_coil
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
        # inner-layer Δ block (match.f signs)
        mat[idx3, idx3] = -delta1
        mat[idx3, idx4] = delta2
        mat[idx4, idx3] = -delta1
        mat[idx4, idx4] = -delta2
    end

    cof = mat \ rmat
    residual = norm(mat * cof - rmat) / max(norm(rmat), 1e-300)
    return cof[1:2msing, :], cof[(2msing+1):4msing, :], residual
end
