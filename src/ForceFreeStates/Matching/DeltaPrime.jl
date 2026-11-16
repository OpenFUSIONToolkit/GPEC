"""
    DeltaPrimeData

The solve's Δ′/outer-region matching payload, in the PEST-3 convention shared by the Riccati
boundary-value problem and the Galerkin solve, so consumers never branch on which integrator ran.

Matrices are indexed by the surfaces the formalism solved across, core→edge: `result.surfaces` for
Riccati, its in-domain in-band subset for Galerkin (`gal_resonant_surfaces`). Side-major orderings
run `[L_s1, R_s1, L_s2, R_s2, …]`.

## Fields

  - `matrix::Matrix{ComplexF64}` - Inter-surface Δ′ of shape (msing × msing) in PEST3
    convention, the tearing↔tearing parity projection of `raw`. Same as `Delta` of the
    PEST-3 block set. Both formalisms.
  - `raw::Matrix{ComplexF64}` - Raw outer-region matching matrix D′ of shape
    (2msing × 2msing), side-major on both axes. Both formalisms.
  - `coil::Matrix{ComplexF64}` - Edge coil-response matrix of shape
    (2msing × numpert_total); column k is the resonant small-solution response at each
    surface side to a unit source on edge poloidal mode k. Riccati fills it from the
    vacuum-edge BVP, Galerkin from its `rpec_flag` columns. Empty when neither ran.
  - `A::Union{Nothing,Matrix{ComplexF64}}` - PEST-3 interchange↔interchange block
    (msing × msing). Galerkin only; `nothing` for Riccati, which persists only `raw` and
    recovers the blocks on demand via `pest3_decompose`.
  - `B::Union{Nothing,Matrix{ComplexF64}}` - PEST-3 interchange↔tearing block; see `A`.
  - `Gamma::Union{Nothing,Matrix{ComplexF64}}` - PEST-3 tearing↔interchange block; see `A`.
"""
struct DeltaPrimeData
    matrix::Matrix{ComplexF64}
    raw::Matrix{ComplexF64}
    coil::Matrix{ComplexF64}
    A::Union{Nothing,Matrix{ComplexF64}}
    B::Union{Nothing,Matrix{ComplexF64}}
    Gamma::Union{Nothing,Matrix{ComplexF64}}
end

"""
    OuterSolutionBasis

Outer-region solutions ξ(ψ) that a `MatchProblem` combines into the matched solution, from the Riccati Δ′
boundary-value problem or the Galerkin solve. Solution `2j-1` (`2j`) carries a unit big solution left (right) of
surface `j` and none elsewhere; the rest, when present, carry no big solution and a unit displacement of one edge
mode each.

## Fields

  - `psi::Vector{Float64}`, `q::Vector{Float64}` - Radial grid, core to edge, and its safety factor.
  - `issing::Vector{Bool}` - Grid points on a singular surface, where the solutions are left zero.
  - `xi::Array{ComplexF64,3}`, `xi_deriv::Array{ComplexF64,3}` - `(mpert, ngrid, nsol)` ξ and dξ/dψ.
  - `xi_cut::Array{ComplexF64,3}` - `(mpert, ngrid, nsol)` ξ without the leading-order big and small resonant terms
    near each surface, the background the inner-layer solution is added to. Empty when not computed.
  - `cut_range::Matrix{Float64}` - `(msing, 2)` ψ bounds around each surface outside which `xi_cut` equals `xi`.
    Empty when not computed.
"""
struct OuterSolutionBasis
    psi::Vector{Float64}
    q::Vector{Float64}
    issing::Vector{Bool}
    xi::Array{ComplexF64,3}
    xi_deriv::Array{ComplexF64,3}
    xi_cut::Array{ComplexF64,3}
    cut_range::Matrix{Float64}
end
