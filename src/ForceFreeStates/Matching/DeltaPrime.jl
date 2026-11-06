"""
    PEST3Blocks

PEST-3 parity blocks of a raw D′ [Glasser-Kolemen 2018 PoP 25, 032501, Eq. 31], each (msing × msing).

## Fields

  - `A` - Interchange↔interchange block.
  - `B` - Interchange↔tearing block.
  - `Gamma` - Tearing↔interchange block.
  - `Delta` - Tearing↔tearing block, the tearing Δ′.
"""
struct PEST3Blocks
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    Gamma::Matrix{ComplexF64}
    Delta::Matrix{ComplexF64}
end

"""
    pest3_decompose(raw) -> PEST3Blocks

PEST-3 parity blocks of the side-major raw D′ `raw` (2msing × 2msing). No Pletzer–Dewar ½ prefactor, as in
Fortran RDCON.
"""
function pest3_decompose(raw::AbstractMatrix{ComplexF64})
    m = size(raw, 1) ÷ 2
    size(raw) == (2m, 2m) || throw(ArgumentError("pest3_decompose: raw must be 2m × 2m, got $(size(raw))"))
    LL = raw[1:2:end, 1:2:end]
    LR = raw[1:2:end, 2:2:end]
    RL = raw[2:2:end, 1:2:end]
    RR = raw[2:2:end, 2:2:end]
    return PEST3Blocks(RR + RL + LR + LL, RR - RL + LR - LL, RR + RL - LR - LL, RR - RL - LR + LL)
end

"""
    DeltaPrimeData(raw_freeboundary, raw_fixedboundary, ef, edge_response)

The solve's Δ′/outer-region matching payload. Matrices are indexed by the surfaces the formalism solved
across, core→edge; side-major orderings run `[L_s1, R_s1, L_s2, R_s2, …]`. An empty matrix (or `nothing`)
means the solve did not compute it. The PEST-3 blocks are built from the raw matrices.

## Fields

  - `raw_freeboundary::Matrix{ComplexF64}` - Raw D′ (2msing × 2msing) with the vacuum (or wall) at the edge.
  - `raw_fixedboundary::Matrix{ComplexF64}` - Raw D′ (2msing × 2msing) with ξ(edge) = 0, a conformal wall on the plasma.
  - `ef::Matrix{ComplexF64}` - Error-field response Δ_EF (2msing × numpert_total): column k is the small-solution
    coefficient at each surface side for a unit edge displacement in mode k with the edge otherwise fixed. It pairs
    with `raw_fixedboundary` in matching.
  - `edge_response::Matrix{ComplexF64}` - Edge momentum Fξ′ + Kξ (numpert_total × 2msing) of the `raw_fixedboundary`
    solutions: column k has unit big-solution coefficient on surface side k and ξ(edge) = 0.
  - `freeboundary::Union{Nothing,PEST3Blocks}` - PEST-3 blocks of `raw_freeboundary`.
  - `fixedboundary::Union{Nothing,PEST3Blocks}` - PEST-3 blocks of `raw_fixedboundary`.
"""
struct DeltaPrimeData
    raw_freeboundary::Matrix{ComplexF64}
    raw_fixedboundary::Matrix{ComplexF64}
    ef::Matrix{ComplexF64}
    edge_response::Matrix{ComplexF64}
    freeboundary::Union{Nothing,PEST3Blocks}
    fixedboundary::Union{Nothing,PEST3Blocks}
end

DeltaPrimeData(raw_freeboundary::AbstractMatrix, raw_fixedboundary::AbstractMatrix, ef::AbstractMatrix, edge_response::AbstractMatrix) =
    DeltaPrimeData(raw_freeboundary, raw_fixedboundary, ef, edge_response,
        isempty(raw_freeboundary) ? nothing : pest3_decompose(raw_freeboundary),
        isempty(raw_fixedboundary) ? nothing : pest3_decompose(raw_fixedboundary))
