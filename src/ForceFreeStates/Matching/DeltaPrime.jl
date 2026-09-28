"""
    DeltaPrimeData

The solve's Δ′/outer-region matching payload, in one formalism-independent layout. The
Riccati/STRIDE boundary-value problem and the RDCON Galerkin solve compute the same
quantities in the same PEST-3 convention — the four parity blocks are the identical ±
combination of the raw side-major matrix in both (`pest3_decompose`, Riccati/DeltaPrimeBVP.jl, and
`gal_pest3_blocks`, GalerkinSolve.jl, both porting Fortran `gal_write_pest3_data`) — so
consumers never branch on which integrator ran.

Every matrix is indexed by the singular surfaces the producing formalism actually solved
across, ordered core→edge: `result.surfaces` for Riccati, the in-domain in-band subset of
it for Galerkin (`gal_resonant_surfaces`). Side-major orderings run
`[L_s1, R_s1, L_s2, R_s2, …]`.

## Fields

  - `matrix::Matrix{ComplexF64}` - Inter-surface Δ′ of shape (msing × msing) in PEST3
    convention, the tearing↔tearing parity projection of `raw`. Same as `Delta` of the
    PEST-3 block set. Both formalisms.
  - `raw::Matrix{ComplexF64}` - Raw outer-region matching matrix D′ of shape
    (2msing × 2msing), side-major on both axes. Both formalisms.
  - `coil::Matrix{ComplexF64}` - Edge coil-response matrix of shape
    (2msing × numpert_total); column k is the resonant small-solution response at each
    surface side to a unit source on edge poloidal mode k. Riccati fills it from the
    vacuum-edge BVP, Galerkin from the `gal_rpec_flag` columns (transposed at pack time
    from the (numpert_total × 2msing) block the Galerkin solve produces). Empty when
    neither ran.
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
