# SystemSpec.jl
#
# Model-agnostic specification of an inner-layer system for the shared
# Wasow–Galerkin engine. A `SystemSpec` is the seam between a physics model
# (single-fluid GGJ today, higher-order toroidal two-fluid drift-MHD later) and
# the generic FEM + Wasow-asymptotic machinery. The engine reads sizes and calls
# the supplied callbacks; it contains no model-specific physics or dimensions.
#
# Conventions assumed by the engine:
#   - The first-order asymptotic system is `u' = (1/x)(A₀ + A₁/x + … )u`, of size
#     `n`. After the eigenbasis rotation `J_i = T⁻¹ A_i T`, the system splits into
#     a leading **algebraic** block (indices `alg_idx`, dimension `n_alg`) whose
#     A₀-eigenvalues cluster at zero / power-like, and an **exponential** block
#     (indices `exp_idx`). The engine requires the algebraic indices to come
#     first and be contiguous (`alg_idx = 1:n_alg`).
#   - The FEM solves a second-order system in `ncomp` physical components,
#     `I·u'' + V·u' + U·u = 0`, discretized with Hermite elements of order `np`.

"""
    SystemSpec

Bundle of sizes and callbacks describing one inner-layer system to the shared
Wasow–Galerkin engine. Parameterized on the callback types so the build path
stays concrete.

# Sizes

  - `n`       — first-order asymptotic system size (GGJ: 6).
  - `ncomp`   — physical second-order components for the FEM (GGJ: 3).
  - `np`      — Hermite element order (GGJ: 3, cubic → 4 DOFs/node).
  - `nterms`  — number of asymptotic coefficient matrices `A₀ … A_{nterms-1}`
    (GGJ: 3).
  - `n_alg`   — algebraic-subspace dimension (GGJ: 2).
  - `alg_idx`, `exp_idx` — index partition of the `n` rotated components into the
    algebraic and exponential blocks (GGJ: `[1,2]`, `[3,4,5,6]`).

# Callbacks

  - `asymptotic_matrices(params, Q) -> NTuple/Vector{Matrix}` of `nterms` `n×n`
    matrices `(A₀, A₁, …)`. Replaces the hardcoded GGJ `A0/A1/A2`.
  - `eigenbasis(params, Q) -> (T, Tinv)`, `n×n`, block-diagonalizing `A₀` into the
    algebraic/exponential split (algebraic first).
  - `exponents(params, Q) -> Vector` of length `n_alg`, the Frobenius exponents at
    infinity (GGJ: Mercier roots `3/2 ± √(−D_I)`).
  - `coeffs(params, Q, x) -> (I, U, V)`, `ncomp×ncomp`, the FEM coefficient
    matrices. Replaces the GGJ `_physical_uv`.
  - `physical(U, dU, x) -> (ua, dua)`, each `ncomp×n_alg`, mapping the first-order
    asymptotic state and its derivative to the physical FEM variables and their
    derivatives. Replaces the GGJ `_physical_ua_dua`.
  - `rescale(Δ, params) -> AbstractVector` applied to the raw matching data
    (GGJ: `rescale_delta`).

# Boundary / domain

  - `bc::ParityBC` — parity selectors for the half-domain solve.
  - `domain::Symbol` — `:half` (parity decomposition) or `:full`
    (`[-xmax, +xmax]`, parity-coupled).
  - `component_parity::Vector{Int}` — per-component sign `σ_c ∈ {+1, -1}` under
    `x → -x`, used by the full-domain path to continue the asymptotic basis from
    the right end to the left: `u_c(-x) = σ_c u_c(|x|)`, `u'_c(-x) = -σ_c u'_c(|x|)`.
    For a parity-symmetric model it is derivable from `bc` (a `NEUMANN` component
    is even → `+1`, a `DIRICHLET` component is odd → `-1`); a parity-broken model
    supplies its own (its negative-side asymptotics are then genuinely
    independent — a C_LAYER Phase-4 input).
"""
struct SystemSpec{FA,FE,FX,FC,FP,FR}
    n::Int
    ncomp::Int
    np::Int
    nterms::Int
    n_alg::Int
    alg_idx::Vector{Int}
    exp_idx::Vector{Int}
    asymptotic_matrices::FA
    eigenbasis::FE
    exponents::FX
    coeffs::FC
    physical::FP
    rescale::FR
    bc::ParityBC
    domain::Symbol
    component_parity::Vector{Int}
end

"""
    component_parity_from_bc(bc::ParityBC) -> Vector{Int}

Derive the per-component parity `σ_c` from a parity-symmetric `bc` using its first
(odd) sector: a `NEUMANN` component is even (`σ = +1`), a `DIRICHLET` component is
odd (`σ = -1`). (The second sector gives `-σ`, the same parity operation.)
"""
component_parity_from_bc(bc::ParityBC) = Int[s == NEUMANN ? 1 : -1 for s in bc.parities[1]]

"""
    SystemSpec(; n, ncomp, np, nterms, n_alg, alg_idx, exp_idx,
               asymptotic_matrices, eigenbasis, exponents, coeffs, physical,
               rescale, bc, domain=:half, component_parity=nothing)

Keyword constructor for [`SystemSpec`](@ref). Validates the algebraic/exponential
partition: `alg_idx` must be `1:n_alg`, and `alg_idx ∪ exp_idx` must be `1:n`. If
`component_parity` is `nothing` it is derived from `bc` via
[`component_parity_from_bc`](@ref).
"""
function SystemSpec(; n::Int, ncomp::Int, np::Int, nterms::Int, n_alg::Int,
    alg_idx::AbstractVector{<:Integer}, exp_idx::AbstractVector{<:Integer},
    asymptotic_matrices, eigenbasis, exponents, coeffs, physical,
    rescale, bc::ParityBC, domain::Symbol=:half,
    component_parity::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    alg = collect(Int, alg_idx)
    exp = collect(Int, exp_idx)
    alg == collect(1:n_alg) || throw(ArgumentError("SystemSpec: alg_idx must be 1:n_alg (algebraic block first and contiguous), got $alg"))
    sort(vcat(alg, exp)) == collect(1:n) || throw(ArgumentError("SystemSpec: alg_idx ∪ exp_idx must partition 1:n"))
    length(alg) == n_alg || throw(ArgumentError("SystemSpec: length(alg_idx) must equal n_alg"))
    domain in (:half, :full) || throw(ArgumentError("SystemSpec: domain must be :half or :full, got $domain"))
    cp = component_parity === nothing ? component_parity_from_bc(bc) : collect(Int, component_parity)
    length(cp) == ncomp || throw(ArgumentError("SystemSpec: component_parity must have length ncomp = $ncomp, got $(length(cp))"))
    return SystemSpec(n, ncomp, np, nterms, n_alg, alg, exp,
        asymptotic_matrices, eigenbasis, exponents, coeffs, physical,
        rescale, bc, domain, cp)
end

"""
    n_exp(spec::SystemSpec) -> Int

Dimension of the exponential subspace, `n - n_alg`.
"""
n_exp(spec::SystemSpec) = spec.n - spec.n_alg
