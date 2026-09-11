const INV_SQRT2 = 1 / √2

"""
Column `cp·e_p + cq·e_q` of the symmetry-adapted basis, at `slot` of `block`. A grid point the
reflection fixes has `q == p` and `cq == 0`.
"""
struct SymmetryColumn
    p::Int
    q::Int
    cp::ComplexF64
    cq::ComplexF64
    block::Int
    slot::Int
end

"""
    StellSymBasis

Unitary change of basis `U` for residue class `k` making `U†D̂ₖU` real, and block-diagonal when the
class is self-conjugate. Both Laplace kernels depend only on `|r_obs − r_src|` and `n_src·(r_obs − r_src)`,
so a stellarator-symmetric boundary gives

    D̂ₖ[σp, σq] = ω^{k(a_p − a_q)} · conj(D̂ₖ[p, q]),   ω = exp(-2πi/nfp),

with `σ` the within-period reflection `(θ, ζ) → (−θ, −ζ)` and `a_p = 0` on the `ζ = 0` plane, `1`
elsewhere. Pairing `p` with `σp` into a symmetric and an antisymmetric column makes the result real;
a self-conjugate class ([`is_self_conjugate`](@ref)) has real `D̂ₖ` and splits by the sign of `ω^{k a_p}`.

## Fields

  - `σ_map`: `σ` as a grid-index map of `1:num_points_per_fp`
  - `σ_phase`: `ω^{k a_p}`, relating a pair's two operator rows
  - `pair_reps`: one grid index per reflection pair (partner is `σ_map[p]`)
  - `columns`, `pair_columns`: basis columns, and the range each pair contributes
  - `block_sizes`: columns per surface in each block (`length` is 1 or 2)
"""
struct StellSymBasis
    σ_map::Vector{Int}
    σ_phase::Vector{ComplexF64}
    pair_reps::Vector{Int}
    columns::Vector{SymmetryColumn}
    pair_columns::Vector{UnitRange{Int}}
    block_sizes::Vector{Int}
end

"""Whether residue class `k` is its own conjugate (`ω^k = ±1`), so `D̂ₖ` and the field-period phases are real."""
is_self_conjugate(k::Integer, nfp::Integer) = mod(2k, nfp) == 0

"""
    get_conjugate_groups(classes, nfp) -> Vector{Vector{Int}}

Each class is a mode family `k = mod(n, nfp)`. Group each with its conjugate family
`mod(nfp - k, nfp)` when that class is present. Self-conjugate families
([`is_self_conjugate`](@ref)) never pair.

Returns a vector of groups; each group is a vector of indices into `classes`, with the
representative first and its conjugate partner second when paired (`[i]` or `[i, j]`).
"""
function get_conjugate_groups(classes::AbstractVector{<:Integer}, nfp::Integer)
    groups = Vector{Int}[]
    taken = falses(length(classes))
    for (i, k) in enumerate(classes)
        taken[i] && continue
        taken[i] = true
        j = is_self_conjugate(k, nfp) ? nothing : findfirst(==(mod(nfp - k, nfp)), classes)
        if j === nothing || taken[j]
            push!(groups, [i])
        else
            taken[j] = true
            push!(groups, [i, j])
        end
    end
    return groups
end

"""
Grid index of `(θ, ζ) → (−θ, −ζ)` for the linear index `p = i_θ + mtheta·(i_ζ − 1)`.
"""
function reflect_index(p::Int, mtheta::Int, nzeta::Int)
    i_θ, i_ζ = mod1(p, mtheta), (p - 1) ÷ mtheta + 1
    return mod1(2 - i_θ, mtheta) + mtheta * (mod1(2 - i_ζ, nzeta) - 1)
end

"""
    stell_sym_map(plasma, wall, nfp) -> Union{Vector{Int},Nothing}

Within-period map of `(θ, ζ) → (−θ, −ζ)` if both surfaces are stellarator symmetric, `nothing`
otherwise. A symmetric boundary matches to round-off; an asymmetric one is off by a fraction of the
minor radius, so the threshold is not delicate.
"""
function stell_sym_map(plasma::PlasmaGeometry3D, wall::WallGeometry3D, nfp::Int)
    # This must be called after expand_field_periods, so nzeta_full is the full-torus nzeta
    mtheta, nzeta_full = plasma.mtheta, plasma.nzeta
    @assert nzeta_full % nfp == 0 "full-torus nzeta ($nzeta_full) must be divisible by nfp ($nfp)"
    num_points_full = mtheta * nzeta_full
    reflected = [reflect_index(p, mtheta, nzeta_full) for p in 1:num_points_full]

    function is_symmetric(r)
        # Cartesian x is even under (θ,ζ) → (−θ,−ζ); y and z are odd.
        tol = 1e-10 * maximum(abs, r)
        return all(p -> abs(r[reflected[p], 1] - r[p, 1]) ≤ tol && abs(r[reflected[p], 2] + r[p, 2]) ≤ tol && abs(r[reflected[p], 3] + r[p, 3]) ≤ tol, 1:num_points_full)
    end

    # Check if the boundary(s) are stellarator symmetric
    # return the mapping if so, otherwise return nothing
    if is_symmetric(plasma.r) && (wall.nowall || is_symmetric(wall.r))
        nzeta = nzeta_full ÷ nfp
        return [reflect_index(p, mtheta, nzeta) for p in 1:(mtheta*nzeta)]
    else
        return nothing
    end
end

"""
    StellSymBasis(σ_map, mtheta, k, nfp)

Build the symmetry-adapted basis for toroidal residue class `k`.
"""
function StellSymBasis(σ_map::Vector{Int}, mtheta::Int, k::Int, nfp::Int)
    npts = length(σ_map)
    self_conjugate = is_self_conjugate(k, nfp)
    ωk = cis(-2π * k / nfp)

    # a_p tags the field-period twist in D̂ₖ[σp,σq] = ω^{k(a_p−a_q)} conj(D̂ₖ[p,q]): 0 on ζ=0, else 1
    a(p) = p ≤ mtheta ? 0 : 1
    # Exact ±1 when self-conjugate
    ω = self_conjugate ? ComplexF64(round(real(ωk))) : ωk
    # relative phase between rows p and σp under the reflection identity
    σ_phase = ComplexF64[ω^a(p) for p in 1:npts]

    # Column prefactor: half-twist ω^{k a_p/2} makes U†D̂ₖU real when k is not self-conjugate
    twist(p) = self_conjugate ? ComplexF64(1) : cis(-π * k * a(p) / nfp)

    # Get one representative from each reflection pair
    pair_reps = [p for p in 1:npts if p ≤ σ_map[p]]

    # Build orthonormal columns of U: symmetric first, antisymmetric second; slot numbered within block
    columns = SymmetryColumn[]
    pair_columns = UnitRange{Int}[]
    filled = zeros(Int, self_conjugate ? 2 : 1)  # one block, or two reflection-eigenvalue blocks
    add_column!(p, q, cp, cq, block) = (filled[block] += 1; push!(columns, SymmetryColumn(p, q, cp, cq, block, filled[block])))
    for p in pair_reps
        q = σ_map[p]
        first_col = length(columns) + 1
        
        # Self-conjugate: put ± reflection eigenvalues in opposite blocks; flip assignment if σ_phase < 0
        negative = self_conjugate && real(σ_phase[p]) < 0
        sym_block = negative ? 2 : 1
        anti_block = self_conjugate ? (negative ? 1 : 2) : 1  # non-self-conjugate: everything in block 1
        
        twist_p = twist(p)
        if q == p
            # Fixed point of σ: single column twist_p · e_p (cq = 0)
            add_column!(p, p, twist_p, 0, sym_block)
        else
            # Symmetric combination (e_p + e_q)/√2, times the half-twist
            add_column!(p, q, twist_p * INV_SQRT2, twist_p * INV_SQRT2, sym_block)
            # Antisymmetric (e_p − e_q)/√2; im·twist keeps non-self-conjugate columns orthonormal and U†DU real
            anti_coeff = self_conjugate ? ComplexF64(INV_SQRT2) : im * twist_p * INV_SQRT2
            add_column!(p, q, anti_coeff, -anti_coeff, anti_block)
        end
        # Columns this pair owns — store_kernel_row! only transforms those when writing the pair's rows
        push!(pair_columns, first_col:length(columns))
    end
    return StellSymBasis(σ_map, σ_phase, pair_reps, columns, pair_columns, filled)
end

"""
    store_kernel_row!(blocks, sym_basis, kernel_row, scratch, i_pair, row_index, col_index)

Write one accumulated kernel row into `blocks`. With a [`StellSymBasis`](@ref), reconstruct the
partner row from the reflection identity and emit `U†DU`; `sym_basis === nothing` stores the row
as-is. `i_pair` indexes the observer list (every period-0 point, or one representative per pair).
`row_index`/`col_index` are 1 = plasma, 2 = wall. `scratch` needs `2·npts` when a basis is present.
"""
function store_kernel_row!(
    blocks::AbstractVector{<:AbstractMatrix},
    sym_basis::Union{Nothing,StellSymBasis},
    kernel_row::AbstractVector{<:Number},
    scratch::AbstractVector{ComplexF64},
    i_pair::Int,
    row_index::Int,
    col_index::Int
)
    npts = length(kernel_row)
    if sym_basis === nothing
        # No basis: copy the row into the single untransformed block
        H = blocks[1]
        r = (row_index - 1) * npts + i_pair
        @views H[r, ((col_index-1)*npts+1):(col_index*npts)] .= kernel_row
        return
    end

    (; σ_map, σ_phase, pair_reps, columns, pair_columns, block_sizes) = sym_basis
    p = pair_reps[i_pair]
    partner = @view scratch[1:npts]
    combined = @view scratch[(npts+1):(2npts)]
    pair_phase = σ_phase[p]

    # Partner row at σp via D̂ₖ[σp, σq] = ω^{k(a_p−a_q)} conj(D̂ₖ[p, q]) — avoids quadrature at σp
    @inbounds for x in 1:npts
        partner[x] = pair_phase * conj(σ_phase[x]) * conj(kernel_row[σ_map[x]])
    end

    # Emit U†DU only for the (≤2) columns this reflection pair owns
    for c in pair_columns[i_pair]
        row_col = columns[c]
        H = blocks[row_col.block]
        nsize = block_sizes[row_col.block]
        r = (row_index - 1) * nsize + row_col.slot
        col_offset = (col_index - 1) * nsize

        # Left U†: mix rows p and σp with this column's conjugated coefficients
        cp, cq = conj(row_col.cp), conj(row_col.cq)
        @inbounds for x in 1:npts
            combined[x] = cp * kernel_row[x] + cq * partner[x]
        end

        # Right U on the same block only; real(·) drops a ~0 imaginary residual
        @inbounds for src_col in columns
            src_col.block == row_col.block || continue
            val = src_col.cp * combined[src_col.p]
            src_col.q != src_col.p && (val += src_col.cq * combined[src_col.q])
            H[r, col_offset+src_col.slot] = real(val)
        end
    end
end

"""
    transform_mode_basis!(E_out, E, sym_basis; conjugate=false)

Write `E_out = E·U` into `E_out[b]` per block. `U` is unitary, so substituting `E_out` for `E` leaves the
RHS, the `wv` projection and `I_v` unchanged. `conjugate=true` gives `conj(E)·U` for this class's
conjugate partner. `sym_basis === nothing` is the identity.
"""
function transform_mode_basis!(
    E_out::AbstractVector{<:AbstractMatrix{ComplexF64}},
    E::AbstractMatrix,
    sym_basis::Union{Nothing,StellSymBasis};
    conjugate::Bool=false
)
    # No stellarator basis: identity U, optionally conj(E) for the conjugate partner class
    if sym_basis === nothing
        E_out[1] .= conjugate ? conj.(E) : E
        return
    end
    maybe_conj = conjugate ? conj : identity
    # Each SymmetryColumn is one column of U: E_out[:, slot] = cp·f(E[:,p]) + cq·f(E[:,q])
    for col in sym_basis.columns
        out = @view E_out[col.block][:, col.slot]
        @views out .= col.cp .* maybe_conj.(E[:, col.p])
        col.q != col.p && (@views out .+= col.cq .* maybe_conj.(E[:, col.q]))
    end
end
