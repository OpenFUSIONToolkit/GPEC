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
    StellaratorBasis

Unitary change of basis `U` for toroidal residue class `k` making `U†D̂ₖU` real, and block diagonal
when the class is self-conjugate. Both Laplace kernels depend only on `|r_obs − r_src|` and
`n_src·(r_obs − r_src)`, so a stellarator-symmetric boundary gives

    D̂ₖ[σp, σq] = ω^{k(a_p − a_q)} · conj(D̂ₖ[p, q]),   ω = exp(-2πi/nfp),

with `σ` the within-period reflection `(θ, ζ) → (−θ, −ζ)` and `a_p = 0` on the `ζ = 0` symmetry
plane, `1` elsewhere. Pairing `p` with `σp` into a symmetric and an antisymmetric column makes the
result real; a self-conjugate class (`mod(2k, nfp) == 0`) has real `D̂ₖ` and splits further by the
sign of `ω^{k a_p}`.

## Fields

  - `mirror`: `σ` as a grid-index map of `1:num_points_per_fp`
  - `phase`: `ω^{k a_p}`, the factor relating a pair's two operator rows
  - `pairs`: the two grid points of each reflection pair, equal at a fixed point
  - `columns`, `pair_columns`: the basis columns, and the range each pair contributes
  - `block_size`: columns per surface in each block, so `length` is 1 or 2
"""
struct StellaratorBasis
    mirror::Vector{Int}
    phase::Vector{ComplexF64}
    pairs::Vector{Tuple{Int,Int}}
    columns::Vector{SymmetryColumn}
    pair_columns::Vector{UnitRange{Int}}
    block_size::Vector{Int}
end

"""
    stellarator_mirror(plasma, wall, nfp) -> Union{Vector{Int},Nothing}

Within-period grid-index map of `(θ, ζ) → (−θ, −ζ)` if both surfaces are stellarator symmetric,
`nothing` otherwise. A symmetric boundary matches to round-off and an asymmetric one is off by a
fraction of the minor radius, so the threshold is not delicate.
"""
function stellarator_mirror(plasma::PlasmaGeometry3D, wall::WallGeometry3D, nfp::Int)
    mtheta, nzeta_full = plasma.mtheta, plasma.nzeta
    nzeta_full % nfp == 0 || return nothing
    num_points = mtheta * nzeta_full
    reflected = [mod1(2 - mod1(p, mtheta), mtheta) + mtheta * (mod1(2 - ((p - 1) ÷ mtheta + 1), nzeta_full) - 1) for p in 1:num_points]

    function symmetric(r)
        tol = 1e-10 * maximum(abs, r)
        return all(p -> abs(r[reflected[p], 1] - r[p, 1]) ≤ tol && abs(r[reflected[p], 2] + r[p, 2]) ≤ tol && abs(r[reflected[p], 3] + r[p, 3]) ≤ tol, 1:num_points)
    end
    symmetric(plasma.r) || return nothing
    (wall.nowall || symmetric(wall.r)) || return nothing

    # Within one field period the reflection is composed with a field-period rotation, so the ζ = 0
    # line maps to itself and every other line reflects about it.
    nzeta = nzeta_full ÷ nfp
    return [mod1(2 - mod1(p, mtheta), mtheta) + mtheta * (mod1(nzeta + 2 - ((p - 1) ÷ mtheta + 1), nzeta) - 1) for p in 1:(mtheta*nzeta)]
end

"""
    StellaratorBasis(mirror, mtheta, k, nfp)

Build the symmetry-adapted basis for toroidal residue class `k`.
"""
function StellaratorBasis(mirror::Vector{Int}, mtheta::Int, k::Int, nfp::Int)
    npts = length(mirror)
    self_conjugate = mod(2k, nfp) == 0
    ωk = cis(-2π * k / nfp)

    # The ζ = 0 plane is the symmetry plane and carries no twist; every other line carries one.
    a(p) = ((p - 1) ÷ mtheta + 1) == 1 ? 0 : 1
    phase = ComplexF64[self_conjugate ? round(real(ωk))^a(p) : ωk^a(p) for p in 1:npts]

    # Half-twist ω^{k a_p / 2}, which makes a non-self-conjugate class real. A self-conjugate class is
    # real already and instead splits by the sign of `phase`, so it takes no twist.
    twist(p) = self_conjugate ? ComplexF64(1) : cis(-π * k * a(p) / nfp)

    pairs = Tuple{Int,Int}[]
    seen = falses(npts)
    for p in 1:npts
        seen[p] && continue
        seen[p] = seen[mirror[p]] = true
        push!(pairs, (p, mirror[p]))
    end

    # Symmetric column first, antisymmetric second. Self-conjugate blocks are assigned so that each
    # block collects one eigenvalue of the reflection, which is what makes the operator block diagonal.
    columns = SymmetryColumn[]
    pair_columns = UnitRange{Int}[]
    rt = 1 / √2
    for (p, q) in pairs
        first_col = length(columns) + 1
        # A non-self-conjugate class has a single block; a self-conjugate one sends the symmetric and
        # antisymmetric columns to opposite blocks, swapped where `phase` is negative.
        negative = self_conjugate && real(phase[p]) < 0
        sym_block = negative ? 2 : 1
        anti_block = self_conjugate ? (negative ? 1 : 2) : 1
        c = twist(p)
        if q == p
            push!(columns, SymmetryColumn(p, p, c, 0, sym_block, 0))
        else
            push!(columns, SymmetryColumn(p, q, c * rt, c * rt, sym_block, 0))
            c2 = self_conjugate ? ComplexF64(rt) : im * c * rt
            push!(columns, SymmetryColumn(p, q, c2, -c2, anti_block, 0))
        end
        push!(pair_columns, first_col:length(columns))
    end

    # Number the columns within their block
    nblocks = self_conjugate ? 2 : 1
    filled = zeros(Int, nblocks)
    for (c, col) in enumerate(columns)
        filled[col.block] += 1
        columns[c] = SymmetryColumn(col.p, col.q, col.cp, col.cq, col.block, filled[col.block])
    end
    return StellaratorBasis(mirror, phase, pairs, columns, pair_columns, copy(filled))
end

"""
    emit_row!(dest, sym, row, work, pair, idx_obs, row_index, col_index)

Write one accumulated operator row into `dest`, with `row_index`/`col_index` (1 = plasma, 2 = wall)
selecting the surface sub-block. With `sym === nothing` the row goes in untransformed and `dest`
holds the whole matrix. With a [`StellaratorBasis`](@ref) the row belongs to reflection pair `pair`
and its partner follows from the symmetry, so the caller evaluates only half the observer points;
`work` is then complex scratch of at least `3·length(row)`.
"""
function emit_row!(dest::AbstractVector{<:AbstractMatrix}, ::Nothing, row::AbstractVector{<:Number}, work, pair::Int, idx_obs::Int, row_index::Int, col_index::Int)
    H = dest[1]
    npts = length(row)
    r = (row_index - 1) * npts + idx_obs
    col_offset = (col_index - 1) * npts
    @inbounds for x in eachindex(row)
        H[r, col_offset+x] = row[x]
    end
    return nothing
end

function emit_row!(
    dest::AbstractVector{<:AbstractMatrix{Float64}},
    basis::StellaratorBasis,
    row::AbstractVector{<:Number},
    work::AbstractVector{ComplexF64},
    pair::Int,
    idx_obs::Int,
    row_index::Int,
    col_index::Int
)
    (; mirror, phase, pairs, columns, pair_columns, block_size) = basis
    p, _ = pairs[pair]
    npts = length(row)
    rep = @view work[1:npts]                # this pair's representative row
    partner = @view work[(npts+1):(2npts)]  # its mirror partner, reconstructed
    combined = @view work[(2npts+1):(3npts)]

    @inbounds for x in 1:npts
        rep[x] = row[x]
        partner[x] = phase[p] * conj(phase[x]) * conj(row[mirror[x]])
    end

    # Form U†DU for the columns this pair owns
    for c in pair_columns[pair]
        rc = columns[c]
        H = dest[rc.block]
        nsize = block_size[rc.block]
        r = (row_index - 1) * nsize + rc.slot
        col_offset = (col_index - 1) * nsize

        # Left factor U†: combine the pair's two rows with this column's conjugated coefficients
        cp, cq = conj(rc.cp), conj(rc.cq)
        @inbounds for x in 1:npts
            combined[x] = cp * rep[x] + cq * partner[x]
        end

        # Right factor U: read off each column the same way. Only the diagonal block survives.
        @inbounds for cc in columns
            cc.block == rc.block || continue
            val = cc.cp * combined[cc.p]
            cc.q != cc.p && (val += cc.cq * combined[cc.q])
            H[r, col_offset+cc.slot] = real(val)
        end
    end
    return nothing
end

"""
    transform_mode_basis!(dest, mode_basis, sym; conjugate=false)

Write `Ẽ = E·U` into `dest[b]` per block. `U` is unitary, so substituting `Ẽ` for `E` leaves the
right-hand side, the `wv` projection and `I_v` unchanged. `conjugate=true` gives `conj(E)·U`, what
this class's conjugate partner needs.
"""
function transform_mode_basis!(dest::AbstractVector{<:AbstractMatrix{ComplexF64}}, mode_basis::AbstractMatrix, sym::StellaratorBasis; conjugate::Bool=false)
    f = conjugate ? conj : identity
    for col in sym.columns
        out = @view dest[col.block][:, col.slot]
        @views out .= col.cp .* f.(mode_basis[:, col.p])
        col.q != col.p && (@views out .+= col.cq .* f.(mode_basis[:, col.q]))
    end
    return nothing
end
