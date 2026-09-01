"""
    StellaratorBasis

Change of basis for one toroidal residue class `k` that makes the reduced double-layer operator
`D̂ₖ` real, and block-diagonal when the class is self-conjugate.

A stellarator-symmetric boundary is invariant under `(θ, ζ) → (−θ, −ζ)` with `Z → −Z`, i.e. under
the rotation `diag(1, −1, −1)`. Both Laplace kernels depend only on `|r_obs − r_src|` and
`n_src·(r_obs − r_src)`, so the operator inherits the symmetry. Under the field-period fold the
involution acts *antilinearly* on the reduced operator,

    D̂ₖ[σp, σq] = ω^{k(aₚ − a_q)} · conj(D̂ₖ[p, q]),   ω = exp(-2πi/nfp),

with `σ` the within-period involution and `a = 0` on the `ζ = 0` symmetry plane, `1` elsewhere.
Two cases follow, and the operator memory halves in both:

  - **Self-conjugate class** (`mod(2k, nfp) == 0`, which is every class when `nfp == 1`): `ω^k = ±1`
    and `D̂ₖ` is real, commuting with the real signed involution `J = ΛΠ`, `Λ[p] = (ω^k)^{aₚ}`. The
    operator splits into the two `J`-eigenspaces — two real blocks of roughly half size each.
  - **Otherwise**: the involution is antiunitary with `Θ² = +1`, so the half-twist `λ = ω^{k a / 2}`,
    the parity basis, and a factor `i` on the odd half make the operator real at full size.

The partner row of an orbit is never evaluated — it follows from the relation above — so the kernel
visits only half the observer points.

## Fields

  - `σ`: within-period involution as a grid-index permutation of `1:num_points_per_fp`
  - `self_conjugate`: whether the class splits into two blocks
  - `λ`: half-twist `ω^{k a_p / 2}` per grid point (non-self-conjugate case)
  - `Λ`: real sign `(ω^k)^{a_p}` per grid point (self-conjugate case)
  - `orbit_rep`, `orbit_partner`: the two grid points of each involution orbit, equal for a fixed point
  - `orbit_cols`: the one or two basis columns each orbit produces, as indices into `block`/`slot`
  - `col_p`, `col_q`, `col_cp`, `col_cq`: basis column `c` is `col_cp[c]·e_{col_p[c]} + col_cq[c]·e_{col_q[c]}`,
    with `col_q == col_p` and `col_cq == 0` for a fixed point of `σ`
  - `block`, `slot`: destination block of each basis column and its position within that block
  - `block_size`: number of basis columns per surface in each block
"""
struct StellaratorBasis
    σ::Vector{Int}
    self_conjugate::Bool
    λ::Vector{ComplexF64}
    Λ::Vector{Int}
    orbit_rep::Vector{Int}
    orbit_partner::Vector{Int}
    orbit_cols::Vector{Vector{Int}}
    col_p::Vector{Int}
    col_q::Vector{Int}
    col_cp::Vector{ComplexF64}
    col_cq::Vector{ComplexF64}
    block::Vector{Int}
    slot::Vector{Int}
    block_size::Vector{Int}
end

"""
    stellarator_involution(plasma, wall, nfp) -> Union{Vector{Int},Nothing}

Within-period grid-index involution `σ` if both surfaces are stellarator symmetric, `nothing`
otherwise. The test compares `r(−θ, −ζ)` against `diag(1,−1,−1)·r(θ, ζ)` on the full torus; a
symmetric boundary matches to round-off while an asymmetric one is off by a fraction of the minor
radius, so the threshold is not delicate.
"""
function stellarator_involution(plasma::PlasmaGeometry3D, wall::WallGeometry3D, nfp::Int)
    mtheta, nzeta_full = plasma.mtheta, plasma.nzeta
    nzeta_full % nfp == 0 || return nothing
    num_points = mtheta * nzeta_full
    σ_full = [mod1(2 - mod1(p, mtheta), mtheta) + mtheta * (mod1(2 - ((p - 1) ÷ mtheta + 1), nzeta_full) - 1) for p in 1:num_points]

    function symmetric(r)
        tol = 1e-10 * maximum(abs, r)
        return all(p -> abs(r[σ_full[p], 1] - r[p, 1]) ≤ tol && abs(r[σ_full[p], 2] + r[p, 2]) ≤ tol && abs(r[σ_full[p], 3] + r[p, 3]) ≤ tol, 1:num_points)
    end
    symmetric(plasma.r) || return nothing
    (wall.nowall || symmetric(wall.r)) || return nothing

    # Within one field period the involution is composed with a field-period rotation, so the ζ = 0
    # line maps to itself and every other line reflects about it.
    nzeta = nzeta_full ÷ nfp
    return [mod1(2 - mod1(p, mtheta), mtheta) + mtheta * (mod1(nzeta + 2 - ((p - 1) ÷ mtheta + 1), nzeta) - 1) for p in 1:(mtheta*nzeta)]
end

"""
    StellaratorBasis(σ, mtheta, k, nfp)

Build the basis for toroidal residue class `k` from the within-period involution `σ`.
"""
function StellaratorBasis(σ::Vector{Int}, mtheta::Int, k::Int, nfp::Int)
    npts = length(σ)
    a(p) = ((p - 1) ÷ mtheta + 1) == 1 ? 0 : 1 # ζ = 0 plane is the symmetry plane and carries no twist
    self_conjugate = mod(2k, nfp) == 0
    ωk = cis(-2π * k / nfp)
    λ = ComplexF64[cis(-π * k * a(p) / nfp) for p in 1:npts]
    Λ = Int[round(Int, real(ωk))^a(p) for p in 1:npts]

    orbit_rep = Int[]
    orbit_partner = Int[]
    seen = falses(npts)
    for p in 1:npts
        seen[p] && continue
        seen[p] = true
        seen[σ[p]] = true
        push!(orbit_rep, p)
        push!(orbit_partner, σ[p])
    end

    # Each orbit contributes one column (fixed point) or a symmetric/antisymmetric pair. In the
    # self-conjugate case the pair splits by J-eigenvalue ±Λ[p]; otherwise everything is one block.
    block = Int[]
    col_p = Int[]
    col_q = Int[]
    col_cp = ComplexF64[]
    col_cq = ComplexF64[]
    orbit_cols = [Int[] for _ in orbit_rep]
    rt = 1 / √2
    function add_column!(blk, p, q, cp, cq)
        push!(block, blk)
        push!(col_p, p)
        push!(col_q, q)
        push!(col_cp, cp)
        push!(col_cq, cq)
        return length(block)
    end

    for (o, (p, q)) in enumerate(zip(orbit_rep, orbit_partner))
        # Symmetric combination first, antisymmetric second; the half-twist is absorbed into the
        # coefficients when the class is not self-conjugate.
        c1 = self_conjugate ? ComplexF64(1) : λ[p]
        blk1 = self_conjugate && Λ[p] == -1 ? 2 : 1
        if q == p
            push!(orbit_cols[o], add_column!(blk1, p, p, c1, 0))
        else
            push!(orbit_cols[o], add_column!(blk1, p, q, c1 * rt, c1 * rt))
            c2 = self_conjugate ? ComplexF64(rt) : im * λ[p] * rt
            blk2 = self_conjugate ? (Λ[p] == -1 ? 1 : 2) : 1
            push!(orbit_cols[o], add_column!(blk2, p, q, c2, -c2))
        end
    end

    nb = self_conjugate ? 2 : 1
    block_size = [count(==(b), block) for b in 1:nb]
    slot = zeros(Int, length(block))
    filled = zeros(Int, nb)
    for c in eachindex(block)
        filled[block[c]] += 1
        slot[c] = filled[block[c]]
    end
    return StellaratorBasis(σ, self_conjugate, λ, Λ, orbit_rep, orbit_partner, orbit_cols, col_p, col_q, col_cp, col_cq, block, slot, block_size)
end

"""
    emit_symmetric_row!(dest, basis, row, work, orbit, row_index, col_index)

Scatter the raw operator row of one involution-orbit representative into the transformed blocks.

`row` holds the untransformed operator row over one source surface and `work` is scratch of the same
length. `dest[b]` receives block `b`, with `row_index`/`col_index` (1 = plasma, 2 = wall) selecting
the surface sub-block. The partner row is reconstructed from `row`, so the caller evaluates only the
orbit representatives.
"""
function emit_symmetric_row!(
    dest::AbstractVector{<:AbstractMatrix{Float64}},
    basis::StellaratorBasis,
    row::AbstractVector{<:Number},
    work::AbstractVector{ComplexF64},
    orbit::Int,
    row_index::Int,
    col_index::Int
)
    (; σ, self_conjugate, λ, Λ, orbit_rep, orbit_partner, orbit_cols, block, slot) = basis
    p, q = orbit_rep[orbit], orbit_partner[orbit]
    fixed_row = p == q
    rt = √2

    # Reconstruct the partner row from the representative: real and signed when the class is
    # self-conjugate, conjugated in the half-twisted frame otherwise.
    if self_conjugate
        @inbounds for x in eachindex(row)
            work[x] = Λ[p] * Λ[x] * real(row[σ[x]])
        end
    else
        cλp = conj(λ[p])
        @inbounds for x in eachindex(row)
            work[x] = cλp * row[x] * λ[x]
        end
    end

    for (n, c) in enumerate(orbit_cols[orbit])
        blk = block[c]
        H = dest[blk]
        nsize = basis.block_size[blk]
        r = (row_index - 1) * nsize + slot[c]
        col_offset = (col_index - 1) * nsize
        even_row = n == 1
        sgn = even_row ? 1.0 : -1.0
        @inbounds for oc in eachindex(orbit_rep)
            x, y = orbit_rep[oc], orbit_partner[oc]
            fixed_col = x == y
            if self_conjugate
                vx = fixed_row ? real(row[x]) : (real(row[x]) + sgn * real(work[x])) / rt
                vy = fixed_col ? 0.0 : (fixed_row ? real(row[y]) : (real(row[y]) + sgn * real(work[y])) / rt)
                for (m, c2) in enumerate(orbit_cols[oc])
                    # Only the diagonal block survives; the rest vanishes by the symmetry
                    block[c2] == blk || continue
                    H[r, col_offset+slot[c2]] = fixed_col ? vx : (m == 1 ? (vx + vy) / rt : (vx - vy) / rt)
                end
            else
                cx = work[x]
                cy = fixed_col ? cx : work[y]
                for (m, c2) in enumerate(orbit_cols[oc])
                    val = if fixed_col
                        fixed_row ? real(cx) : rt * (even_row ? real(cx) : imag(cx))
                    elseif fixed_row
                        m == 1 ? rt * real(cx) : -rt * imag(cx)
                    elseif even_row
                        m == 1 ? real(cx) + real(cy) : imag(cy) - imag(cx)
                    else
                        m == 1 ? imag(cx) + imag(cy) : real(cx) - real(cy)
                    end
                    H[r, col_offset+slot[c2]] = val
                end
            end
        end
    end
    return nothing
end

"""
    emit_plain_row!(dest, row, idx_obs, row_index, col_index)

Write one accumulated operator row into the untransformed operator, the `sym === nothing` fall
through. `dest` is a one-element block list holding the full matrix.
"""
function emit_plain_row!(dest::AbstractVector{<:AbstractMatrix}, row::AbstractVector{<:Number}, idx_obs::Int, row_index::Int, col_index::Int)
    H = dest[1]
    npts = length(row)
    r = (row_index - 1) * npts + idx_obs
    col_offset = (col_index - 1) * npts
    @inbounds for x in eachindex(row)
        H[r, col_offset+x] = row[x]
    end
    return nothing
end

"""
    transform_mode_basis!(dest, basis_matrix, sym)

Express the Fourier mode basis in the symmetry-adapted basis, `Ẽ = E·U`, writing block `b` into
`dest[b]`. Substituting `Ẽ` for `E` carries the change of basis through the right-hand side, the
`wv` projection and `I_v` unchanged, since `U` is unitary.
"""
function transform_mode_basis!(dest::AbstractVector{<:AbstractMatrix{ComplexF64}}, basis_matrix::AbstractMatrix, sym::StellaratorBasis)
    for c in eachindex(sym.block)
        out = @view dest[sym.block[c]][:, sym.slot[c]]
        p, q = sym.col_p[c], sym.col_q[c]
        @views out .= sym.col_cp[c] .* basis_matrix[:, p]
        q != p && (@views out .+= sym.col_cq[c] .* basis_matrix[:, q])
    end
    return nothing
end
