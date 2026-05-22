# FullDomain.jl
#
# Full-domain (parity-coupled) inner-layer solve for the Wasow–Galerkin engine.
#
# Whereas the half-domain path folds the problem to X ∈ [0, Xmax] and imposes a
# parity boundary condition at the rational surface (X = 0), the full-domain path
# discretizes X ∈ [−Xmax, +Xmax] directly, with asymptotic matching at *both*
# ends and no parity reduction. The inner-layer coefficient matrices are regular
# at X = 0 (the rational-surface singularity is resolved in the stretched
# coordinate), so X = 0 is an ordinary interior point.
#
# Implementation (reflect-and-merge): the right half (X ∈ [0, Xmax]) and the left
# half — reflected to S = −X ∈ [0, Xmax], where it is a standard right-oriented
# half-domain — are each assembled with the *same* validated half-domain
# machinery, then merged at the shared center node X = 0. Continuity there ties
# the two halves' value DOFs (same) and derivative DOFs (opposite sign, since
# d/dS = −d/dX); the two halves' X = 0 IBP surface terms then cancel, as required
# for an interior point. The left half's operator in S is `I u_SS − V u_S + U u`
# (V flips sign) evaluated at X = −S, and its asymptotic basis is the
# parity-continued one (component parity σ from `spec.component_parity`).
#
# The result is the 2×2 end-matching matrix `M` (see [`FullDomainMatching`](@ref)).
# `M` is the general deliverable: for a parity-broken two-fluid layer it is
# asymmetric by design. For a parity-symmetric system (GGJ) it is symmetric, and
# [`parity_recombine`](@ref) recovers the half-domain `(Δ_even, Δ_odd)` pair — the
# validation oracle proving the assembly is correct, not the production extraction.

using SparseArrays

"""
    FullDomainMatching

Matching data from a full-domain inner-layer solve. Field `M` is the 2×2
end-matching matrix: `M[i,j]` is the (raw, pre-rescale) asymptotic amplitude at
end `i` (1 = −Xmax, 2 = +Xmax) in response to driving at end `j`. For a
parity-symmetric system `M` is symmetric and recombines via
[`parity_recombine`](@ref) to the half-domain parity pair; for a parity-broken
system `M` is asymmetric and is itself the matching deliverable.
"""
struct FullDomainMatching
    M::SMatrix{2,2,ComplexF64}
    xmax::Float64
    ndim::Int
end

"""
    parity_recombine(fdm::FullDomainMatching) -> (λ_sym, λ_anti)

Project the end-matching matrix `M` onto the symmetric `(1, 1)` and antisymmetric
`(1, -1)` parity eigenvectors, returning the matching amplitudes
`λ_sym = ½ Σ M` and `λ_anti = ½ (M₁₁ − M₁₂ − M₂₁ + M₂₂)`. For a parity-symmetric
system (`M = [a b; b a]`) these are `a+b` and `a-b`, the matching data of the two
parity sectors (equivalent to the eigenvalues of the symmetrized `M`). Which
sector (even / odd) each corresponds to is model-specific; for GGJ, `λ_sym` is
the odd-sector and `λ_anti` the even-sector amplitude.
"""
function parity_recombine(fdm::FullDomainMatching)
    M = fdm.M
    λ_sym = (M[1, 1] + M[1, 2] + M[2, 1] + M[2, 2]) / 2
    λ_anti = (M[1, 1] - M[1, 2] - M[2, 1] + M[2, 2]) / 2
    return λ_sym, λ_anti
end

# Left-half spec in the reflected coordinate S = −X: the operator becomes
# `I u_SS − V u_S + U u` (V flips sign), evaluated at the physical point X = −S,
# and the asymptotic basis is continued by the component parity σ.
function _reflected_spec(spec::SystemSpec)
    σ = spec.component_parity
    coeffsL = function (params, Q, s)
        Imat, Umat, Vmat = spec.coeffs(params, Q, -s)
        return Imat, Umat, -Vmat
    end
    physicalL = function (U, dU, s)
        ua, dua = spec.physical(U, dU, s)
        return σ .* ua, σ .* dua
    end
    return SystemSpec(; n=spec.n, ncomp=spec.ncomp, np=spec.np, nterms=spec.nterms,
        n_alg=spec.n_alg, alg_idx=spec.alg_idx, exp_idx=spec.exp_idx,
        asymptotic_matrices=spec.asymptotic_matrices, eigenbasis=spec.eigenbasis,
        exponents=spec.exponents, coeffs=coeffsL, physical=physicalL,
        rescale=spec.rescale, bc=spec.bc, domain=:half, component_parity=σ)
end

# Assemble one half-domain (right-oriented in its own coordinate) into the global
# sparse system via the DOF map `gmap` and sign `sgn`, with RHS driving in column
# `col`. Mirrors the half-domain assembly but emits triplets and applies no
# parity boundary condition (continuity at X = 0 is handled by the shared center
# DOFs in `gmap`/`sgn`).
function _assemble_half_coo!(Is::Vector{Int}, Js::Vector{Int}, Vs::Vector{ComplexF64},
    rhs::Matrix{ComplexF64}, ws::GalerkinWorkspace, spec::SystemSpec, params,
    Q::ComplexF64, cache::WasowCache, gmap::Vector{Int}, sgn::Vector{Int}, col::Int;
    nq::Int, tol_res::Float64)
    nc = spec.ncomp
    quad_nodes, quad_weights = gausslobatto(nq + 1)
    addA! = (i, j, v) -> (push!(Is, gmap[i]); push!(Js, gmap[j]); push!(Vs, sgn[i] * sgn[j] * v))
    addR! = (i, v) -> (rhs[gmap[i], col] += sgn[i] * v)

    for ix in 1:ws.nx
        cell = ws.cells[ix]

        if cell.np >= 0
            np_eff = cell.np
            cell_mat = zeros(ComplexF64, nc, nc, np_eff + 1, np_eff + 1)
            _gauss_quad!(cell_mat, cell, quad_nodes, quad_weights, spec, params, Q)
            for ip in 0:np_eff, ipert in 1:nc
                i = cell.map[ipert, ip+1]
                i > ws.ndim && continue
                for jp in 0:np_eff, jpert in 1:nc
                    j = cell.map[jpert, jp+1]
                    j > ws.ndim && continue
                    addA!(i, j, cell_mat[ipert, jpert, ip+1, jp+1])
                end
            end
        end

        if cell.etype in (CT_EXT, CT_EXT1, CT_EXT2)
            if cell.etype == CT_EXT
                cell_mat_ext = zeros(ComplexF64, nc, nc, cell.np + 2, cell.np + 2)
                cell_rhs_ext = zeros(ComplexF64, nc, cell.np + 2)
            else
                cell_mat_ext = zeros(ComplexF64, nc, nc, cell.np + 1, cell.np + 1)
                cell_rhs_ext = zeros(ComplexF64, nc, cell.np + 1)
            end
            _extension!(cell_mat_ext, cell_rhs_ext, cell, quad_nodes, quad_weights, spec, params, Q, cache)
            npp = size(cell_mat_ext, 3) - 1
            for ip in 0:npp, ipert in 1:nc
                i = ip < size(cell.map, 2) ? cell.map[ipert, ip+1] : cell.emap[1]
                (cell.etype == CT_EXT && ip == cell.np + 1 && ipert > 1) && continue
                i > ws.ndim && continue
                for jp in 0:npp, jpert in 1:nc
                    j = jp < size(cell.map, 2) ? cell.map[jpert, jp+1] : cell.emap[1]
                    (cell.etype == CT_EXT && jp == cell.np + 1 && jpert > 1) && continue
                    j > ws.ndim && continue
                    addA!(i, j, cell_mat_ext[ipert, jpert, ip+1, jp+1])
                end
                addR!(i, cell_rhs_ext[ipert, ip+1])
            end
        end

        if cell.etype == CT_RES
            val_11, val_12 = _resonant_integral(cell, spec, params, Q, cache; tol=tol_res)
            e = cell.emap[1]
            if e <= ws.ndim
                addA!(e, e, val_11)
                addR!(e, -val_12)
            end
        end
    end

    # Surface term at the outer (Xmax) matching boundary of this half.
    res_cell = ws.cells[ws.nx]
    xl = res_cell.x_lsode
    ua_e, dua_e = _eval_physical(spec, cache, xl)
    Imat_e, _, _ = spec.coeffs(params, Q, xl)
    e = res_cell.emap[1]
    if e <= ws.ndim
        addA!(e, e, -(transpose(ua_e[:, 1]) * Imat_e * dua_e[:, 1]))
        addR!(e, transpose(ua_e[:, 1]) * Imat_e * dua_e[:, 2])
    end

    # IBP surface term at X = 0 (this half's inner boundary). The two halves'
    # contributions cancel under the center DOF identification (value shared,
    # derivative sign-flipped), leaving no boundary term at the interior point.
    cell1 = ws.cells[1]
    Imat0, _, _ = spec.coeffs(params, Q, cell1.x_left)
    for ipert in 1:nc, jpert in 1:nc
        i = cell1.map[ipert, 1]
        j = cell1.map[jpert, 2]
        (i <= ws.ndim && j <= ws.ndim) && addA!(i, j, Imat0[ipert, jpert])
    end
end

# Build the global DOF map for a half: each local DOF gets a global index and a
# sign. Center DOFs (cell 1's value [ip=0] and derivative [ip=1] at X = 0) reuse
# the shared `cval`/`cder` global indices; the derivative carries `der_sign`
# (−1 for the reflected left half). All other DOFs get fresh global indices from
# the `gid` counter (a 1-element Ref so it threads across both halves).
function _half_dof_map(ws::GalerkinWorkspace, nc::Int, cval::Vector{Int}, cder::Vector{Int},
    der_sign::Int, gid::Base.RefValue{Int})
    g = zeros(Int, ws.ndim)
    sgn = ones(Int, ws.ndim)
    cv = [ws.cells[1].map[ipert, 1] for ipert in 1:nc]
    cd = [ws.cells[1].map[ipert, 2] for ipert in 1:nc]
    for d in 1:ws.ndim
        kv = findfirst(==(d), cv)
        if kv !== nothing
            g[d] = cval[kv]
            continue
        end
        kd = findfirst(==(d), cd)
        if kd !== nothing
            g[d] = cder[kd]
            sgn[d] = der_sign
            continue
        end
        gid[] += 1
        g[d] = gid[]
    end
    return g, sgn
end

"""
    solve_galerkin_full(spec::SystemSpec, params, Q::ComplexF64;
                        kmax=8, nx=512, nq=4, pfac=1.0, cutoff=5, xfac=1.0,
                        tol_res=1e-5) -> FullDomainMatching

Full-domain inner-layer solve for `spec` on X ∈ [−Xmax, +Xmax] (no parity
reduction), via the reflect-and-merge construction described in the file header.
Returns the 2×2 end-matching matrix wrapped in [`FullDomainMatching`](@ref).
"""
function solve_galerkin_full(spec::SystemSpec, params, Q::ComplexF64;
    kmax::Int=8, nx::Int=512, nq::Int=4, pfac::Float64=1.0,
    cutoff::Int=5, xfac::Float64=1.0, tol_res::Float64=1e-5)
    spec.np == 3 || throw(ArgumentError("solve_galerkin_full: Hermite order np must be 3 (cubic); got $(spec.np)"))
    nc = spec.ncomp

    info = xmax_3level(spec, params, Q; kmax=kmax, xfac=xfac)
    cache = info.cache
    specL = _reflected_spec(spec)

    # Both halves share the same grid structure (the left in S = −X).
    wsR = _build_grid_and_workspace(spec, nx, info.xmax, info.dx1, info.dx2, pfac, cutoff, 1)
    wsL = _build_grid_and_workspace(specL, nx, info.xmax, info.dx1, info.dx2, pfac, cutoff, 1)

    # Shared center (X = 0) global DOFs: value (continuous) and derivative.
    gid = Ref(0)
    cval = Int[(gid[] += 1) for _ in 1:nc]
    cder = Int[(gid[] += 1) for _ in 1:nc]
    gR, sgnR = _half_dof_map(wsR, nc, cval, cder, 1, gid)    # right half, d/dX
    gL, sgnL = _half_dof_map(wsL, nc, cval, cder, -1, gid)   # left half, d/dS = −d/dX
    ndim_global = gid[]

    Is = Int[]
    Js = Int[]
    Vs = ComplexF64[]
    rhs = zeros(ComplexF64, ndim_global, 2)
    # Column 2 = drive right end; column 1 = drive left end.
    _assemble_half_coo!(Is, Js, Vs, rhs, wsR, spec, params, Q, cache, gR, sgnR, 2; nq=nq, tol_res=tol_res)
    _assemble_half_coo!(Is, Js, Vs, rhs, wsL, specL, params, Q, cache, gL, sgnL, 1; nq=nq, tol_res=tol_res)

    A = sparse(Is, Js, Vs, ndim_global, ndim_global)
    # Both driving columns share the factorization; solve them together.
    X = lu(A) \ rhs   # X[:,1] = response to left drive, X[:,2] = response to right drive

    leg = gL[wsL.cells[wsL.nx].emap[1]]   # left matching DOF (X = −Xmax)
    reg = gR[wsR.cells[wsR.nx].emap[1]]   # right matching DOF (X = +Xmax)
    # M[i,j]: amplitude at end i (1=left, 2=right) for drive j (1=left, 2=right).
    M = SMatrix{2,2,ComplexF64}(X[leg, 1], X[reg, 1], X[leg, 2], X[reg, 2])
    return FullDomainMatching(M, info.xmax, ndim_global)
end

solve_galerkin_full(spec::SystemSpec, params, Q::Number; kwargs...) =
    solve_galerkin_full(spec, params, ComplexF64(Q); kwargs...)
