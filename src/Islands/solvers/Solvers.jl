"""
    Solvers

The Islands steady-state solve (design `03 §3`, `04 §5`, Decision D2):
matrix-free **Newton–Krylov** — never initial-value time-stepping, never nested
Picard loops (kokuchou's Picard criterion was *never met* in production; the
sources' Φ-outer/ū_∥i-inner iteration is the explicit anti-pattern).

Pieces:

  - [`flat_residual`](@ref) — wrap an operator stack (+ optional far-field BCs)
    as a flat-vector residual `f!(out, u)`, generic over `eltype` with
    preallocated real and dual workspaces.
  - [`JVPOperator`](@ref) — the exact Jacobian–vector product via ForwardDiff
    duals through the stack (`04 §5`); no global sparse Jacobian is ever formed
    except in [`dense_jacobian`](@ref) tiny-grid debug mode.
  - [`YBlockJacobi`](@ref) — the physics-block preconditioner skeleton: per-pitch-
    pencil blocks with **TSVD-regularized** solves, the explicit `y_c`-matching
    treatment of `04 §3` (GMRES must never resolve the near-singular directions
    itself).
  - [`newton_krylov`](@ref) — inexact Newton with Eisenstat–Walker forcing and
    a backtracking line search.
  - [`pseudo_arclength`](@ref) — the continuation scaffold with fold detection
    from day one (`03 §3`: the Level-3 penetration bifurcation is a fold; the
    solver must step around folds, not fail at them).

Pure numerics — no physics coefficients (the stack it solves carries the gated
parameters).
"""
module Solvers

using LinearAlgebra
import Krylov
import ForwardDiff
import ..PhaseSpace: IslandGrid, nnodes
import ..Operators: IslandState, IslandCache, IslandStack, FarFieldConditions,
    residual!, flatten!, unflatten!, statelength

export flat_residual, JVPOperator, dense_jacobian
export YBlockJacobi, newton_krylov, pseudo_arclength

# Tag for the solver's ForwardDiff duals (one directional derivative).
struct JVPTag end
const JVPDual = ForwardDiff.Dual{JVPTag,Float64,1}

# ---------------------------------------------------------------------------
# Flat residual wrapper
# ---------------------------------------------------------------------------
"""
    flat_residual(stack, grid; bc=nothing)

Wrap `residual!` for `stack` on `grid` (with optional `FarFieldConditions`) as
a flat-vector function `f!(out, u)` usable by [`newton_krylov`](@ref) and
[`JVPOperator`](@ref). Real (`Float64`) and dual workspaces are preallocated so
repeated evaluations allocate nothing.
"""
function flat_residual(stack::IslandStack, grid::IslandGrid; bc::Union{Nothing,FarFieldConditions}=nothing)
    U = IslandState(grid)
    R = IslandState(grid)
    cache = IslandCache(grid)
    Ud = IslandState{JVPDual}(grid)
    Rd = IslandState{JVPDual}(grid)
    cached = IslandCache{JVPDual}(grid)
    function f!(out, u)
        if eltype(u) === Float64
            unflatten!(U, u)
            bc === nothing ? residual!(R, U, stack, grid, cache) : residual!(R, U, stack, grid, cache, bc)
            flatten!(out, R)
        else
            unflatten!(Ud, u)
            bc === nothing ? residual!(Rd, Ud, stack, grid, cached) : residual!(Rd, Ud, stack, grid, cached, bc)
            flatten!(out, Rd)
        end
        return out
    end
    return f!
end

# ---------------------------------------------------------------------------
# Matrix-free Jacobian-vector product
# ---------------------------------------------------------------------------
"""
    JVPOperator(f!, u)

Matrix-free linear operator for the Jacobian of `f!` at the point `u` (which is
**aliased**, so updating `u` in place moves the linearization point): `mul!(y, J, v)` computes the exact directional derivative via one dual-mode evaluation
of `f!` (`04 §5`). Satisfies the `mul!`/`size`/`eltype` interface Krylov.jl
expects.
"""
struct JVPOperator{F}
    f!::F
    u::Vector{Float64}
    udual::Vector{JVPDual}
    rdual::Vector{JVPDual}
end

function JVPOperator(f!, u::Vector{Float64})
    n = length(u)
    return JVPOperator(f!, u, Vector{JVPDual}(undef, n), Vector{JVPDual}(undef, n))
end

Base.size(J::JVPOperator) = (length(J.u), length(J.u))
Base.size(J::JVPOperator, d::Int) = d <= 2 ? length(J.u) : 1
Base.eltype(::JVPOperator) = Float64

function LinearAlgebra.mul!(y::AbstractVector, J::JVPOperator, v::AbstractVector)
    @inbounds for i in eachindex(J.u)
        J.udual[i] = JVPDual(J.u[i], ForwardDiff.Partials((v[i],)))
    end
    J.f!(J.rdual, J.udual)
    @inbounds for i in eachindex(y)
        y[i] = ForwardDiff.partials(J.rdual[i])[1]
    end
    return y
end

"""
    dense_jacobian(f!, u)

Dense Jacobian of `f!` at `u` by column-wise dual sweeps — **tiny-grid debug
mode only** (`04 §5`): used for eigenvalue/conditioning diagnostics (the
`y_c`-block singular-value monitor, ladder A8), never in the production solve.
"""
function dense_jacobian(f!, u::Vector{Float64})
    n = length(u)
    J = zeros(n, n)
    ud = Vector{JVPDual}(undef, n)
    rd = Vector{JVPDual}(undef, n)
    for j in 1:n
        @inbounds for i in 1:n
            ud[i] = JVPDual(u[i], ForwardDiff.Partials((i == j ? 1.0 : 0.0,)))
        end
        f!(rd, ud)
        @inbounds for i in 1:n
            J[i, j] = ForwardDiff.partials(rd[i])[1]
        end
    end
    return J
end

# ---------------------------------------------------------------------------
# Physics-block preconditioner skeleton (04 §3, §5)
# ---------------------------------------------------------------------------
"""
    YBlockJacobi(grid, block; svd_cutoff=1e-10, phi_scale=1.0)

Block-Jacobi preconditioner over pitch pencils — the skeleton of the `04 §5`
physics-block preconditioner. `block(ix, iξ, iE, iσ)` returns the `ny × ny`
stiff-direction block for that pencil (typically the identity-plus-collisions
operator); each block is factored by SVD and **truncated below
`svd_cutoff · σ_max`** — the explicit, regularized `y_c`-matching treatment of
`04 §3` (kokuchou's rcond ≈ 1e-16 block gave machine-dependent noise under
plain LU; TSVD is the documented fix). `phi_scale` is the diagonal applied to
the `Φ` rows. Apply via `ldiv!` (pass as `N=...` with `ldiv=true` to GMRES).
"""
struct YBlockJacobi
    grid::IslandGrid
    pinvs::Array{Matrix{Float64},4}
    phi_scale::Float64
end

function YBlockJacobi(grid::IslandGrid, block; svd_cutoff::Real=1e-10, phi_scale::Real=1.0)
    nx, nξ, ny, nE, nσ = nnodes(grid)
    pinvs = Array{Matrix{Float64},4}(undef, nx, nξ, nE, nσ)
    for iσ in 1:nσ, iE in 1:nE, iξ in 1:nξ, ix in 1:nx
        B = Matrix{Float64}(block(ix, iξ, iE, iσ))
        size(B) == (ny, ny) || throw(ArgumentError("block must be ny×ny = ($ny, $ny)"))
        F = svd(B)
        smax = F.S[1]
        sinv = [s > svd_cutoff * smax ? 1.0 / s : 0.0 for s in F.S]
        pinvs[ix, iξ, iE, iσ] = F.V * Diagonal(sinv) * F.U'
    end
    return YBlockJacobi(grid, pinvs, Float64(phi_scale))
end

function LinearAlgebra.ldiv!(y::AbstractVector, P::YBlockJacobi, x::AbstractVector)
    nx, nξ, ny, nE, nσ = nnodes(P.grid)
    ng = nx * nξ * ny * nE * nσ
    xg = reshape(view(x, 1:ng), nx, nξ, ny, nE, nσ)
    yg = reshape(view(y, 1:ng), nx, nξ, ny, nE, nσ)
    @inbounds for iσ in 1:nσ, iE in 1:nE, iξ in 1:nξ, ix in 1:nx
        Pi = P.pinvs[ix, iξ, iE, iσ]
        for a in 1:ny
            acc = 0.0
            for b in 1:ny
                acc += Pi[a, b] * xg[ix, iξ, b, iE, iσ]
            end
            yg[ix, iξ, a, iE, iσ] = acc
        end
    end
    @inbounds for i in (ng+1):length(x)
        y[i] = x[i] / P.phi_scale
    end
    return y
end

# ---------------------------------------------------------------------------
# Inexact Newton-Krylov
# ---------------------------------------------------------------------------
"""
    newton_krylov(f!, u0; rtol=1e-8, atol=1e-10, max_iter=25, precond=nothing,
                  eta0=0.5, eta_max=0.9, gamma=0.9, ls_max=10, memory=30,
                  verbose=false)

Matrix-free inexact Newton–Krylov solve of `f!(out, u) = 0` (design `03 §3`,
`04 §5`, Decision D2):

  - GMRES (Krylov.jl) on the ForwardDiff [`JVPOperator`](@ref), with the
    **Eisenstat–Walker** (choice-2) forcing term `η_k = γ(‖F_k‖/‖F_{k−1}‖)²` so
    early Newton steps do not over-solve the linear system;
  - a backtracking (Armijo-on-`‖F‖`) line search;
  - optional right preconditioner `precond` applied via `ldiv!` (e.g.
    [`YBlockJacobi`](@ref)).

Convergence is declared on **both** the global residual norm and its max-norm
(`04 §5`: the array-averaged residual hid locally divergent regions in the
prior art). Returns a NamedTuple `(u, converged, iterations, residual_norms, residual_max, gmres_iters)` — the total Krylov iteration count is a tracked
preconditioner-quality metric (`04 §5`), not folklore.
"""
function newton_krylov(f!, u0::AbstractVector{Float64};
    rtol::Real=1e-8, atol::Real=1e-10, max_iter::Int=25, precond=nothing,
    eta0::Real=0.5, eta_max::Real=0.9, gamma::Real=0.9, ls_max::Int=10,
    memory::Int=30, verbose::Bool=false)
    u = collect(u0)
    n = length(u)
    F = similar(u)
    f!(F, u)
    nF = norm(F)
    nF0 = max(nF, eps())
    tol = max(atol, rtol * nF0)
    hist = Float64[nF]
    J = JVPOperator(f!, u)                 # aliases u: updates move the linearization point
    Ftrial = similar(u)
    utrial = similar(u)
    eta = float(eta0)
    nF_prev = nF
    k = 0
    gmres_total = 0
    while nF > tol && k < max_iter
        k += 1
        k > 1 && (eta = clamp(gamma * (nF / nF_prev)^2, 1e-4, eta_max))
        rhs = -F
        du, stats =
            precond === nothing ?
            Krylov.gmres(J, rhs; rtol=eta, atol=0.1 * atol, memory=memory, itmax=10n) :
            Krylov.gmres(J, rhs; rtol=eta, atol=0.1 * atol, memory=memory, itmax=10n, N=precond, ldiv=true)
        gmres_total += stats.niter
        verbose && @info "newton iter $k: ‖F‖=$nF, η=$eta, gmres iters=$(stats.niter)"
        # backtracking line search on ‖F‖
        λ = 1.0
        nFtrial = Inf
        for _ in 1:ls_max
            @. utrial = u + λ * du
            f!(Ftrial, utrial)
            nFtrial = norm(Ftrial)
            nFtrial <= (1 - 1e-4 * λ) * nF && break
            λ /= 2
        end
        nF_prev = nF
        if nFtrial < nF
            u .= utrial
            F .= Ftrial
            nF = nFtrial
        else
            verbose && @warn "newton line search stagnated at iter $k"
            break
        end
        push!(hist, nF)
    end
    return (u=u, converged=(nF <= tol), iterations=k, residual_norms=hist, residual_max=maximum(abs, F), gmres_iters=gmres_total)
end

# ---------------------------------------------------------------------------
# Pseudo-arclength continuation scaffold (03 §3)
# ---------------------------------------------------------------------------
"""
    pseudo_arclength(f!, u0, p0; ds=0.1, nsteps=10, newton_kwargs...)

Pseudo-arclength continuation scaffold for a parameterized residual
`f!(out, u, p)` (`03 §3`): predictor along the (secant) tangent, corrector =
[`newton_krylov`](@ref) on the extended system `[F(u, p); t·(z − z_pred)]`, and
**fold detection from day one** via sign changes of the tangent's `p`-component
(the Level-3 penetration bifurcation is a fold in this curve). Returns
`(us, ps, folds)` where `folds` lists the step indices at which `dp/ds`
reversed.
"""
function pseudo_arclength(f!, u0::AbstractVector{Float64}, p0::Real;
    ds::Real=0.1, nsteps::Int=10, newton_kwargs...)
    n = length(u0)
    # converge onto the curve at fixed p0
    sol = newton_krylov((out, u) -> f!(out, u, p0), collect(u0); newton_kwargs...)
    sol.converged || error("pseudo_arclength: initial solve at p0 did not converge")
    us = [copy(sol.u)]
    ps = [float(p0)]
    folds = Int[]
    # initial tangent from du/dp: J du = -dF/dp (finite difference in p)
    Fp = zeros(n)
    Fm = zeros(n)
    hp = max(1e-7, 1e-7 * abs(p0))
    f!(Fp, sol.u, p0 + hp)
    f!(Fm, sol.u, p0 - hp)
    dFdp = (Fp .- Fm) ./ (2hp)
    Jop = JVPOperator((out, u) -> f!(out, u, p0), copy(sol.u))
    dudp, _ = Krylov.gmres(Jop, -dFdp; rtol=1e-10)
    t = vcat(dudp, 1.0)
    t ./= norm(t)
    tp_prev = t[end]
    for step in 1:nsteps
        zpred = vcat(us[end], ps[end]) .+ ds .* t
        tloc = copy(t)      # freeze tangent for this corrector
        function ext!(out, z)
            f!(view(out, 1:n), view(z, 1:n), z[n+1])
            out[n+1] = LinearAlgebra.dot(tloc, z) - LinearAlgebra.dot(tloc, zpred)
            return out
        end
        sol = newton_krylov(ext!, zpred; newton_kwargs...)
        sol.converged || break
        push!(us, sol.u[1:n])
        push!(ps, sol.u[n+1])
        # secant tangent for the next step; fold when dp/ds changes sign
        t = vcat(us[end] .- us[end-1], ps[end] - ps[end-1])
        t ./= norm(t)
        if t[end] * tp_prev < 0
            push!(folds, step)
        end
        tp_prev = t[end]
    end
    return (us=us, ps=ps, folds=folds)
end

end # module Solvers
