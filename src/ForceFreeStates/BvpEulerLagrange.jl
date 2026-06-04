"""
BvpEulerLagrange.jl — MIRK collocation boundary-value-problem path for the ideal
Euler–Lagrange fundamental matrix (issue #251).

Instead of the serial shooting integration (`eulerlagrange_integration`) or the
parallel-FM/Riccati paths, this solves the linear fundamental-matrix ODE

    Y'(ψ) = L(ψ) Y(ψ),   Y = (U₁, U₂),   Y(ψ_axis) = (0, I)

with native-complex MIRK6 collocation. The right-hand side reuses [`sing_der!`](@ref)
(ideal path); only the discretization changes (global collocation + adaptive mesh
vs. adaptive RK shooting with Gaussian reduction).

**Per-column decomposition.** The fundamental matrix has N = `numpert_total` columns,
each an independent solution `yⱼ(ψ) ∈ ℂ^{2N}` of the *same* linear operator. `sing_der!`
is column-separable (each output column depends only on the matching input column), so
each column is solved as its own 2N-dimensional `BVProblem`. This keeps the collocation
Jacobian small (2N per node instead of 2N²), sidesteps the sparse-AD-through-LAPACK
pathology of the monolithic matrix BVP, and is embarrassingly parallel across columns.

The assembled matrix is packed into an [`OdeState`](@ref) in the axis (EL) basis — the
convention [`free_run!`](@ref) and the PerturbedEquilibrium / FieldReconstruction
pipeline consume — so no downstream code changes are required. Because `wp = U₂U₁⁻¹`
and the resonant-flux projections are invariant under right-multiplication of the
fundamental matrix, any correct spanning of the regular-at-axis subspace reproduces
`et` and the resonant flux.

Stage 1 (current): single segment `[ψ_axis, ψ_lim]` with no singular surface inside the
integration domain. Stage 2 adds per-segment collocation with asymptotic-basis matching
at each `q = m/n` crossing.

Activated by `ctrl.use_bvp = true` (opt-in; takes precedence over `use_parallel` /
`use_riccati`).
"""

# MIRK promotes the mesh time to the state eltype, so for a native-complex problem f!
# receives ψ as a ComplexF64 (zero imaginary part); it can also be a ForwardDiff.Dual for
# internal bookkeeping. sing_der! needs the bare Float64 ψ: the EL operator depends on ψ
# only through the (real-argument) equilibrium splines, and the collocation Jacobian is
# taken w.r.t. the state, not the mesh time — so stripping to the real value is exact.
_psi_value(x::Real) = Float64(x)
_psi_value(x::Complex) = Float64(real(x))
_psi_value(x) = _psi_value(x.value)   # ForwardDiff.Dual fallback

# Env-gated flushed debug print to diagnose first-solve compile vs solve time.
@inline _bvpdbg(msg) = haskey(ENV, "BVP_DEBUG") && (println(stderr, "[BVPDBG] ", msg); flush(stderr))

"""
    bvp_eulerlagrange_integration(ctrl, equil, ffit, intr) -> OdeState

Solve the ideal Euler–Lagrange fundamental matrix as a set of per-column MIRK
collocation BVPs and return a populated [`OdeState`](@ref) in the axis (EL) basis.
Stage 1: asserts no singular surface lies strictly inside `[ψ_axis, ψ_lim]`.
"""
function bvp_eulerlagrange_integration(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars, intr::ForceFreeStatesInternal)

    N = intr.numpert_total
    odet = OdeState(N, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)

    # Axis initialization: sets odet.psifac (Newton on qlow), U₂ = I, ising_start.
    initialize_el_at_axis!(odet, ctrl, equil.profiles, intr)

    psi0 = odet.psifac
    psi1 = intr.psilim
    psi1 > psi0 || error("bvp_eulerlagrange_integration: empty domain ψ ∈ [$psi0, $psi1]")

    n_inside = count(s -> psi0 < s.psifac < psi1, intr.sing)
    if n_inside > 0
        error("bvp_eulerlagrange_integration: $n_inside singular surface(s) inside [ψ_axis, ψ_lim]; " *
              "multipoint crossing (Stage 2) not yet implemented. Use use_parallel/use_riccati for this case.")
    end

    # Solve all N columns over the single (no-crossing) segment, then assemble + store.
    chunk = IntegrationChunk(; psi_start=psi0, psi_end=psi1, needs_crossing=false, ising=0)
    sols = _bvp_solve_segment_columns(ctrl, equil, ffit, intr, odet, chunk)

    grid = _merge_column_grids(sols, psi0, psi1)
    params = (ctrl, equil, ffit, intr, odet, chunk)
    _materialize_odestate_columns!(odet, sols, grid, params, intr)

    # Final edge state for wp = U₂U₁⁻¹.
    _assemble_fm_at!(odet.u, sols, psi1, N)
    odet.psifac = psi1
    odet.q = equil.profiles.q_spline(psi1)

    odet.nzero = evaluate_stability_criterion!(odet, equil.profiles)

    if ctrl.verbose
        @info "BVP MIRK6 (per-column) complete: $(length(grid)) ψ-nodes over [$(round(psi0; digits=4)), $(round(psi1; digits=4))]"
    end

    return odet
end

"""
    _bvp_solve_segment_columns(ctrl, equil, ffit, intr, odet, chunk) -> Vector

Solve the N independent column BVPs `yⱼ' = L(ψ)yⱼ` over `[chunk.psi_start, chunk.psi_end]`
with axis IC `yⱼ(ψ_start) = (U₁=0, U₂=eⱼ)`, returning the vector of MIRK solution objects
(one per column). The IC for column `j` is taken from `odet.u[:, j, :]` so the same routine
serves Stage-2 segments whose start state is a post-crossing fundamental matrix.
"""
function _bvp_solve_segment_columns(ctrl::ForceFreeStatesControl, equil::Equilibrium.PlasmaEquilibrium,
    ffit::FourFitVars, intr::ForceFreeStatesInternal, odet::OdeState, chunk::IntegrationChunk)

    N = intr.numpert_total
    psi0 = chunk.psi_start
    psi1 = chunk.psi_end
    params = (ctrl, equil, ffit, intr, odet, chunk)
    dt = ctrl.bvp_dt > 0 ? ctrl.bvp_dt : (psi1 - psi0) / ctrl.bvp_init_intervals
    tol = ctrl.eulerlagrange_tolerance
    # Dense complex finite-diff Jacobians. The block-bidiagonal collocation structure is NOT
    # exploited here: the symbolic tracer cannot see through sing_der!'s LAPACK calls, and a
    # finite-diff DenseSparsityDetector proved far slower than the dense path at these sizes.
    # Dense cost is O((2N·nnodes)³); keep the per-segment mesh modest. Banded linear-solve
    # acceleration is tracked as a follow-up (see PR #256 perf notes).
    jac_alg = BVPJacobianAlgorithm(; bc_diffmode=AutoFiniteDiff(), nonbc_diffmode=AutoFiniteDiff())

    maxcol = haskey(ENV, "BVP_MAXCOL") ? min(N, parse(Int, ENV["BVP_MAXCOL"])) : N

    _bvpdbg("segment columns: N=$N domain=[$psi0,$psi1] dt=$dt — entering loop")
    sols = Vector{Any}(undef, N)
    for j in 1:N
        ic = vcat(odet.u[:, j, 1], odet.u[:, j, 2])   # 2N-vector: (U₁ col j ; U₂ col j)
        _bvpdbg("col $j START")
        tcol = @elapsed sols[j] = _bvp_solve_one_column(ic, psi0, psi1, params, N, dt, tol, jac_alg, ctrl)
        _bvpdbg("col $j DONE t=$(round(tcol; digits=2))s")
        if ctrl.verbose
            @info "BVP col $j/$N: nodes=$(length(sols[j].t)) retcode=$(sols[j].retcode) t=$(round(tcol; digits=2))s"
            flush(stderr)
        end
        if j >= maxcol
            @warn "BVP_MAXCOL=$maxcol diagnostic cap: solved $j columns, copying for the rest"
            for k in (j+1):N
                sols[k] = sols[j]
            end
            break
        end
    end
    return sols
end

"Solve a single column BVP `y' = L(ψ)y`, `y(ψ_start) = ic` (all BCs at the left end)."
function _bvp_solve_one_column(ic::Vector{ComplexF64}, psi0::Float64, psi1::Float64,
    params, N::Int, dt::Float64, tol::Float64, jac_alg, ctrl::ForceFreeStatesControl)

    # Persistent (N,N,2) scratch with only column 1 populated; sing_der! is column-separable
    # so the action on this single column is read back from slot 1.
    um = zeros(ComplexF64, N, N, 2)
    dm = zeros(ComplexF64, N, N, 2)
    function f!(du, u, _p, psi)
        @inbounds for i in 1:N
            um[i, 1, 1] = u[i]
            um[i, 1, 2] = u[N+i]
        end
        sing_der!(dm, um, params, _psi_value(psi))
        @inbounds for i in 1:N
            du[i] = dm[i, 1, 1]
            du[N+i] = dm[i, 1, 2]
        end
        return nothing
    end
    function bc!(res, u, _p, _t)
        uL = u(psi0)
        @inbounds @. res = uL - ic
        return nothing
    end

    prob = BVProblem(f!, bc!, copy(ic), (psi0, psi1))
    alg = MIRK6(; jac_alg=jac_alg, max_num_subintervals=ctrl.bvp_max_intervals)
    _bvpdbg("  solve() START dt=$dt adaptive=$(ctrl.bvp_adaptive) tol=$tol")
    sol = solve(prob, alg; dt=dt, abstol=tol, reltol=tol, adaptive=ctrl.bvp_adaptive, maxiters=ctrl.bvp_maxiters)
    _bvpdbg("  solve() END nodes=$(length(sol.t)) retcode=$(sol.retcode)")
    if Symbol(sol.retcode) != :Success
        @warn "BVP column solve retcode $(sol.retcode) on [$psi0, $psi1]; result may be inaccurate"
    end
    return sol
end

"Merged, sorted, unique ψ-grid spanning all column collocation meshes (captures every column's refinement)."
function _merge_column_grids(sols, psi0::Float64, psi1::Float64)
    g = Float64[]
    for s in sols
        for t in s.t
            push!(g, _psi_value(t))   # sol.t may be ComplexF64 for a native-complex BVP
        end
    end
    push!(g, psi0); push!(g, psi1)
    sort!(g)
    unique!(g)
    return g
end

"Assemble the (N,N,2) fundamental matrix at ψ from the per-column interpolants into `dest`."
function _assemble_fm_at!(dest::Array{ComplexF64,3}, sols, psi::Float64, N::Int)
    @inbounds for j in 1:N
        yj = sols[j](psi)
        for i in 1:N
            dest[i, j, 1] = yj[i]
            dest[i, j, 2] = yj[N+i]
        end
    end
    return dest
end

"""
    _materialize_odestate_columns!(odet, sols, grid, params, intr)

Populate `odet.u_store`, `odet.ud_store`, `odet.psi_store`, `odet.q_store`, and `odet.step`
from the per-column solutions evaluated on the common `grid`. At each node the derivative
arrays (`ud[:,:,1] = Ξ'_Ψ`, `ud[:,:,2] = Ξ_s`) are recomputed by [`sing_der!`](@ref) on the
assembled matrix, matching the shooting path's save contract.
"""
function _materialize_odestate_columns!(odet::OdeState, sols, grid::Vector{Float64}, params, intr::ForceFreeStatesInternal)
    N = intr.numpert_total
    ng = length(grid)
    while size(odet.u_store, 4) < ng
        resize_storage!(odet)
    end

    um = zeros(ComplexF64, N, N, 2)
    du = zeros(ComplexF64, N, N, 2)
    @inbounds for k in 1:ng
        psi = grid[k]
        _assemble_fm_at!(um, sols, psi, N)
        sing_der!(du, um, params, psi)          # fills odet.ud and sets odet.q = q(psi)
        odet.psi_store[k] = psi
        odet.q_store[k] = odet.q
        @views odet.u_store[:, :, :, k] .= um
        @views odet.ud_store[:, :, :, k] .= odet.ud
    end

    odet.step = ng
    trim_storage!(odet)
    return odet
end
