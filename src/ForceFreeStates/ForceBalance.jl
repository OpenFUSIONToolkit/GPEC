# ForceBalance.jl: the force-balance problem and its solve.

# The solve's layer-overlap cap and kinetic model come from Tearing and KineticForces, which load after
# ForceFreeStates, so they reach it through these functions: Tearing adds the `layer_overlap_scan` method for a
# `SLAYERModel`, and KineticForces the kinetic-model methods.
"""
    layer_overlap_scan(equil, model, ns)

The binding resistive-layer overlap scan over the toroidal mode numbers `ns`, with `model`'s layer closures; its
`psihigh` caps the domain. Methods live in `Tearing`.
"""
function layer_overlap_scan end

"""
    kinetic_grid_inputs(source, ctrl, equil, intr; rationals) -> (kinetic_profiles, pinned)

The kinetic profiles the two-pass ψ grid refines on and the kinetic-resonance surfaces it pins, for the kinetic
`source` of the solve (any source, or `nothing`). The method lives in `KineticForces`.
"""
function kinetic_grid_inputs end

"""
    kinetic_model_source(model, equil, intr) -> (calculated_source, axis_validity_psi_c)

The callback that evaluates a kinetic model's matrices for [`evaluate_kinetic_matrices`](@ref), and the model's
axis-validity boundary. The method for `KineticForcesControl` lives in `KineticForces`.
"""
function kinetic_model_source end

"""
    ForceBalanceProblem(equil; nn, wall=nothing, kwargs...)

The perturbed force-balance problem posed on an equilibrium, whose solutions are the force-free
(or, with a `kinetic` source, kinetic) perturbed states. With kinetic terms the force operator
is not self-adjoint, so this is not always an energy extremization. The integrator passed to
[`solve`](@ref) sets the numerics; inner-layer matching is a separate `MatchProblem` on the finished solve.

`nn` is the toroidal mode number or range and `wall` the vacuum wall shape. Remaining keywords describe physics and workflow controls;
numerical settings belong on the integrator passed to `solve`. `nn_low`/`nn_high` are rejected — they
come from `nn`.

## Fields

  - `equil::Equilibrium.PlasmaEquilibrium` - The equilibrium the problem is posed on.
  - `wall::Union{Nothing,Vacuum.WallSettings}` - Vacuum wall shape for the free-boundary energies.
  - `ctrl_kwargs::Dict{Symbol,Any}` - `ForceFreeStatesControl` keywords, `nn` already folded in.
"""
struct ForceBalanceProblem
    equil::Equilibrium.PlasmaEquilibrium
    wall::Union{Nothing,Vacuum.WallSettings}
    ctrl_kwargs::Dict{Symbol,Any}
end

function ForceBalanceProblem(
    equil::Equilibrium.PlasmaEquilibrium;
    nn::Union{Int,AbstractUnitRange{Int}},
    wall::Union{Nothing,Vacuum.WallSettings}=nothing,
    kwargs...
)
    ctrl_kwargs = Dict{Symbol,Any}(kwargs)
    allowed = setdiff(fieldnames(ForceFreeStatesControl), (:algorithm,))
    for name in keys(ctrl_kwargs)
        name in allowed || throw(ArgumentError("`$name` is not a ForceBalanceProblem keyword; numerical settings belong on the integrator."))
    end
    (haskey(ctrl_kwargs, :nn_low) || haskey(ctrl_kwargs, :nn_high)) &&
        error("the toroidal mode range comes from the `nn` keyword; drop nn_low/nn_high")
    ctrl_kwargs[:nn_low] = first(nn)
    ctrl_kwargs[:nn_high] = last(nn)
    return ForceBalanceProblem(equil, wall, ctrl_kwargs)
end

"""
    solve(prob::ForceBalanceProblem, alg; verbose=false) -> ForceFreeStatesResult
    solve(equil, alg; nn, verbose=false, kwargs...) -> ForceFreeStatesResult

Solve the perturbed-plasma [`ForceBalanceProblem`](@ref) with the formalism `alg` —
[`Forward`](@ref), [`Riccati`](@ref) or [`Galerkin`](@ref) — and return the published
[`ForceFreeStatesResult`](@ref). This is the scripting entry point; it runs the same stages
a `gpec.toml` run of `main` does and produces the same result object. The second form is
sugar building the problem from an equilibrium and the problem keywords in one call.

Knobs owned by `alg` are rejected as `ForceFreeStatesControl` keywords. Kinetic runs
(a `kinetic` source) take `Forward()`. `kinetic=KineticForcesControl(; ...)` evaluates the
KineticForces model with those settings on the equilibrium's kinetic profiles (`kinetic_file`, or
`attach_kinetic_profiles!`); `kinetic=TestKineticMatrices(; scale)` uses synthetic test matrices; `kinetic=km` reuses
the [`KineticMatrices`](@ref) from [`kinetic_matrices`](@ref).

```julia
eq = PlasmaEquilibrium("input.geqdsk"; jac_type="hamada")
prob = ForceBalanceProblem(eq; nn=1, delta_mlow=8, delta_mhigh=8, vac_flag=true)
ffs = solve(prob, Riccati())
ffs = solve(eq, Riccati(); nn=1, vac_flag=true)   # equivalent one-line form
```
"""
function CommonSolve.solve(prob::ForceBalanceProblem, alg::AbstractIntegrator; verbose::Bool=false)
    total_start = time()
    ctrl, intr, equil, metric, mats, layer_overlap = _prepare_solve(prob, alg; verbose)
    result = run_force_free_states(ctrl, equil, mats, intr, metric; cut_solution=alg isa Galerkin && alg.cut_solution)
    result = _with(result; problem=prob, algorithm=alg, layer_overlap=layer_overlap)

    @info "Force-Free States completed in $(@sprintf("%.3f", time() - total_start)) s"

    return result
end

"""
    kinetic_matrices(prob::ForceBalanceProblem, alg=Forward(); verbose=false) -> KineticMatrices

Evaluate the kinetic matrices of `prob`'s kinetic source once, running the part of [`solve`](@ref) before the
Euler-Lagrange integration. Pass them back as the problem's `kinetic` to solve without evaluating them again: the solve
is identical to one with the source itself, and may change `kinetic_factor`.

```julia
km = kinetic_matrices(ForceBalanceProblem(eq; nn=1, kinetic=KineticForcesControl()))
scan = [solve(ForceBalanceProblem(eq; nn=1, kinetic=km, kinetic_factor=f), Forward()) for f in (0.5, 1.0, 2.0)]
```
"""
function kinetic_matrices(prob::ForceBalanceProblem, alg::Forward=Forward(); verbose::Bool=false)
    get(prob.ctrl_kwargs, :kinetic, nothing) === nothing &&
        error("kinetic_matrices: the problem has no kinetic source; pass kinetic=KineticForcesControl(...) or TestKineticMatrices(; scale)")
    return last(_prepare_solve(prob, alg; verbose))
end

# Everything solve does before the integration: the control, the mode space, the (re-formed) equilibrium, the metric,
# the matrices, the layer-overlap scan and the kinetic matrices.
function _prepare_solve(prob::ForceBalanceProblem, alg::AbstractIntegrator; verbose::Bool=false)
    equil = prob.equil
    ctrl_kwargs = copy(prob.ctrl_kwargs)
    ctrl = ForceFreeStatesControl(; algorithm=alg, ctrl_kwargs...)
    validate_algorithm(ctrl)

    ctrl.kinetic === nothing && ctrl.kinetic_factor != 1 &&
        error("kinetic_factor scales the kinetic matrices, but no kinetic source is set: pass kinetic=... " *
              "([ForceFreeStates] kinetic = \"KineticForces\")")
    # A kinetic model (not supplied or synthetic matrices) is evaluated on the equilibrium's kinetic profiles.
    ctrl.kinetic === nothing || ctrl.kinetic isa Union{KineticMatrices,TestKineticMatrices} || equil.kinetic !== nothing ||
        error("the KineticForces kinetic source needs kinetic profiles on the equilibrium: set kinetic_file or call attach_kinetic_profiles!")
    ctrl.kinetic isa Union{Nothing,KineticMatrices,TestKineticMatrices} || Equilibrium.has_ntv_fields(equil.kinetic) ||
        error("the KineticForces kinetic source needs n_e, T_i, T_e and omega_E in the kinetic profiles; the kinetic file lacks at least one")
    ctrl.kinetic !== nothing && !(alg isa Forward) &&
        error("$(nameof(typeof(alg)))() does not support kinetic runs; use Forward().")

    # GPEC differentiates the spline of F, so single-precision FPOL rounding becomes noise in Δ′.
    eq_config = equil.config
    if eq_config.eq_type in Equilibrium.EFIT_KINDS && alg isa Union{Riccati,Galerkin} &&
       isfile(eq_config.eq_filename) && Equilibrium.gfile_fpol_single_precision(eq_config.eq_filename)
        @warn "The g-file's FPOL is single precision. Its rounding puts noise of order ±2–3 on Δ′ at 257 radial " *
              "nodes; regenerate the file in double precision for converged Δ′." file = eq_config.eq_filename
    end

    intr = ForceFreeStatesInternal(; verbose)
    intr.wall_settings = prob.wall

    resolve_mode_space!(intr, ctrl)

    equil = maybe_reform_equilibrium(equil, intr, ctrl)

    metric, mats, layer_overlap, km = prepare_force_free_states!(intr, ctrl, equil)
    return ctrl, intr, equil, metric, mats, layer_overlap, km
end

"""
    solve(equil::PlasmaEquilibrium, alg; nn, verbose=false, kwargs...) -> ForceFreeStatesResult

Convenience form of [`solve`](@ref): builds the [`ForceBalanceProblem`](@ref) from the
equilibrium and the problem keywords, then solves it with `alg`.
"""
CommonSolve.solve(equil::Equilibrium.PlasmaEquilibrium, alg::AbstractIntegrator; verbose::Bool=false, kwargs...) =
    CommonSolve.solve(ForceBalanceProblem(equil; kwargs...), alg; verbose)

"""
    maybe_reform_equilibrium(equil, intr, ctrl) -> equil

Two-pass auto grid: measure the pass-1 equilibrium's curvature (profiles, geometry, and the kinetic
model's view of its attached kinetic profiles), pin knots on rational surfaces, and re-form on the refined
grid with `Equilibrium.reform` from the source the equilibrium remembers. Returns `equil` untouched
when its configuration wants a single pass.
"""
function maybe_reform_equilibrium(
    equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal,
    ctrl::ForceFreeStatesControl
)
    eq_config = equil.config
    Equilibrium.wants_two_pass(eq_config) || return equil

    mandatory = rational_psi_nodes(equil; nlow=intr.nlow, nhigh=intr.nhigh)
    # Kinetic-resonance nodes are located here rather than at the call site so the validity
    # boundary that filters them is cleared against this pass's rationals (intr.sing is empty yet).
    kinetic_profiles, pinned = kinetic_grid_inputs(kinetic_source(ctrl.kinetic), ctrl, equil, intr; rationals=mandatory)
    # Smallest |n| in the run sets the widest matching half-stencil dpsi = singfac_min/(n_min·|q′|),
    # so the rational-surface brackets clear a zone large enough for every mode.
    n_min = minimum(abs(n) for n in intr.nlow:intr.nhigh if n != 0)
    isempty(pinned) ||
        @info "Pinning $(length(pinned)) kinetic-resonance surfaces into the ψ grid: $(round.(sort(pinned); digits=3))"
    psi_nodes = Equilibrium.refined_psi_grid(equil;
        tau=eq_config.psi_accuracy, kin=kinetic_profiles, mandatory=mandatory, pinned=pinned,
        singfac_min=ctrl.singfac_min, n_min=n_min)
    equil = Equilibrium.reform(equil, psi_nodes)
    implied = Equilibrium.implied_knot_count(equil; tau=eq_config.psi_accuracy, kin=kinetic_profiles)
    if implied > 1.5 * (length(psi_nodes) - 1)
        @warn "Two-pass psi grid: refined equilibrium implies $implied knots vs $(length(psi_nodes) - 1) used — " *
              "pass 1 may have under-sampled a feature; consider tightening psi_accuracy"
    end
    @info "Two-pass psi grid: $(length(psi_nodes)) knots, $(length(mandatory)) rational surfaces pinned (n=$(_mode_range_label(intr)))"

    return equil
end

"""
    prepare_force_free_states!(intr, ctrl, equil) -> (metric, mats, overlap, km)

Set up the force-free-states solve on `intr`: integration limits, the surviving singular
surfaces and their GGJ coefficients, the poloidal mode range, and the metric plus
Euler-Lagrange (and, when requested, kinetic) matrices. `km` is the run's [`KineticMatrices`](@ref), or `nothing`.
"""
function prepare_force_free_states!(
    intr::ForceFreeStatesInternal,
    ctrl::ForceFreeStatesControl,
    equil::Equilibrium.PlasmaEquilibrium
)
    # Resistive-layer overlap: where adjacent rational surfaces' layers run into each other bounds the domain, when asked.
    cap = ctrl.psilim_from_layer_overlap
    overlap = cap === false ? nothing : layer_overlap_scan(equil, cap === true ? InnerLayer.SLAYERModel() : cap, intr.nlow:intr.nhigh)

    # psilim and qlim, the integration limits
    sing_lim!(intr, ctrl, equil; psilim_cap=overlap === nothing ? nothing : overlap.psihigh)

    sing_find!(intr, equil)

    # Lower bound from qlow for every integrator, then keep only surfaces inside [psilow, psilim],
    # so the Δ' BVP sees only crossable ones.
    sing_min!(intr, ctrl, equil)
    remove_singular_surfs!(intr; psimin=intr.psilow)

    # GGJ geometric coefficients (E, F, G, H, K, M) of each surviving singular surface; `ggj_parameters` adds τ_A / τ_R.
    if intr.msing > 0
        resist_eval_all!(intr, equil)
    end

    # Poloidal mode range; delta_mhigh is doubled to match Fortran.
    delta_mhigh = 2 * ctrl.delta_mhigh
    if ctrl.delta_mlow < 0 || ctrl.delta_mhigh < 0
        error("Negative delta_mlow or delta_mhigh not allowed")
    end
    intr.mlow = trunc(Int, min(intr.nlow * equil.params.qmin, 0)) - 4 - ctrl.delta_mlow
    intr.mhigh = trunc(Int, intr.nhigh * equil.params.qmax) + delta_mhigh
    intr.mpert = intr.mhigh - intr.mlow + 1
    intr.numpert_total = intr.mpert * intr.npert

    if intr.verbose
        @info "Run parameters:\n" *
              "   q0 = $(@sprintf("%.3f", equil.params.q0)), qmin = $(@sprintf("%.3f", equil.params.qmin)), qmax = $(@sprintf("%.3f", equil.params.qmax)), q95 = $(@sprintf("%.3f", equil.params.q95))\n" *
              "   qlim = $(@sprintf("%.3f", intr.qlim)), psilim = $(@sprintf("%.3f", intr.psilim))\n" *
              "   betat = $(@sprintf("%.3f", equil.params.betat)), betan = $(@sprintf("%.3f", equil.params.betan)), betap1 = $(@sprintf("%.3f", equil.params.betap1))\n" *
              "   mlow = $(@sprintf("%4i", intr.mlow)), mhigh = $(@sprintf("%4i", intr.mhigh)), mpert = $(@sprintf("%4i", intr.mpert))\n" *
              "   nlow = $(@sprintf("%4i", intr.nlow)), nhigh = $(@sprintf("%4i", intr.nhigh)), npert = $(@sprintf("%4i", intr.npert))"
    end

    metric = make_metric(equil, intr.mpert)

    if intr.verbose
        @info "Computing F, G, and K matrices"
    end

    mats = build_matrix_splines(equil, intr, metric)

    km = nothing
    if ctrl.kinetic isa KineticMatrices
        check_kinetic_matrices(ctrl.kinetic, intr, metric)
        km = ctrl.kinetic
    elseif ctrl.kinetic isa TestKineticMatrices
        intr.verbose && @info "Computing kinetic matrices (synthetic test matrices, scale $(ctrl.kinetic.scale))"
        km = evaluate_kinetic_matrices(ctrl, equil, mats, intr, metric)
    elseif ctrl.kinetic !== nothing
        intr.verbose && @info "Computing kinetic matrices (KineticForces model)"
        calculated_cb, axis_psi_c = kinetic_model_source(ctrl.kinetic, equil, intr)
        km = evaluate_kinetic_matrices(ctrl, equil, mats, intr, metric; calculated_source=calculated_cb, axis_validity_psi_c=axis_psi_c)
    end
    km === nothing || (mats = build_kinetic_matrix_splines(mats, equil, intr, metric, km, ctrl.kinetic_factor))

    # Kinetically displaced singular surfaces (zeros of det(F̄)) for ODE crossings; singfac_min == 0 keeps a single chunk.
    if ctrl.kinetic !== nothing && ctrl.singfac_min > 0
        find_kinetic_singular_surfaces!(mats, equil, intr)
    end

    return metric, mats, overlap, km
end
