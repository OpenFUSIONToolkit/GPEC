# ForceBalance.jl: the force-balance problem and its solve. The solve prepares the equilibrium with the Tearing
# layer-overlap cap and the KineticForces model, so it lives with the top module rather than in ForceFreeStates.
#
# Included directly into the GeneralizedPerturbedEquilibrium module.

using .ForceFreeStates: ForceFreeStatesInternal, ForceFreeStatesControl, ForceFreeStatesResult
using .ForceFreeStates: sing_lim!, sing_min!, sing_find!, remove_singular_surfs!
using .ForceFreeStates: make_metric, build_matrix_splines, build_kinetic_matrix_splines, find_kinetic_singular_surfaces!
using .ForceFreeStates: KineticMatrices, evaluate_kinetic_matrices, check_kinetic_matrices, kinetic_source
using .ForceFreeStates: resolve_mode_space!, run_force_free_states, _mode_range_label

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
function solve(prob::ForceBalanceProblem, alg::ForceFreeStates.AbstractIntegrator; verbose::Bool=false)
    total_start = time()
    ctrl, intr, equil, metric, mats, layer_overlap = _prepare_solve(prob, alg; verbose)
    result = run_force_free_states(ctrl, equil, mats, intr, metric; cut_solution=alg isa ForceFreeStates.Galerkin && alg.cut_solution)
    result = ForceFreeStates._with(result; problem=prob, algorithm=alg, layer_overlap=layer_overlap)

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
function kinetic_matrices(prob::ForceBalanceProblem, alg::ForceFreeStates.Forward=ForceFreeStates.Forward(); verbose::Bool=false)
    get(prob.ctrl_kwargs, :kinetic, nothing) === nothing &&
        error("kinetic_matrices: the problem has no kinetic source; pass kinetic=KineticForcesControl(...) or TestKineticMatrices(; scale)")
    return last(_prepare_solve(prob, alg; verbose))
end

# Everything solve does before the integration: the control, the mode space, the (re-formed) equilibrium, the metric,
# the matrices, the layer-overlap scan and the kinetic matrices.
function _prepare_solve(prob::ForceBalanceProblem, alg::ForceFreeStates.AbstractIntegrator; verbose::Bool=false)
    equil = prob.equil
    ctrl_kwargs = copy(prob.ctrl_kwargs)
    ctrl = ForceFreeStatesControl(; algorithm=alg, ctrl_kwargs...)
    ForceFreeStates.validate_algorithm(ctrl)

    source = kinetic_source(ctrl.kinetic)
    ctrl.kinetic isa KineticForces.KineticForcesControl && equil.kinetic === nothing &&
        error("the KineticForces kinetic source needs kinetic profiles on the equilibrium: set kinetic_file or call attach_kinetic_profiles!")
    ctrl.kinetic !== nothing && !(alg isa ForceFreeStates.Forward) &&
        error("$(nameof(typeof(alg)))() does not support kinetic runs; use Forward().")

    # GPEC differentiates the spline of F, so single-precision FPOL rounding becomes noise in Δ′.
    eq_config = equil.config
    if eq_config.eq_type in Equilibrium.EFIT_KINDS && alg isa Union{ForceFreeStates.Riccati,ForceFreeStates.Galerkin} &&
       isfile(eq_config.eq_filename) && Equilibrium.gfile_fpol_single_precision(eq_config.eq_filename)
        @warn "The g-file's FPOL is single precision. Its rounding puts noise of order ±2–3 on Δ′ at 257 radial " *
              "nodes; regenerate the file in double precision for converged Δ′." file = eq_config.eq_filename
    end

    intr = ForceFreeStatesInternal(; verbose)
    intr.wall_settings = prob.wall

    resolve_mode_space!(intr, ctrl)

    # The kinetic model's control, when there is one, sets the views of the attached kinetic profiles.
    kf_ctrl = source isa KineticForces.KineticForcesControl ? source : KineticForces.KineticForcesControl()

    equil = maybe_reform_equilibrium(equil, intr, ctrl; kf_ctrl)

    metric, mats, layer_overlap, km = prepare_force_free_states!(intr, ctrl, equil)
    return ctrl, intr, equil, metric, mats, layer_overlap, km
end

"""
    solve(equil::PlasmaEquilibrium, alg; nn, verbose=false, kwargs...) -> ForceFreeStatesResult

Convenience form of [`solve`](@ref): builds the [`ForceBalanceProblem`](@ref) from the
equilibrium and the problem keywords, then solves it with `alg`.
"""
solve(equil::Equilibrium.PlasmaEquilibrium, alg::ForceFreeStates.AbstractIntegrator; verbose::Bool=false, kwargs...) =
    solve(ForceBalanceProblem(equil; kwargs...), alg; verbose)

"""
    maybe_reform_equilibrium(equil, intr, ctrl; kf_ctrl=KineticForcesControl()) -> equil

Two-pass auto grid: measure the pass-1 equilibrium's curvature (profiles, geometry, and the NTV
view of its attached kinetic profiles), pin knots on rational surfaces, and re-form on the refined
grid with `Equilibrium.reform` from the source the equilibrium remembers. Returns `equil` untouched
when its configuration wants a single pass.
"""
function maybe_reform_equilibrium(
    equil::Equilibrium.PlasmaEquilibrium,
    intr::ForceFreeStatesInternal,
    ctrl::ForceFreeStatesControl;
    kf_ctrl::KineticForces.KineticForcesControl=KineticForces.KineticForcesControl()
)
    eq_config = equil.config
    Equilibrium.wants_two_pass(eq_config) || return equil
    kinetic_profiles = KineticForces.ntv_profiles(kf_ctrl, equil)

    mandatory = ForceFreeStates.rational_psi_nodes(equil; nlow=intr.nlow, nhigh=intr.nhigh)
    # Kinetic-resonance nodes are located here rather than at the call site so the validity
    # boundary that filters them is cleared against this pass's rationals (intr.sing is empty yet).
    pinned = KineticForces.resonance_grid_nodes(ctrl, kf_ctrl, kinetic_profiles, KineticForces.ntv_species(kf_ctrl, equil), equil, intr;
        rationals=mandatory)
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
    overlap = cap === false ? nothing : Tearing.layer_overlap_scan(equil, cap === true ? SLAYERModel() : cap, intr.nlow:intr.nhigh)

    # psilim and qlim, the integration limits
    sing_lim!(intr, ctrl, equil; psilim_cap=overlap === nothing ? nothing : overlap.psihigh)

    sing_find!(intr, equil)

    # Lower bound from qlow for every integrator, then keep only surfaces inside [psilow, psilim],
    # so the Δ' BVP sees only crossable ones.
    sing_min!(intr, ctrl, equil)
    remove_singular_surfs!(intr; psimin=intr.psilow)

    # GGJ geometric coefficients (E, F, G, H, K, M) of each surviving singular surface; `ggj_parameters` adds τ_A / τ_R.
    if intr.msing > 0
        ForceFreeStates.resist_eval_all!(intr, equil)
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
    elseif ctrl.kinetic isa ForceFreeStates.TestKineticMatrices
        intr.verbose && @info "Computing kinetic matrices (synthetic test matrices, scale $(ctrl.kinetic.scale))"
        km = evaluate_kinetic_matrices(ctrl, equil, mats, intr, metric)
    elseif ctrl.kinetic !== nothing
        intr.verbose && @info "Computing kinetic matrices (KineticForces model)"
        # Pass KineticForces in as a callback: ForceFreeStates loads first and cannot import it.
        kf_ctrl = ctrl.kinetic
        kinetic_profiles = KineticForces.ntv_profiles(kf_ctrl, equil)
        species = KineticForces.ntv_species(kf_ctrl, equil)
        # One boundary for the whole run: the widest-orbit species, moved clear of any rational
        # surface whose window the transition band would otherwise cut through.
        axis_psi_c = KineticForces.axis_validity_boundary(kf_ctrl, species, kinetic_profiles, equil,
            Float64[sng.psifac for sng in intr.sing])
        calculated_cb =
            (c, e, i, m, f; psis::Vector{Float64}=Float64[]) ->
                KineticForces.compute_calculated_kinetic_matrices(
                    c, e, i, m, f;
                    kf_ctrl=kf_ctrl, kinetic_profiles=kinetic_profiles, species=species, psis=psis,
                    axis_psi_c=axis_psi_c)
        km = evaluate_kinetic_matrices(ctrl, equil, mats, intr, metric; calculated_source=calculated_cb, axis_validity_psi_c=axis_psi_c)
    end
    km === nothing || (mats = build_kinetic_matrix_splines(mats, equil, intr, metric, km, ctrl.kinetic_factor))

    # Kinetically displaced singular surfaces (zeros of det(F̄)) for ODE crossings; singfac_min == 0 keeps a single chunk.
    if ctrl.kinetic !== nothing && ctrl.singfac_min > 0
        find_kinetic_singular_surfaces!(mats, equil, intr)
    end

    return metric, mats, overlap, km
end
