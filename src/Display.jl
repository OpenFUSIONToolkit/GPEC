# Concise REPL display for the published result types: a one-line `show(io, x)` used inside
# containers (the named tuple `main` returns, nested fields) and a short labelled `text/plain`
# summary. Arrays are summarized by shape or range and never printed.
#
# Included directly into the GeneralizedPerturbedEquilibrium module, after every submodule.

using .ForceFreeStates: SolutionProfiles, FreeBoundaryResult, DeltaPrimeData, GalerkinResult, MatchResult, SingType, OdeState
using .PerturbedEquilibrium: PerturbedEquilibriumState
using .KineticForces: MethodResult, KineticForcesState
using .ErrorFields: MonteCarloResult, RiskResult, EFCCoupling
using .Tearing: GrowthRateResult

_fmt(x::Real) = @sprintf("%.4g", x)
_fmt(::Nothing) = "—"
_fmt(z::Complex) = iszero(imag(z)) ? _fmt(real(z)) : string(_fmt(real(z)), imag(z) < 0 ? " - " : " + ", _fmt(abs(imag(z))), "im")

_span(v) = isempty(v) ? "empty" : string("[", _fmt(minimum(v)), ", ", _fmt(maximum(v)), "]")
# "name ∈ [a, b]", or "name = a" when the values are a single point.
_in(name, v) = !isempty(v) && minimum(v) == maximum(v) ? "$name = $(_fmt(first(v)))" : "$name ∈ $(_span(v))"
_shape(a) = join(size(a), "×")
_plural(k, word) = string(k, " ", word, k == 1 ? "" : "s")

# Comma-joined items, truncated after `max` with a count of the rest.
function _list(items; max::Int=6)
    s = join(Iterators.take(items, max), ", ")
    extra = length(items) - max
    return extra > 0 ? string(s, ", … ", extra, " more") : s
end

_row(io, label, value) = print(io, "\n  ", rpad(label, 17), value)

# Range of `ψ` where `mask` holds, as "k/N points, ψ ∈ [a, b]"; "none" when nowhere.
function _where(psi, mask)
    k = count(mask)
    k == 0 && return "none"
    return string(k, "/", length(mask), " points, ", _in("ψ", psi[mask]))
end

# Equilibrium

function _eq_source(eq::PlasmaEquilibrium)
    c = eq.config
    eq.analytic === nothing || return "analytic $(c.eq_type)"
    return string(c.eq_type, " ", basename(c.eq_filename))
end

const _KINETIC_PROFILES = (:n_i, :n_e, :T_i, :T_e, :omega_E, :omega_tor, :chi_e, :chi_i, :chi_phi, :zeff, :nu_i, :nu_e)
_kinetic_names(k::KineticProfileData) = [String(f) for f in _KINETIC_PROFILES if getfield(k, f) !== nothing]

function Base.show(io::IO, eq::PlasmaEquilibrium)
    p = eq.params
    return print(io, "PlasmaEquilibrium(", _eq_source(eq), ", q95=", _fmt(p.q95), ", βN=", _fmt(p.betan), ")")
end

function Base.show(io::IO, ::MIME"text/plain", eq::PlasmaEquilibrium)
    p = eq.params
    print(io, "PlasmaEquilibrium  ", _eq_source(eq))
    _row(io, "Safety factor", "q₀ = $(_fmt(p.q0)), q_min = $(_fmt(p.qmin)), q₉₅ = $(_fmt(p.q95)), q_edge = $(_fmt(p.qa))")
    betat = p.betat === nothing ? "—" : _fmt(100 * p.betat) * " %"
    _row(io, "Beta", "β_N = $(_fmt(p.betan)), β_t = $betat, β_p = $(_fmt(p.betap1)), l_i = $(_fmt(p.li1))")
    _row(io, "Field/current", "B_T = $(_fmt(p.bt0)) T, I_p = $(_fmt(p.crnt)) MA")
    _row(io, "Shape", "R = $(_fmt(p.rmean)) m, a = $(_fmt(p.amean)) m, κ = $(_fmt(p.kappa)), δ = $(_fmt(p.delta1)) / $(_fmt(p.delta2))")
    _row(io, "Grid", "$(length(eq.rzphi_xs)) ψ × $(length(eq.rzphi_ys)) θ, ψ ∈ $(_span(eq.rzphi_xs)), $(eq.config.jac_type) coordinates")
    _row(io, "Kinetic", eq.kinetic === nothing ? "no profiles attached" : _list(_kinetic_names(eq.kinetic); max=12))
    return nothing
end

function Base.show(io::IO, k::KineticProfileData)
    return print(io, "KineticProfileData(", length(k.psi), " ψ points; ", _list(_kinetic_names(k); max=12), ")")
end

# LocalStability

_ballooning_margin(r::LocalStability.LocalStabilityResult) =
    filter(isfinite, r.alpha_critical ./ r.alpha)

function _ballooning_note(r::LocalStability.LocalStabilityResult)
    isempty(r.ballooning_psi) && return "α-boundary scan not run"
    window = r.ballooning_psi[isfinite.(r.alpha)]
    isempty(window) && return "α-boundary scan window held no points"
    margin = _ballooning_margin(r)
    isempty(margin) && return "no α_crit found on $(length(window)) scan points ($(_in("ψ", window)))"
    return "min α_crit/α = $(_fmt(minimum(margin))) over $(length(window)) scan points ($(_in("ψ", window)))"
end

function Base.show(io::IO, r::LocalStability.LocalStabilityResult)
    bal = isempty(r.ballooning_psi) ? "no α scan" : isempty(_ballooning_margin(r)) ? "α scan (no boundary)" : "α scan"
    return print(io, "LocalStabilityResult(", length(r.psi), " ψ, Mercier unstable ", count(>(0), r.D_I), ", ", bal, ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::LocalStability.LocalStabilityResult)
    print(io, "LocalStabilityResult  ", length(r.psi), " ψ points ∈ ", _span(r.psi))
    _row(io, "Mercier", "D_I > 0 (unstable): " * _where(r.psi, r.D_I .> 0))
    _row(io, "Resistive", "D_R > 0 (unstable): " * _where(r.psi, r.D_R .> 0))
    _row(io, "Ballooning", _ballooning_note(r))
    return nothing
end

# ForceFreeStates

_surface_label(s::SingType) = string(s.m[1], "/", s.n[1])

function Base.show(io::IO, s::SingType)
    return print(io, "SingType(m/n = ", _surface_label(s), ", ψ = ", _fmt(s.psifac), ", q = ", _fmt(s.q), ")")
end

function Base.show(io::IO, s::SolutionProfiles)
    return print(io, "SolutionProfiles(", s.basis, ", ", s.step, " ψ nodes ∈ ", _span(s.psi_store), ", ξ ", _shape(s.u_store), ")")
end

function Base.show(io::IO, fb::FreeBoundaryResult)
    return print(io, "FreeBoundaryResult(et₁ = ", _fmt(real(first(fb.et))), ", ", _plural(length(fb.et), "eigenmode"), ")")
end

function Base.show(io::IO, d::DeltaPrimeData)
    source = d.A === nothing ? "Riccati" : "Galerkin"
    return print(io, "DeltaPrimeData(", source, ", ", _shape(d.matrix), ")")
end

function Base.show(io::IO, g::GalerkinResult)
    return print(io, "GalerkinResult(", _plural(g.msing, "surface"), ", solver = ", g.solver, ")")
end

function Base.show(io::IO, o::OdeState)
    return print(io, "OdeState(", o.step, " stored steps, ", o.nzero, " fixed-boundary zero crossings)")
end

_integrator_name(r::ForceFreeStates.ForceFreeStatesResult) = nameof(typeof(r.algorithm))

_wall_label(::Nothing) = "none"
_wall_label(w::Vacuum.ConformalWall) = "ConformalWall: a = $(_fmt(w.standoff_fraction)) (standoff in minor radii)"
_wall_label(w::Vacuum.WallSettings) = "custom wall ($(nameof(typeof(w))))"

# How the singularities at the rationals were closed, naming the inner-layer model of a matched closure.
function _closure_label(r::ForceFreeStates.ForceFreeStatesResult)
    r.closure === :matched && r.match !== nothing && return "matched closure ($(InnerLayer.model_name(r.match.model)))"
    return "$(r.closure) closure"
end

function _physics_label(r::ForceFreeStates.ForceFreeStatesResult)
    k = r.control.kinetic
    k === nothing && return "ideal MHD"
    factor = r.control.kinetic_factor == 1 ? "" : ", factor $(_fmt(r.control.kinetic_factor))"
    return "kinetic ($(nameof(typeof(k)))$factor)"
end
_nrange(r) = r.nlow == r.nhigh ? "n = $(r.nlow)" : "n = $(r.nlow):$(r.nhigh)"

function Base.show(io::IO, r::ForceFreeStates.ForceFreeStatesResult)
    print(io, "ForceFreeStatesResult(", _integrator_name(r), ", ", _nrange(r), ", ", _plural(length(r.surfaces), "surface"))
    r.control.kinetic === nothing || print(io, ", kinetic")
    r.closure === :matched && r.match !== nothing && print(io, ", matched ", InnerLayer.model_name(r.match.model))
    r.free_boundary === nothing || print(io, ", et₁ = ", _fmt(real(first(r.free_boundary.et))))
    return print(io, ")")
end

function _solution_note(r::ForceFreeStates.ForceFreeStatesResult)
    r.solution === nothing || return "ξ profiles stored ($(r.solution.step) ψ nodes, $(r.solution.basis) basis)"
    return r.integrator === :riccati ? "no ξ profiles (Riccati stores none)" : "no ξ profiles ($(r.integrator) without a match)"
end

function _free_boundary_note(r::ForceFreeStates.ForceFreeStatesResult)
    fb = r.free_boundary
    if fb === nothing
        r.closure === :matched && return "none (matched closure publishes no ideal δW)"
        return r.control.vac_flag ? "not published by the $(r.integrator) integrator" : "not computed (vac_flag = false)"
    end
    verdict = real(first(fb.et)) < 0 ? "unstable" : "stable"
    return "et₁ = $(_fmt(real(first(fb.et)))) ($verdict), ep₁ = $(_fmt(real(first(fb.ep)))), ev₁ = $(_fmt(real(first(fb.ev)))), vacuum λ = $(_fmt(fb.vacuum_eigenvalue))"
end

function Base.show(io::IO, ::MIME"text/plain", r::ForceFreeStates.ForceFreeStatesResult)
    print(io, "ForceFreeStatesResult  ", _integrator_name(r), ", ", _closure_label(r))
    _row(io, "Physics", _physics_label(r))
    _row(io, "Modes", "$(_nrange(r)), m = $(r.mlow):$(r.mhigh) ($(r.numpert_total) total)")
    _row(io, "Domain", "ψ ∈ [$(_fmt(r.psilow)), $(_fmt(r.psilim))], q_lim = $(_fmt(r.qlim))" * (r.layer_overlap === nothing ? "" : " (capped by layer overlap)"))
    surf = isempty(r.surfaces) ? "none" : _list(["$(_surface_label(s)) @ ψ = $(_fmt(s.psifac))" for s in r.surfaces]; max=4)
    _row(io, "Surfaces", "$(length(r.surfaces)): " * surf)
    _row(io, "Free boundary", _free_boundary_note(r))
    if r.delta_prime !== nothing
        _row(io, "Δ′ diagonal", _list([_fmt(real(r.delta_prime.matrix[i, i])) for i in 1:minimum(size(r.delta_prime.matrix))]; max=4))
    end
    _row(io, "Solution", _solution_note(r))
    r.galerkin === nothing || _row(io, "Galerkin", "$(_plural(r.galerkin.msing, "surface")), $(r.galerkin.solver) factorization")
    r.match === nothing || _row(io, "Match", sprint(show, r.match))
    r.kinetic.kmsing > 0 && _row(io, "Kinetic", "$(_plural(r.kinetic.kmsing, "kinetic singular surface"))")
    _row(io, "Wall", _wall_label(r.wall_settings))
    return nothing
end

function Base.show(io::IO, m::MatchResult)
    return print(io, "MatchResult(", InnerLayer.model_name(m.model), ", ", _plural(size(m.deltar, 1), "surface"), ", residual = ", _fmt(m.residual), ")")
end

function Base.show(io::IO, ::MIME"text/plain", m::MatchResult)
    print(io, "MatchResult  ", InnerLayer.model_name(m.model))
    _row(io, "Surfaces", "$(size(m.deltar, 1)) matched, $(size(m.cout, 2)) edge drives")
    _row(io, "Residual", _fmt(m.residual))
    _row(io, "rpec eigs", isempty(m.rpec_eig) ? "none" : _list(_fmt.(m.rpec_eig); max=3))
    outer = isempty(m.xi) ? "not kept" : "$(_shape(m.xi)) (one column per edge drive)"
    _row(io, "Stored", "matched outer ξ $outer, inner profiles $(isempty(m.inner_psi) ? "not kept" : "on $(length(m.inner_psi)) surfaces")")
    return nothing
end

# PerturbedEquilibrium

_resonances(r::PlasmaResponse) = ["$(m)/$(n)" for (m, n) in zip(r.rational_m_res, r.rational_n)]

function Base.show(io::IO, r::PlasmaResponse)
    return print(io, "PlasmaResponse(", length(r.m_modes), " modes, ", _plural(length(r.rational_psi), "resonant surface"), ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::PlasmaResponse)
    print(io, "PlasmaResponse  on ", r.ffs)
    _row(io, "Modes", "$(length(r.m_modes)) (m = $(_span(r.m_modes)), n = $(_span(r.n_modes)))")
    _row(io, "Operators", isempty(r.permeability_flux) ? "not built (needs a free-boundary result)" : "permeability, inductances $(_shape(r.permeability_flux))")
    _row(io, "Resonances", isempty(r.rational_psi) ? "none" : _list(_resonances(r)))
    _row(io, "Coupling", isempty(r.C_resonant_area_weighted_field) ? "not computed" : "C $(_shape(r.C_resonant_area_weighted_field))")
    isempty(r.dominant_singular_values) || _row(io, "Dominant σ", _list(_fmt.(r.dominant_singular_values); max=3))
    _row(io, "Torque resp.", isempty(r.T_xe) ? "not computed" : "T_xe $(_shape(r.T_xe))")
    r.odet_from_gal && _row(io, "Basis", "Galerkin-native solution")
    return nothing
end

function _forcing_label(s::PerturbedEquilibriumState)
    isempty(s.coil_sets) || return "coils: " * _list([c.name for c in s.coil_sets])
    s.forcing === nothing && return "none"
    return string(nameof(typeof(s.forcing)))
end

function Base.show(io::IO, s::PerturbedEquilibriumState)
    print(io, "PerturbedEquilibriumState(", _forcing_label(s), ", ", _plural(length(s.response.rational_psi), "resonant surface"))
    isempty(s.chirikov_parameter) || print(io, ", max σ_chir = ", _fmt(maximum(s.chirikov_parameter)))
    return print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", s::PerturbedEquilibriumState)
    r = s.response
    print(io, "PerturbedEquilibriumState  forcing from ", _forcing_label(s))
    _row(io, "Energy", "vacuum $(_fmt(s.vacuum_energy)) J, surface $(_fmt(s.surface_energy)) J, plasma $(_fmt(s.plasma_energy)) J")
    _row(io, "Torque", "$(_fmt(s.toroidal_torque)) N·m (boundary response)")
    if isempty(s.island_half_width)
        _row(io, "Resonances", "singular coupling not computed")
    else
        _row(io, "Resonances", _list(_resonances(r)))
        _row(io, "|b̄_res| [T]", _list(_fmt.(abs.(s.resonant_area_weighted_field))))
        _row(io, "Island w/2 [ψ_N]", _list(_fmt.(s.island_half_width)))
        _row(io, "Chirikov", "max $(_fmt(maximum(s.chirikov_parameter)))")
    end
    isempty(s.dominant_forcing_overlap) || _row(io, "Dominant", "overlap with mode 1 = $(_fmt(abs(first(s.dominant_forcing_overlap)))) T")
    _row(io, "Profiles", s.xi_modes === nothing ? "not stored" : "ξ, b on $(length(s.psi_grid)) ψ nodes")
    s.torque_response === nothing || _row(io, "Torque resp.", "computed")
    iszero(s.regularization_width) || _row(io, "Regularized", "width $(_fmt(s.regularization_width))")
    _row(io, "Response", sprint(show, r))
    return nothing
end

# KineticForces

function Base.show(io::IO, m::MethodResult)
    return print(io, "MethodResult(", m.method, ", n = ", m.nn, ", T_φ = ", _fmt(real(m.total_torque)), " N·m, δW_k = ", _fmt(m.total_energy), " J)")
end

Base.show(io::IO, s::KineticForcesState) = print(io, "KineticForcesState(", _list(sort!(collect(keys(s.method_results)))), ")")

function _method_lines(io, s::KineticForcesState; indent="")
    for k in sort!(collect(keys(s.method_results)))
        m = s.method_results[k]
        _row(io, indent * k, "T_φ = $(_fmt(real(m.total_torque))) N·m, δW_k = $(_fmt(m.total_energy)) J")
    end
end

function Base.show(io::IO, r::KineticForcesResult)
    return print(io, "KineticForcesResult(", _list(sort!(collect(keys(r.total.method_results)))), isempty(r.species) ? "" : ", $(length(r.species)) species", ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::KineticForcesResult)
    print(io, "KineticForcesResult  ", _plural(length(r.total.method_results), "method"), isempty(r.species) ? "" : ", summed over $(length(r.species)) species")
    _method_lines(io, r.total)
    for (label, st) in r.species
        _row(io, "  " * label, sprint(show, st))
    end
    _row(io, "Axis cutoff", "ψ_c = $(_fmt(r.axis_psi_c))")
    return nothing
end

# ErrorFields

function Base.show(io::IO, mc::MonteCarloResult)
    return print(io, "MonteCarloResult(", mc.nsample, " samples, mean |δ| = ", _fmt(mc.abs_delta_sampled_mean), ", with EFC ", _fmt(mc.abs_delta_efc_sampled_mean), ")")
end

function Base.show(io::IO, r::RiskResult)
    return print(io, "RiskResult(P_lock = ", _fmt(r.locking_probability_percent), " %, with EFC ", _fmt(r.locking_probability_efc_percent), " %)")
end

function Base.show(io::IO, c::EFCCoupling)
    return print(io, "EFCCoupling(", c.coil_name, ", |δ| = ", _fmt(c.delta_per_kat), "/kA, T = ", _fmt(c.torque_full_per_kat2), " N·m/kA²)")
end

function Base.show(io::IO, r::ErrorFieldsResult)
    print(io, "ErrorFieldsResult(", _plural(length(r.sensitivities.coil_names), "coil"))
    r.risk === nothing || print(io, ", P_lock = ", _fmt(r.risk.locking_probability_percent), " %")
    return print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::ErrorFieldsResult)
    print(io, "ErrorFieldsResult  ", _plural(length(r.sensitivities.coil_names), "coil"))
    _row(io, "Dominant σ", isempty(r.dominant.singular_values) ? "none" : _list(_fmt.(r.dominant.singular_values); max=3))
    if r.monte_carlo === nothing
        _row(io, "Monte Carlo", "not run")
    else
        mc = r.monte_carlo
        _row(
            io,
            "Monte Carlo",
            "$(mc.nsample) samples; |δ|/B_T as designed $(_fmt(mc.abs_delta_total_as_designed)), mean $(_fmt(mc.abs_delta_sampled_mean)), with EFC $(_fmt(mc.abs_delta_efc_sampled_mean))"
        )
    end
    if r.risk === nothing
        _row(io, "Locking risk", "not computed")
    else
        rk = r.risk
        _row(
            io,
            "Locking risk",
            "$(_fmt(rk.locking_probability_percent)) %, with EFC $(_fmt(rk.locking_probability_efc_percent)) %, as designed $(_fmt(rk.locking_probability_as_designed_percent)) %"
        )
        _row(io, "Threshold", "fit |δ|/B_T = $(_fmt(rk.threshold_fit))")
    end
    r.scan === nothing || _row(io, "Tolerance scan", "computed")
    r.couplings === nothing || _row(io, "EFC coils", _list([c.coil_name for c in r.couplings]))
    return nothing
end

# Tearing

function Base.show(io::IO, g::GrowthRateResult)
    return print(io, "GrowthRateResult(γ = ", _fmt(g.gamma), " 1/s, ω = ", _fmt(g.omega), " rad/s, ", g.classification, ")")
end

_tearing_mode(r::TearingResult) = r.coupled_extraction === nothing ? "uncoupled" : "coupled"

# The root extraction behind each entry of `gamma`: one per surface uncoupled, the single global root coupled.
_extractions(r::TearingResult) = r.coupled_extraction === nothing ? r.per_surface_extraction : [r.coupled_extraction]

# Whether each entry of `gamma` found no root; its γ and ω are then 0 placeholders.
function _no_root(r::TearingResult)
    ex = _extractions(r)
    length(ex) == length(r.gamma) || return falses(length(r.gamma))
    return [:no_root in g.warning_flags for g in ex]
end

function _root_names(r::TearingResult)
    r.coupled_extraction === nothing || return ["coupled"]
    length(r.rational_q) == length(r.gamma) && return ["q = $(_fmt(q))" for q in r.rational_q]
    return ["#$i" for i in eachindex(r.gamma)]
end

_rooted(v, none) = [n ? "—" : _fmt(x) for (x, n) in zip(v, none)]

function Base.show(io::IO, r::TearingResult)
    r.enabled || return print(io, "TearingResult(disabled)")
    print(io, "TearingResult(", InnerLayer.model_name(r.model), ", ", _tearing_mode(r), ", ", _plural(length(r.params), "surface"))
    found = r.gamma[.!_no_root(r)]
    isempty(found) || print(io, ", max γ = ", _fmt(maximum(found)), " 1/s")
    return print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::TearingResult)
    r.enabled || return print(io, "TearingResult  disabled (the analysis did not run)")
    print(io, "TearingResult  ", InnerLayer.model_name(r.model), ", ", _tearing_mode(r))
    isempty(r.rational_q) || _row(io, "Surfaces", "q = " * _list(_fmt.(r.rational_q)))
    none = _no_root(r)
    found = r.gamma[.!none]
    verdict = isempty(found) ? "  (no roots found)" : any(>(0), found) ? "  (unstable)" : "  (stable)"
    _row(io, "γ [1/s]", _list(_rooted(r.gamma, none)) * verdict)
    _row(io, "ω [rad/s]", _list(_rooted(r.omega, none)))
    isempty(r.dp_matrix) || _row(io, "Δ′ diagonal", _list([_fmt(real(r.dp_matrix[i, i])) for i in 1:minimum(size(r.dp_matrix))]))
    isempty(r.layer_widths) || _row(io, "Layer δ [m]", _list([_fmt(w.delta_s_m) for w in r.layer_widths]))
    ex = _extractions(r)
    if length(ex) == length(r.gamma)
        flagged = ["$name: $(join(g.warning_flags, ", "))" for (name, g) in zip(_root_names(r), ex) if !isempty(g.warning_flags)]
        isempty(flagged) || _row(io, "Warnings", _list(flagged; max=4))
    end
    crf = r.critical_resonant_field
    crf.enabled && _row(io, "B_r,crit [T]", _list(_fmt.(crf.br_crit_tesla)))
    isempty(r.scan_data) || _row(io, "Scan data", "$(length(r.scan_data)) stored")
    return nothing
end

# Every field of each type with a text/plain summary, split into the fields the summary reports and
# those it leaves out on purpose. The display tests require the split to cover `fieldnames` exactly,
# so a new field fails them until it is placed in one list or the other.
const _DISPLAY_FIELDS = Dict{Any,@NamedTuple{shown::Tuple, omitted::Tuple}}(
    PlasmaEquilibrium => (shown=(:config, :params, :rzphi_xs, :rzphi_ys, :analytic, :kinetic),
        omitted=(:profiles, :geometry, :rzphi_rsquared, :rzphi_offset, :rzphi_nu, :rzphi_jac, :eqfun_B, :eqfun_metric1, :eqfun_metric2,
            :ro, :zo, :psio, :ingest)),
    LocalStability.LocalStabilityResult => (shown=(:psi, :D_I, :D_R, :ballooning_psi, :alpha, :alpha_critical),
        omitted=(:ballooning_Delta_prime, :di_psi, :problem, :algorithm)),
    ForceFreeStates.ForceFreeStatesResult => (
        shown=(:algorithm, :integrator, :closure, :control, :mlow, :mhigh, :nlow, :nhigh, :numpert_total, :psilow, :psilim, :qlim,
            :wall_settings, :surfaces, :kinetic, :match, :solution, :free_boundary, :delta_prime, :galerkin, :layer_overlap),
        omitted=(:problem, :equil, :mpert, :npert, :q1lim, :metric, :mats, :bpen, :diagnostics, :wp)),
    MatchResult => (shown=(:cout, :deltar, :rpec_eig, :residual, :xi, :inner_psi, :model),
        omitted=(:cin, :bpen, :reconnected_flux, :xi_deriv, :inner_xi, :inner_b, :inner_params, :problem)),
    PlasmaResponse => (
        shown=(:ffs, :m_modes, :n_modes, :odet_from_gal, :permeability_flux, :C_resonant_area_weighted_field, :rational_psi, :rational_m_res,
            :rational_n, :dominant_singular_values, :T_xe),
        omitted=(:compute_singular_coupling, :compute_torque_response, :flux_matrix, :plasma_inductance_flux, :surface_inductance_flux,
            :plasma_inductance, :surface_inductance, :permeability, :reluctance, :rootarea_to_area_weight, :surface_area, :C_resonant_current,
            :C_island_width_sq, :C_penetrated_area_weighted_field, :C_delta_prime, :coupling_flux, :C_coeffs, :rational_q, :rational_surface_idx,
            :rational_area, :dominant_right_singular_vectors, :dominant_left_singular_vectors, :dominant_rational_index)),
    PerturbedEquilibriumState => (
        shown=(:response, :forcing, :coil_sets, :regularization_width, :psi_grid, :xi_modes, :resonant_area_weighted_field, :island_half_width,
            :chirikov_parameter, :dominant_forcing_overlap, :vacuum_energy, :surface_energy, :plasma_energy, :toroidal_torque, :torque_response),
        omitted=(:forcing_modes, :b_modes, :b_n_modes, :xi_n_modes, :resonant_current, :island_width_sq, :penetrated_area_weighted_field,
            :delta_prime, :forcing_solution_weights, :forcing_b, :forcing_b_rootarea, :forcing_b_area, :response_b, :response_b_rootarea,
            :response_b_area)),
    KineticForcesResult => (shown=(:total, :species, :axis_psi_c), omitted=(:pe, :control)),
    ErrorFieldsResult => (shown=(:sensitivities, :dominant, :monte_carlo, :risk, :scan, :couplings), omitted=(:tolerances, :response, :settings)),
    TearingResult => (
        shown=(:enabled, :model, :params, :rational_q, :dp_matrix, :omega, :gamma, :per_surface_extraction, :coupled_extraction, :layer_widths,
            :scan_data, :critical_resonant_field),
        omitted=(:problem, :config, :rational_psi, :Q_root, :omega_E, :q_shift))
)
