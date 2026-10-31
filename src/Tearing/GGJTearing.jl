# GGJTearing.jl
#
# `solve(TearingProblem, GGJModel)`: tearing growth rates from the full GGJ matching
# determinant (Dispersion.ggj_growth_rates), each surface on its own when uncoupled,
# the leading `msing_max` surfaces together when coupled.

# The raw side-major Δ′ the determinant needs, after checking it is aligned with ordered surfaces.
function _ggj_delta_prime_raw(ffs)
    ffs.npert == 1 || throw(ArgumentError("GGJ tearing is single-n; the result spans n = $(ffs.nlow):$(ffs.nhigh)"))
    surfaces = ffs.surfaces
    psi = Float64[s.psifac for s in surfaces]
    issorted(psi; lt=<=) || throw(ArgumentError("GGJ tearing needs distinct surfaces ordered core to edge; got ψ = $psi"))
    dp = ffs.delta_prime
    dp === nothing && throw(ArgumentError("GGJ tearing needs the raw Δ′ matrix; solve with Riccati() and vac_flag=true, or Galerkin(; rpec_flag=true)"))
    size(dp.raw) == (2length(surfaces), 2length(surfaces)) ||
        throw(ArgumentError("GGJ tearing: Δ′ raw block is $(size(dp.raw, 1))×$(size(dp.raw, 2)) but the result carries $(length(surfaces)) surfaces"))
    all(isfinite, dp.raw) || throw(ArgumentError("GGJ tearing: the raw Δ′ matrix has non-finite entries"))
    return dp.raw
end

# η and ρ given on the problem, as MatchProblem takes them: a scalar or one value per surface, core to edge.
function _ggj_per_surface(value, name, count)
    values = value isa Real ? fill(Float64(value), count) : collect(Float64, value)
    length(values) == count || throw(ArgumentError("GGJ tearing: $name has length $(length(values)), expected $count"))
    all(v -> isfinite(v) && v > 0, values) || throw(ArgumentError("GGJ tearing: $name must be positive and finite"))
    return values
end

# η and ρ at each surface: the problem's values where given, else from the equilibrium's kinetic data,
# η by the resistivity closure the SLAYER layer uses and ρ = m_i·n_i of the main ion.
function _ggj_layer_inputs(prob::TearingProblem)
    ffs = prob.ffs
    surfaces = ffs.surfaces
    count = length(surfaces)
    InnerLayer._require_layer_inputs("GGJ tearing", ffs.equil; prob.eta, prob.rho)
    eta = prob.eta === nothing ? InnerLayer.layer_eta(ffs.equil, surfaces) : prob.eta
    rho = prob.rho === nothing ? InnerLayer.layer_rho(ffs.equil, surfaces) : prob.rho
    return _ggj_per_surface(eta, "eta", count), _ggj_per_surface(rho, "rho", count)
end

# E×B rotation Ω_E per unit n at each surface: the kinetic data's omega_E (NaN where there is none), replaced on the
# surfaces whose m/n `problem.omega_E_kHz` lists.
function _ggj_omega_E(prob::TearingProblem)
    ffs = prob.ffs
    data = ffs.equil.kinetic
    data === nothing || data.omega_E !== nothing ||
        @warn "GGJ tearing: the kinetic data have no omega_E; surfaces without an omega_E_kHz override assume Ω_E = 0, so their lab-frame ω equals their E×B-frame ω."
    return _omega_E_per_surface(prob, ["$(only(s.m))/$(only(s.n))" for s in ffs.surfaces], InnerLayer.layer_omega_E(ffs.equil, ffs.surfaces))
end

# The SLAYER-only physics fields must stay at their defaults, so a GGJ solve never silently ignores one.
const _SLAYER_ONLY_FIELDS = (:critical_resonant_field,)

function _check_ggj_inputs(prob::TearingProblem, config::TearingConfig)
    defaults = TearingProblem(prob.ffs)
    for key in _SLAYER_ONLY_FIELDS
        getfield(prob, key) == getfield(defaults, key) || throw(ArgumentError("TearingProblem: `$key` is a SLAYER input; GGJModel does not use it"))
    end
    prob.gamma > 0 && isfinite(prob.gamma) || throw(ArgumentError("TearingProblem: gamma (ratio of specific heats) must be positive and finite"))
    config.gamma_range === nothing && throw(ArgumentError("GGJ tearing needs TearingConfig(; gamma_range=(γ_min, γ_max)) in 1/s"))
    config.omega_range === nothing && throw(ArgumentError("GGJ tearing needs TearingConfig(; omega_range=(ω_min, ω_max)) in rad/s"))
    config.root_classification in (:auto, :winding) || throw(ArgumentError("GGJ tearing classifies roots by winding number"))
    config.scan_mode === :amr && isempty(config.boxes) || throw(ArgumentError("GGJ tearing scans one adaptive box; scan_mode and boxes are SLAYER settings"))
    return nothing
end

function CommonSolve.solve(prob::TearingProblem, model::GGJModel; config::TearingConfig=TearingConfig())
    validate(prob, config)
    _check_ggj_inputs(prob, config)
    ffs = prob.ffs
    surfaces = ffs.surfaces
    isempty(surfaces) && return empty_tearing_result(prob, config, model)
    raw = _ggj_delta_prime_raw(ffs)
    count = length(surfaces)
    eta, rho = _ggj_layer_inputs(prob)
    params = [ggj_parameters(s, ffs.equil; eta=eta[k], rho=rho[k], gamma=prob.gamma, ising=k) for (k, s) in enumerate(surfaces)]
    coupled = prob.coupling_mode === :coupled
    Ω_E = _ggj_omega_E(prob)
    # A surface with no known rotation applies no shift.
    Ω_shift = [isnan(Ω) ? 0.0 : Ω for Ω in Ω_E]
    n = ffs.nlow

    # Coupled: each layer sees the lab-frame growth rate in its own E×B frame, s_k = s + i·n·Ω_E,k, so differential
    # rotation enters the determinant. Uncoupled: each surface with its own 2×2 block of Δ′, solved in its own frame.
    determinants = if coupled
        couplings = [surface_coupling(model, p, 0.0; omega_E=Ω_shift[k], n) for (k, p) in enumerate(params)]
        [Dispersion.multi_surface_coupling_full(couplings, raw; msing_max=min(prob.msing_max, count))]
    else
        [Dispersion.multi_surface_coupling_full([surface_coupling(model, params[k], 0.0)], raw[(2k-1):2k, (2k-1):2k]) for k in 1:count]
    end
    extractions = GrowthRateResult[]
    scans = Union{ScanResult,AMRResult}[]
    for det in determinants
        extraction, scan = Dispersion.ggj_growth_rates(det; gamma_range=config.gamma_range, omega_range=config.omega_range,
            nre0=config.nre, nim0=config.nim, passes=config.amr_passes, max_cells=config.amr_max_cells, polish=config.polish_roots)
        push!(extractions, extraction)
        config.store_scan && push!(scans, scan)
    end
    Q_root = ComplexF64[e.Q_root for e in extractions]
    omega = Float64[e.omega for e in extractions]
    if !coupled
        # Each uncoupled layer is solved in its own E×B frame; report its root in the lab frame, ω_lab = ω + n·Ω_E.
        for (k, e) in enumerate(extractions)
            :no_root in e.warning_flags && continue
            omega[k] += n * Ω_shift[k]
            Q_root[k] += n * Ω_shift[k] / InnerLayer.GGJ.q0(params[k])
        end
    end
    return TearingResult(prob, true, config, model, params,
        Float64[s.psifac for s in surfaces], Float64[s.q for s in surfaces], Matrix{ComplexF64}(ffs.delta_prime.matrix),
        Q_root, omega, Float64[e.gamma for e in extractions],
        coupled ? GrowthRateResult[] : extractions, coupled ? only(extractions) : nothing,
        LayerWidths[], Ω_E, zeros(count), scans, empty_critical_resonant_field_result())
end
