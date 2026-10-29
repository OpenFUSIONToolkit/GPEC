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
    data = ffs.equil.kinetic
    (prob.eta === nothing || prob.rho === nothing) && data === nothing &&
        throw(ArgumentError("GGJ tearing: give eta and rho, or attach kinetic profiles to the equilibrium"))
    eta = if prob.eta !== nothing
        _ggj_per_surface(prob.eta, "eta", count)
    else
        loaded = _slayer_profiles(data, prob)
        InnerLayer.SLAYER.layer_resistivity(ffs.equil, surfaces, loaded.profiles; zeff=_layer_zeff(loaded, data),
            resistivity_model=_build_resistivity_model(data.resistivity_model), lnLambda_form=data.lnLambda_form)
    end
    rho = if prob.rho !== nothing
        _ggj_per_surface(prob.rho, "rho", count)
    else
        data.n_i === nothing && throw(ArgumentError("GGJ tearing: ρ = m_i·n_i needs an n_i profile in the kinetic data, or give rho"))
        n_i = cubic_interp(collect(Float64, data.psi), data.n_i)
        _layer_ion_mass(data) * Utilities.M_P .* Float64[n_i(s.psifac) for s in surfaces]
    end
    return _ggj_per_surface(eta, "eta", count), _ggj_per_surface(rho, "rho", count)
end

# The SLAYER-only physics fields must stay at their defaults, so a GGJ solve never silently ignores one.
const _SLAYER_ONLY_FIELDS = (:D_c_type, :P_perp_model, :D_R, :D_geo, :theta_sample, :omega_E_kHz, :critical_resonant_field)

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
    couplings = [surface_coupling(model, p, 0.0) for p in params]

    # Uncoupled: each surface with its own 2×2 block of Δ′. Coupled: the leading msing_max surfaces together.
    determinants = if prob.coupling_mode === :uncoupled
        [Dispersion.multi_surface_coupling_full(couplings[k:k], raw[(2k-1):2k, (2k-1):2k]) for k in 1:count]
    else
        [Dispersion.multi_surface_coupling_full(couplings, raw; msing_max=min(prob.msing_max, count))]
    end
    # The result reports ω = Re(Q)/τ, the negative of the determinant's Im(s), so the box flips.
    omega_range = (-config.omega_range[2], -config.omega_range[1])
    extractions = GrowthRateResult[]
    scans = Union{ScanResult,AMRResult}[]
    for det in determinants
        extraction, scan = Dispersion.ggj_growth_rates(det; gamma_range=config.gamma_range, omega_range=omega_range,
            nre0=config.nre, nim0=config.nim, passes=config.amr_passes, max_cells=config.amr_max_cells, polish=config.polish_roots)
        push!(extractions, extraction)
        config.store_scan && push!(scans, scan)
    end
    coupled = prob.coupling_mode === :coupled
    return TearingResult(prob, true, config, model, params,
        Float64[s.psifac for s in surfaces], Float64[s.q for s in surfaces], Matrix{ComplexF64}(ffs.delta_prime.matrix),
        ComplexF64[e.Q_root for e in extractions], Float64[e.omega for e in extractions], Float64[e.gamma for e in extractions],
        coupled ? GrowthRateResult[] : extractions, coupled ? only(extractions) : nothing,
        LayerWidths[], fill(NaN, count), zeros(count), scans, empty_critical_resonant_field_result())
end
