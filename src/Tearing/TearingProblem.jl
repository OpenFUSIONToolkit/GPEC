# TearingProblem.jl
#
# Tearing growth rates from a finished solve: root-find Δ_inner(Q) = Δ′_outer per surface.

"""
    TearingProblem(ffs; eta=nothing, rho=nothing, gamma=5/3, kwargs...)

Tearing stability of a finished force-balance solve. Keywords describe surface coupling and the
GGJ layer inputs; the SLAYER layer closures belong to [`SLAYERModel`](@ref) and scan settings to
[`TearingConfig`](@ref). The plasma itself (profiles, species,
Z_eff and transport defaults, confinement time and resistivity closure) is the equilibrium's kinetic
data, `ffs.equil.kinetic` (see [`KineticProfileData`](@ref)), so every layer model sees one plasma.
Solve with `solve(problem, GGJModel(); config)` or `solve(problem, SLAYERModel(); config)`.
GGJ needs the raw Δ′ matrix (Riccati with `vac_flag=true`, or Galerkin with `rpec_flag=true`)
and takes no rotation: its roots are in the plasma frame. SLAYER reports lab-frame roots.

## Fields

  - `ffs` - Outer solution, retained by reference.
  - `coupling_mode`, `msing_max` - Uncoupled or coupled analysis and its maximum number of surfaces.
  - `omega_E_kHz` - SLAYER per-surface rotation overrides keyed by m/n.
  - `critical_resonant_field` - The SLAYER penetration-threshold calculation, a `CriticalResonantFieldControl`; `nothing` (default) skips it.
  - `eta`, `rho` - GGJ resistivity in Ω·m and mass density in kg/m³, a scalar or one value per surface
    (core to edge), as in [`MatchProblem`](@ref). `nothing` (default) derives them from the kinetic data:
    η from its resistivity closure, the one SLAYER uses, and ρ = m_i·n_i of its main ion. Unused by `SLAYERModel`.
  - `gamma` - GGJ ratio of specific heats.
"""
@kwdef struct TearingProblem{R,E,D}
    ffs::R
    eta::E = nothing
    rho::D = nothing
    gamma::Float64 = 5/3
    coupling_mode::Symbol = :uncoupled
    msing_max::Int = 3
    omega_E_kHz::Dict{String,Float64} = Dict{String,Float64}()
    critical_resonant_field::Union{Nothing,CriticalResonantFieldControl} = nothing
end

TearingProblem(ffs; eta=nothing, rho=nothing, kwargs...) =
    TearingProblem{typeof(ffs),typeof(eta),typeof(rho)}(; ffs=ffs, eta=eta, rho=rho, kwargs...)

function _tearing_params(model::SLAYERModel, equil, surfaces, loaded)
    data = equil.kinetic
    # χ⊥,e/χ⊥,i/χ_φ from the kinetic file when present, else the kinetic data's defaults.
    chi = _slayer_chi_inputs(loaded, model, data)
    # B_T = F(ψ)/(2π·R₀) per surface comes from the equilibrium's F-spline (bt=nothing).
    return build_slayer_inputs(equil, surfaces, loaded.profiles;
        mu_i=_layer_ion_mass(data),
        zeff=_layer_zeff(loaded, data),
        chi_perp_e=chi.chi_perp_e,
        chi_perp_i=chi.chi_perp_i,
        chi_tor=chi.chi_tor,
        tau_E=data.tau_E,
        P_perp_model=model.P_perp_model,
        dr_val=model.D_R,
        dgeo_val=model.D_geo,
        dc_type=model.D_c_type,
        theta=model.theta_sample,
        resistivity_model=_build_resistivity_model(data.resistivity_model),
        lnLambda_form=data.lnLambda_form)
end

# The layer models take one main ion: its mass in proton masses.
function _layer_ion_mass(data::KineticProfileData)
    length(data.ion_species) == 1 ||
        throw(ArgumentError("The tearing layers take one main-ion species; the kinetic data list $(length(data.ion_species))"))
    return Float64(only(data.ion_species).m)
end

# Z_eff(ψ) for the layers, reporting where it came from: the file's profile, the n_i/n_e closure, or the default.
function _layer_zeff(loaded, data::KineticProfileData)
    if loaded.zeff_source === :scalar
        @warn "Tearing: kinetic file has no usable zeff or n_i profile (dataset absent or all-zero); " *
              "using zeff_default = $(data.zeff_default) at every surface."
        return data.zeff_default
    end
    zeff_origin = loaded.zeff_source === :file ? "zeff profile" : "n_i/n_e with impurity_charge = $(data.impurity_charge)"
    @info "Tearing: Z_eff(ψ) from the kinetic file's $zeff_origin"
    return loaded.zeff
end

"""
    solve(prob::TearingProblem, model) -> TearingResult

Tearing growth rates per surface with the inner layer of `model`.
"""
function CommonSolve.solve(prob::TearingProblem, model::SLAYERModel; config::TearingConfig=TearingConfig())
    config.gamma_range === nothing && config.omega_range === nothing || throw(ArgumentError("SLAYER uses Q_re_range and Q_im_range."))
    prob.eta === nothing && prob.rho === nothing && prob.gamma == 5/3 ||
        throw(ArgumentError("SLAYER derives its layer inputs from transport profiles; eta, rho, and gamma overrides are GGJ settings."))
    ffs = prob.ffs
    equil = ffs.equil
    surfaces = ffs.surfaces

    validate(prob, config)
    isempty(surfaces) && return empty_tearing_result(prob, config, model)

    equil.kinetic === nothing &&
        error("TearingProblem: the equilibrium carries no kinetic profiles; set kinetic_file in [Equilibrium] or attach_kinetic_profiles!")
    loaded = _slayer_profiles(equil.kinetic, model)

    params = _tearing_params(model, equil, surfaces, loaded)

    # Δ' matrix: prefer the full inter-surface matrix; fall back to a
    # diagonal built from each SingType's scalar delta_prime.
    dpm = ffs.delta_prime === nothing ? Matrix{ComplexF64}(undef, 0, 0) : ffs.delta_prime.matrix
    dp = if !isempty(dpm) && size(dpm) == (length(params), length(params))
        Matrix{ComplexF64}(dpm)
    else
        # The full Δ' matrix is unavailable (e.g. the parallel-FM stage that
        # populates it was not run). The scalar-diagonal fallback uses
        # `sing.delta_prime`, which is a coarse per-surface stub; surfaces
        # with no entry default to Δ'=0, giving γ computed from zero drive.
        n_missing = count(s -> isempty(s.delta_prime), surfaces)
        @warn(
            "SLAYER: delta_prime_matrix is empty or wrong-sized " *
            "($(size(dpm)) vs " *
            "($(length(params)),$(length(params)))); falling back to the " *
            "diagonal `sing.delta_prime` stub. Growth rates use a coarse " *
            "per-surface Δ' and may be unreliable" *
            (n_missing > 0 ? "; $n_missing surface(s) have NO Δ' entry and " *
                             "default to Δ'=0 (zero tearing drive)." : ".")
        )
        M = zeros(ComplexF64, length(params), length(params))
        for (k, s) in enumerate(surfaces)
            M[k, k] = isempty(s.delta_prime) ? 0.0 + 0im : s.delta_prime[1]
        end
        M
    end

    rational_psi = Float64[surfaces[p.ising].psifac for p in params]
    rational_q = Float64[surfaces[p.ising].q for p in params]
    # E×B rotation per unit n at each surface, for the Doppler shifts and the lab-frame roots;
    # empty when the kinetic file carries none.
    omega_E = loaded.has_omega_E ? Float64[loaded.profiles.omega_E(ψ) for ψ in rational_psi] : Float64[]
    loaded.has_omega_E || @warn "SLAYER: the kinetic file has no omega_E; surfaces without an omega_E_kHz override assume Ω_E = 0, so their lab-frame ω equals their E×B-frame ω."
    result = run_slayer_from_inputs(model, params, dp, prob, config;
        rational_psi=rational_psi, rational_q=rational_q, omega_E=omega_E)
    return result
end

function validate(problem::TearingProblem, config::TearingConfig)
    problem.coupling_mode in (:uncoupled, :coupled) || throw(ArgumentError("Invalid tearing coupling_mode."))
    problem.msing_max >= 1 || throw(ArgumentError("msing_max must be positive."))
    all(isfinite, values(problem.omega_E_kHz)) || throw(ArgumentError("Rotation overrides must be finite."))
    problem.critical_resonant_field === nothing || problem.critical_resonant_field.n >= 3 || throw(ArgumentError("Critical-resonant-field scans require at least three points."))
    config.scan_mode in (:amr, :brute_force) || throw(ArgumentError("Invalid tearing scan_mode."))
    config.root_classification in (:auto, :threshold, :winding) || throw(ArgumentError("Invalid root_classification."))
    config.nre >= 2 && config.nim >= 2 && config.amr_passes >= 0 || throw(ArgumentError("Invalid tearing scan resolution."))
    return nothing
end
