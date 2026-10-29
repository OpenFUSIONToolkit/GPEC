# TearingProblem.jl
#
# Tearing growth rates from a finished solve: root-find Δ_inner(Q) = Δ′_outer per surface.

"""
    TearingProblem(ffs; eta=nothing, rho=nothing, gamma=5/3, kwargs...)

Tearing stability of a finished force-balance solve. Physical keywords describe surface coupling,
species, transport and rotation; scan settings belong to [`TearingConfig`](@ref).
Solve with `solve(problem, GGJModel(); config)` or `solve(problem, SLAYERModel(); config)`.
GGJ needs the raw Δ′ matrix (Riccati with `vac_flag=true`, or Galerkin with `rpec_flag=true`)
and takes no rotation: its roots are in the plasma frame. SLAYER takes its profiles from the
equilibrium's kinetic data and reports lab-frame roots.

## Fields

  - `ffs` - Outer solution, retained by reference.
  - `coupling_mode`, `msing_max` - Uncoupled or coupled analysis and its maximum number of surfaces.
  - `ion_mass`, `zeff`, `impurity_charge` - Species mass and charge inputs.
  - `bt`, `chi_perp_e`, `chi_perp_i`, `chi_tor`, `tau_E`, `P_perp_model` - SLAYER field and transport inputs.
  - `D_c_type`, `D_R`, `D_geo`, `theta_sample` - SLAYER critical-Delta and geometry inputs.
  - `resistivity_model`, `lnLambda_form` - SLAYER resistivity closure.
  - `omega_E_kHz` - SLAYER per-surface rotation overrides keyed by m/n.
  - `critical_resonant_field` - Optional SLAYER penetration-threshold calculation.
  - `eta`, `rho` - GGJ resistivity in Ω·m and mass density in kg/m³, a scalar or one value per surface
    (core to edge), as in [`MatchProblem`](@ref); required by `GGJModel`, unused by `SLAYERModel`.
  - `gamma` - GGJ ratio of specific heats.
"""
@kwdef struct TearingProblem{R,E,D}
    ffs::R
    eta::E = nothing
    rho::D = nothing
    gamma::Float64 = 5/3
    coupling_mode::Symbol = :uncoupled
    msing_max::Int = 3
    D_c_type::Symbol = :none
    bt::Union{Float64,Nothing} = nothing
    ion_mass::Float64 = 2.0
    zeff::Float64 = 1.0
    impurity_charge::Float64 = 6.0
    chi_perp_e::Float64 = 1.0
    chi_perp_i::Union{Float64,Nothing} = nothing
    chi_tor::Float64 = 1.0
    tau_E::Union{Float64,Nothing} = nothing
    P_perp_model::Symbol = :D_perp
    D_R::Union{Float64,Nothing} = nothing
    D_geo::Union{Float64,Nothing} = nothing
    theta_sample::Float64 = 0.0
    resistivity_model::Symbol = :sauter
    lnLambda_form::Symbol = :nrl
    omega_E_kHz::Dict{String,Float64} = Dict{String,Float64}()
    critical_resonant_field::CriticalResonantFieldControl = CriticalResonantFieldControl()
end

TearingProblem(ffs; eta=nothing, rho=nothing, kwargs...) =
    TearingProblem{typeof(ffs),typeof(eta),typeof(rho)}(; ffs=ffs, eta=eta, rho=rho, kwargs...)

function _tearing_params(::SLAYERModel, equil, surfaces, loaded, control)
    # `equil.config.b0_norm` is a NORMALIZATION (commonly exactly 1.0), not the toroidal
    # field, so substituting it here silently ran the layer physics at B_T = 1 T. Pass the
    # control value through instead: `nothing` makes build_slayer_inputs compute the
    # physical B_T = F(psi)/(2*pi*R_0) per surface from the equilibrium's F-spline, which is
    # what its docstring already prescribes.
    bt = control.bt
    # χ⊥,e/χ⊥,i/χ_φ from the kinetic file when present, else the scalar fallbacks.
    chi = _slayer_chi_inputs(loaded, control)
    # Z_eff(ψ) from the kinetic file when it carries n_i, else the scalar fallback.
    zeff = loaded.zeff === nothing ? control.zeff : loaded.zeff
    if loaded.zeff_source === :scalar
        @warn "SLAYER: kinetic file has no usable zeff or n_i profile (dataset absent or all-zero); " *
              "using the scalar control.zeff = $(control.zeff) at every surface."
    else
        zeff_origin = loaded.zeff_source === :file ? "zeff profile" : "n_i/n_e with impurity_charge = $(control.impurity_charge)"
        @info "SLAYER: Z_eff(ψ) from the kinetic file's $zeff_origin"
    end
    return build_slayer_inputs(equil, surfaces, loaded.profiles;
        bt=bt,
        mu_i=control.ion_mass,
        zeff=zeff,
        chi_perp_e=chi.chi_perp_e,
        chi_perp_i=chi.chi_perp_i,
        chi_tor=chi.chi_tor,
        tau_E=control.tau_E,
        P_perp_model=control.P_perp_model,
        dr_val=control.D_R,
        dgeo_val=control.D_geo,
        dc_type=control.D_c_type,
        theta=control.theta_sample,
        resistivity_model=_build_resistivity_model(control.resistivity_model),
        lnLambda_form=control.lnLambda_form)
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
    loaded = _slayer_profiles(equil.kinetic, prob)

    params = _tearing_params(model, equil, surfaces, loaded, prob)

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
    problem.D_c_type in (:none, :lar, :fitzpatrick, :toroidal) || throw(ArgumentError("Invalid critical-Delta model."))
    problem.P_perp_model in (:chi_perp_e, :chi_perp_i, :P_phi, :D_perp, :c_beta, :tau_E) || throw(ArgumentError("Invalid perpendicular transport model."))
    for value in (problem.chi_perp_e, problem.chi_perp_i, problem.chi_tor)
        value === nothing || (isfinite(value) && value > 0) || throw(ArgumentError("Thermal and momentum diffusivities must be positive and finite."))
    end
    if problem.P_perp_model === :tau_E
        problem.tau_E !== nothing && isfinite(problem.tau_E) && problem.tau_E > 0 || throw(ArgumentError("The tau_E model requires a positive confinement time."))
    end
    problem.impurity_charge > 1 || throw(ArgumentError("Impurity charge must exceed one."))
    all(isfinite, values(problem.omega_E_kHz)) || throw(ArgumentError("Rotation overrides must be finite."))
    problem.critical_resonant_field.n >= 3 || throw(ArgumentError("Critical-resonant-field scans require at least three points."))
    config.scan_mode in (:amr, :brute_force) || throw(ArgumentError("Invalid tearing scan_mode."))
    config.root_classification in (:auto, :threshold, :winding) || throw(ArgumentError("Invalid root_classification."))
    config.nre >= 2 && config.nim >= 2 && config.amr_passes >= 0 || throw(ArgumentError("Invalid tearing scan resolution."))
    return nothing
end
