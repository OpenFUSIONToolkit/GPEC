# TearingProblem.jl
#
# Tearing growth rates from a finished solve: root-find Δ_inner(Q) = Δ′_outer per surface.

"""
    TearingProblem(ffs; kwargs...)

Tearing-mode growth rates from a finished force-free-states solve: hold the outer Δ′ fixed
and find the growth rate at which the inner layer matches it. Solve with
`solve(prob, SLAYERModel())`; GGJ growth-rate extraction is not implemented yet. Keywords are
[`SLAYERControl`](@ref) fields, the `[SLAYER]` deck knobs; kinetic profiles come from
`profile_file`. Without a full Δ′ matrix the per-surface Δ′ stubs are used, with a warning.

## Fields

  - `ffs` - The finished solve; duck-typed on `equil`, `surfaces`, `delta_prime`, `dir_path`, so tests can use stand-ins.
  - `control::SLAYERControl` - The tearing knobs.
"""
struct TearingProblem{R}
    ffs::R
    control::SLAYERControl
end

function TearingProblem(ffs::ForceFreeStatesResult; kwargs...)
    haskey(kwargs, :inner_model) && error("TearingProblem: the inner-layer model is the solve argument; drop inner_model")
    return TearingProblem(ffs, SLAYERControl(; enabled=true, kwargs...))
end

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
        tau_E=something(control.tau_E, NaN),
        P_perp_model=control.P_perp_model,
        dr_val=control.D_R,
        dgeo_val=control.D_geo,
        dc_type=control.D_c_type,
        theta=control.theta_sample,
        resistivity_model=_build_resistivity_model(control.resistivity_model),
        lnLambda_form=control.lnLambda_form)
end

"""
    solve(prob::TearingProblem, model) -> SLAYERResult

Tearing growth rates per surface with the inner layer of `model`.
"""
function CommonSolve.solve(prob::TearingProblem, model::InnerLayer.InnerLayerModel)
    control = prob.control
    ffs = prob.ffs
    equil = ffs.equil
    surfaces = ffs.surfaces

    model isa SLAYERModel || error("TearingProblem: GGJ tearing growth rates are not currently supported; use SLAYERModel()")
    validate(control)
    control.enabled || return empty_slayer_result(control)
    isempty(surfaces) && return empty_slayer_result(control)

    loaded = _load_profiles(control, ffs.dir_path)

    params = _tearing_params(model, equil, surfaces, loaded, control)

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
    return run_slayer_from_inputs(model, params, dp, control;
        rational_psi=rational_psi, rational_q=rational_q, omega_E=omega_E)
end
