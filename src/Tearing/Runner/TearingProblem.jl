# TearingProblem.jl
#
# The free-eigenvalue tearing solve in the problem/model grammar: a TearingProblem holds a
# finished force-free-states result and the matching-procedure control; the inner-layer
# model passed to `solve` evaluates Δ(Q), and the growth rate is root-found where
# Δ_inner(Q) = Δ'_outer. This file IS the orchestration (profiles → per-surface parameters
# → Δ' conditioning → the scan core); the model stays a typed object end-to-end, and the
# deck's `inner_model` string is translated to a model at the deck boundary, never below.

"""
    TearingProblem(ffs; kwargs...)

The free-eigenvalue tearing problem posed on a finished force-free-states solve: hold the
outer Δ′ fixed and root-find the growth rate where the inner-layer response matches it.
This is the WHAT; the inner-layer model passed to [`solve`](@ref) — `SLAYER()` or
`GGJ(; solver=:shooting|:galerkin)` — is the HOW. Keyword arguments are
[`SLAYERControl`](@ref) fields (the matching procedure: scan mode and Q-domain, coupling
mode, critical-Δ convention, extraction filters, plasma-composition knobs, and the
`profile_file` override); `enabled` is implied by posing the problem.

Kinetic profiles come from `profile_file` when it is set, otherwise from the profiles
attached to the equilibrium (`ffs.equil.kinetic`). The outer Δ′ comes from
`ffs.delta_prime`; a result without one (or with the wrong surface count) falls back to the
per-surface scalar stubs with a loud warning, exactly as the deck path always has.

## Fields

  - `ffs` - The force-free-states result supplying the equilibrium, surfaces and Δ′.
  - `control::SLAYERControl` - The matching-procedure control (single source of truth; its
    `inner_model` key is deck vocabulary resolved at the deck boundary and never read here).
"""
# The result field is deliberately duck-typed (anything carrying equil/surfaces/
# delta_prime/dir_path), so tests can drive the solve with lightweight stand-ins.
struct TearingProblem{R}
    ffs::R
    control::SLAYERControl
end

TearingProblem(ffs::ForceFreeStatesResult; kwargs...) =
    TearingProblem(ffs, SLAYERControl(; enabled=true, kwargs...))

# Kinetic profiles from the splines attached to the equilibrium, in the layer builders'
# convention: temperatures back in eV, `omega` the E×B rotation, and the per-surface
# diamagnetic inputs zeroed (they are recomputed from equilibrium gradients downstream,
# exactly as the file loader does). χ profiles are not carried by the attachment, so the
# scalar model fallbacks apply.
function _profiles_from_equilibrium(kp)
    xs = kp.xs
    E_CHG = Utilities.PhysicalConstants.E_CHG
    return (profiles=KineticProfiles(;
            psi=xs,
            n_e=[kp.ne_spline(ψ) for ψ in xs],
            T_e=[kp.Te_spline(ψ) / E_CHG for ψ in xs],
            T_i=[kp.Ti_spline(ψ) / E_CHG for ψ in xs],
            omega=[kp.omegaE_spline(ψ) for ψ in xs],
            omega_e=zeros(length(xs)),
            omega_i=zeros(length(xs))),
        chi_perp=nothing, chi_tor=nothing)
end

# Per-surface parameter building and the dispatch tag, keyed on the user-facing model
# config. GGJ is genuinely toroidal/ψ-based; SLAYER is the slab layer with χ transport.
_tearing_tag(model::GGJ) =
    model.solver in (:shooting, :galerkin) ? InnerLayer.GGJModel(; solver=model.solver) :
    error("tearing with GGJ needs solver=:shooting or :galerkin (got :$(model.solver); the :ray backend has no tearing dispersion path)")
_tearing_tag(::SLAYER) = InnerLayer.SLAYERModel(; variant=:fitzpatrick)

function _tearing_params(::GGJ, equil, surfaces, loaded, control)
    lp = layer_parameters(surfaces, equil; profiles=loaded.profiles,
        mu_i=control.mu_i,
        zeff=control.zeff,
        resistivity_model=_build_resistivity_model(control.resistivity_model),
        lnLambda_form=control.lnLambda_form)
    return [ggj_parameters(s, equil; eta=lp.eta[k], rho=lp.rho[k], ising=k) for (k, s) in enumerate(surfaces)]
end

function _tearing_params(model::SLAYER, equil, surfaces, loaded, control)
    # `equil.config.b0exp` is a NORMALIZATION (commonly exactly 1.0), not the toroidal
    # field: `control.bt = nothing` makes build_slayer_inputs compute the physical
    # B_T = F(psi)/(2*pi*R_0) per surface from the equilibrium's F-spline.
    # χ⊥/χ_φ from the kinetic file when present, else the model's scalar fallbacks.
    chi_perp = loaded.chi_perp === nothing ? model.chi_perp : loaded.chi_perp
    chi_tor = loaded.chi_tor === nothing ? model.chi_tor : loaded.chi_tor
    (loaded.chi_perp === nothing || loaded.chi_tor === nothing) && @warn(
        "SLAYER: no usable chi_e/chi_phi profile(s) (dataset absent or all-zero); " *
        "using the scalar chi_perp/chi_tor fallback for the missing one(s).")
    return build_slayer_inputs(equil, surfaces, loaded.profiles;
        bt=control.bt,
        mu_i=control.mu_i,
        zeff=control.zeff,
        chi_perp=chi_perp,
        chi_tor=chi_tor,
        dr_val=control.dr_val,
        dgeo_val=control.dgeo_val,
        dc_type=control.dc_type,
        theta=control.theta_sample,
        resistivity_model=_build_resistivity_model(control.resistivity_model),
        lnLambda_form=control.lnLambda_form)
end

"""
    solve(prob::TearingProblem, model) -> SLAYERResult

Run the tearing analysis: source the kinetic profiles, build the per-surface layer
parameters for `model`, condition the outer Δ′ (full matrix when the result carries one,
the per-surface diagonal stub fallback otherwise), and root-find the growth rates with the
scan core. `model` is a user-facing inner-layer config — [`SLAYER`](@ref) or
[`GGJ`](@ref) with a `:shooting`/`:galerkin` backend.
"""
function CommonSolve.solve(prob::TearingProblem, model::InnerLayer.InnerLayerModel)
    control = prob.control
    ffs = prob.ffs
    equil = ffs.equil
    surfaces = ffs.surfaces

    validate(control)
    control.enabled || return empty_slayer_result(control)
    isempty(surfaces) && return empty_slayer_result(control)

    loaded = if !isempty(control.profile_file)
        _load_profiles(control, ffs.dir_path)
    elseif equil.kinetic !== nothing
        _profiles_from_equilibrium(equil.kinetic)
    else
        error("TearingProblem: no kinetic profiles — set profile_file, or attach profiles " *
              "to the equilibrium with attach_kinetic_profiles!(equil, file)")
    end

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
    return run_slayer_from_inputs(_tearing_tag(model), params, dp, control;
        rational_psi=rational_psi, rational_q=rational_q)
end
