"""
    ErrorFieldsResult

The error-field assessment of a plasma response, from [`error_fields`](@ref). Products that were not requested are
`nothing`.

## Fields

  - `sensitivities::CoilSensitivities` - The coil sets' rigid-body sensitivities on the control surface.
  - `dominant::DominantCoupling` - The singular modes of the resonant coupling the overlaps project on.
  - `tolerances::Union{Nothing,ToleranceSet}` - The manufacturing tolerances, validated against the coil sets.
  - `monte_carlo::Union{Nothing,MonteCarloResult}` - The tolerance Monte Carlo; needs `tolerances`.
  - `risk::Union{Nothing,RiskResult}` - The locking risk; needs the Monte Carlo and a `scenario`.
  - `scan::Union{Nothing,ToleranceScan}` - The locking risk against a tolerance scale; needs `RiskControl.scan_scales`.
  - `couplings::Union{Nothing,Vector{EFCCoupling}}` - The correction arrays' couplings and NTV torques; needs `NTVControl.efc_coils`.
  - `response::PerturbedEquilibrium.PlasmaResponse` - The plasma response the assessment was made on.
  - `settings::NamedTuple` - The settings it was made with: `ctrl`, `monte_carlo`, `risk`, `ntv`, `kinetic`,
    `scenario` and `regularization_width`, as given to `error_fields`.
"""
struct ErrorFieldsResult
    sensitivities::CoilSensitivities
    dominant::DominantCoupling
    tolerances::Union{Nothing,ToleranceSet}
    monte_carlo::Union{Nothing,MonteCarloResult}
    risk::Union{Nothing,RiskResult}
    scan::Union{Nothing,ToleranceScan}
    couplings::Union{Nothing,Vector{EFCCoupling}}
    response::PerturbedEquilibrium.PlasmaResponse
    settings::NamedTuple
end

"""
    error_fields(resp, coil_sets, coil_config; ctrl=ErrorFieldsControl(), tolerances=nothing,
                 monte_carlo=MonteCarloControl(), scenario=nothing, risk=RiskControl(), ntv=NTVControl(),
                 kinetic=KineticForcesControl(), regularization_width=0.0, verbose=false) -> ErrorFieldsResult

Assess the error field of `coil_sets` on the plasma response `resp`: the coil sensitivities always; the tolerance
Monte Carlo when `tolerances` (a [`ToleranceSet`](@ref)) is given; the locking risk when a `scenario` (a
[`ScenarioParameters`](@ref)) is given too; and the couplings of the correction arrays named in `ntv.efc_coils`, whose
NTV torque uses the `kinetic` settings and the perturbed-equilibrium `regularization_width`. `coil_config` sets the
boundary grid the coil spectra are evaluated on. The coil sets in `ctrl.exclude_coils` and `ntv.efc_coils` are left
out of the error field. Call `save_to_hdf5(target, ef)` to save the result.

```julia
resp = plasma_response(ffs)
ef = error_fields(resp, coil_sets, ForcingTerms.CoilConfig(ft_ctrl); tolerances=read_tolerance_toml("tolerances.toml"))
```
"""
function error_fields(
    resp::PerturbedEquilibrium.PlasmaResponse,
    coil_sets::Vector{CoilSet},
    coil_config::CoilConfig;
    ctrl::ErrorFieldsControl=ErrorFieldsControl(),
    tolerances::Union{Nothing,ToleranceSet}=nothing,
    monte_carlo::MonteCarloControl=MonteCarloControl(),
    scenario::Union{Nothing,ScenarioParameters}=nothing,
    risk::RiskControl=RiskControl(),
    ntv::NTVControl=NTVControl(),
    kinetic::KineticForces.KineticForcesControl=KineticForces.KineticForcesControl(),
    regularization_width::Real=0.0,
    verbose::Bool=false
)
    ffs = resp.ffs
    equil = ffs.equil
    rc = ResonantCoupling(resp)
    sens = compute_coil_sensitivities(coil_sets, rc, equil, coil_config, ctrl; psi=ffs.psilim, b_t0=equil.params.bt0, verbose)
    excluded = unique(String[ctrl.exclude_coils; ntv.efc_coils])

    if tolerances !== nothing
        tolerances = validate_tolerances(tolerances, sens.coil_names)
        check_excluded_tolerances(tolerances, excluded)
        @info "Tolerances: $(length(tolerances.coils)) coil sets, $(length(tolerances.groups)) coherent groups"
    end

    # The run's Monte Carlo is the full-window, dominant-mode summary; other windows are re-run post hoc.
    dom = dominant_coupling(rc)
    table = without_coils(sensitivity_table(sens, dom), excluded)
    isempty(excluded) || @info "Error field: $(join(table.coil_names, ", ")); excluded $(join(excluded, ", "))"
    mc = nothing
    if tolerances !== nothing
        mc_start = time()
        mc = run_monte_carlo(table, tolerances, coil_sets, monte_carlo)
        @info "Monte Carlo: $(monte_carlo.nbatch) × $(monte_carlo.nsample) samples in $(@sprintf("%.2f", time() - mc_start)) s; " *
              "⟨|δ|⟩ = $(@sprintf("%.3e", mc.abs_delta_sampled_mean)) intrinsic, $(@sprintf("%.3e", mc.abs_delta_efc_sampled_mean)) corrected " *
              "(nominal $(@sprintf("%.3e", mc.abs_delta_total_as_designed)))"
    end

    lock = nothing
    scan = nothing
    if mc !== nothing && scenario !== nothing
        n_scaling = single_toroidal_mode(ffs.nlow, ffs.nhigh; where="the in-run locking risk of this run")
        sc = threshold_scaling(; n=n_scaling, year=risk.year, dataset=risk.dataset, fit=risk.fit)
        risk_start = time()
        lock = locking_risk(mc, sc, scenario; ctrl=risk)
        @info "Locking risk ($(scaling_label(sc))): threshold $(@sprintf("%.3e", lock.threshold_fit)); " *
              "P_lock = $(@sprintf("%.2f", lock.locking_probability_percent)) % intrinsic, $(@sprintf("%.2f", lock.locking_probability_efc_percent)) % corrected, " *
              "$(@sprintf("%.2f", lock.locking_probability_as_designed_percent)) % as designed ($(@sprintf("%.2f", time() - risk_start)) s)"
        if !isempty(risk.scan_scales)
            scan_start = time()
            scan = tolerance_scan(table, tolerances, coil_sets, monte_carlo, sc, scenario; scales=risk.scan_scales, risk_ctrl=risk)
            @info "Tolerance scan over $(length(scan.tolerance_scale)) scales in $(@sprintf("%.2f", time() - scan_start)) s"
        end
    end

    # Correction-coil couplings: overlap per kAt and the NTV torque of the full and residual fields.
    couplings = nothing
    if !isempty(ntv.efc_coils)
        efc_sets = [cs for cs in coil_sets if cs.name in ntv.efc_coils]
        missing = setdiff(ntv.efc_coils, [cs.name for cs in efc_sets])
        isempty(missing) || error("NTVControl efc_coils not among the coil sets: $(join(missing, ", "))")
        ntv_start = time()
        couplings = efc_couplings(resp, efc_sets, rc, dom, coil_config, kinetic; method=ntv.method, verbose, regularization_width,
            rotation_scan=ntv.rotation_scan, scan_points=ntv.rotation_scan_points, scan_max_points=ntv.rotation_scan_max_points,
            scan_tolerance=ntv.rotation_scan_tolerance, span_factor=ntv.rotation_span_factor, offset_factor=ntv.rotation_offset_factor)
        @info "NTV couplings of $(length(couplings)) correction arrays in $(@sprintf("%.1f", time() - ntv_start)) s"
    end

    settings = (; ctrl, monte_carlo, risk, ntv, kinetic, scenario, regularization_width=Float64(regularization_width))
    return ErrorFieldsResult(sens, dom, tolerances, mc, lock, scan, couplings, resp, settings)
end
