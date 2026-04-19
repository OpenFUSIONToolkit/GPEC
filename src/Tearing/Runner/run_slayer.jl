# Runner.jl
#
# Top-level orchestration for the SLAYER tearing-mode analysis. Given a
# fully-solved `PlasmaEquilibrium` + `ForceFreeStatesInternal` (which
# supplies the rational-surface list and the outer-region Δ' matrix) + a
# populated `SLAYERControl`, `run_slayer` loads kinetic profiles, builds
# per-surface SLAYER parameters, runs the requested scan mode, extracts
# growth rates by contour intersection, and returns a `SLAYERResult`.
#
# A secondary entry point `run_slayer_from_inputs` takes pre-built
# per-surface parameters + a Δ' matrix and bypasses the
# equilibrium-driven `build_slayer_inputs` step. This is what the test
# suite drives; it keeps the end-to-end code covered without requiring a
# full equilibrium solve in every test.

# ---------------------------------------------------------------------
# Profile loading dispatch
# ---------------------------------------------------------------------
function _load_profiles(control::SLAYERControl, toml_section::AbstractDict,
                         dir_path::AbstractString)
    if control.profile_source === :inline
        haskey(toml_section, "profiles") ||
            error("run_slayer: profile_source=:inline but no " *
                  "[SLAYER.profiles] subsection found in gpec.toml")
        return kinetic_profiles_from_toml(toml_section["profiles"])
    elseif control.profile_source === :h5
        isempty(control.profile_file) &&
            error("run_slayer: profile_source=:h5 but profile_file is empty")
        h5path = isabspath(control.profile_file) ? control.profile_file :
                 joinpath(dir_path, control.profile_file)
        return kinetic_profiles_from_h5(h5path; group=control.profile_group)
    end
    error("run_slayer: unknown profile_source=$(control.profile_source)")
end

# ---------------------------------------------------------------------
# Inner-layer model factory
# ---------------------------------------------------------------------
function _build_inner_model(name::Symbol)
    if name === :slayer_fitzpatrick
        return SLAYERModel(variant=:fitzpatrick)
    elseif name === :ggj_shooting
        return GGJModel(solver=:shooting)
    elseif name === :ggj_galerkin
        return GGJModel(solver=:galerkin)
    end
    throw(ArgumentError("_build_inner_model: unknown model $name"))
end

# ---------------------------------------------------------------------
# Scan dispatch
# ---------------------------------------------------------------------
function _run_scan(f, control::SLAYERControl)
    if control.scan_mode === :brute_force
        return brute_force_scan(f, control.Q_re_range, control.Q_im_range;
                                 nre=control.nre, nim=control.nim)
    elseif control.scan_mode === :amr
        return amr_scan(f, control.Q_re_range, control.Q_im_range;
                         nre0=control.nre, nim0=control.nim,
                         passes=control.amr_passes,
                         max_cells=control.amr_max_cells)
    end
    throw(ArgumentError("_run_scan: unknown scan_mode=$(control.scan_mode)"))
end

# ---------------------------------------------------------------------
# Surface-coupling builder — dispatches on model type to thread the
# correct `scale` and `tauk` through the Dispersion API.
# ---------------------------------------------------------------------
function _build_surface_coupling(model, params::SLAYERParameters, dp_diag)
    # For both SLAYER and GGJ models, `surface_coupling` has a method that
    # auto-fills scale and tauk based on the parameter type — SLAYER uses
    # lu^(1/3) and params.tauk; GGJ defaults to 1.0/1.0.
    if model isa SLAYERModel
        return surface_coupling(model, params, dp_diag; dc=params.dc_tmp)
    else
        # For GGJ we need GGJParameters — SLAYER params don't map there.
        # This path exists only for type-compatibility; calling it in
        # practice raises at the surface_coupling dispatch level.
        error("_build_surface_coupling: non-SLAYER inner models require " *
              "an upstream GGJParameters conversion that is not yet " *
              "implemented. Use inner_model=:slayer_fitzpatrick.")
    end
end

# ---------------------------------------------------------------------
# Core analysis entry point that takes pre-built parameters.
# ---------------------------------------------------------------------
"""
    run_slayer_from_inputs(params::Vector{SLAYERParameters},
                            dp_matrix::AbstractMatrix,
                            control::SLAYERControl) -> SLAYERResult

Run the SLAYER tearing analysis given pre-built per-surface
`SLAYERParameters` and the outer-region Δ' matrix. Bypasses the
equilibrium-driven `build_slayer_inputs` step — use this when the
parameters are already known (e.g. in unit tests or when rebuilding
from cached HDF5 output).
"""
function run_slayer_from_inputs(params::Vector{SLAYERParameters},
                                 dp_matrix::AbstractMatrix,
                                 control::SLAYERControl)
    validate(control)
    control.enabled || return empty_slayer_result(control)
    isempty(params) && return empty_slayer_result(control)

    n = length(params)
    size(dp_matrix) == (n, n) ||
        throw(ArgumentError("run_slayer: dp_matrix size $(size(dp_matrix)) " *
                             "≠ ($n, $n)"))
    dp = Matrix{ComplexF64}(dp_matrix)

    model = _build_inner_model(control.inner_model)

    # Per-surface SurfaceCoupling objects
    scs = [_build_surface_coupling(model, params[k], dp[k, k]) for k in 1:n]

    Q_root = ComplexF64[]
    omega_Hz = Float64[]
    gamma_Hz = Float64[]
    per_surface_extraction = GrowthRateResult[]
    coupled_extraction = nothing
    scan_data_list = Any[]

    if control.coupling_mode === :uncoupled
        for sc in scs
            scan = _run_scan(sc, control)
            gr   = find_growth_rates(scan, sc.tauk;
                    pole_threshold=control.pole_threshold,
                    filter_above_poles=control.filter_above_poles,
                    filter_outside_re=control.filter_outside_re)
            push!(Q_root, gr.Q_root)
            push!(omega_Hz, gr.omega_Hz)
            push!(gamma_Hz, gr.gamma_Hz)
            push!(per_surface_extraction, gr)
            control.store_scan && push!(scan_data_list, scan)
        end

    elseif control.coupling_mode === :coupled
        m_use = min(control.msing_max, n)
        mc = multi_surface_coupling(scs, dp; ref_idx=1, msing_max=m_use)
        scan = _run_scan(mc, control)
        ref_tauk = scs[1].tauk
        gr = find_growth_rates(scan, ref_tauk;
                pole_threshold=control.pole_threshold,
                filter_above_poles=control.filter_above_poles,
                filter_outside_re=control.filter_outside_re)
        push!(Q_root, gr.Q_root)
        push!(omega_Hz, gr.omega_Hz)
        push!(gamma_Hz, gr.gamma_Hz)
        coupled_extraction = gr
        control.store_scan && push!(scan_data_list, scan)
    end

    return SLAYERResult(true, control, params, dp,
                         Q_root, omega_Hz, gamma_Hz,
                         per_surface_extraction, coupled_extraction,
                         scan_data_list)
end

# ---------------------------------------------------------------------
# Full pipeline: equilibrium + ForceFreeStates → parameters → analysis
# ---------------------------------------------------------------------
"""
    run_slayer(equil, ffs_intr, control, toml_section;
                dir_path="./") -> SLAYERResult

Orchestrate the full SLAYER analysis against a solved
`PlasmaEquilibrium` and `ForceFreeStatesInternal`. Kinetic profiles are
loaded according to `control.profile_source` (either inline from
`toml_section["profiles"]` or from the HDF5 file `control.profile_file`
relative to `dir_path`). Per-surface parameters are built via
`build_slayer_inputs`; the outer-region Δ' matrix is pulled from
`ffs_intr.delta_prime_matrix` (or, if empty, from the diagonal
`sing.delta_prime` entries).

Returns an `enabled=false` `SLAYERResult` when `control.enabled` is
false.
"""
function run_slayer(equil, ffs_intr, control::SLAYERControl,
                     toml_section::AbstractDict; dir_path::AbstractString="./")
    validate(control)
    control.enabled || return empty_slayer_result(control)
    isempty(ffs_intr.sing) && return empty_slayer_result(control)

    profiles = _load_profiles(control, toml_section, dir_path)

    bt = control.bt === nothing ? equil.config.b0exp : control.bt
    params = build_slayer_inputs(equil, ffs_intr.sing, profiles;
                                  bt=bt,
                                  mu_i=control.mu_i,
                                  zeff=control.zeff,
                                  chi_perp=control.chi_perp,
                                  chi_tor=control.chi_tor,
                                  dr_val=control.dr_val,
                                  dgeo_val=control.dgeo_val,
                                  dc_type=control.dc_type,
                                  theta=control.theta_sample)

    # Δ' matrix: prefer the parallel-FM STRIDE-style full matrix; fall
    # back to a diagonal built from each SingType's scalar delta_prime.
    dp = if !isempty(ffs_intr.delta_prime_matrix) &&
            size(ffs_intr.delta_prime_matrix) == (length(params), length(params))
        Matrix{ComplexF64}(ffs_intr.delta_prime_matrix)
    else
        M = zeros(ComplexF64, length(params), length(params))
        for (k, s) in enumerate(ffs_intr.sing)
            M[k, k] = isempty(s.delta_prime) ? 0.0+0im : s.delta_prime[1]
        end
        M
    end

    return run_slayer_from_inputs(params, dp, control)
end
