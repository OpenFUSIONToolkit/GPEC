# SLAYERTearing.jl
#
# Tearing scan core: χ inputs, the slab Δ′ reference conversion, `run_slayer_from_inputs`
# (scan + growth-rate extraction), and the critical resonant field.

# χ⊥,e/χ⊥,i/χ_φ for the layer: the kinetic file's profiles where present, else the kinetic data's
# `chi_*_default`. With `warn`, a default is reported only for a χ the selected P_perp model
# (and, for χ⊥,e, the critical Δ) actually uses.
function _slayer_chi_inputs(loaded, control, data::KineticProfileData; warn::Bool=true)
    model = control.P_perp_model
    pick(f, s) = f === nothing ? s : f
    model in (:chi_perp_i, :D_perp) && loaded.chi_perp_i === nothing && data.chi_i_default === nothing &&
        throw(
            ArgumentError(
                "SLAYER: P_perp_model = $model needs the ion diffusivity χ⊥,i: add a chi_i profile to the " *
                "kinetic file, set chi_i_default on the kinetic data, or choose P_perp_model = \"chi_perp_e\"."
            )
        )
    model === :tau_E && data.tau_E === nothing &&
        throw(ArgumentError("SLAYER: P_perp_model = tau_E needs the confinement time tau_E on the kinetic data"))
    if warn
        uses_e = model in (:chi_perp_e, :D_perp) || control.D_c_type !== :none
        uses_i = model in (:chi_perp_i, :D_perp)
        uses_tor = model !== :tau_E
        fallbacks = String[]
        uses_e && loaded.chi_perp_e === nothing && push!(fallbacks, "chi_e → chi_e_default = $(data.chi_e_default)")
        uses_i && loaded.chi_perp_i === nothing && push!(fallbacks, "chi_i → chi_i_default = $(data.chi_i_default)")
        uses_tor && loaded.chi_tor === nothing && push!(fallbacks, "chi_phi → chi_phi_default = $(data.chi_phi_default)")
        isempty(fallbacks) || @warn "SLAYER: no usable kinetic-file profile (dataset absent or all-zero) for " *
                                    "$(join(fallbacks, ", ")); using the default."
        control.chi_perp_Dc !== nothing && control.D_c_type !== :none &&
            @info "SLAYER: the critical Δ uses chi_perp_Dc = $(control.chi_perp_Dc) m²/s for χ⊥,e; P_perp keeps its own χ inputs."
        model === :tau_E && @info "SLAYER: P_perp_model = tau_E sets P_perp and P_tor from a uniform " *
              "χ_E = (3/16)a²/τ̄_E with τ̄_E = $(data.tau_E) s; the χ_φ profile is not used."
    end
    return (; chi_perp_e=pick(loaded.chi_perp_e, data.chi_e_default),
        chi_perp_i=pick(loaded.chi_perp_i, data.chi_i_default),
        chi_tor=pick(loaded.chi_tor, data.chi_phi_default))
end

# Surface label for warnings; GGJ params carry only the singular-surface index `ising`.
_surface_label(p::SLAYERParameters) = "m=$(p.m), n=$(p.n)"
_surface_label(p::InnerLayerParameters) = "ising=$(getfield(p, :ising))"

# Is this a GGJ inner-layer model? Its growth rates come from `ggj_growth_rates`, not this runner.
_is_ggj(::GGJModel) = true
_is_ggj(::Any) = false

# Scan dispatch
function _run_scan(f, control::TearingConfig)
    if control.scan_mode === :brute_force
        return brute_force_scan(f, control.Q_re_range, control.Q_im_range;
            nre=control.nre, nim=control.nim)
    elseif control.scan_mode === :amr
        if !isempty(control.boxes)
            # Multi-box layout: pole threshold 10 × median(|Δ|) over a coarse 16×6 sample of the boxes' union.
            ω_lo = minimum(b[1] for b in control.boxes)
            ω_hi = maximum(b[2] for b in control.boxes)
            γ_lo = minimum(b[3] for b in control.boxes)
            γ_hi = maximum(b[4] for b in control.boxes)
            coarse_pts = ComplexF64[ComplexF64(ω, γ)
                                    for ω in range(ω_lo, ω_hi; length=16)
                                    for γ in range(γ_lo, γ_hi; length=6)]
            coarse_Δ = ComplexF64[ComplexF64(f(q)) for q in coarse_pts]
            finite = filter(z -> isfinite(z) && abs(z) < 1e30, coarse_Δ)
            pole_thr = isempty(finite) ? 1e8 : 10.0 * median(abs.(finite))
            boxes_in = [((b[1], b[2]), (b[3], b[4])) for b in control.boxes]
            return multi_box_amr_scan(f, boxes_in;
                pole_magnitude_threshold=pole_thr,
                prescreen_nre=control.multi_box_prescreen_n,
                prescreen_nim=control.multi_box_prescreen_n,
                nre0=control.nre, nim0=control.nim,
                passes=control.amr_passes,
                max_cells=control.amr_max_cells,
                max_cells_action=:warn_truncate) |>
                   as_amr_result        # downstream expects AMRResult
        end
        return amr_scan(f, control.Q_re_range, control.Q_im_range;
            nre0=control.nre, nim0=control.nim,
            passes=control.amr_passes,
            max_cells=control.amr_max_cells)
    end
    throw(ArgumentError("_run_scan: unknown scan_mode=$(control.scan_mode)"))
end

# Surface coupling for SLAYER: scale = lu^(1/3), tauk from the surface, dc from the χ‖ proxy.
function _build_surface_coupling(model::SLAYERModel, params::SLAYERParameters,
    dp_diag, q_shift::Real)
    return surface_coupling(model, params, dp_diag; dc=params.dc_tmp, q_shift=q_shift)
end

# "m/n" label keying `control.omega_E_kHz`; GGJ parameters carry no mode numbers.
_mn_label(p::SLAYERParameters) = "$(p.m)/$(p.n)"
_mn_label(::InnerLayerParameters) = nothing

# Per-surface Ω_E per unit n [rad/s]: the kinetic file's `omega_E` (NaN if none), overridden by `omega_E_kHz` by m/n.
function _omega_E_per_surface(control, labels::AbstractVector, omega_E_file::AbstractVector{<:Real})
    n = length(labels)
    isempty(omega_E_file) || length(omega_E_file) == n ||
        throw(ArgumentError("run_slayer: omega_E has $(length(omega_E_file)) entries but " *
                            "$n rational surfaces were analysed."))
    Ω_E = isempty(omega_E_file) ? fill(NaN, n) : Float64.(omega_E_file)
    isempty(control.omega_E_kHz) && return Ω_E
    any(isnothing, labels) && throw(ArgumentError("run_slayer: omega_E_kHz is keyed by m/n, which these surfaces do not carry."))
    unknown = setdiff(keys(control.omega_E_kHz), labels)
    isempty(unknown) ||
        throw(ArgumentError("run_slayer: omega_E_kHz lists m/n $(sort(collect(unknown))) matching no analysed surface; " *
                            "the analysed surfaces are $(labels)."))
    for (k, label) in enumerate(labels)
        haskey(control.omega_E_kHz, label) && (Ω_E[k] = 2π * 1e3 * control.omega_E_kHz[label])
    end
    return Ω_E
end

"""
    _q_shift(p::SLAYERParameters, Ω_E) -> Float64

Real offset added to a surface's scanned inner-layer `Q` for E×B rotation `Ω_E` per unit n [rad/s].
The layer's `Q` is the mode frequency in the local E×B frame (Burgess et al. 2026, Table 1), while
the coupled scan runs in the lab frame, so

```
Q_layer = τ_k·ω_lab − τ_k·n·Ω_E    ⇒    q_shift = −τ_k·n·Ω_E
```
"""
function _q_shift(p::SLAYERParameters, Ω_E::Real)
    iszero(Ω_E) || p.n >= 1 ||
        throw(ArgumentError("run_slayer: E×B rotation Ω_E=$(Ω_E) rad/s needs the toroidal mode number, but SLAYERParameters has n=$(p.n)"))
    return -p.tauk * p.n * Ω_E
end

# Reference-length conversion of the outer Δ' for the slab layer
"""
    delta_prime_to_rs_reference(dp_matrix, params) -> Matrix{ComplexF64}

Convert the outer-region Δ' matrix from its ψ_N reference length (STRIDE/BVP convention) to the
r_s-based `x̂ = (r − r_s)/r_s` reference of the slab layer (Fitzpatrick 2023 convention, shared by
`dc_tmp` and the `S^(1/3)` Δ(Q) scale). With the Mercier exponent `α = √(−D_I)` (Glasser, Greene &
Johnson 1975, Eqs. 16, 24; Glasser, Wang & Park 2016, Eq. 26) and `K = r_s·(dψ_N/dr)|_s`,

    Δ̂_ij = K_i^(1/2+α_i) · Δ'_ij · K_j^(α_j−1/2)

which reduces to `Δ̂ = r_s·Δ'` at α = 1/2. The off-diagonal split is a similarity transform that
leaves every dispersion root unchanged. `K` and `α` come from `SLAYERParameters.k_ref` /
`.alpha_mercier`; hand-built parameters default to `k_ref = 1`, making this the identity.
"""
function delta_prime_to_rs_reference(dp_matrix::AbstractMatrix,
    params::AbstractVector{<:SLAYERParameters})
    dl = [p.k_ref^(0.5 + p.alpha_mercier) for p in params]
    dr = [p.k_ref^(p.alpha_mercier - 0.5) for p in params]
    return Diagonal(dl) * Matrix{ComplexF64}(dp_matrix) * Diagonal(dr)
end

# Core analysis entry point on pre-built parameters
"""
    run_slayer_from_inputs(model::InnerLayerModel, params, dp_matrix, problem, config;
                           rational_psi, rational_q, omega_E) -> TearingResult

Run the tearing scan and root extraction for SLAYER `model` on pre-built per-surface parameters
and the outer-region Δ' matrix, skipping the `TearingProblem` setup. `omega_E` is the E×B angular
frequency per unit n at each surface in `rad/s`, overridden by `problem.omega_E_kHz`; empty means
unknown (no shift, reported as NaN). In coupled mode it Doppler-shifts each surface's Q; an uncoupled
layer is solved in its own E×B frame and its root moved by τ_k·n·Ω_E. Roots are reported in the lab frame.
"""
function run_slayer_from_inputs(model::InnerLayer.InnerLayerModel,
    params::AbstractVector{<:InnerLayerParameters},
    dp_matrix::AbstractMatrix,
    problem::TearingProblem, config::TearingConfig;
    rational_psi::Vector{Float64}=Float64[],
    rational_q::Vector{Float64}=Float64[],
    omega_E::Vector{Float64}=Float64[])
    validate(problem, config)
    _is_ggj(model) && throw(ArgumentError(
        "run_slayer_from_inputs: this runner scans SLAYER's tearing-only residual, which gives no GGJ growth rate. " *
        "Use Dispersion.ggj_growth_rates on multi_surface_coupling_full, which matches both parity channels, and " *
        "solve_inner for the GGJ layer's Δ."))
    isempty(params) && return empty_tearing_result(problem, config, model)

    n = length(params)
    size(dp_matrix) == (n, n) ||
        throw(ArgumentError("run_slayer: dp_matrix size $(size(dp_matrix)) " *
                            "≠ ($n, $n)"))
    dp = Matrix{ComplexF64}(dp_matrix)

    # Without this, a parameter-type mismatch is an opaque MethodError in `_build_surface_coupling`.
    expected_P = SLAYERParameters
    all(p -> p isa expected_P, params) ||
        throw(
            ArgumentError(
                "run_slayer: a $(typeof(model)) inner model requires " *
                "$(expected_P) per-surface parameters, but got eltype " *
                "$(eltype(params)). Build them with build_slayer_inputs.")
        )

    # Convert Δ' from its ψ_N reference length to the r_s-based convention shared by the layer Δ(Q) and the
    # critical-Δ (see `delta_prime_to_rs_reference`).
    dp = delta_prime_to_rs_reference(dp, params)

    # E×B rotation only Doppler-shifts the coupled determinant: each uncoupled layer is already
    # solved in its own plasma frame, so a shift there would only relabel the reported frequency.
    coupled = problem.coupling_mode === :coupled
    Ω_E = _omega_E_per_surface(problem, [_mn_label(p) for p in params], omega_E)
    # A surface with no known rotation applies no shift.
    Ω_shift = [isnan(Ω) ? 0.0 : Ω for Ω in Ω_E]
    q_shifts = coupled ? Float64[_q_shift(params[k], Ω_shift[k]) for k in 1:n] : zeros(Float64, n)
    scs = [_build_surface_coupling(model, params[k], dp[k, k], q_shifts[k]) for k in 1:n]
    # Coupled Q is scanned in the reference surface's normalization.
    ref_idx = 1
    # In the scanned Q, surface k's layer response moves by τ_ref·n·Ω_E,k. Heuristic check only:
    # the root also carries the diamagnetic Re(Q), so a window can pass and still miss it.
    if any(!iszero, q_shifts)
        m_use = min(problem.msing_max, n)
        doppler = [-q_shifts[k] * scs[ref_idx].tauk / scs[k].tauk for k in 1:m_use]
        lo, hi = config.Q_re_range
        any(d -> !(lo <= d <= hi), doppler) && @warn(
            "SLAYER: E×B Doppler offsets Re(Q)=$(round.(doppler; digits=3)) fall outside " *
            "Q_re_range=$(config.Q_re_range); widen it or the coupled root may be missed.")
    end

    # Diagnostic layer thickness [m] from the del_s Riccati solve (SLAYER only).
    layer_widths = eltype(params) <: SLAYERParameters ?
                   LayerWidths[slayer_layer_thickness(params[k]) for k in 1:n] :
                   LayerWidths[]

    Q_root = ComplexF64[]
    omega = Float64[]
    gamma = Float64[]
    per_surface_extraction = GrowthRateResult[]
    coupled_extraction = nothing
    scan_data_list = Union{ScanResult,AMRResult}[]

    # Adaptive pole threshold 10 × median(|Δ|) over the scan; the median is robust to near-pole samples.
    function _pole_threshold_for(scan)
        config.pole_threshold_adaptive || return config.pole_threshold
        Δ_arr = hasproperty(scan, :Δ) ? scan.Δ : nothing
        Δ_arr === nothing && return config.pole_threshold
        finite = filter(z -> isfinite(z) && abs(z) < 1e30, Δ_arr)
        isempty(finite) && return config.pole_threshold
        return 10.0 * median(abs.(finite))
    end

    classification = config.root_classification === :auto ? :threshold : config.root_classification
    winding = classification === :winding

    if problem.coupling_mode === :uncoupled
        for (k, sc) in enumerate(scs)
            scan = _run_scan(sc, config)
            pthr = _pole_threshold_for(scan)
            gr = find_growth_rates(scan, sc.tauk;
                pole_threshold=pthr,
                filter_above_poles=config.filter_above_poles,
                filter_outside_re=config.filter_outside_re,
                gamma_gap_threshold=config.gamma_gap_threshold,
                residual=(config.polish_roots || winding) ? sc : nothing,
                polish=config.polish_roots,
                validity_rtol=config.validity_rtol,
                classification=classification)
            :no_root in gr.warning_flags && @warn(
                "SLAYER: no usable growth-rate root found for surface " *
                "$(_surface_label(params[k])); reported γ=0 is a " *
                "placeholder, not a physical result — check scan grid / " *
                "pole_threshold.")
            # The layer is solved in its own E×B frame; report the root in the lab frame, Q_lab = Q + τ_k·n·Ω_E.
            q_lab = -_q_shift(params[k], Ω_shift[k])
            push!(Q_root, gr.Q_root + q_lab)
            push!(omega, :no_root in gr.warning_flags ? gr.omega : gr.omega + q_lab / params[k].tauk)
            push!(gamma, gr.gamma)
            push!(per_surface_extraction, gr)
            config.store_scan && push!(scan_data_list, scan)
        end

    elseif problem.coupling_mode === :coupled
        m_use = min(problem.msing_max, n)
        mc = multi_surface_coupling(scs, dp; ref_idx=ref_idx, msing_max=m_use)
        scan = _run_scan(mc, config)
        pthr = _pole_threshold_for(scan)
        ref_tauk = scs[ref_idx].tauk
        # No polishing or validity gate: det(D'−D(Q)) is ill-conditioned, so the root is the raw contour
        # intersection with its interpolation error. :winding uses the determinant only to count zeros and poles.
        gr = find_growth_rates(scan, ref_tauk;
            pole_threshold=pthr,
            filter_above_poles=config.filter_above_poles,
            filter_outside_re=config.filter_outside_re,
            gamma_gap_threshold=config.gamma_gap_threshold,
            residual=winding ? mc : nothing,
            polish=false,
            classification=classification)
        :no_root in gr.warning_flags && @warn(
            "SLAYER: no usable growth-rate root found for the coupled " *
            "$(m_use)-surface determinant; reported γ=0 is a placeholder, " *
            "not a physical result — check scan grid / pole_threshold.")
        push!(Q_root, gr.Q_root)
        push!(omega, gr.omega)
        push!(gamma, gr.gamma)
        coupled_extraction = gr
        config.store_scan && push!(scan_data_list, scan)
    end

    return TearingResult(problem, true, config, model, params, rational_psi, rational_q, dp,
        Q_root, omega, gamma,
        per_surface_extraction, coupled_extraction,
        layer_widths, Ω_E, q_shifts, scan_data_list,
        run_critical_resonant_field(model, params, Ω_E, real.(diag(dp)), problem.critical_resonant_field))
end

# Critical resonant field (torque balance)
"""
    run_critical_resonant_field(model, params, omega_E, dp_diag, ctrl) -> CriticalResonantFieldResult

Critical normalized resonant field b_r/B_φ for error-field penetration at each SLAYER
surface, from the torque balance of Cole and Fitzpatrick, Phys. Plasmas 13, 032503 (2006),
Eq. 62. The magnetic Prandtl number is the layer's `P_tor`, and the natural rotation is the
E×B frequency of the m/n mode, Q0 = τ_k·n·Ω_E, with `omega_E` the resolved Ω_E per unit n at
each surface [rad/s]. The outer stability is α = S^(-1/3)·(Δ_crit − r_s Δ'), with `dp_diag` the
r_s-referenced Δ' diagonal and Δ_crit the layer's critical-Δ offset, as in the dispersion relation.
A surface whose Ω_E is unknown (NaN), or that is tearing-unstable (α ≤ 0), is skipped with a
warning; if no surface has a known Ω_E, the analysis cannot run and this throws.
"""
function run_critical_resonant_field(model::InnerLayer.InnerLayerModel, params::AbstractVector{<:InnerLayerParameters},
    omega_E::AbstractVector{<:Real}, dp_diag::AbstractVector{<:Real}, ctrl::Union{Nothing,CriticalResonantFieldControl})
    ctrl === nothing && return empty_critical_resonant_field_result()
    model isa SLAYERModel{:fitzpatrick} && eltype(params) <: SLAYERParameters ||
        throw(ArgumentError("run_critical_resonant_field: the critical resonant field needs the SLAYER " *
                            "fitzpatrick model and SLAYER parameters; got $(typeof(model))."))
    nsurf = length(params)
    length(dp_diag) == nsurf ||
        throw(ArgumentError("run_critical_resonant_field: dp_diag has $(length(dp_diag)) entries for $nsurf surfaces."))
    length(omega_E) == nsurf ||
        throw(ArgumentError("run_critical_resonant_field: omega_E has $(length(omega_E)) entries for $nsurf surfaces."))
    nsurf > 0 && all(isnan, omega_E) &&
        throw(ArgumentError("run_critical_resonant_field: the critical resonant field needs the E×B rotation, but the " *
                            "kinetic file has no omega_E and no [SLAYER] omega_E_kHz entry covers the surfaces."))
    Q_peak, br_crit = fill(NaN, nsurf), fill(NaN, nsurf)
    # Cole Sec. IV: ω0 is the mode frequency in the E×B frame; diamagnetic parts enter via Q_e, Q_i.
    Q0 = Float64[p.tauk * p.n * Ω for (p, Ω) in zip(params, omega_E)]
    scan = CriticalResonantFieldScan[]
    for (k, p) in enumerate(params)
        alpha = (p.dc_tmp - dp_diag[k]) / p.lu^(1 / 3)
        tb = TorqueBalance(model, p, Q0[k], p.P_tor, p.lu, 2.0 / p.sval_r^2, alpha)
        Qs, bal, Q_peak[k], br_crit[k], _, Δs = torque_balance_scan(tb; Qmin=ctrl.Qmin, Qmax=ctrl.Qmax, n=ctrl.n)
        ctrl.store_scan && push!(scan, CriticalResonantFieldScan(collect(Float64, Qs), bal, Δs))
    end
    br_crit_tesla = br_crit .* Float64[abs(p.bt) for p in params]
    return CriticalResonantFieldResult(true, Int[p.ising for p in params], Q_peak, br_crit, br_crit_tesla, Q0, scan)
end

