# run_slayer.jl
#
# Tearing scan core: profile loading, the slab Δ′ reference conversion, and
# `run_slayer_from_inputs` (scan + growth-rate extraction). `TearingProblem` builds its inputs.

# ---------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------
# Read kinetic profiles through the shared `Equilibrium.read_kinetic_file`
# reader and adapt them to the SLAYER inner layer. Returns the spline-based
# `KineticProfiles` plus χ⊥,e(ψ)/χ⊥,i(ψ)/χ_φ(ψ) callables built from the file's
# `chi_e`/`chi_i`/`chi_phi` (or `nothing` when the file carries no usable χ, in which
# case the caller falls back to the scalar `control.chi_perp_e`/`chi_perp_i`/`chi_tor`), and
# a Z_eff(ψ) callable: the file's own `zeff` profile when it carries one, else
# derived from its `n_i/n_e` (or `nothing` when the file carries neither, in which
# case the caller falls back to the scalar `control.zeff`), and whether the file
# carries an `omega_E` profile.
function _load_profiles(control::SLAYERControl, dir_path::AbstractString)
    isempty(control.profile_file) &&
        error("run_slayer: [SLAYER] profile_file is empty — point it at a " *
              "kinetic-profile file (HDF5 GPEC kinetic schema or ASCII table).")
    path = isabspath(control.profile_file) ? control.profile_file :
           joinpath(dir_path, control.profile_file)
    data = read_kinetic_file(path; group=control.profile_group)

    for (name, v) in (("n_e", data.n_e), ("T_e", data.T_e), ("T_i", data.T_i))
        v === nothing &&
            error("run_slayer: kinetic file '$path' is missing required " *
                  "dataset '$name' for the SLAYER inner layer.")
    end
    # `omega_E` carries the E×B rotation Ω_E (per unit n) when the file has it, else zero (no shift);
    # ω_*e/ω_*i are derived per surface from the density and temperature splines by `build_slayer_inputs`.
    npsi = length(data.psi)
    omega_E = data.omega_E === nothing ? zeros(npsi) : data.omega_E
    profiles = KineticProfiles(; psi=data.psi, n_e=data.n_e, T_e=data.T_e,
        T_i=data.T_i, omega_E=omega_E)

    # χ⊥(ψ)/χ_φ(ψ) splines from the file. A χ array that is absent OR all-zero
    # is treated as "not provided" — χ must be positive (χ=0 ⇒ τ_⊥→∞), and the
    # all-zero sentinel lets a file keep the chi_e/chi_phi keys while deferring
    # to the scalar control.chi_perp_e/chi_tor fallback. Build the spline only
    # from a usable (present, not all-zero) array.
    psi_xs = collect(Float64, data.psi)
    function _chi_spline(name, v)
        (v === nothing || all(iszero, v)) && return nothing
        minimum(v) > 0 || error("run_slayer: kinetic file '$path' has $name with minimum $(minimum(v)) ≤ 0; diffusivities must be positive")
        return cubic_interp(psi_xs, collect(Float64, v))
    end
    model = control.P_perp_model
    uses_e = model in (:chi_perp_e, :D_perp) || control.D_c_type !== :none
    chi_perp_e = uses_e ? _chi_spline("chi_e", data.chi_e) : nothing
    chi_perp_i = model in (:chi_perp_i, :D_perp) ? _chi_spline("chi_i", data.chi_i) : nothing
    chi_tor = model !== :tau_E ? _chi_spline("chi_phi", data.chi_phi) : nothing
    # Z_eff(ψ): a profile supplied in the file wins, since a real impurity mix is not recoverable
    # from n_i/n_e. Otherwise quasi-neutrality with one impurity species of charge `impurity_charge` and a
    # charge-1 main ion, the single-ion closure the kinetic spline set uses:
    # Z_eff = impurity_charge − (n_i/n_e)(impurity_charge − 1).
    _usable(v) = v !== nothing && !all(iszero, v)
    zeff = if _usable(data.zeff)
        minimum(data.zeff) >= 1.0 - 1e-8 ||
            error("run_slayer: kinetic file '$path' has zeff = $(minimum(data.zeff)) < 1 — unphysical")
        cubic_interp(psi_xs, collect(Float64, data.zeff))
    elseif !_usable(data.n_i)
        nothing
    else
        z = [ne > 0 ? control.impurity_charge - (ni / ne) * (control.impurity_charge - 1.0) : control.impurity_charge for (ni, ne) in zip(data.n_i, data.n_e)]
        minimum(z) >= 1.0 - 1e-8 ||
            error("run_slayer: kinetic file '$path' has n_i > n_e, giving Z_eff = $(minimum(z)) < 1 — unphysical")
        cubic_interp(psi_xs, z)
    end
    zeff_source = _usable(data.zeff) ? :file : zeff === nothing ? :scalar : :closure
    return (profiles=profiles, chi_perp_e=chi_perp_e, chi_perp_i=chi_perp_i, chi_tor=chi_tor, zeff=zeff, zeff_source=zeff_source,
        has_omega_E=(data.omega_E !== nothing))
end

# χ⊥,e/χ⊥,i/χ_φ for the layer: the kinetic file's profiles where present, else the control's
# scalar fallbacks. With `warn`, a fallback is reported only for a χ the selected P_perp model
# (and, for χ⊥,e, the critical Δ) actually uses.
function _slayer_chi_inputs(loaded, control::SLAYERControl; warn::Bool=true)
    model = control.P_perp_model
    pick(f, s) = f === nothing ? s : f
    model in (:chi_perp_i, :D_perp) && loaded.chi_perp_i === nothing && control.chi_perp_i === nothing &&
        throw(
            ArgumentError(
                "SLAYER: P_perp_model = $model needs the ion diffusivity χ⊥,i: add a chi_i profile to the " *
                "kinetic file, set [SLAYER] chi_perp_i, or choose P_perp_model = \"chi_perp_e\"."
            )
        )
    if warn
        uses_e = model in (:chi_perp_e, :D_perp) || control.D_c_type !== :none
        uses_i = model in (:chi_perp_i, :D_perp)
        uses_tor = model !== :tau_E
        fallbacks = String[]
        uses_e && loaded.chi_perp_e === nothing && push!(fallbacks, "chi_e → chi_perp_e = $(control.chi_perp_e)")
        uses_i && loaded.chi_perp_i === nothing && push!(fallbacks, "chi_i → chi_perp_i = $(control.chi_perp_i)")
        uses_tor && loaded.chi_tor === nothing && push!(fallbacks, "chi_phi → chi_tor = $(control.chi_tor)")
        isempty(fallbacks) || @warn "SLAYER: no usable kinetic-file profile (dataset absent or all-zero) for " *
                                    "$(join(fallbacks, ", ")); using the scalar fallback."
        model === :tau_E && @info "SLAYER: P_perp_model = tau_E sets P_perp and P_tor from a uniform " *
              "χ_E = (3/16)a²/τ̄_E with τ̄_E = $(control.tau_E) s; the χ_φ profile is not used."
    end
    return (; chi_perp_e=pick(loaded.chi_perp_e, control.chi_perp_e),
        chi_perp_i=pick(loaded.chi_perp_i, control.chi_perp_i),
        chi_tor=pick(loaded.chi_tor, control.chi_tor))
end

# ---------------------------------------------------------------------
# Inner-layer model factory
# ---------------------------------------------------------------------
function _build_inner_model(name::Symbol)
    if name === :slayer_fitzpatrick
        return SLAYERModel(; variant=:fitzpatrick)
    end
    throw(ArgumentError("_build_inner_model: unknown model $name"))
end

# Map the TOML resistivity_model symbol to a NeoResistivityModel instance.
function _build_resistivity_model(name::Symbol)
    name === :sauter && return SauterNeoModel()
    name === :redl && return RedlNeoModel()
    name === :spitzer && return SpitzerModel()
    name === :spitzer_harm && return SpitzerHarmModel()
    throw(ArgumentError("_build_resistivity_model: unknown model $name"))
end

# Human-readable surface label for warnings. SLAYER params carry (m, n);
# GGJ params carry only the singular-surface index `ising`.
_surface_label(p::SLAYERParameters) = "m=$(p.m), n=$(p.n)"
_surface_label(p::InnerLayerParameters) = "ising=$(getfield(p, :ising))"

# Is this a GGJ inner-layer model? Its growth rates come from `ggj_growth_rates`, not this runner.
_is_ggj(::GGJModel) = true
_is_ggj(::Any) = false

# Winding number of an isolated zero of the dispersion residual, for
# root_classification = :winding. The SLAYER residual is a function of conj(Q)
# (solve_inner evaluates conj(Δ̂(ĝ)); see Riccati.jl), so a zero winds −1.
_zero_winding(::SLAYERModel{:fitzpatrick}) = -1
_zero_winding(model) =
    throw(ArgumentError("root_classification=:winding is not available for inner model $(typeof(model))"))

# ---------------------------------------------------------------------
# Scan dispatch
# ---------------------------------------------------------------------
function _run_scan(f, control::SLAYERControl)
    if control.scan_mode === :brute_force
        return brute_force_scan(f, control.Q_re_range, control.Q_im_range;
            nre=control.nre, nim=control.nim)
    elseif control.scan_mode === :amr
        if !isempty(control.boxes)
            # Multi-box stripe layout. Pole magnitude threshold for the
            # activity check is derived from a coarse 16×6 sample of the
            # union of all boxes, using 10 × median(|Δ|) (median is robust
            # to near-pole sample inflation; see the adaptive threshold note
            # below).
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
            # Convert NTuple{4,Float64} → ((ω_lo,ω_hi),(γ_lo,γ_hi)) tuples
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

# ---------------------------------------------------------------------
# Surface-coupling builder — dispatches on model type to thread the
# correct `scale` and `tauk` through the Dispersion API.
# ---------------------------------------------------------------------
# SLAYER: scale = lu^(1/3), tauk from the surface, dc from the χ‖ proxy.
function _build_surface_coupling(model::SLAYERModel, params::SLAYERParameters,
    dp_diag, q_shift::Real)
    return surface_coupling(model, params, dp_diag; dc=params.dc_tmp, q_shift=q_shift)
end

# "m/n" label keying `control.omega_E_kHz`; GGJ parameters carry no mode numbers.
_mn_label(p::SLAYERParameters) = "$(p.m)/$(p.n)"
_mn_label(::InnerLayerParameters) = nothing

# Per-surface E×B angular frequency Ω_E per unit n [rad/s]: the kinetic file's `omega_E` at each
# rational surface (NaN when there is none), replaced on every surface whose m/n `control.omega_E_kHz` lists.
function _omega_E_per_surface(control::SLAYERControl, params::AbstractVector, omega_E_file::AbstractVector{<:Real})
    n = length(params)
    isempty(omega_E_file) || length(omega_E_file) == n ||
        throw(ArgumentError("run_slayer: omega_E has $(length(omega_E_file)) entries but " *
                            "$n rational surfaces were analysed."))
    Ω_E = isempty(omega_E_file) ? fill(NaN, n) : Float64.(omega_E_file)
    isempty(control.omega_E_kHz) && return Ω_E
    labels = [_mn_label(p) for p in params]
    any(isnothing, labels) &&
        throw(ArgumentError("run_slayer: omega_E_kHz is keyed by m/n, which $(eltype(params)) surfaces do not carry."))
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

The conjugated layer Δ and the opposite-sign diamagnetic inputs (`Q_e = −τ_k·ω_*e`) flip together
and leave this sign unchanged.
"""
function _q_shift(p::SLAYERParameters, Ω_E::Real)
    iszero(Ω_E) || p.n >= 1 ||
        throw(ArgumentError("run_slayer: E×B rotation Ω_E=$(Ω_E) rad/s needs the toroidal mode number, but SLAYERParameters has n=$(p.n)"))
    return -p.tauk * p.n * Ω_E
end
# The runner builds SLAYER surfaces only; a GGJ surface takes its rotation from `surface_coupling(...; omega_E, n)`.
_q_shift(::InnerLayerParameters, ::Real) = 0.0

# ---------------------------------------------------------------------
# Reference-length conversion of the outer Δ' for the slab layer
# ---------------------------------------------------------------------
"""
    delta_prime_to_rs_reference(dp_matrix, params) -> Matrix{ComplexF64}

Convert the outer-region Δ' matrix from its ψ_N reference length (the
STRIDE/BVP convention: Frobenius coefficients normalized per unit Δψ_N) to
the r_s-based `x̂ = (r − r_s)/r_s` reference the slab layer works in
(Fitzpatrick 2023 convention — the same reference used by `dc_tmp` and by
the `S^(1/3)` Δ(Q) scale, both built on r_s).

Near surface `k` the outer solution combines the large and small Frobenius
solutions with the Mercier exponent `α = √(−D_I)`, `D_I = E + F + H − 1/4`
(Glasser, Greene & Johnson 1975, Eqs. 16, 24; STRIDE's `alpha`). The displacement
scales as `|x|^(−1/2 ± α)` (Glasser, Wang & Park 2016, Eq. 26), so the
flux-like variable `ψ ∝ x·ξ` of the slab Δ' definition scales as
`A_L·|x|^(1/2−α) + A_S·|x|^(1/2+α)`. Rescaling the radial variable
`x_ψ = K·x̂` with `K = r_s·(dψ_N/dr)|_s` maps the flux coefficients as
`Â_L = A_L·K^(1/2−α)` and `Â_S = A_S·K^(1/2+α)`, so the response matrix
(small coefficient at surface `i` per unit large coefficient at surface `j`)
transforms as

    Δ̂_ij = K_i^(1/2+α_i) · Δ'_ij · K_j^(α_j−1/2)

whose diagonal is `K^(2α)·Δ'_kk`; at D_I = −1/4 (α = 1/2) this reduces to
the textbook `Δ̂ = r_s·Δ'_phys`. The diagonal factor is the exponent gap and
is the same whichever variable carries the coefficients; using the
displacement instead shifts both exponents by one and changes the
off-diagonal split by `K_i/K_j`, a diagonal similarity transform `D·Δ'·D⁻¹`
that leaves every uncoupled and coupled dispersion root unchanged. `K` and
`α` are carried per surface in `SLAYERParameters.k_ref` / `.alpha_mercier`;
hand-built parameters default to `k_ref = 1`, making the conversion the
identity.
"""
function delta_prime_to_rs_reference(dp_matrix::AbstractMatrix,
    params::AbstractVector{<:SLAYERParameters})
    dl = [p.k_ref^(0.5 + p.alpha_mercier) for p in params]
    dr = [p.k_ref^(p.alpha_mercier - 0.5) for p in params]
    return Diagonal(dl) * Matrix{ComplexF64}(dp_matrix) * Diagonal(dr)
end

# ---------------------------------------------------------------------
# Core analysis entry point that takes pre-built parameters.
# ---------------------------------------------------------------------
"""
    run_slayer_from_inputs(model::InnerLayerModel, params, dp_matrix, control;
                           rational_psi, rational_q, omega_E) -> SLAYERResult

Run the tearing analysis scan core given the inner-layer dispatch `model`, the pre-built
per-surface parameters, and the outer-region Δ' matrix. Bypasses the orchestration of the
`TearingProblem` solve — use this when the parameters are already known (e.g. in unit
tests or when rebuilding from cached HDF5 output). The model is a typed object end to
end; nothing below this point reads `control.inner_model`. `omega_E` is the E×B angular
frequency per unit n at each surface [rad/s], replaced on any surface whose m/n
`control.omega_E_kHz` lists; empty means none is known. A surface with no known value applies no
shift and is reported as NaN. In coupled mode it Doppler-shifts each surface's inner-layer Q; in
both modes the resolved value is reported as `result.omega_E`. Roots are reported in the lab frame
in both modes: an uncoupled layer is solved in its own E×B frame and its root is then moved by
τ_k·n·Ω_E (Ω_E = 0 where it is unknown), so `γ` is the same and ω carries n·Ω_E.
"""
function run_slayer_from_inputs(model::InnerLayer.InnerLayerModel,
    params::AbstractVector{<:InnerLayerParameters},
    dp_matrix::AbstractMatrix,
    control::SLAYERControl;
    rational_psi::Vector{Float64}=Float64[],
    rational_q::Vector{Float64}=Float64[],
    omega_E::Vector{Float64}=Float64[])
    validate(control)
    _is_ggj(model) && throw(ArgumentError(
        "run_slayer_from_inputs: this runner scans SLAYER's tearing-only residual, which gives no GGJ growth rate. " *
        "Use Dispersion.ggj_growth_rates on multi_surface_coupling_full, which matches both parity channels, and " *
        "ggj_inner_deltas for the GGJ layer's Δ."))
    control.enabled || return empty_slayer_result(control)
    isempty(params) && return empty_slayer_result(control)

    n = length(params)
    size(dp_matrix) == (n, n) ||
        throw(ArgumentError("run_slayer: dp_matrix size $(size(dp_matrix)) " *
                            "≠ ($n, $n)"))
    dp = Matrix{ComplexF64}(dp_matrix)

    # Guard: the inner-layer model and the parameter eltype must match, or
    # `_build_surface_coupling` throws an opaque MethodError downstream.
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
    coupled = control.coupling_mode === :coupled
    Ω_E = _omega_E_per_surface(control, params, omega_E)
    # A surface with no known rotation applies no shift.
    Ω_shift = [isnan(Ω) ? 0.0 : Ω for Ω in Ω_E]
    q_shifts = coupled ? Float64[_q_shift(params[k], Ω_shift[k]) for k in 1:n] : zeros(Float64, n)
    scs = [_build_surface_coupling(model, params[k], dp[k, k], q_shifts[k]) for k in 1:n]
    # Coupled Q is scanned in the reference surface's normalization.
    ref_idx = 1
    # In the scanned Q, surface k's layer response moves by τ_ref·n·Ω_E,k. Heuristic check only:
    # the root also carries the diamagnetic Re(Q), so a window can pass and still miss it.
    if any(!iszero, q_shifts)
        m_use = min(control.msing_max, n)
        doppler = [-q_shifts[k] * scs[ref_idx].tauk / scs[k].tauk for k in 1:m_use]
        lo, hi = control.Q_re_range
        any(d -> !(lo <= d <= hi), doppler) && @warn(
            "SLAYER: E×B Doppler offsets Re(Q)=$(round.(doppler; digits=3)) fall outside " *
            "Q_re_range=$(control.Q_re_range); widen it or the coupled root may be missed.")
    end

    # Per-surface resistive layer thickness [m] via the del_s Riccati solve.
    # Independent of the dispersion scan / coupling mode — a pure diagnostic.
    # SLAYER-only: the del_s Riccati is defined on SLAYERParameters. GGJ
    # surfaces carry no analogous layer-thickness diagnostic, so leave it empty.
    layer_widths = eltype(params) <: SLAYERParameters ?
                   LayerWidths[slayer_layer_thickness(params[k]) for k in 1:n] :
                   LayerWidths[]

    Q_root = ComplexF64[]
    omega_Hz = Float64[]
    gamma_Hz = Float64[]
    per_surface_extraction = GrowthRateResult[]
    coupled_extraction = nothing
    scan_data_list = Union{ScanResult,AMRResult}[]

    # Helper: compute the pole_threshold actually passed to find_growth_rates.
    # When `control.pole_threshold_adaptive` is true, override with
    # `10 × median(|Δ|)` over the scan's dispersion residual array.
    #
    # The median formulation is robust against pre-screen samples landing
    # near a pole. A single near-pole sample inflates `|mean(Δ)|` by orders
    # of magnitude (and `|mean|` further collapses on oscillating residuals
    # whose phases cancel in the complex sum). 10 × median(|Δ|) reflects
    # "10× the typical residual magnitude" with median robust to both
    # pathologies.
    function _pole_threshold_for(scan)
        control.pole_threshold_adaptive || return control.pole_threshold
        # ScanResult and AMRResult both carry `.Δ` — abstract over both
        Δ_arr = hasproperty(scan, :Δ) ? scan.Δ : nothing
        Δ_arr === nothing && return control.pole_threshold
        finite = filter(z -> isfinite(z) && abs(z) < 1e30, Δ_arr)
        isempty(finite) && return control.pole_threshold
        return 10.0 * median(abs.(finite))
    end

    winding = control.root_classification === :winding
    zw = winding ? _zero_winding(model) : -1

    if control.coupling_mode === :uncoupled
        for (k, sc) in enumerate(scs)
            scan = _run_scan(sc, control)
            pthr = _pole_threshold_for(scan)
            gr = find_growth_rates(scan, sc.tauk;
                pole_threshold=pthr,
                filter_above_poles=control.filter_above_poles,
                filter_outside_re=control.filter_outside_re,
                gap_kHz_threshold=control.gap_kHz_threshold,
                residual=(control.polish_roots || winding) ? sc : nothing,
                polish=control.polish_roots,
                validity_rtol=control.validity_rtol,
                classification=control.root_classification,
                zero_winding=zw)
            :no_root in gr.warning_flags && @warn(
                "SLAYER: no usable growth-rate root found for surface " *
                "$(_surface_label(params[k])); reported γ=0 is a " *
                "placeholder, not a physical result — check scan grid / " *
                "pole_threshold.")
            # The layer is solved in its own E×B frame; report the root in the lab frame, Q_lab = Q + τ_k·n·Ω_E.
            q_lab = -_q_shift(params[k], Ω_shift[k])
            push!(Q_root, gr.Q_root + q_lab)
            push!(omega_Hz, :no_root in gr.warning_flags ? gr.omega_Hz : gr.omega_Hz + q_lab / params[k].tauk)
            push!(gamma_Hz, gr.gamma_Hz)
            push!(per_surface_extraction, gr)
            control.store_scan && push!(scan_data_list, scan)
        end

    elseif control.coupling_mode === :coupled
        m_use = min(control.msing_max, n)
        mc = multi_surface_coupling(scs, dp; ref_idx=ref_idx, msing_max=m_use)
        scan = _run_scan(mc, control)
        pthr = _pole_threshold_for(scan)
        ref_tauk = scs[ref_idx].tauk
        # Coupled path: no root polishing / validity gate. The m×m coupled
        # determinant det(D'−D(Q)) is ill-conditioned (its magnitude floors well
        # above zero), so |det|-based polishing is a no-op and the residual-scale
        # gate is unreliable. The coupled root is therefore the raw contour
        # intersection and carries the triangulation's interpolation error. With
        # root_classification = :winding the determinant is evaluated around
        # each intersection only to count its zeros and poles.
        gr = find_growth_rates(scan, ref_tauk;
            pole_threshold=pthr,
            filter_above_poles=control.filter_above_poles,
            filter_outside_re=control.filter_outside_re,
            gap_kHz_threshold=control.gap_kHz_threshold,
            residual=winding ? mc : nothing,
            polish=false,
            classification=control.root_classification,
            zero_winding=zw)
        :no_root in gr.warning_flags && @warn(
            "SLAYER: no usable growth-rate root found for the coupled " *
            "$(m_use)-surface determinant; reported γ=0 is a placeholder, " *
            "not a physical result — check scan grid / pole_threshold.")
        push!(Q_root, gr.Q_root)
        push!(omega_Hz, gr.omega_Hz)
        push!(gamma_Hz, gr.gamma_Hz)
        coupled_extraction = gr
        control.store_scan && push!(scan_data_list, scan)
    end

    return SLAYERResult(true, control, model, params, rational_psi, rational_q, dp,
        Q_root, omega_Hz, gamma_Hz,
        per_surface_extraction, coupled_extraction,
        layer_widths, Ω_E, q_shifts, scan_data_list,
        run_critical_resonant_field(model, params, Ω_E, real.(diag(dp)), control.critical_resonant_field))
end

# ---------------------------------------------------------------------
# Critical resonant field (torque balance)
# ---------------------------------------------------------------------
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
    omega_E::AbstractVector{<:Real}, dp_diag::AbstractVector{<:Real}, ctrl::CriticalResonantFieldControl)
    ctrl.enabled || return empty_critical_resonant_field_result()
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

# ---------------------------------------------------------------------
# GGJ parity-Δ diagnostic
# ---------------------------------------------------------------------
"""
    ggj_inner_deltas(params::AbstractVector{GGJParameters}, Q::Number;
                     solver=:galerkin) -> Vector{NamedTuple}

Evaluate both parity channels of the GGJ inner-layer matching data at the
complex normalized growth rate `Q` for each rational surface. Returns a
vector of `(ising, tearing, interchange)` named tuples, where `tearing`
(GWP Δ₋, odd) is the reconnecting channel and `interchange` (GWP Δ₊, even) the
non-reconnecting Glasser-stabilization channel (see `InnerLayerResponse`).

Both parity Δ's are physical and directly accessible here. For GGJ growth
rates use `Dispersion.ggj_growth_rates`, which matches both channels through
the full 4m×4m determinant.
"""
function ggj_inner_deltas(params::AbstractVector{GGJParameters}, Q::Number;
    solver::Symbol=:galerkin)
    model = GGJModel(; solver=solver)
    out = Vector{NamedTuple{(:ising, :tearing, :interchange),
        Tuple{Int,ComplexF64,ComplexF64}}}(undef, length(params))
    for (k, p) in enumerate(params)
        r = solve_inner(model, p, ComplexF64(Q))
        out[k] = (ising=p.ising, tearing=r.tearing, interchange=r.interchange)
    end
    return out
end
