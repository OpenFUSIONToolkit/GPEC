# Runner.jl
#
# Top-level orchestration for the SLAYER tearing-mode analysis. Given a
# `ForceFreeStatesResult` (which supplies the equilibrium, the rational-surface
# list and the outer-region Δ' matrix) + a
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
# Profile loading
# ---------------------------------------------------------------------
# Read kinetic profiles through the shared `Equilibrium.read_kinetic_file`
# reader and adapt them to the SLAYER inner layer. Returns the spline-based
# `KineticProfiles` plus χ⊥(ψ)/χ_φ(ψ) callables built from the file's
# `chi_e`/`chi_phi` (or `nothing` when the file carries no usable χ, in which
# case the caller falls back to the scalar `control.chi_perp`/`chi_tor`).
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
    # ω_*e/ω_*i are recomputed per-surface from equilibrium gradients
    # (compute_omega_star), so the diamagnetic-frequency inputs here are
    # placeholders; `omega` carries the ExB rotation when present.
    npsi = length(data.psi)
    omega = data.omega_E === nothing ? zeros(npsi) : data.omega_E
    profiles = KineticProfiles(; psi=data.psi, n_e=data.n_e, T_e=data.T_e,
        T_i=data.T_i, omega=omega,
        omega_e=zeros(npsi), omega_i=zeros(npsi))

    # χ⊥(ψ)/χ_φ(ψ) splines from the file. A χ array that is absent OR all-zero
    # is treated as "not provided" — χ must be positive (χ=0 ⇒ τ_⊥→∞), and the
    # all-zero sentinel lets a file keep the chi_e/chi_phi keys while deferring
    # to the scalar control.chi_perp/chi_tor fallback. Build the spline only
    # from a usable (present, not all-zero) array.
    psi_xs = collect(Float64, data.psi)
    _chi_spline(v) = (v === nothing || all(iszero, v)) ? nothing :
                     cubic_interp(psi_xs, collect(Float64, v))
    chi_perp = _chi_spline(data.chi_e)
    chi_tor = _chi_spline(data.chi_phi)
    return (profiles=profiles, chi_perp=chi_perp, chi_tor=chi_tor)
end

# ---------------------------------------------------------------------
# Inner-layer model factory
# ---------------------------------------------------------------------
function _build_inner_model(name::Symbol)
    if name === :slayer_fitzpatrick
        return SLAYERModel(; variant=:fitzpatrick)
    elseif name === :ggj_shooting
        return GGJModel(; solver=:shooting)
    elseif name === :ggj_galerkin
        return GGJModel(; solver=:galerkin)
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

# Is this an unvalidated GGJ inner-layer model? Used to gate the γ-extraction
# future-work warning.
_is_ggj(::GGJModel) = true
_is_ggj(::Any) = false

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
    dp_diag)
    return surface_coupling(model, params, dp_diag; dc=params.dc_tmp)
end

# GGJ: scale = 1.0 (rescale_delta applied inside solve_inner), tauk = 1.0,
# dc = 0 (the 4m×4m Pletzer-Dewar residual carries interchange stabilization
# natively). See the GGJ `surface_coupling` method.
function _build_surface_coupling(model::GGJModel, params::GGJParameters,
    dp_diag)
    return surface_coupling(model, params, dp_diag)
end

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

Near surface `k` the tearing-parity outer solution is
`A_L·|x|^(1/2−μ) + A_S·|x|^(1/2+μ)` with `μ = √(−D_I)` (Glasser, Wang &
Park 2016, Eq. 26). Rescaling the radial variable `x_ψ = K·x̂` with
`K = r_s·(dψ_N/dr)|_s` maps the coefficients as `Â_L = A_L·K^(1/2−μ)` and
`Â_S = A_S·K^(1/2+μ)`, so the response matrix (small coefficient at surface
`i` per unit large coefficient at surface `j`) transforms as

    Δ̂_ij = K_i^(1/2+μ_i) · Δ'_ij · K_j^(μ_j−1/2)

whose diagonal is `K^(2μ)·Δ'_kk`; at D_I = −1/4 (μ = 1/2) this reduces to
the textbook `Δ̂ = r_s·Δ'_phys`. `K` and `μ` are carried per surface in
`SLAYERParameters.k_ref` / `.mu_mercier`; hand-built parameters default to
`k_ref = 1`, making the conversion the identity.
"""
function delta_prime_to_rs_reference(dp_matrix::AbstractMatrix,
    params::AbstractVector)
    dl = [p.k_ref^(0.5 + p.mu_mercier) for p in params]
    dr = [p.k_ref^(p.mu_mercier - 0.5) for p in params]
    return Diagonal(dl) * Matrix{ComplexF64}(dp_matrix) * Diagonal(dr)
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
function run_slayer_from_inputs(params::AbstractVector{<:InnerLayerParameters},
    dp_matrix::AbstractMatrix,
    control::SLAYERControl;
    rational_psi::Vector{Float64}=Float64[],
    rational_q::Vector{Float64}=Float64[])
    validate(control)
    control.enabled || return empty_slayer_result(control)
    isempty(params) && return empty_slayer_result(control)

    n = length(params)
    size(dp_matrix) == (n, n) ||
        throw(ArgumentError("run_slayer: dp_matrix size $(size(dp_matrix)) " *
                            "≠ ($n, $n)"))
    dp = Matrix{ComplexF64}(dp_matrix)

    model = _build_inner_model(control.inner_model)

    # Guard: the inner-layer model and the parameter eltype must match, or
    # `_build_surface_coupling` throws an opaque MethodError downstream.
    expected_P = _is_ggj(model) ? GGJParameters : SLAYERParameters
    all(p -> p isa expected_P, params) ||
        throw(
            ArgumentError(
                "run_slayer: inner_model=$(control.inner_model) requires " *
                "$(expected_P) per-surface parameters, but got eltype " *
                "$(eltype(params)). Build inputs with the matching builder " *
                "(build_slayer_inputs for SLAYER, build_ggj_inputs for GGJ).")
        )

    # Slab-layer path: convert Δ' from its ψ_N reference length to the
    # r_s-based convention shared by the layer Δ(Q) and the critical-Δ (see
    # `delta_prime_to_rs_reference`). GGJ is genuinely toroidal/ψ-based
    # (its `rescale_delta` handles inner→outer units natively) — no conversion.
    if !_is_ggj(model)
        dp = delta_prime_to_rs_reference(dp, params)
    end

    # The coupled determinant uses the reduced m×m (tearing-only) form, which
    # drops the interchange channel. For GGJ that channel carries the Glasser
    # interchange stabilization, so coupled-GGJ results omit real physics.
    if _is_ggj(model) && control.coupling_mode === :coupled
        @warn(
            "SLAYER: coupling_mode=:coupled with a GGJ inner model uses the " *
            "reduced m×m tearing-only determinant, which DROPS the GGJ " *
            "interchange (Glasser stabilization) channel — results are " *
            "physically incomplete. Use the full Pletzer-Dewar matching " *
            "(multi_surface_coupling_full) for coupled GGJ studies."
        )
    end

    # GGJ growth-rate extraction by Re/Im contour matching is not yet
    # reliable (the contours do not robustly intersect at a dispersion-relation
    # zero). The scan still runs so the user can inspect/plot the contours and
    # access both parity Δ channels via `ggj_inner_deltas`, but the reported γ
    # is unvalidated and should be treated as future work.
    if _is_ggj(model)
        @warn(
            "SLAYER: GGJ γ-extraction by Re/Im contour matching is " *
            "FUTURE WORK — contours do not reliably intersect at a " *
            "dispersion-relation root. The scan grid and parity Δ channels " *
            "are provided for inspection; reported growth rates are " *
            "unvalidated placeholders."
        )
    end

    # Per-surface SurfaceCoupling objects
    scs = [_build_surface_coupling(model, params[k], dp[k, k]) for k in 1:n]

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

    if control.coupling_mode === :uncoupled
        for (k, sc) in enumerate(scs)
            scan = _run_scan(sc, control)
            pthr = _pole_threshold_for(scan)
            gr = find_growth_rates(scan, sc.tauk;
                pole_threshold=pthr,
                filter_above_poles=control.filter_above_poles,
                filter_outside_re=control.filter_outside_re,
                gap_kHz_threshold=control.gap_kHz_threshold,
                residual=control.polish_roots ? sc : nothing,
                validity_rtol=control.validity_rtol)
            :no_root in gr.warning_flags && @warn(
                "SLAYER: no usable growth-rate root found for surface " *
                "$(_surface_label(params[k])); reported γ=0 is a " *
                "placeholder, not a physical result — check scan grid / " *
                "pole_threshold.")
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
        pthr = _pole_threshold_for(scan)
        ref_tauk = scs[1].tauk
        # Coupled path: no root polishing / validity gate. The m×m coupled
        # determinant det(D'−D(Q)) is ill-conditioned (its magnitude floors well
        # above zero), so |det|-based polishing is a no-op and the residual-scale
        # gate is unreliable. A σ_min-based coupled refinement is a dev follow-up;
        # for now the coupled determinant uses the raw contour extraction (the
        # BLAS pin in the scan still makes it thread-deterministic).
        gr = find_growth_rates(scan, ref_tauk;
            pole_threshold=pthr,
            filter_above_poles=control.filter_above_poles,
            filter_outside_re=control.filter_outside_re,
            gap_kHz_threshold=control.gap_kHz_threshold)
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

    return SLAYERResult(true, control, params, rational_psi, rational_q, dp,
        Q_root, omega_Hz, gamma_Hz,
        per_surface_extraction, coupled_extraction,
        layer_widths, scan_data_list)
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
(GWP Δ_+) is the reconnecting channel and `interchange` (GWP Δ_−) the
non-reconnecting Glasser-stabilization channel (see `InnerLayerResponse`).

This is the supported GGJ diagnostic: both parity Δ's are physical and
directly accessible here. Contour-matching γ extraction from these (the
Re/Im scan intersection) is not yet validated — see the warning in
`run_slayer_from_inputs`.
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

# ---------------------------------------------------------------------
# Full pipeline: equilibrium + ForceFreeStates → parameters → analysis
# ---------------------------------------------------------------------
"""
    run_slayer(result, control; dir_path="./") -> SLAYERResult

Orchestrate the full SLAYER analysis against a `ForceFreeStates.ForceFreeStatesResult`,
reading its equilibrium, singular surfaces and Δ' matrix. Kinetic profiles are
read from `control.profile_file` (relative to `dir_path`) through the shared
`Equilibrium.read_kinetic_file` reader; when the file carries `chi_e`/`chi_phi`
profiles they set χ⊥(ψ)/χ_φ(ψ), otherwise the scalar `control.chi_perp`/
`chi_tor` fallbacks are used.

The toroidal field comes from `control.bt`; leaving it unset (the default) makes
`build_slayer_inputs` evaluate the physical `B_T = F(ψ)/(2π·R₀)` per surface.

Returns an `enabled=false` `SLAYERResult` when `control.enabled` is
false.
"""
function run_slayer(result, control::SLAYERControl; dir_path::AbstractString="./")
    dpm = result.delta_prime === nothing ? Matrix{ComplexF64}(undef, 0, 0) : result.delta_prime.matrix
    return run_slayer(result.equil, result.surfaces, dpm, control; dir_path=dir_path)
end

"""
    run_slayer(equil, surfaces, delta_prime_matrix, control; dir_path="./") -> SLAYERResult

Loose-argument form of [`run_slayer`](@ref), taking the equilibrium, the singular-surface
vector and the outer-region Δ' matrix directly. Per-surface parameters are built via
`build_slayer_inputs`; an empty or wrong-sized `delta_prime_matrix` falls back to a diagonal
built from the `sing.delta_prime` stubs.
"""
function run_slayer(equil, surfaces::AbstractVector, delta_prime_matrix::AbstractMatrix,
    control::SLAYERControl; dir_path::AbstractString="./")
    validate(control)
    control.enabled || return empty_slayer_result(control)
    isempty(surfaces) && return empty_slayer_result(control)

    loaded = _load_profiles(control, dir_path)
    profiles = loaded.profiles

    if control.inner_model in (:ggj_shooting, :ggj_galerkin)
        # GGJ γ-extraction is future work; `run_slayer_from_inputs` emits the
        # warning once the model is built (so direct callers see it too).
        params = build_ggj_inputs(equil, surfaces, profiles;
            mu_i=control.mu_i,
            zeff=control.zeff,
            resistivity_model=_build_resistivity_model(control.resistivity_model),
            lnLambda_form=control.lnLambda_form)
    else
        # `equil.config.b0exp` is a NORMALIZATION (commonly exactly 1.0), not the toroidal
        # field, so substituting it here silently ran the layer physics at B_T = 1 T. Pass the
        # control value through instead: `nothing` makes build_slayer_inputs compute the
        # physical B_T = F(psi)/(2*pi*R_0) per surface from the equilibrium's F-spline, which is
        # what its docstring already prescribes.
        bt = control.bt
        # χ⊥/χ_φ from the kinetic file when present, else the scalar fallbacks.
        chi_perp = loaded.chi_perp === nothing ? control.chi_perp : loaded.chi_perp
        chi_tor = loaded.chi_tor === nothing ? control.chi_tor : loaded.chi_tor
        (loaded.chi_perp === nothing || loaded.chi_tor === nothing) && @warn(
            "SLAYER: kinetic file has no usable chi_e/chi_phi profile(s) " *
            "(dataset absent or all-zero); using the scalar " *
            "control.chi_perp/chi_tor fallback for the missing one(s).")
        params = build_slayer_inputs(equil, surfaces, profiles;
            bt=bt,
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

    # Δ' matrix: prefer the full parallel-FM matrix; fall back to a
    # diagonal built from each SingType's scalar delta_prime.
    dp = if !isempty(delta_prime_matrix) &&
       size(delta_prime_matrix) == (length(params), length(params))
        Matrix{ComplexF64}(delta_prime_matrix)
    else
        # The full Δ' matrix is unavailable (e.g. the parallel-FM stage that
        # populates it was not run). The scalar-diagonal fallback uses
        # `sing.delta_prime`, which is a coarse per-surface stub; surfaces
        # with no entry default to Δ'=0, giving γ computed from zero drive.
        n_missing = count(s -> isempty(s.delta_prime), surfaces)
        @warn(
            "SLAYER: delta_prime_matrix is empty or wrong-sized " *
            "($(size(delta_prime_matrix)) vs " *
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
    return run_slayer_from_inputs(params, dp, control; rational_psi=rational_psi, rational_q=rational_q)
end
