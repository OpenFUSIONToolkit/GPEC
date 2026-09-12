"""
    Risk

From an overlap distribution to a locking risk. The empirical ITPA penetration-threshold
scalings give the overlap δ at which an error field locks as a power law in density, field,
major radius and β_N/l_i, with fitted exponents and their uncertainties; sampling the
exponents turns the threshold into a distribution and its cumulative distribution into the
probability that an overlap δ locks. Convolving that with the Monte Carlo distribution of
`|δ|` gives the locking probability of the assembled machine, and repeating the Monte Carlo
over a range of tolerance scales gives the allowable tolerance for a target risk.

n = 1 fits: Logan et al., "Robustness of the tokamak error field correction tolerance scaling",
Plasma Phys. Control. Fusion 62 (2020) 084001; n = 2 fits: Logan et al., "Empirical scaling of
the n = 2 error field penetration threshold in tokamaks", Nucl. Fusion 60 (2020) 086010.
"""

"""
    ThresholdScaling

One fit of the ITPA error-field penetration threshold,
`δ_thresh = 10^α_c · n_e^α_n · B_T^α_B · R_0^α_R · (β_N/l_i)^α_β`, with `n_e` in 10¹⁹ m⁻³,
`B_T` in tesla and `R_0` in metres. Each exponent carries the fit's standard error.

## Fields

  - `n`: toroidal mode number of the scaling
  - `dataset`: plasma dataset the fit was made on (`"O,L"` ohmic and L-mode, `"O,L,H"` with H-modes, ...)
  - `fit`: fitting method (`"OLS"`, `"DSOLS"` downsampled, `"WLS"` weighted)
  - `alpha_c`, `alpha_n`, `alpha_b`, `alpha_r`, `alpha_beta`: `(value, standard error)` of each exponent
"""
struct ThresholdScaling
    n::Int
    dataset::String
    fit::String
    alpha_c::Tuple{Float64,Float64}
    alpha_n::Tuple{Float64,Float64}
    alpha_b::Tuple{Float64,Float64}
    alpha_r::Tuple{Float64,Float64}
    alpha_beta::Tuple{Float64,Float64}
end

# The published ITPA fits, keyed "n=<n> <dataset> <fit>".
const ITPA_THRESHOLD_SCALINGS = Dict{String,ThresholdScaling}(
    "n=1 O,L OLS" => ThresholdScaling(1, "O,L", "OLS", (-3.75, 0.05), (0.63, 0.09), (-0.98, 0.12), (0.15, 0.08), (-0.13, 0.10)),
    "n=1 O,L DSOLS" => ThresholdScaling(1, "O,L", "DSOLS", (-3.39, 0.06), (0.58, 0.08), (-1.08, 0.10), (0.19, 0.07), (0.26, 0.10)),
    "n=1 O,L WLS" => ThresholdScaling(1, "O,L", "WLS", (-3.46, 0.05), (0.64, 0.06), (-1.14, 0.08), (0.20, 0.07), (0.15, 0.07)),
    "n=1 O,L,H OLS" => ThresholdScaling(1, "O,L,H", "OLS", (-3.64, 0.04), (0.60, 0.08), (-0.95, 0.08), (0.12, 0.08), (-0.30, 0.05)),
    "n=1 O,L,H DSOLS" => ThresholdScaling(1, "O,L,H", "DSOLS", (-3.58, 0.04), (0.45, 0.06), (-0.94, 0.08), (0.09, 0.07), (-0.15, 0.05)),
    "n=1 O,L,H WLS" => ThresholdScaling(1, "O,L,H", "WLS", (-3.62, 0.04), (0.53, 0.06), (-0.95, 0.07), (0.14, 0.08), (-0.19, 0.05)),
    "n=2 O,L WLS" => ThresholdScaling(2, "O,L", "WLS", (-3.36, 0.06), (1.07, 0.09), (-1.52, 0.2), (1.46, 0.09), (0.36, 0.11)),
    "n=2 O,L,-C WLS" => ThresholdScaling(2, "O,L,-C", "WLS", (-2.98, 0.05), (0.93, 0.08), (-1.28, 0.15), (0.0, 0.0), (0.41, 0.08)),
    "n=2 O,L,N WLS" => ThresholdScaling(2, "O,L,N", "WLS", (-3.16, 0.05), (0.64, 0.06), (-1.14, 0.08), (0.20, 0.07), (0.15, 0.07))
)

"""
    threshold_scaling(; n=1, dataset="O,L", fit="WLS") -> ThresholdScaling

Look up a published ITPA fit; the available keys are those of `ITPA_THRESHOLD_SCALINGS`.
"""
function threshold_scaling(; n::Int=1, dataset::AbstractString="O,L", fit::AbstractString="WLS")
    key = "n=$n $dataset $fit"
    haskey(ITPA_THRESHOLD_SCALINGS, key) ||
        throw(ArgumentError("no ITPA threshold scaling \"$key\"; available: $(join(sort(collect(keys(ITPA_THRESHOLD_SCALINGS))), ", "))"))
    return ITPA_THRESHOLD_SCALINGS[key]
end

"""
    ScenarioParameters

Operating point the threshold scaling is evaluated at. Density is not an ideal-MHD equilibrium
output and must be given; the other four default from the equilibrium when built with
`ScenarioParameters(equil; n_e)`.

## Fields

  - `n_e`: electron density, 10¹⁹ m⁻³
  - `b_t0`: toroidal field magnitude on axis, tesla
  - `r_0`: major radius of the magnetic axis, metres
  - `beta_n`: normalized beta
  - `l_i`: normalized internal inductance (`l_i(1)`)
"""
struct ScenarioParameters
    n_e::Float64
    b_t0::Float64
    r_0::Float64
    beta_n::Float64
    l_i::Float64
    function ScenarioParameters(n_e, b_t0, r_0, beta_n, l_i)
        all(>(0), (n_e, b_t0, r_0, beta_n, l_i)) || throw(ArgumentError("scenario parameters must be positive (got n_e=$n_e, B_T0=$b_t0, R_0=$r_0, β_N=$beta_n, l_i=$l_i)"))
        return new(n_e, b_t0, r_0, beta_n, l_i)
    end
end

"""
    ScenarioParameters(equil::PlasmaEquilibrium; n_e, b_t0=equil.params.bt0, r_0=equil.ro, beta_n=equil.params.betan, l_i=equil.params.li1)
    ScenarioParameters(h5path; n_e, kwargs...)

Build the operating point from an equilibrium (or a run's `Equilibrium/` scalars), with the
density supplied and any other parameter overridable.
"""
function ScenarioParameters(equil::Equilibrium.PlasmaEquilibrium; n_e::Real, b_t0::Real=equil.params.bt0, r_0::Real=equil.ro,
    beta_n::Real=equil.params.betan, l_i::Real=equil.params.li1)
    return ScenarioParameters(n_e, b_t0, r_0, beta_n, l_i)
end

function ScenarioParameters(h5path::AbstractString; n_e::Real, kwargs...)
    vals = h5open(h5path, "r") do f
        (b_t0=read(f["Equilibrium/B_T_axis"]), r_0=read(f["Equilibrium/R_axis"]), beta_n=read(f["Equilibrium/beta_N"]), l_i=read(f["Equilibrium/l_i_1"]))
    end
    merged = merge(vals, values(kwargs))
    return ScenarioParameters(n_e, merged.b_t0, merged.r_0, merged.beta_n, merged.l_i)
end

_threshold(sc::ThresholdScaling, scen::ScenarioParameters, αc, αn, αb, αr, αβ) =
    10.0^αc * scen.n_e^αn * scen.b_t0^αb * scen.r_0^αr * (scen.beta_n / scen.l_i)^αβ

"""
    nominal_threshold(sc::ThresholdScaling, scen::ScenarioParameters) -> Float64

The penetration threshold at the fitted exponents.
"""
nominal_threshold(sc::ThresholdScaling, scen::ScenarioParameters) = _threshold(sc, scen, sc.alpha_c[1], sc.alpha_n[1], sc.alpha_b[1], sc.alpha_r[1], sc.alpha_beta[1])

"""
    threshold_samples(rng, sc, scen; nsample=1_000_000, dist="normal") -> Vector{Float64}

Thresholds with the five exponents drawn around their fitted values: `"normal"` (Gaussian with
the standard error), `"flat"` (uniform within one standard error), or `"normal_truncated"`
(Gaussian, redrawn beyond 1.5 standard errors). The exponents' uncertainties are far from small
relative to the sensitivity of a power law, so the distribution is sampled rather than
propagated to first order.
"""
function threshold_samples(rng::AbstractRNG, sc::ThresholdScaling, scen::ScenarioParameters; nsample::Int=1_000_000, dist::AbstractString="normal")
    draw = if dist == "normal"
        () -> randn(rng)
    elseif dist == "flat"
        () -> 2 * rand(rng) - 1
    elseif dist == "normal_truncated"
        () -> begin
            x = randn(rng)
            while abs(x) > 1.5
                x = randn(rng)
            end
            x
        end
    else
        throw(ArgumentError("dist must be \"normal\", \"flat\", or \"normal_truncated\" (got \"$dist\")"))
    end
    out = Vector{Float64}(undef, nsample)
    for i in 1:nsample
        out[i] = _threshold(sc, scen,
            sc.alpha_c[1] + sc.alpha_c[2] * draw(), sc.alpha_n[1] + sc.alpha_n[2] * draw(), sc.alpha_b[1] + sc.alpha_b[2] * draw(),
            sc.alpha_r[1] + sc.alpha_r[2] * draw(), sc.alpha_beta[1] + sc.alpha_beta[2] * draw())
    end
    return out
end

"""
    RiskControl

Settings of the locking-risk evaluation, the `[ErrorFields.Risk]` TOML table.

## Fields

  - `dataset`, `fit`: which ITPA threshold fit to use (the toroidal mode number is the run's)
  - `distribution`: how the fit exponents are sampled — `"normal"`, `"flat"`, or `"normal_truncated"`
  - `nsample_threshold`: threshold samples
  - `seed`: seed of the threshold sampling
  - `scan_scales`: tolerance multipliers of the allowable-tolerance scan (empty: no scan); each
    is a fresh Monte Carlo with every shift and tilt tolerance multiplied by the scale
"""
Base.@kwdef struct RiskControl
    dataset::String = "O,L"
    fit::String = "WLS"
    distribution::String = "normal"
    nsample_threshold::Int = 1_000_000
    seed::Int = 1
    scan_scales::Vector{Float64} = Float64[]
end

"""
    RiskResult

The locking risk of a Monte Carlo overlap distribution under one threshold scaling. All
probabilities are percentages.

## Fields

  - `bin_edges`: the Monte Carlo's `|δ|` grid `[nbins + 1]`
  - `threshold_pdf`: probability density of the sampled penetration threshold on that grid `[nbins]`
  - `p_lock_given_delta`: probability that an overlap equal to each bin edge locks, the
    threshold's cumulative distribution `[nbins + 1]`
  - `threshold_nominal`: the threshold at the fitted exponents
  - `plock`, `plock_efc`: locking probability of the intrinsic and of the corrected distribution,
    `100 ∫ pdf(δ) P(lock|δ) dδ`, batch averages
  - `plock_batches`, `plock_efc_batches`: the same per Monte Carlo batch `[nbatch]`; their ranges
    are the statistical error bars
  - `plock_nominal`: risk of the as-designed machine, `100 P(lock|δ_nominal)`
  - `plock_sharp`: risk if the threshold were exactly its nominal value, `100 P(|δ| > threshold_nominal)`
  - `scaling`: the threshold fit used
"""
struct RiskResult
    bin_edges::Vector{Float64}
    threshold_pdf::Vector{Float64}
    p_lock_given_delta::Vector{Float64}
    threshold_nominal::Float64
    plock::Float64
    plock_efc::Float64
    plock_batches::Vector{Float64}
    plock_efc_batches::Vector{Float64}
    plock_nominal::Float64
    plock_sharp::Float64
    scaling::ThresholdScaling
end

"""
    locking_risk(mc::MonteCarloResult, sc::ThresholdScaling, scen::ScenarioParameters; ctrl=RiskControl()) -> RiskResult
    locking_risk(mc, thresholds::AbstractVector, sc, scen) -> RiskResult

Convolve an overlap distribution with the threshold distribution: `P(lock|δ)` is the fraction of
sampled thresholds below `δ`, and the locking probability is `100 ∫ pdf(δ) P(lock|δ) dδ` over
the Monte Carlo's bins, evaluated per batch. Thresholds are sampled with `ctrl` or passed in.
"""
function locking_risk(mc::MonteCarloResult, sc::ThresholdScaling, scen::ScenarioParameters; ctrl::RiskControl=RiskControl())
    thresholds = threshold_samples(Xoshiro(ctrl.seed), sc, scen; nsample=ctrl.nsample_threshold, dist=ctrl.distribution)
    return locking_risk(mc, thresholds, sc, scen)
end

function locking_risk(mc::MonteCarloResult, thresholds::AbstractVector{<:Real}, sc::ThresholdScaling, scen::ScenarioParameters)
    edges = mc.bin_edges
    sorted = sort(Float64.(thresholds))
    n = length(sorted)
    cdf_at(x) = searchsortedlast(sorted, x) / n
    p_given = [e == 0 ? 0.0 : cdf_at(e) for e in edges]
    threshold_pdf = diff(p_given) ./ diff(edges)
    # Bin-average of P(lock|δ) times the bin's probability mass; the last bin also holds the
    # clamped tail, whose overlaps are at least the last edge.
    weights = (p_given[1:end-1] .+ p_given[2:end]) ./ 2
    widths = diff(edges)
    # Densities integrate to one only to round-off, so the percentage is clamped to [0, 100].
    plock_of(pdf) = clamp(100 * sum(pdf .* widths .* weights), 0.0, 100.0)
    plock_b = [plock_of(view(mc.pdf_batches, :, b)) for b in 1:size(mc.pdf_batches, 2)]
    plock_efc_b = [plock_of(view(mc.pdf_efc_batches, :, b)) for b in 1:size(mc.pdf_efc_batches, 2)]
    nominal = nominal_threshold(sc, scen)
    plock_sharp = clamp(100 * (1 - sum(mc.pdf[edges[2:end].<=nominal] .* widths[edges[2:end].<=nominal])), 0.0, 100.0)
    return RiskResult(copy(edges), threshold_pdf, p_given, nominal, sum(plock_b) / length(plock_b), sum(plock_efc_b) / length(plock_efc_b),
        plock_b, plock_efc_b, 100 * cdf_at(mc.delta_nominal), plock_sharp, sc)
end

"""
    locking_risk(h5path; n_e, psi_low=0.0, psi_high=1.0, mode=1, risk_ctrl=RiskControl(), kwargs...) -> RiskResult

Post-hoc locking risk from a run: the Monte Carlo is re-run from the file for the given window
and mode with [`MonteCarloControl`](@ref) `kwargs`, the threshold fit is chosen by `risk_ctrl`
for the run's toroidal mode number, and the scenario is the run's equilibrium at density `n_e`.
"""
function locking_risk(h5path::AbstractString; n_e::Real, psi_low::Real=0.0, psi_high::Real=1.0, mode::Int=1, risk_ctrl::RiskControl=RiskControl(), kwargs...)
    mc = run_monte_carlo(h5path; psi_low, psi_high, mode, kwargs...)
    n = h5open(f -> Int(read(f["Info/nlow"])), h5path, "r")
    sc = threshold_scaling(; n, dataset=risk_ctrl.dataset, fit=risk_ctrl.fit)
    return locking_risk(mc, sc, ScenarioParameters(h5path; n_e); ctrl=risk_ctrl)
end

"""
    ToleranceScan

Locking risk against a multiplier of every shift and tilt tolerance, from repeated Monte
Carlos on the same sensitivities. Percentages throughout.

## Fields

  - `scale`: tolerance multipliers `[nscale]`
  - `plock`, `plock_efc`: locking probability at each scale, batch averages
  - `plock_spread`, `plock_efc_spread`: range of the batch values at each scale
  - `plock_nominal`: risk of the as-designed machine (independent of the scale)
"""
struct ToleranceScan
    scale::Vector{Float64}
    plock::Vector{Float64}
    plock_efc::Vector{Float64}
    plock_spread::Vector{Float64}
    plock_efc_spread::Vector{Float64}
    plock_nominal::Float64
end

"""
    tolerance_scan(table, ts, coil_sets, mc_ctrl, sc, scen; scales, risk_ctrl=RiskControl()) -> ToleranceScan
    tolerance_scan(h5path; scales, psi_low=0.0, psi_high=1.0, mode=1, n_e, kwargs...) -> ToleranceScan

Run the Monte Carlo once per tolerance multiplier in `scales` and evaluate the locking risk of
each with one threshold sampling. The file form takes the run's tolerances, coil geometry and
equilibrium scalars from `gpec.h5`; `kwargs` are [`MonteCarloControl`](@ref) fields, and the
threshold fit is chosen with `risk_ctrl`.
"""
function tolerance_scan(table::SensitivityTable, ts::ToleranceSet, coil_sets::Vector{CoilSet}, mc_ctrl::MonteCarloControl,
    sc::ThresholdScaling, scen::ScenarioParameters; scales::AbstractVector{<:Real}, risk_ctrl::RiskControl=RiskControl())
    isempty(scales) && throw(ArgumentError("tolerance_scan needs at least one scale"))
    all(>=(0), scales) || throw(ArgumentError("tolerance scales must be ≥ 0"))
    thresholds = threshold_samples(Xoshiro(risk_ctrl.seed), sc, scen; nsample=risk_ctrl.nsample_threshold, dist=risk_ctrl.distribution)
    fields = (f => getfield(mc_ctrl, f) for f in fieldnames(MonteCarloControl) if f != :tolerance_scale)
    plock = Float64[]
    plock_efc = Float64[]
    spread = Float64[]
    spread_efc = Float64[]
    nominal = 0.0
    for s in scales
        mc = run_monte_carlo(table, ts, coil_sets, MonteCarloControl(; fields..., tolerance_scale=Float64(s)))
        risk = locking_risk(mc, thresholds, sc, scen)
        push!(plock, risk.plock)
        push!(plock_efc, risk.plock_efc)
        push!(spread, maximum(risk.plock_batches) - minimum(risk.plock_batches))
        push!(spread_efc, maximum(risk.plock_efc_batches) - minimum(risk.plock_efc_batches))
        nominal = risk.plock_nominal
    end
    return ToleranceScan(Float64.(collect(scales)), plock, plock_efc, spread, spread_efc, nominal)
end

function tolerance_scan(h5path::AbstractString; scales::AbstractVector{<:Real}, psi_low::Real=0.0, psi_high::Real=1.0, mode::Int=1,
    n_e::Real, risk_ctrl::RiskControl=RiskControl(), kwargs...)
    ts = read_tolerance_snapshot(h5path)
    ts === nothing && throw(ArgumentError("$h5path carries no tolerance snapshot (the run named no tolerance_file)"))
    table = sensitivity_table(h5path; psi_low, psi_high, mode)
    coil_sets, n = h5open(h5path, "r") do f
        haskey(f, "Input/RawInputs/Coils") || throw(ArgumentError("$h5path has no Input/RawInputs/Coils snapshot"))
        sets = CoilSet[]
        ForcingTerms.load_coils_from_h5_group!(sets, f["Input/RawInputs/Coils"])
        sets, Int(read(f["Info/nlow"]))
    end
    sc = threshold_scaling(; n, dataset=risk_ctrl.dataset, fit=risk_ctrl.fit)
    scen = ScenarioParameters(h5path; n_e)
    return tolerance_scan(table, ts, coil_sets, MonteCarloControl(; kwargs...), sc, scen; scales, risk_ctrl)
end

"""
    allowable_tolerance(scan::ToleranceScan, target_percent; corrected=false) -> Float64

The tolerance multiplier at which the locking risk reaches `target_percent`, by linear
interpolation in the logarithms of both the scale and the risk between the two scan points
that bracket the target (the scan need not be monotonic; the first bracketing pair from the
smallest scale is used). Returns `NaN` when no pair brackets the target. `corrected` reads the
error-field-corrected curve.
"""
function allowable_tolerance(scan::ToleranceScan, target_percent::Real; corrected::Bool=false)
    target_percent > 0 || throw(ArgumentError("target_percent must be positive"))
    p = corrected ? scan.plock_efc : scan.plock
    lt = log10(target_percent)
    for i in 1:length(scan.scale)-1
        p1, p2 = p[i], p[i+1]
        (p1 > 0 && p2 > 0) || continue
        l1, l2 = log10(p1), log10(p2)
        (min(l1, l2) <= lt <= max(l1, l2)) || continue
        l1 == l2 && return scan.scale[i]
        f = (lt - l1) / (l2 - l1)
        return 10^(log10(scan.scale[i]) + f * (log10(scan.scale[i+1]) - log10(scan.scale[i])))
    end
    return NaN
end
