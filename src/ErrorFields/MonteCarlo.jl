"""
    MonteCarlo

The tolerance Monte Carlo: recombine a [`SensitivityTable`](@ref) with a [`ToleranceSet`](@ref)
by drawing every coil set's misalignment within its tolerance many times and histogramming the
resulting dominant-mode overlap `|δ|`. Because the overlap is linear in the misalignments, one
sample is a handful of complex multiply-adds per coil set; no field is recomputed. Two
histograms are accumulated per run, the intrinsic `|δ|` and the corrected one in which every
correctable contribution is divided by the error-field-correction factor.

The projection the table was built with (ψ_N window, singular mode) is an analysis choice, so
the run writes the full-window summary and [`run_monte_carlo`](@ref) re-evaluates any other
choice in memory or from `gpec.h5` in about a second.
"""

"""
    MonteCarloControl

Settings of the tolerance Monte Carlo, the `[ErrorFields.MonteCarlo]` TOML table.

## Fields

  - `nsample`: samples per batch
  - `nbatch`: independent batches; their spread is the statistical error bar of every derived quantity
  - `seed`: base seed; batch `b` uses `Xoshiro(hash((seed, b)))`, so results are bit-identical for
    any thread count
  - `nbins`: histogram bins, linear on `[0, delta_max]`
  - `delta_max`: upper edge of the histogram; `0` means `1.5 ×` the worst-case alignment
    `Σ(|δ_nominal| + tolerance × |sensitivity|)`, which the tolerance draws cannot exceed (only
    the Gaussian uncertainties and the unattributed budget can, and those land in the last bin)
  - `tolerance_scale`: multiplies every shift and tilt tolerance, for allowable-tolerance scans;
    uncertainties and the unattributed budget are not scaled
  - `coil_subset`: coil set names whose tolerances are sampled; every other coil set contributes
    only its nominal overlap, and a coherent group is sampled only if all its members are listed.
    Empty means all
"""
Base.@kwdef struct MonteCarloControl
    nsample::Int = 1_000_000
    nbatch::Int = 10
    seed::Int = 1
    nbins::Int = 300
    delta_max::Float64 = 0.0
    tolerance_scale::Float64 = 1.0
    coil_subset::Vector{String} = String[]
end

"""
    MonteCarloResult

Histograms of the dominant-mode overlap magnitude over the sampled misalignments.

## Fields

  - `bin_edges`: `|δ|` bin edges `[nbins + 1]`; samples above the last edge are counted in the last bin
  - `pdf`, `pdf_efc`: probability density of the intrinsic and of the corrected `|δ|`, averaged
    over batches `[nbins]`
  - `pdf_batches`, `pdf_efc_batches`: the same per batch `[nbins × nbatch]`
  - `delta_nominal`: `|Σ δ_nominal|`, the as-designed overlap with every coil at its nominal position
  - `delta_worst`: the worst-case alignment bound used to size the histogram
  - `mean_abs_delta`, `mean_abs_delta_efc`: sample means of the two magnitudes
  - `clamped_fraction`: fraction of samples whose intrinsic `|δ|` landed beyond `bin_edges[end]`
    (the corrected histogram clamps too but is not separately tallied)
  - `nsample`, `nbatch`, `seed`: as run
"""
struct MonteCarloResult
    bin_edges::Vector{Float64}
    pdf::Vector{Float64}
    pdf_efc::Vector{Float64}
    pdf_batches::Matrix{Float64}
    pdf_efc_batches::Matrix{Float64}
    delta_nominal::Float64
    delta_worst::Float64
    mean_abs_delta::Float64
    mean_abs_delta_efc::Float64
    clamped_fraction::Float64
    nsample::Int
    nbatch::Int
    seed::Int
end

const _ADDITIVE = 1
const _CYLINDER = 2

const _HISTOGRAM_MARGIN = 1.5   # default delta_max = _HISTOGRAM_MARGIN × the worst-case alignment
const _WORST_CASE_SIGMA = 3     # standard deviations at which Gaussian uncertainties enter the worst-case bound

_model_code(tolerance_model::AbstractString) = tolerance_model == "cylinder" ? _CYLINDER : _ADDITIVE
_radial_p(radial_shape::AbstractString) = randpow(radial_distribution(radial_shape))

# Everything the sample loop needs, resolved to plain arrays once per run: no names, no
# dictionaries, no dispatch inside the loop.
struct _CoilTerms
    delta0::Vector{ComplexF64}
    Sx::Vector{ComplexF64}
    Sy::Vector{ComplexF64}
    Tx::Vector{ComplexF64}
    Ty::Vector{ComplexF64}
    shift_tol::Vector{Float64}      # m, scaled
    tilt_tol::Vector{Float64}       # deg, scaled
    shift_sigma::Vector{Float64}    # m
    tilt_sigma::Vector{Float64}     # deg
    p_radial::Vector{Float64}
    model::Vector{Int}
    z_top::Vector{Float64}
    correctable::Vector{Bool}
end

struct _GroupTerms
    Sx::Vector{ComplexF64}          # Σ_members S_x
    Sy::Vector{ComplexF64}
    Tx::Vector{ComplexF64}
    Ty::Vector{ComplexF64}
    Rx::Vector{ComplexF64}          # Σ_members (z_m − z_pivot)·S_x, the lateral shift of a rigid rotation
    Ry::Vector{ComplexF64}
    shift_tol::Vector{Float64}
    tilt_tol::Vector{Float64}
    p_radial::Vector{Float64}
    model::Vector{Int}
    z_top::Vector{Float64}
    phase_index::Vector{Int}
    nphase::Int
    correctable::Vector{Bool}
end

"""
    run_monte_carlo(table, tolerances, coil_sets, ctrl=MonteCarloControl()) -> MonteCarloResult
    run_monte_carlo(h5path; psi_low=0.0, psi_high=1.0, mode=1, kwargs...) -> MonteCarloResult

Sample every coil set's misalignment within its tolerance and histogram the dominant-mode
overlap. Per sample and coil set `c` with sensitivities `S = (S_x, S_y)` per metre and
`T = (T_x, T_y)` per degree,

```
δ = Σ_c [δ_nominal,c + S_c·(Δ_c + u_c) + T_c·(θ_c + v_c)] + Σ_g [S_g·(Δ_g + Δ_rot) + T_g·θ_g] + δ_other
```

where `(Δ_c, θ_c)` is the coil's own draw (additive: independent shift and tilt disks; cylinder:
[`sample_cylinder`](@ref)), `u_c`, `v_c` its Gaussian placement uncertainties, `(Δ_g, θ_g)` one
draw shared by every member of coherent group `g` (groups with equal `phase_group` share the
direction), `Δ_rot = −(z_m − z_pivot)·(θy_g + iθx_g)` the lateral shift a rigid rotation of the
group about its pivot (in the sense `apply_transforms` uses) gives a member at height `z_m`, and
`δ_other` the unattributed budget with a random direction. `S·Δ` means `S_x·real(Δ) + S_y·imag(Δ)`.
The corrected histogram divides every correctable term (coils and groups not listed as
uncorrectable, and the unattributed budget) by `efc_factor`.

`coil_sets` supply the nominal radii for tilt tolerances given in metres and the heights of
group members; coil sets of the run without a tolerance block contribute their nominal overlap
only. The file form rebuilds the coupling from `gpec.h5`, windows it, projects the stored
linearization, and reads the tolerance snapshot the run echoed; `kwargs` are
[`MonteCarloControl`](@ref) fields.
"""
function run_monte_carlo(table::SensitivityTable, ts::ToleranceSet, coil_sets::Vector{CoilSet}, ctrl::MonteCarloControl=MonteCarloControl())
    ctrl.nsample > 0 && ctrl.nbatch > 0 && ctrl.nbins > 0 || throw(ArgumentError("nsample, nbatch and nbins must be positive"))
    ctrl.tolerance_scale >= 0 || throw(ArgumentError("tolerance_scale must be ≥ 0"))
    validate_tolerances(ts, table.coil_names)
    coils, groups = _resolve_terms(table, ts, coil_sets, ctrl)
    other = ts.other_field
    p_other = _radial_p(other.radial_shape)
    efc = ts.efc_factor

    delta_nominal = abs(sum(table.delta_nominal))
    delta_worst = _worst_case(table, coils, groups, other)
    delta_max = ctrl.delta_max > 0 ? ctrl.delta_max : _HISTOGRAM_MARGIN * delta_worst
    delta_max > 0 || throw(ArgumentError("the histogram range is zero: no nominal overlap, tolerance, or budget to sample"))
    edges = collect(range(0.0, delta_max; length=ctrl.nbins + 1))
    width = edges[2] - edges[1]

    counts = zeros(Int, ctrl.nbins, ctrl.nbatch)
    counts_efc = zeros(Int, ctrl.nbins, ctrl.nbatch)
    sums = zeros(ctrl.nbatch)
    sums_efc = zeros(ctrl.nbatch)
    clamped = zeros(Int, ctrl.nbatch)

    Threads.@threads for b in 1:ctrl.nbatch
        rng = Xoshiro(hash((ctrl.seed, b)))
        cnt = view(counts, :, b)
        cnt_efc = view(counts_efc, :, b)
        s = 0.0
        s_efc = 0.0
        nclamp = 0
        phases = zeros(4, groups.nphase)
        for _ in 1:ctrl.nsample
            δ_corr, δ_uncorr = _sample_delta!(rng, phases, coils, groups, other, p_other)
            a = abs(δ_corr + δ_uncorr)
            a_efc = abs(δ_corr / efc + δ_uncorr)
            s += a
            s_efc += a_efc
            i = min(floor(Int, a / width) + 1, ctrl.nbins)
            a >= delta_max && (nclamp += 1)
            cnt[i] += 1
            cnt_efc[min(floor(Int, a_efc / width) + 1, ctrl.nbins)] += 1
        end
        sums[b] = s
        sums_efc[b] = s_efc
        clamped[b] = nclamp
    end

    norm = ctrl.nsample * width
    pdf_batches = counts ./ norm
    pdf_efc_batches = counts_efc ./ norm
    total = ctrl.nsample * ctrl.nbatch
    return MonteCarloResult(edges, vec(sum(pdf_batches; dims=2)) ./ ctrl.nbatch, vec(sum(pdf_efc_batches; dims=2)) ./ ctrl.nbatch,
        pdf_batches, pdf_efc_batches, delta_nominal, delta_worst, sum(sums) / total, sum(sums_efc) / total,
        sum(clamped) / total, ctrl.nsample, ctrl.nbatch, ctrl.seed)
end

function run_monte_carlo(h5path::AbstractString; psi_low::Real=0.0, psi_high::Real=1.0, mode::Int=1, kwargs...)
    ts = read_tolerance_snapshot(h5path)
    ts === nothing && throw(ArgumentError("$h5path carries no tolerance snapshot (the run named no tolerance_file)"))
    table = sensitivity_table(h5path; psi_low, psi_high, mode)
    coil_sets = h5open(h5path, "r") do f
        haskey(f, "Input/RawInputs/Coils") || throw(ArgumentError("$h5path has no Input/RawInputs/Coils snapshot"))
        sets = CoilSet[]
        ForcingTerms.load_coils_from_h5_group!(sets, f["Input/RawInputs/Coils"])
        sets
    end
    return run_monte_carlo(table, ts, coil_sets, MonteCarloControl(; kwargs...))
end

# One sample: returns (correctable, uncorrectable) complex overlaps. `phases` is scratch for
# the per-phase-group directions (shift, tilt, cylinder top, cylinder bottom).
@inline function _sample_delta!(rng::AbstractRNG, phases::Matrix{Float64}, coils::_CoilTerms, groups::_GroupTerms, other::OtherFieldBudget, p_other::Float64)
    δc = zero(ComplexF64)
    δu = zero(ComplexF64)
    @inbounds for c in eachindex(coils.delta0)
        if coils.model[c] == _CYLINDER
            shift, tilt = sample_cylinder(rng, coils.shift_tol[c], coils.z_top[c], coils.p_radial[c])
        else
            shift, tilt = sample_additive(rng, coils.shift_tol[c], coils.p_radial[c], coils.tilt_tol[c], coils.p_radial[c])
        end
        shift += sample_uncertainty(rng, coils.shift_sigma[c])
        tilt += sample_uncertainty(rng, coils.tilt_sigma[c])
        d = coils.delta0[c] + coils.Sx[c] * real(shift) + coils.Sy[c] * imag(shift) + coils.Tx[c] * real(tilt) + coils.Ty[c] * imag(tilt)
        coils.correctable[c] ? (δc += d) : (δu += d)
    end
    @inbounds for k in 1:groups.nphase, q in 1:4
        phases[q, k] = 2π * rand(rng)
    end
    @inbounds for g in eachindex(groups.Sx)
        k = groups.phase_index[g]
        if groups.model[g] == _CYLINDER
            shift, tilt = sample_cylinder(rng, groups.shift_tol[g], groups.z_top[g], groups.p_radial[g]; phase_top=phases[3, k], phase_bot=phases[4, k])
        else
            shift, tilt = sample_additive(rng, groups.shift_tol[g], groups.p_radial[g], groups.tilt_tol[g], groups.p_radial[g];
                phase_shift=phases[1, k], phase_tilt=phases[2, k])
        end
        # A rigid rotation by (θx, θy) about the pivot moves a member at height h by −h·(θy + iθx).
        θ = deg2rad(tilt)
        d = groups.Sx[g] * real(shift) + groups.Sy[g] * imag(shift) + groups.Tx[g] * real(tilt) + groups.Ty[g] * imag(tilt) -
            groups.Rx[g] * imag(θ) - groups.Ry[g] * real(θ)
        groups.correctable[g] ? (δc += d) : (δu += d)
    end
    if other.magnitude > 0 || other.sigma > 0
        amp = other.magnitude * rand(rng)^p_other + other.sigma * randn(rng)
        δc += amp * cis(2π * rand(rng))
    end
    return δc, δu
end

function _resolve_terms(table::SensitivityTable, ts::ToleranceSet, coil_sets::Vector{CoilSet}, ctrl::MonteCarloControl)
    names = table.coil_names
    index = Dict(n => i for (i, n) in enumerate(names))
    set_of = Dict(cs.name => cs for cs in coil_sets)
    for n in names
        haskey(set_of, n) || throw(ArgumentError("no coil set geometry named \"$n\" among the given coil sets"))
    end
    active(n) = isempty(ctrl.coil_subset) || n in ctrl.coil_subset
    uncorrectable = Set(ts.uncorrectable_coils)
    scale = ctrl.tolerance_scale

    n = length(names)
    shift_tol = zeros(n)
    tilt_tol = zeros(n)
    shift_sigma = zeros(n)
    tilt_sigma = zeros(n)
    p_radial = ones(n)
    model = fill(_ADDITIVE, n)
    z_top = zeros(n)
    for t in ts.coils
        i = index[t.name]
        active(t.name) || continue
        cs = set_of[t.name]
        shift_tol[i] = scale * t.shift_tol_m
        tilt_tol[i] = scale * tilt_tolerance_deg(t, cs)
        shift_sigma[i] = t.shift_sigma_m
        tilt_sigma[i] = tilt_tolerance_deg(t.tilt_sigma, t.tilt_units, cs)
        p_radial[i] = _radial_p(t.radial_shape)
        model[i] = _model_code(t.tolerance_model)
        z_top[i] = t.cylinder_half_height_m
    end
    coils = _CoilTerms(collect(table.delta_nominal), table.shift[1, :], table.shift[2, :], table.tilt[1, :], table.tilt[2, :],
        shift_tol, tilt_tol, shift_sigma, tilt_sigma, p_radial, model, z_top, [!(nm in uncorrectable) for nm in names])

    kept = [g for g in ts.groups if all(active, g.members)]
    ng = length(kept)
    gSx = zeros(ComplexF64, ng)
    gSy = zeros(ComplexF64, ng)
    gTx = zeros(ComplexF64, ng)
    gTy = zeros(ComplexF64, ng)
    gRx = zeros(ComplexF64, ng)
    gRy = zeros(ComplexF64, ng)
    g_shift = zeros(ng)
    g_tilt = zeros(ng)
    g_p = ones(ng)
    g_model = fill(_ADDITIVE, ng)
    g_ztop = zeros(ng)
    g_corr = trues(ng)
    labels = unique(g.phase_group for g in kept)
    phase_index = [findfirst(==(g.phase_group), labels) for g in kept]
    for (gi, g) in enumerate(kept)
        for m in g.members
            i = index[m]
            h = _set_center(set_of[m])[3] - g.rotation_center_z_m
            gSx[gi] += table.shift[1, i]
            gSy[gi] += table.shift[2, i]
            gTx[gi] += table.tilt[1, i]
            gTy[gi] += table.tilt[2, i]
            gRx[gi] += h * table.shift[1, i]
            gRy[gi] += h * table.shift[2, i]
            m in uncorrectable && (g_corr[gi] = false)
        end
        g_shift[gi] = scale * g.shift_tol_m
        # A group tilt in metres is converted through the first member's radius, as the OMFIT
        # tables did; give group tilts in degrees when members differ in size.
        g_tilt[gi] = scale * tilt_tolerance_deg(g.tilt_tol, g.tilt_units, set_of[g.members[1]])
        g_p[gi] = _radial_p(g.radial_shape)
        g_model[gi] = _model_code(g.tolerance_model)
        g_ztop[gi] = g.cylinder_half_height_m
    end
    groups = _GroupTerms(gSx, gSy, gTx, gTy, gRx, gRy, g_shift, g_tilt, g_p, g_model, g_ztop, phase_index, length(labels), g_corr)
    return coils, groups
end

# Worst-case alignment bound used to size the histogram: every term at its tolerance edge and in
# phase. Gaussian uncertainties enter at _WORST_CASE_SIGMA standard deviations, not a hard limit.
function _worst_case(table::SensitivityTable, coils::_CoilTerms, groups::_GroupTerms, other::OtherFieldBudget)
    w = sum(abs, table.delta_nominal)
    for c in eachindex(coils.delta0)
        s_in = max(abs(coils.Sx[c]), abs(coils.Sy[c]))
        t_in = max(abs(coils.Tx[c]), abs(coils.Ty[c]))
        tilt_reach = coils.model[c] == _CYLINDER ? rad2deg(atan(coils.shift_tol[c] / coils.z_top[c])) : coils.tilt_tol[c]
        w += s_in * (coils.shift_tol[c] + _WORST_CASE_SIGMA * coils.shift_sigma[c]) + t_in * (tilt_reach + _WORST_CASE_SIGMA * coils.tilt_sigma[c])
    end
    for g in eachindex(groups.Sx)
        s_in = max(abs(groups.Sx[g]), abs(groups.Sy[g]))
        # Rx, Ry carry the rotation term's per-radian sensitivity; deg2rad(1) folds it into T's per-degree units.
        t_in = max(abs(groups.Tx[g]), abs(groups.Ty[g])) + deg2rad(1) * max(abs(groups.Rx[g]), abs(groups.Ry[g]))
        tilt_reach = groups.model[g] == _CYLINDER ? rad2deg(atan(groups.shift_tol[g] / groups.z_top[g])) : groups.tilt_tol[g]
        w += s_in * groups.shift_tol[g] + t_in * tilt_reach
    end
    return w + other.magnitude + _WORST_CASE_SIGMA * other.sigma
end
