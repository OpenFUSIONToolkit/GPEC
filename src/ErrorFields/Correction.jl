# The linear correction requirement of a set of correction arrays. Included after NTVLimits.jl (reads an EFCCoupling).

"""
    CorrectionRequirement

Field names follow the ErrorFields result grammar `<quantity>[_instance][_efc][_statistic][_unit]` (see the manual's
"Result names"); every field name is also its HDF5 dataset name.

The linear correction requirement of one error-field source against a set of correction arrays, on
the rational surfaces of one run. Each needed current is given as a complex factor on the array's
current as given (`−1` on an array identical to the source cancels it exactly) and in kilo-ampere-turns.

## Fields

  - `source_name`, `array_names`: the coil sets, `[narray]` for the arrays
  - `rational_psi`, `rational_q`, `rational_m`, `rational_n`: the surfaces, `[nsurface]`
  - `current_factor_dominant`: the factor on each array *alone* that cancels the source's
    dominant-mode overlap, `−δ_source / δ_array`; `NaN` for an array with no overlap `[narray]`.
    Not a joint solution: for two or more arrays the dominant-mode condition leaves an
    `(narray − 1)`-dimensional family of solutions
  - `current_factor_dominant_minimum_norm`: the member of that family with the smallest total current
    `Σ|I_k|²` `[narray]`, `I_k = −δ_source · conj(δ_k/a_k) / Σ|δ/a|²` with `a_k` the array's ampere-turns
    (`a_k = 1` when any array's ampere-turns are unknown)
  - `current_factor_least_squares`: the factors on every array together that minimize the resonant
    field on every surface, `argmin ‖C·(b̃_source + Σ_k f_k b̃_k)‖` `[narray]`; unique when
    `least_squares_rank == narray`, otherwise the minimizer with the smallest total current in the
    same sense
  - `least_squares_rank`: rank of the `nsurface × narray` system the least squares solves
  - `current_dominant_kat`, `current_dominant_minimum_norm_kat`, `current_least_squares_kat`: the
    three factors times each array's `ampere_turns_kat`, complex kilo-ampere-turns (magnitude and
    toroidal phase) `[narray]`; `NaN` for an array whose ampere-turns are unknown
  - `resonant_field_source_t`: the source's resonant field on each surface, `C·b̃_source`, tesla `[nsurface]`
  - `resonant_field_dominant_t`: the same after each array's dominant-mode correction alone, tesla `[nsurface × narray]`
  - `resonant_field_least_squares_t`: the same after the joint least-squares correction, tesla `[nsurface]`
  - `cosine_similarity`: `|⟨b̃_array, b̃_source⟩| / (‖b̃_array‖ ‖b̃_source‖)`, how much of the source's
    spectrum each array can see at all; `NaN` for a round-off field `[narray]`
"""
struct CorrectionRequirement
    source_name::String
    array_names::Vector{String}
    rational_psi::Vector{Float64}
    rational_q::Vector{Float64}
    rational_m::Vector{Int}
    rational_n::Vector{Int}
    current_factor_dominant::Vector{ComplexF64}
    current_factor_dominant_minimum_norm::Vector{ComplexF64}
    current_factor_least_squares::Vector{ComplexF64}
    least_squares_rank::Int
    current_dominant_kat::Vector{ComplexF64}
    current_dominant_minimum_norm_kat::Vector{ComplexF64}
    current_least_squares_kat::Vector{ComplexF64}
    resonant_field_source_t::Vector{ComplexF64}
    resonant_field_dominant_t::Matrix{ComplexF64}
    resonant_field_least_squares_t::Vector{ComplexF64}
    cosine_similarity::Vector{Float64}
end

"""
    correction_requirement(ctx, source::CoilOverlap, arrays) -> CorrectionRequirement
    correction_requirement(h5path, source_name, array_names; mode=1, coil_sets=nothing, kwargs...) -> CorrectionRequirement

The currents a set of arrays needs to correct one error field, and what is left: each array's solo
dominant-mode factor, the minimum-current dominant-mode solution, and the joint least-squares factors
that minimize the resonant field `C·(b̃_source + Σ_k f_k b̃_k)` over every rational surface, with the
resonant field left after each. Every input is a [`CoilOverlap`](@ref), so any geometry can be a source
or an array. The `h5path` form takes names among the run's coil sets (or `coil_sets` when given) and
evaluates them on the run's control surface. See [`uncorrectable_probability`](@ref) for the
NTV-limited counterpart.
"""
function correction_requirement(ctx::ResonantDriveContext, source::CoilOverlap, arrays::AbstractVector{CoilOverlap})
    isempty(arrays) && throw(ArgumentError("correction_requirement: give at least one correction array"))
    N = length(source.spectrum)
    for o in arrays
        o.mode == source.mode || throw(ArgumentError("correction_requirement: \"$(o.coil_name)\" is projected onto mode $(o.mode), the source onto mode $(source.mode)"))
        length(o.spectrum) == N || throw(DimensionMismatch("correction_requirement: \"$(o.coil_name)\" carries $(length(o.spectrum)) modes, the source $N"))
    end
    size(ctx.rc.C, 2) == N || throw(DimensionMismatch("correction_requirement: the coupling has $(size(ctx.rc.C, 2)) columns, the spectra $N entries"))
    C = ctx.rc.C
    r_source = C * source.spectrum
    R = hcat((C * o.spectrum for o in arrays)...)
    f_dom = [abs(o.delta) > 0 ? -source.delta / o.delta : NaN + NaN * im for o in arrays]
    r_dom = hcat((isnan(f_dom[k]) ? fill(NaN + NaN * im, length(r_source)) : r_source .+ R[:, k] .* f_dom[k] for k in eachindex(arrays))...)
    kat = [o.ampere_turns_kat for o in arrays]
    # Minimum-norm solutions are taken in kilo-ampere-turns when every array's ampere-turns are known,
    # so arrays given at different pattern currents are weighed by physical current, not bookkeeping.
    a = all(k -> isfinite(k) && k > 0, kat) ? kat : ones(length(kat))
    δ = [o.delta for o in arrays] ./ a
    f_mn = sum(abs2, δ) > 0 ? (-source.delta .* conj.(δ) ./ sum(abs2, δ)) ./ a : fill(NaN + NaN * im, length(δ))
    f_ls = -((R ./ a') \ r_source) ./ a
    r_ls = r_source .+ R * f_ls
    n_src = norm(source.spectrum)
    cosines = [(n = norm(o.spectrum); n > 0 && n_src > 0 ? abs(dot(o.spectrum, source.spectrum)) / (n * n_src) : NaN) for o in arrays]
    return CorrectionRequirement(source.coil_name, [o.coil_name for o in arrays], copy(ctx.rc.rational_psi), copy(ctx.rc.rational_q),
        copy(ctx.rc.rational_m), copy(ctx.rc.rational_n), f_dom, f_mn, f_ls, rank(R), f_dom .* kat, f_mn .* kat, f_ls .* kat, r_source, r_dom, r_ls, cosines)
end
function correction_requirement(h5path::AbstractString, source_name::AbstractString, array_names::AbstractVector{<:AbstractString}; mode::Int=1,
    coil_sets=nothing, psi_low::Real=0.0, psi_high::Real=PerturbedEquilibrium.CORE_PSI_HIGH, kwargs...)
    ctx = ResonantDriveContext(h5path; psi_low, psi_high, kwargs...)
    sets = coil_sets === nothing ? ForcingTerms.load_coil_sets(ctx.cfg, 1; equil=ctx.equil) : collect(coil_sets)
    by_name = Dict(cs.name => cs for cs in sets)
    pick(nm) = haskey(by_name, nm) ? by_name[nm] : throw(ArgumentError("no coil set named \"$nm\" (have $(join(sort(collect(keys(by_name))), ", ")))"))
    overlaps = coil_overlaps(ctx, vcat([pick(source_name)], [pick(nm) for nm in array_names]); mode)
    return correction_requirement(ctx, overlaps[1], overlaps[2:end])
end

"""
    needed_current_distribution(mc::MonteCarloResult, array::CoilOverlap) -> NamedTuple

The Monte Carlo's intrinsic `|δ|` histogram re-expressed as the current on `array` that cancels
each sampled overlap's dominant mode: as the factor `|δ| / |δ_array|` on the array's current as
given, fields `current_factor_bin_edges` and `current_factor_pdf` (density per unit factor,
integrating to one), and in kilo-ampere-turns through the array's `ampere_turns_kat`, fields
`current_bin_edges_kat` and `current_pdf_per_kat` (`NaN` when the ampere-turns are unknown).
"""
function needed_current_distribution(mc::MonteCarloResult, array::CoilOverlap)
    a = abs(array.delta)
    a > 0 || throw(ArgumentError("needed_current_distribution: array \"$(array.coil_name)\" has no dominant-mode overlap to correct with"))
    kat = array.ampere_turns_kat
    return (; current_factor_bin_edges=mc.abs_delta_bin_edges ./ a, current_factor_pdf=mc.abs_delta_pdf .* a,
        current_bin_edges_kat=mc.abs_delta_bin_edges .* (kat / a), current_pdf_per_kat=mc.abs_delta_pdf .* (a / kat))
end

"""
    uncorrectable_probability(mc::MonteCarloResult, c::EFCCoupling; delta_threshold, torque_budget, safety_factor=1.0, model=:auto, rotation_exponent=1.0) -> NamedTuple

How often the sampled machine's overlap exceeds what the array can correct: the fraction of the
Monte Carlo's intrinsic `|δ|` beyond `max_correctable_overlap`'s `with_ntv` limit for the array,
as a percentage. The explicit-correction counterpart of the corrected locking probability's
`efc_factor` model. Fields `abs_delta_max_correctable`, `probability_percent`, and the `model` resolved.
"""
function uncorrectable_probability(mc::MonteCarloResult, c::EFCCoupling; delta_threshold::Real, torque_budget::Real, safety_factor::Real=1.0,
    model::Symbol=:auto, rotation_exponent::Real=1.0)
    limits = max_correctable_overlap(c; delta_threshold, torque_budget, safety_factor, model, rotation_exponent)
    limit = limits.with_ntv
    edges = mc.abs_delta_bin_edges
    widths = diff(edges)
    beyond = 0.0
    for i in eachindex(widths)
        lo, hi = edges[i], edges[i+1]
        hi <= limit && continue
        beyond += mc.abs_delta_pdf[i] * (lo >= limit ? widths[i] : hi - limit)
    end
    # Samples past the last edge sit in the last bin; if the limit is beyond the grid nothing exceeds it.
    return (; abs_delta_max_correctable=limit, probability_percent=isfinite(limit) ? clamp(100 * beyond, 0.0, 100.0) : 0.0, model=_resolve_model(c, model))
end
