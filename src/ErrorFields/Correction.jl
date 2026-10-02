# The linear correction requirement: what currents the correction arrays need to cancel an error
# field on every rational surface, what remains, and whether the Monte Carlo's overlaps exceed what
# the NTV-limited arrays can correct. Lives after NTVLimits.jl because the exceedance reads an EFCCoupling.

"""
    CorrectionRequirement

Field names follow the ErrorFields result grammar `<quantity>[_instance][_efc][_statistic][_unit]` (see the manual's
"Result names"); every field name is also its HDF5 dataset name.

The linear correction requirement of one error-field source against a set of correction arrays, on
the rational surfaces of one run. Currents are complex multiples of each array's current as given
(magnitude and toroidal phase), so a factor of `−1` on an array identical to the source is the
exact cancellation; multiply by the array's ampere-turns for a current.

## Fields

  - `source_name`, `array_names`: the coil sets, `[narray]` for the arrays
  - `rational_psi`, `rational_q`, `rational_m`, `rational_n`: the surfaces, `[nsurface]`
  - `current_factor_dominant`: the factor on each array alone that cancels the source's
    dominant-mode overlap, `−δ_source / δ_array`; `NaN` for an array with no overlap `[narray]`
  - `current_factor_least_squares`: the factors on every array together that minimize the resonant
    field on every surface, `argmin ‖C·(b̃_source + Σ_k f_k b̃_k)‖` `[narray]`
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
    current_factor_least_squares::Vector{ComplexF64}
    resonant_field_source_t::Vector{ComplexF64}
    resonant_field_dominant_t::Matrix{ComplexF64}
    resonant_field_least_squares_t::Vector{ComplexF64}
    cosine_similarity::Vector{Float64}
end

"""
    correction_requirement(ctx, source::CoilOverlap, arrays) -> CorrectionRequirement
    correction_requirement(h5path, source_name, array_names; mode=1, coil_sets=nothing, kwargs...) -> CorrectionRequirement

What it takes to correct one error field with a set of arrays, and what is left. The dominant-mode
answer for one array is the ratio `−δ_source / δ_array`; the question worth a function is the joint
one over every rational surface: the complex factors on all arrays that minimize the resonant field
`C·(b̃_source + Σ_k f_k b̃_k)` in the least-squares sense (`\\` on the `nsurface × narray` system; the
minimum-norm solution when there are more arrays than surfaces), the resonant field left on each
surface after it and after each array's dominant-mode correction alone, and how much of the
source's spectrum each array can see. Every input is a [`CoilOverlap`](@ref) carrying its
spectrum, so any geometry can be a source or an array, not only the run's coils. The `h5path`
form takes names among the run's coil sets, or among `coil_sets` when given, and evaluates them
on the run's control surface.

Compare the needed factors with the NTV-limited allowance of the same array through
[`uncorrectable_probability`](@ref), which asks how often the Monte Carlo's overlap exceeds what
the array can correct.
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
    f_ls = -(R \ r_source)
    r_ls = r_source .+ R * f_ls
    n_src = norm(source.spectrum)
    cosines = [(n = norm(o.spectrum); n > 0 && n_src > 0 ? abs(dot(o.spectrum, source.spectrum)) / (n * n_src) : NaN) for o in arrays]
    return CorrectionRequirement(source.coil_name, [o.coil_name for o in arrays], copy(ctx.rc.rational_psi), copy(ctx.rc.rational_q),
        copy(ctx.rc.rational_m), copy(ctx.rc.rational_n), f_dom, f_ls, r_source, r_dom, r_ls, cosines)
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

The Monte Carlo's intrinsic `|δ|` histogram re-expressed as the factor on `array`'s current that
cancels each sampled overlap's dominant mode, `|δ| / |δ_array|`: fields `current_factor_bin_edges`
and `current_factor_pdf` (density per unit factor, integrating to one). The needed current for the
as-built machine is one number; over the tolerance samples it is this distribution.
"""
function needed_current_distribution(mc::MonteCarloResult, array::CoilOverlap)
    a = abs(array.delta)
    a > 0 || throw(ArgumentError("needed_current_distribution: array \"$(array.coil_name)\" has no dominant-mode overlap to correct with"))
    return (; current_factor_bin_edges=mc.abs_delta_bin_edges ./ a, current_factor_pdf=mc.abs_delta_pdf .* a)
end

"""
    uncorrectable_probability(mc::MonteCarloResult, c::EFCCoupling; delta_threshold, torque_budget, safety_factor=1.0, model=:auto, rotation_exponent=1.0) -> NamedTuple

How often the sampled machine's overlap exceeds what the array can correct: the fraction of the
Monte Carlo's intrinsic `|δ|` beyond `max_correctable_overlap`'s `with_ntv` limit for the array,
with the limit it used, as a percentage. This is the explicit-correction counterpart of the
corrected locking probability's `efc_factor` model: it asks not whether the field divided by a
factor still locks, but whether the array, spending its own NTV torque, can reach the field at
all. Fields `abs_delta_max_correctable`, `probability_percent`, and the `model` resolved.
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
