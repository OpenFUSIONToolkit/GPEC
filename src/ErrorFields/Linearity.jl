# The linear sensitivity model checked at finite displacements. Lives after MonteCarlo.jl because it
# converts tolerances the way the sampler does (group_tilt_deg) and rebuilds file inputs with it.

"""
    LinearityCheck

Field names follow the ErrorFields result grammar `<quantity>[_instance][_efc][_statistic][_unit]` (see the manual's
"Result names"); every field name is also its HDF5 dataset name.

The linear sensitivity model against finite rigid displacements, one row per coil set (or
coherent group), degree of freedom, scale and sign. Vectors are all `[nrow]`.

## Fields

  - `name`: coil set or coherent group
  - `kind`: `:shift` or `:tilt`
  - `axis`: 1 for x, 2 for y (a group's rows use axis 1)
  - `scale`: the displacement as a multiple of the tolerance
  - `displacement`: the signed displacement applied, metres for a shift and degrees for a tilt
  - `delta_as_designed`: δ of the undisplaced geometry
  - `delta_linear`: the model's prediction `δ_as_designed + S·Δ` (or `T·θ`, with a group's lateral
    rotation term as the Monte Carlo applies it)
  - `delta_actual`: δ of the displaced geometry from a fresh Biot–Savart pass
  - `relative_error`: `|δ_actual − δ_linear| / |δ_linear − δ_as_designed|`, the error of the
    predicted change; `NaN` where the linear change vanishes
"""
struct LinearityCheck
    name::Vector{String}
    kind::Vector{Symbol}
    axis::Vector{Int}
    scale::Vector{Float64}
    displacement::Vector{Float64}
    delta_as_designed::Vector{ComplexF64}
    delta_linear::Vector{ComplexF64}
    delta_actual::Vector{ComplexF64}
    relative_error::Vector{Float64}
end

"""
    linearity_check(ctx, coil_sets, tolerances; scales=(1.0, 2.0), mode=1, ctrl=ErrorFieldsControl()) -> LinearityCheck
    linearity_check(h5path; scales=(1.0, 2.0), mode=1, psi_low=0.0, psi_high=CORE_PSI_HIGH, ctrl=ErrorFieldsControl(), kwargs...) -> LinearityCheck

How linear the overlap is out to the tolerance edge, which is the premise every post-hoc sweep rests
on and which the stored finite-difference residuals only test at the step. Each coil set with a
tolerance block is displaced by `±scale × tolerance` along each in-plane shift axis and about each
in-plane tilt axis (a cylinder coil's tilt range is the angle its axis line can reach,
`atan(shift_tol / half_height)`); each coherent group is shifted and rigidly rotated about its pivot
as one body at its own tolerance; the overlap of the displaced geometry is recomputed and compared
with the linear prediction from [`sensitivity_table`](@ref). Include the largest scale a tolerance
scan will use, since that is the regime it relies on. Cost: one Biot–Savart pass per row, so it is
opt-in and not part of the run. The file form rebuilds the context, the coil geometry and the
tolerance snapshot from `gpec.h5`.
"""
function linearity_check(ctx::ResonantDriveContext, coil_sets::AbstractVector{CoilSet}, ts::ToleranceSet;
    scales=(1.0, 2.0), mode::Int=1, ctrl::ErrorFieldsControl=ErrorFieldsControl())
    sets = collect(coil_sets)
    validate_tolerances(ts, [cs.name for cs in sets])
    all(>(0), scales) || throw(ArgumentError("scales must be positive multiples of the tolerance"))
    table = sensitivity_table(compute_coil_sensitivities(ctx, sets, ctrl), ctx.dom; mode)
    set_of = Dict(cs.name => cs for cs in sets)
    index = Dict(nm => i for (i, nm) in enumerate(table.coil_names))
    overlap(cs) = coupling_overlap(ctx.dom, applied_spectrum(cs, ctx.rc, ctx.grids))[mode] / ctx.b_t0
    S, T = table.shift_sensitivity_per_m, table.tilt_sensitivity_per_deg
    name = String[]
    kind = Symbol[]
    axis = Int[]
    scale = Float64[]
    disp = Float64[]
    δ0s = ComplexF64[]
    lins = ComplexF64[]
    acts = ComplexF64[]
    function record!(nm, k, a, s, Δ, δ0, lin, act)
        push!(name, nm)
        push!(kind, k)
        push!(axis, a)
        push!(scale, s)
        push!(disp, Δ)
        push!(δ0s, δ0)
        push!(lins, lin)
        push!(acts, act)
    end
    for t in ts.coils
        cs = set_of[t.name]
        i = index[t.name]
        pivot = ctrl.rotation_center == "set" ? _set_center(cs) : nothing
        tilt_tol = t.tolerance_model == "cylinder" ? rad2deg(atan(t.shift_tol_m / t.cylinder_half_height_m)) : tilt_tolerance_deg(t, cs)
        δ0 = table.delta_as_designed[i]
        for (k, tol, M) in ((:shift, t.shift_tol_m, S), (:tilt, tilt_tol, T)), a in 1:2, s in scales, sgn in (-1, 1)
            Δ = sgn * s * tol
            Δ == 0 && continue
            record!(t.name, k, a, s, Δ, δ0, δ0 + M[a, i] * Δ, overlap(_rigidly_perturbed(cs, k, a, Δ, pivot)))
        end
    end
    for g in ts.groups
        members = [set_of[m] for m in g.members]
        idx = [index[m] for m in g.members]
        δ0 = sum(table.delta_as_designed[idx])
        tilt_tol = g.tolerance_model == "cylinder" ? rad2deg(atan(g.shift_tol_m / g.cylinder_half_height_m)) : group_tilt_deg(g, set_of)
        pivot = (0.0, 0.0, g.rotation_center_z_m)
        for (k, tol) in ((:shift, g.shift_tol_m), (:tilt, tilt_tol)), s in scales, sgn in (-1, 1)
            Δ = sgn * s * tol
            Δ == 0 && continue
            if k === :shift
                lin = δ0 + sum(S[1, i] * Δ for i in idx)
                act = sum(overlap(_rigidly_perturbed(cs, :shift, 1, Δ, nothing)) for cs in members)
            else
                # A rigid rotation about the pivot on the machine axis tilts each member and shifts it
                # laterally by −h·θ, the term the Monte Carlo carries as R; the geometry pass does both at once.
                θ = deg2rad(Δ)
                lin = δ0 + sum(T[1, i] * Δ - (_set_center(set_of[g.members[j]])[3] - g.rotation_center_z_m) * S[2, i] * θ for (j, i) in enumerate(idx))
                act = sum(overlap(_rigidly_perturbed(cs, :tilt, 1, Δ, pivot)) for cs in members)
            end
            record!(g.name, k, 1, s, Δ, δ0, lin, act)
        end
    end
    change = abs.(lins .- δ0s)
    rel = [c > 0 ? abs(a - l) / c : NaN for (a, l, c) in zip(acts, lins, change)]
    return LinearityCheck(name, kind, axis, scale, disp, δ0s, lins, acts, rel)
end
function linearity_check(h5path::AbstractString; scales=(1.0, 2.0), mode::Int=1, psi_low::Real=0.0, psi_high::Real=PerturbedEquilibrium.CORE_PSI_HIGH,
    ctrl::ErrorFieldsControl=ErrorFieldsControl(), kwargs...)
    _, ts, coil_sets = _monte_carlo_inputs(h5path; psi_low, psi_high, mode)
    ctx = ResonantDriveContext(h5path; psi_low, psi_high, kwargs...)
    return linearity_check(ctx, coil_sets, ts; scales, mode, ctrl)
end
