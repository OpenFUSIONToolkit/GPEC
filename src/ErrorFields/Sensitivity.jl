"""
    compute_coil_sensitivities(coil_sets, rc, equil, cfg, ctrl=ErrorFieldsControl(); psi, b_t0, verbose=false) -> CoilSensitivities

Linearize the control-surface spectrum of every coil set in `coil_sets` with respect to its six
rigid-body degrees of freedom, exactly, from one pass of the Biot-Savart field and its gradient at
the points of the surface `psi` (the solve's integration limit, `ffs.psilim`) sampled once per
toroidal mode of `rc`. Moving a set by `h·ê` changes its field at a fixed point by `−h (ê·∇)B`, and
rotating it by `θ` about `â` through the pivot `p` by `θ (â × B − ((â × (x − p))·∇)B)`; the surface
and its normals stay fixed. Tilt taps rotate about the pivot named by `ctrl.rotation_center`, in
degrees; `b_t0` is the axis toroidal field (tesla) for the δ normalization. Errors on a set
carrying no current.

Spectra are on `rc`'s (m, n) ordering, conformed with [`rootarea_field`](@ref), so
[`coupling_overlap`](@ref) applies directly.
"""
function compute_coil_sensitivities(
    coil_sets::Vector{CoilSet},
    rc::ResonantCoupling,
    equil::Equilibrium.PlasmaEquilibrium,
    cfg::CoilConfig,
    ctrl::ErrorFieldsControl=ErrorFieldsControl();
    psi::Float64,
    b_t0::Float64,
    verbose::Bool=false
)
    isempty(coil_sets) && throw(ArgumentError("compute_coil_sensitivities: no coil sets given"))
    ctrl.rotation_center in ("conductor", "set") ||
        throw(ArgumentError("rotation_center must be \"conductor\" or \"set\" (got \"$(ctrl.rotation_center)\")"))
    b_t0 > 0 || throw(ArgumentError("b_t0 must be a positive field magnitude (got $b_t0)"))
    for cs in coil_sets
        all(iszero, cs.currents) &&
            throw(ArgumentError("coil set \"$(cs.name)\" carries no current; its spectrum and sensitivities would vanish identically"))
    end

    N = length(rc.m_modes)
    nset = length(coil_sets)
    grids = forcing_grids(rc, equil, cfg; psi)
    m_low, m_high = extrema(rc.m_modes)

    nominal = zeros(ComplexF64, N, nset)
    shift = zeros(ComplexF64, N, 3, nset)
    tilt = zeros(ComplexF64, N, 3, nset)

    for (j, cs) in enumerate(coil_sets)
        t_start = time()
        set_pivot = ctrl.rotation_center == "set" ? _set_center(cs) : nothing
        strands = ForcingTerms.PrecomputedSegments[]
        pivots = NTuple{3,Float64}[]
        for c in 1:cs.ncoil, k in 1:cs.s
            iszero(cs.currents[c]) && continue
            push!(strands, ForcingTerms.precompute_segments(cs, c, k, cs.currents[c] * cs.nw))
            strand = (view(cs.x, c, k, :), view(cs.y, c, k, :), view(cs.z, c, k, :))
            push!(pivots, something(set_pivot, ForcingTerms._arc_length_center(strand...)))
        end

        # Modes of the nominal field, then the shifts along x, y, z, then the tilts about the three axes.
        modes = [ForcingMode[] for _ in 1:7]
        for (n, grid) in grids
            F = _rigid_motion_fields(strands, pivots, grid)
            for k in 1:7
                bn = zeros(grid.grid.mtheta, grid.grid.nzeta)
                ForcingTerms.project_normal_flux!(bn, view(F, 1, k, :), view(F, 2, k, :), grid.grid)
                append!(modes[k], ForcingTerms.fourier_decompose_bn(bn, grid.grid, n, m_low, m_high))
            end
        end
        nominal[:, j] = rootarea_field(rc, modes[1])
        for a in 1:3
            shift[:, a, j] = rootarea_field(rc, modes[1+a])
            tilt[:, a, j] = rootarea_field(rc, modes[4+a])
        end

        verbose &&
            @info "  $(cs.name): |b̃| = $(@sprintf("%.3e", norm(nominal[:, j]))) T, " *
                  "max |∂b̃/∂x| = $(@sprintf("%.3e", maximum(norm, eachslice(shift[:, :, j]; dims=2)))) T/m, " *
                  "max |∂b̃/∂θ| = $(@sprintf("%.3e", maximum(norm, eachslice(tilt[:, :, j]; dims=2)))) T/deg ($(@sprintf("%.2f", time() - t_start)) s)"
    end

    return CoilSensitivities(
        [cs.name for cs in coil_sets], copy(rc.m_modes), copy(rc.n_modes), b_t0,
        nominal, shift, tilt,
        [maximum(abs, cs.currents) for cs in coil_sets], [cs.nw for cs in coil_sets],
        [ForcingTerms.nominal_major_radius(cs) for cs in coil_sets]
    )
end

# Rotation axes of the single-axis tilts of `apply_transforms`: tiltx turns about +x, tilty about −y, tiltz about +z.
const _TILT_AXES = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))

_cross(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
_matvec(J, v) = (J[1] * v[1] + J[4] * v[2] + J[7] * v[3], J[2] * v[1] + J[5] * v[2] + J[8] * v[3], J[3] * v[1] + J[6] * v[2] + J[9] * v[3])

"""
    _rigid_motion_fields(strands, pivots, grid::CoilForcingGrid) -> Array{Float64,3}

Cylindrical `(B_R, B_Z)` at each point of `grid` of the strands together, and their derivatives under each rigid
motion, `[2, 7, nobs]`: the field, then its change per metre of shift along x, y, z, then per degree of tilt about
`_TILT_AXES` with each strand turning about its entry of `pivots`.
"""
function _rigid_motion_fields(strands::Vector{ForcingTerms.PrecomputedSegments}, pivots::Vector{NTuple{3,Float64}}, grid::ForcingTerms.CoilForcingGrid)
    nobs = length(grid.obs_R)
    F = zeros(2, 7, nobs)
    Threads.@threads for i in 1:nobs
        sp, cp = sincos(grid.obs_phi[i])
        taps = _rigid_motion_point((grid.obs_R[i] * cp, grid.obs_R[i] * sp, grid.obs_Z[i]), strands, pivots)
        for k in 1:7
            F[1, k, i] = taps[k][1] * cp + taps[k][2] * sp
            F[2, k, i] = taps[k][3]
        end
    end
    return F
end

# Field and rigid-motion derivatives at the Cartesian point x, in the order of `_rigid_motion_fields`.
function _rigid_motion_point(x::NTuple{3,Float64}, strands, pivots)
    b = (0.0, 0.0, 0.0)
    J = ntuple(_ -> 0.0, Val(9))
    t1 = t2 = t3 = (0.0, 0.0, 0.0)
    @inbounds for s in eachindex(strands)
        bs, Js = ForcingTerms._accumulate_precomputed_gradient(x[1], x[2], x[3], strands[s])
        b = b .+ bs
        J = J .+ Js
        d = x .- pivots[s]
        t1 = t1 .+ _cross(_TILT_AXES[1], bs) .- _matvec(Js, _cross(_TILT_AXES[1], d))
        t2 = t2 .+ _cross(_TILT_AXES[2], bs) .- _matvec(Js, _cross(_TILT_AXES[2], d))
        t3 = t3 .+ _cross(_TILT_AXES[3], bs) .- _matvec(Js, _cross(_TILT_AXES[3], d))
    end
    deg = π / 180
    return (b, (-J[1], -J[2], -J[3]), (-J[4], -J[5], -J[6]), (-J[7], -J[8], -J[9]), t1 .* deg, t2 .* deg, t3 .* deg)
end

"""
    compute_coil_sensitivities(ctx::ResonantDriveContext, coil_sets, ctrl=ErrorFieldsControl(); verbose=false) -> CoilSensitivities

Sweep `coil_sets` against an already-assembled [`ResonantDriveContext`](@ref), reusing its boundary grids.
Equivalent to the five-argument method and the way to apply a grid or ψ-window override.
"""
function compute_coil_sensitivities(ctx::ResonantDriveContext, coil_sets::Vector{CoilSet}, ctrl::ErrorFieldsControl=ErrorFieldsControl(); verbose::Bool=false)
    return compute_coil_sensitivities(coil_sets, ctx.rc, ctx.equil, ctx.cfg, ctrl; psi=ctx.psilim, b_t0=ctx.b_t0, verbose)
end

"""
    _rigidly_perturbed(cs, kind, axis, h, pivot) -> CoilSet

The coil set moved rigidly by `h` along one degree of freedom: `kind = :shift` translates every
conductor by `h` metres along `axis` (x, y, z); `kind = :tilt` rotates every conductor by `h`
degrees about `axis` through `pivot`, or about its own arc-length centre when `pivot === nothing`.
Both go through `apply_transforms` with the toroidal modulation that makes the motion rigid
(`n_tilt = 0` for the shift, `n_tilt = 1` for the tilt).
"""
function _rigidly_perturbed(cs::CoilSet, kind::Symbol, axis::Int, h::Real, pivot)
    per_conductor = fill(Float64(h), cs.ncoil)
    component(a) = a == axis ? per_conductor : Float64[]
    if kind === :shift
        cfg = CoilSetConfig(; shiftx=component(1), shifty=component(2), shiftz=component(3))
        return apply_transforms(cs, cfg; n_tilt=0)
    end
    nom(i) = pivot === nothing ? Float64[] : fill(pivot[i], cs.ncoil)
    cfg = CoilSetConfig(; tiltx=component(1), tilty=component(2), tiltz=component(3),
        rotation_center_x=nom(1), rotation_center_y=nom(2), rotation_center_z=nom(3), tilt_in_meters=false)
    return apply_transforms(cs, cfg; n_tilt=1)
end

"""
    _set_center(cs::CoilSet) -> (x0, y0, z0)

Arc-length-weighted centre of the whole coil set, over every conductor and strand: the pivot of
a rigid tilt of a multi-filament winding pack.
"""
function _set_center(cs::CoilSet)
    total = 0.0
    cx = cy = cz = 0.0
    for j in 1:cs.ncoil, k in 1:cs.s, l in 1:(cs.nsec-1)
        dl = hypot(cs.x[j, k, l+1] - cs.x[j, k, l], cs.y[j, k, l+1] - cs.y[j, k, l], cs.z[j, k, l+1] - cs.z[j, k, l])
        cx += dl * (cs.x[j, k, l] + cs.x[j, k, l+1]) / 2
        cy += dl * (cs.y[j, k, l] + cs.y[j, k, l+1]) / 2
        cz += dl * (cs.z[j, k, l] + cs.z[j, k, l+1]) / 2
        total += dl
    end
    total > 0 || return (0.0, 0.0, 0.0)
    return (cx / total, cy / total, cz / total)
end

"""
    sensitivity_table(sens::CoilSensitivities, dom::DominantCoupling; mode=1) -> SensitivityTable
    sensitivity_table(h5path; psi_low=CORE_PSI_LOW, psi_high=CORE_PSI_HIGH, mode=1) -> SensitivityTable

Project a coil linearization onto singular mode `mode` of `dom` and normalize by the axis
toroidal field: `δ = dot(V[:, mode], b̃) / B_T0` for the nominal spectrum and each derivative,
the dimensionless overlap and its sensitivities per coil set. The window and mode are analysis
choices, so re-evaluate freely on the same `sens`; the file form rebuilds the coupling from
`gpec.h5`, windows it to `psi_low ≤ ψ_N ≤ psi_high`, and reads the stored linearization.
"""
function sensitivity_table(sens::CoilSensitivities, dom::DominantCoupling; mode::Int=1)
    1 <= mode <= length(dom.singular_values) ||
        throw(ArgumentError("mode $mode is outside the $(length(dom.singular_values)) singular modes of the decomposition"))
    size(dom.right_singular_vectors, 1) == length(sens.m_modes) ||
        throw(DimensionMismatch("the decomposition acts on $(size(dom.right_singular_vectors, 1)) modes but the sensitivities carry $(length(sens.m_modes))"))
    PerturbedEquilibrium.check_mode_basis(dom, sens.m_modes, sens.n_modes, "sensitivity_table")
    v = dom.right_singular_vectors[:, mode]
    project(b̃) = dot(v, b̃) / sens.b_t0
    nset = length(sens.coil_names)

    delta_as_designed = [project(view(sens.field_as_designed, :, j)) for j in 1:nset]
    shift = [project(view(sens.shift_sensitivity_per_m, :, a, j)) for a in 1:3, j in 1:nset]
    tilt = [project(view(sens.tilt_sensitivity_per_deg, :, a, j)) for a in 1:3, j in 1:nset]
    inplane_rms(S) = [sqrt((abs2(S[1, j]) + abs2(S[2, j])) / 2) for j in 1:nset]
    cancelling(S) = [cancelling_offset(delta_as_designed[j], S[1, j], S[2, j])[i] for i in 1:2, j in 1:nset]

    per_deg = inplane_rms(tilt)
    # A tilt of one degree sweeps the rim through r·π/180 metres, the form mechanical tolerances take.
    mm_per_deg = [sens.major_radius_m[j] * pi / 180 * 1000 for j in 1:nset]
    per_mm_rim = [mm_per_deg[j] > 0 ? per_deg[j] / mm_per_deg[j] : 0.0 for j in 1:nset]

    return SensitivityTable(copy(sens.coil_names), mode, delta_as_designed, shift, tilt,
        1e-3 .* inplane_rms(shift), per_deg, per_mm_rim, cancelling(shift), cancelling(tilt))
end

function sensitivity_table(h5path::AbstractString; psi_low::Real=PerturbedEquilibrium.CORE_PSI_LOW, psi_high::Real=PerturbedEquilibrium.CORE_PSI_HIGH, mode::Int=1)
    rc = ResonantCoupling(h5path)
    dom = dominant_coupling(rc; psi_low, psi_high)
    return sensitivity_table(CoilSensitivities(h5path), dom; mode)
end

"""
    without_coils(table::SensitivityTable, names) -> SensitivityTable

The table with the named coil sets (e.g. correction arrays) dropped, leaving the error-field
sources the Monte Carlo, worst case and scans use. Every name must be in the table.
"""
function without_coils(table::SensitivityTable, names)
    names = String[String(n) for n in names]
    unknown = setdiff(names, table.coil_names)
    isempty(unknown) || throw(ArgumentError("coil sets to exclude not among the run's coil sets ($(join(table.coil_names, ", "))): $(join(unknown, ", "))"))
    keep = [i for (i, n) in enumerate(table.coil_names) if !(n in names)]
    return SensitivityTable(table.coil_names[keep], table.mode, table.delta_as_designed[keep], table.shift_sensitivity_per_m[:, keep], table.tilt_sensitivity_per_deg[:, keep],
        table.abs_delta_shift_per_mm[keep], table.abs_delta_tilt_per_deg[keep], table.abs_delta_rim_per_mm[keep], table.cancelling_shift_m[:, keep],
        table.cancelling_tilt_deg[:, keep])
end

"""
    excluded_coil_names(ef_raw::AbstractDict) -> Vector{String}
    excluded_coil_names(h5path::AbstractString) -> Vector{String}

The coil sets a run leaves out of its error field: `[ErrorFields] exclude_coils` together with
the correction arrays in `[ErrorFields.NTV] efc_coils`, read from the parsed `[ErrorFields]`
section of a deck or from the deck recorded in a run file (none when the file carries no deck).
"""
function excluded_coil_names(ef_raw::AbstractDict)
    listed = vcat(get(ef_raw, "exclude_coils", String[]), get(get(ef_raw, "NTV", Dict{String,Any}()), "efc_coils", String[]))
    return unique(String[String(n) for n in listed])
end
function excluded_coil_names(h5path::AbstractString)
    return h5open(h5path, "r") do f
        haskey(f, "Input/gpec_toml_raw") || return String[]
        excluded_coil_names(get(TOML.parse(read(f["Input/gpec_toml_raw"])), "ErrorFields", Dict{String,Any}()))
    end
end

"""
    cancelling_offset(δ_as_designed, S_x, S_y) -> (Δx, Δy)

The real in-plane displacement that cancels a coil set's nominal overlap to linear order,
`S_x·Δx + S_y·Δy = −δ_as_designed`, as the least-squares solution of the 2×2 real system on the
real and imaginary parts. For an axisymmetric hoop (`S_y = −i·S_x`) the solution is
`conj(−δ_as_designed / S_x)`; dropping the conjugate mirrors the offset about the x axis. A degenerate
pair returns the minimum-norm solution.
"""
function cancelling_offset(δ_as_designed::Number, S_x::Number, S_y::Number)
    A = [real(S_x) real(S_y); imag(S_x) imag(S_y)]
    Δ = pinv(A) * [-real(δ_as_designed), -imag(δ_as_designed)]
    return (Δ[1], Δ[2])
end
