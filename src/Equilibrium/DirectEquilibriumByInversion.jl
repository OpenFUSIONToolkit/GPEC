"""
Contour-tracing flux surface reconstruction for the direct (EFIT) equilibrium path.

Traces level sets of ψ(R,Z) using marching squares (`Contour.jl`), resamples each
closed curve to a uniform geometric-angle grid, constructs an `InverseRunInput`,
and delegates to the existing `equilibrium_solver(::InverseRunInput)`.

Select via `eq_type = "efit_by_inversion"` in `gpec.toml`.
"""

import Contour as Ctr

"""
    _is_closed_curve(curve)

Returns true if the Contour.jl curve is closed (first ≈ last vertex).
"""
function _is_closed_curve(curve)
    verts = Ctr.vertices(curve)
    length(verts) < 3 && return false
    dx = abs(verts[1][1] - verts[end][1])
    dz = abs(verts[1][2] - verts[end][2])
    return dx < 1e-8 && dz < 1e-8
end

"""
    _point_in_polygon_verts(verts, r_test, z_test)

Ray-casting point-in-polygon test on a `Ctr.vertices` vector.
"""
function _point_in_polygon_verts(verts, r_test::Float64, z_test::Float64)::Bool
    n = length(verts)
    inside = false
    j = n
    @inbounds for i in 1:n
        ri, zi = verts[i]
        rj, zj = verts[j]
        if (zi > z_test) != (zj > z_test)
            r_cross = (rj - ri) * (z_test - zi) / (zj - zi) + ri
            if r_test < r_cross
                inside = !inside
            end
        end
        j = i
    end
    return inside
end

"""
    select_plasma_contour(curve_list, ro, zo)

From the list of Contour.jl curves at a given ψ level, return the closed curve
whose interior contains the magnetic axis (ro, zo).

Returns `nothing` if no qualifying closed curve is found.
"""
function select_plasma_contour(curve_list, ro::Float64, zo::Float64)
    for curve in curve_list
        _is_closed_curve(curve) || continue
        verts = Ctr.vertices(curve)
        _point_in_polygon_verts(verts, ro, zo) && return curve
    end
    return nothing
end

"""
    clamp_psihigh_to_separatrix(raw_profile) -> (clamped_psihigh, was_adjusted)

Binary-searches for the highest psihigh ≤ raw_profile.config.psihigh at which the
ψ level set is still a closed curve in the EFIT grid. Returns the safe value and a
Bool indicating whether any clamping occurred.
"""
function clamp_psihigh_to_separatrix(raw_profile::DirectRunInput)
    psihigh = raw_profile.config.psihigh
    ψ_coarse = raw_profile.psi_in.nodal_derivs.partials[1, :, :]

    has_closed_contour(ψ_high) = any(
        _is_closed_curve,
        Ctr.lines(Ctr.contour(raw_profile.psi_in_xs, raw_profile.psi_in_ys,
                              ψ_coarse, raw_profile.psio * (1.0 - ψ_high)))
    )

    has_closed_contour(psihigh) && return (psihigh, false)

    lo = min(0.99, psihigh - 0.1)
    hi = psihigh
    for _ in 1:52
        hi - lo < 1e-10 && break
        mid = (lo + hi) / 2.0
        has_closed_contour(mid) ? (lo = mid) : (hi = mid)
    end
    return (lo, true)
end

"""
    resample_contour_to_theta_grid!(R_out, Z_out, curve, ro, zo, theta_grid)

Resample a Contour.jl curve onto a uniform geometric-angle grid, writing results
directly into the pre-allocated `R_out` and `Z_out` vectors (views into R_table/Z_table).
Sorts vertices by geometric angle η, appends a periodic wrap point, fits cubic splines
R(η) and Z(η), then samples at each `theta_grid` point.
"""
function resample_contour_to_theta_grid!(
    R_out::AbstractVector{Float64}, Z_out::AbstractVector{Float64},
    curve, ro::Float64, zo::Float64, theta_grid::AbstractVector{Float64}
)
    verts = Ctr.vertices(curve)
    nv = length(verts)

    # Drop duplicate closing vertex (Contour.jl closed curves repeat first==last)
    if nv > 1
        r1, z1 = verts[1]
        rn, zn = verts[nv]
        if abs(r1 - rn) < 1e-8 && abs(z1 - zn) < 1e-8
            nv -= 1
        end
    end

    # Compute angles and extract R/Z in a single pass
    η_buf = Vector{Float64}(undef, nv)
    R_buf = Vector{Float64}(undef, nv)
    Z_buf = Vector{Float64}(undef, nv)
    @inbounds for i in 1:nv
        r, z = verts[i]
        η_buf[i] = mod(atan(z - zo, r - ro), 2π)
        R_buf[i] = r
        Z_buf[i] = z
    end

    idx = sortperm(η_buf)

    # Build extended (nv+1) sorted arrays with periodic wrap appended
    n_ext = nv + 1
    η_ext = Vector{Float64}(undef, n_ext)
    R_ext = Vector{Float64}(undef, n_ext)
    Z_ext = Vector{Float64}(undef, n_ext)
    @inbounds for i in 1:nv
        k = idx[i]
        η_ext[i] = η_buf[k]
        R_ext[i] = R_buf[k]
        Z_ext[i] = Z_buf[k]
    end
    η_ext[n_ext] = η_ext[1] + 2π
    R_ext[n_ext] = R_ext[1]
    Z_ext[n_ext] = Z_ext[1]

    R_spl = cubic_interp(η_ext, R_ext; bc=PeriodicBC(), extrap=WrapExtrap(), search=LinearBinary())
    Z_spl = cubic_interp(η_ext, Z_ext; bc=PeriodicBC(), extrap=WrapExtrap(), search=LinearBinary())

    # theta_grid is in turns [0, 1]; sample the geometric-angle spline at 2π*θ radians.
    # Monotonically increasing → shared hint gives O(1) lookups per pass.
    hint = Ref(1)
    @inbounds for k in eachindex(theta_grid, R_out)
        R_out[k] = R_spl(2π * theta_grid[k]; hint=hint)
    end
    hint[] = 1
    @inbounds for k in eachindex(theta_grid, Z_out)
        Z_out[k] = Z_spl(2π * theta_grid[k]; hint=hint)
    end
end

"""
    classify_topology(raw_profile, psio; xpt_threshold=0.05) → Symbol

Classify the plasma topology as `:limited`, `:sn_lower`, `:sn_upper`, or `:double_null`
by evaluating ψ on the (R, Z) grid and finding the minimum in each Z half-domain.

An x-point is detected when the minimum ψ in a half-domain falls below `xpt_threshold * psio`.
"""
function classify_topology(raw_profile::DirectRunInput, psio::Float64;
                           xpt_threshold::Float64=0.05)
    R_grid = raw_profile.psi_in_xs
    Z_grid = raw_profile.psi_in_ys
    nZ = length(Z_grid)
    mid_Z = nZ ÷ 2
    hint = (Ref(1), Ref(1))

    min_ψ_lower = Inf
    for jZ in 1:mid_Z
        hint[1][] = 1  # reset R hint at start of each row; Z hint advances naturally
        z = Z_grid[jZ]
        for R in R_grid
            ψ = raw_profile.psi_in((R, z); hint=hint)
            min_ψ_lower = min(min_ψ_lower, ψ)
        end
    end

    min_ψ_upper = Inf
    for jZ in (mid_Z+1):nZ
        hint[1][] = 1
        z = Z_grid[jZ]
        for R in R_grid
            ψ = raw_profile.psi_in((R, z); hint=hint)
            min_ψ_upper = min(min_ψ_upper, ψ)
        end
    end

    has_lower = min_ψ_lower / psio < xpt_threshold
    has_upper = min_ψ_upper / psio < xpt_threshold
    if has_lower && has_upper
        return :double_null
    elseif has_lower
        return :sn_lower
    elseif has_upper
        return :sn_upper
    else
        return :limited
    end
end

"""
    make_multi_stretched_grid(lo, hi, centers, n, β) → Vector{Float64}

Build a 1D grid of length `n` on `[lo, hi]` with sinh-stretching toward each point in
`centers`. The domain is split at each interior concentration point; each resulting
sub-interval uses:
- symmetric (double-ended) sinh when both endpoints are concentration points
- one-sided sinh fine at the concentration-point end otherwise

Domain boundaries (`lo`, `hi`) are treated as concentration points only if they appear in
`centers`. This produces a grid that is simultaneously fine near the magnetic axis (an
interior concentration point) and near x-points (which lie at domain boundaries for
the clipped plasma bounding box). Returns a uniform grid if `β ≤ 0`.
"""
function make_multi_stretched_grid(lo::Float64, hi::Float64,
                                    centers::Vector{Float64}, n::Int, β::Float64)
    β ≤ 0.0 && return collect(range(lo, hi; length=n))
    lo_is_conc = any(c -> abs(c - lo) < 1e-12, centers)
    hi_is_conc = any(c -> abs(c - hi) < 1e-12, centers)
    interior   = sort!(unique!(filter(c -> lo < c < hi, copy(centers))))
    nodes      = [lo; interior; hi]
    n_segs     = length(nodes) - 1
    n_segs == 0 && return collect(range(lo, hi; length=n))

    total_len = hi - lo
    # Distribute n points across segments proportionally; last segment takes the remainder
    # so concatenation (dropping n_segs-1 shared endpoints) gives exactly n total points.
    n_segi = Vector{Int}(undef, n_segs)
    allocated = 0
    for k in 1:(n_segs - 1)
        n_segi[k] = max(2, round(Int, n * (nodes[k+1] - nodes[k]) / total_len))
        allocated += n_segi[k]
    end
    n_segi[n_segs] = max(2, n + (n_segs - 1) - allocated)

    result = Float64[]
    for k in 1:n_segs
        a, b = nodes[k], nodes[k+1]
        left_fine  = k > 1 || lo_is_conc
        right_fine = k < n_segs || hi_is_conc
        sub = if left_fine && right_fine
            _sinh_both_ends(a, b, n_segi[k], β)
        elseif left_fine
            _sinh_left(a, b, n_segi[k], β)
        elseif right_fine
            _sinh_right(a, b, n_segi[k], β)
        else
            collect(range(a, b; length=n_segi[k]))
        end
        isempty(result) ? append!(result, sub) : append!(result, sub[2:end])
    end
    return result
end

# sinh grid fine at left end a: x(v) = a + Δ·sinh(β·v)/sinh(β)
function _sinh_left(a::Float64, b::Float64, n::Int, β::Float64)
    Δ = b - a; s = sinh(β)
    [a + Δ * sinh(β * (k - 1) / (n - 1)) / s for k in 1:n]
end

# sinh grid fine at right end b: x(v) = b − Δ·sinh(β·(1−v))/sinh(β)
function _sinh_right(a::Float64, b::Float64, n::Int, β::Float64)
    Δ = b - a; s = sinh(β)
    [b - Δ * sinh(β * (1.0 - (k - 1) / (n - 1))) / s for k in 1:n]
end

# symmetric sinh fine at both ends a and b
function _sinh_both_ends(a::Float64, b::Float64, n::Int, β::Float64)
    Δ = b - a; s = sinh(β)
    grid = Vector{Float64}(undef, n)
    @inbounds for k in 1:n
        v = (k - 1) / (n - 1)
        grid[k] = v ≤ 0.5 ? (a + 0.5Δ * sinh(β * 2v) / s) :
                             (b - 0.5Δ * sinh(β * 2(1.0 - v)) / s)
    end
    return grid
end

"""
    make_stretched_r_grid(r_lo, r_hi, ro, nr, β_r) → Vector{Float64}

Grid of `nr` points on `[r_lo, r_hi]` concentrated near `ro` (magnetic axis).
Thin wrapper around `make_multi_stretched_grid`.
"""
function make_stretched_r_grid(r_lo::Float64, r_hi::Float64, ro::Float64, nr::Int, β_r::Float64)
    return make_multi_stretched_grid(r_lo, r_hi, [ro], nr, β_r)
end

"""
    make_stretched_z_grid(z_lo, z_hi, zo, nz, topology, β_z) → Vector{Float64}

Grid of `nz` points on `[z_lo, z_hi]` concentrated near `zo` (magnetic axis) **and**
near each x-point boundary (at `z_lo` for `:sn_lower`/`:double_null`, at `z_hi` for
`:sn_upper`/`:double_null`). Thin wrapper around `make_multi_stretched_grid`.
"""
function make_stretched_z_grid(z_lo::Float64, z_hi::Float64, zo::Float64, nz::Int,
                                topology::Symbol, β_z::Float64)
    centers = Float64[zo]
    topology ∈ (:sn_lower, :double_null) && push!(centers, z_lo)
    topology ∈ (:sn_upper, :double_null) && push!(centers, z_hi)
    return make_multi_stretched_grid(z_lo, z_hi, centers, nz, β_z)
end

"""
    adaptive_grid_params(raw_profile, ro, zo, r_lo, r_hi, z_lo, z_hi,
                         psilow, Δψ, bbox_curve) → (nr, nz, β_r, β_z)

Physics-based Cartesian grid sizing for contour-tracing equilibrium reconstruction.

Computes required cell widths from the bilinear interpolation error bound
  δψ ≈ (dR²·|ψ_RR| + dZ²·|ψ_ZZ|)/8 ≤ Δψ  →  dR_req = √(8Δψ/|ψ_RR|)
sampled at each LCFS vertex (reusing `bbox_curve`), plus an axis constraint ensuring
the innermost surface (psilow) has ≥ 5 cells per semi-axis on the global grid.

β = acosh(h_max/h_min) from the ratio of coarsest to finest required cell widths.
n = 1 + ⌈span·sinch(β)/h_min⌉ where sinch(β) = β/sinh(β).
"""
function adaptive_grid_params(
    raw_profile::DirectRunInput,
    ro::Float64, zo::Float64,
    r_lo::Float64, r_hi::Float64,
    z_lo::Float64, z_hi::Float64,
    psilow::Float64, Δψ::Float64,
    bbox_curve
)
    psio = raw_profile.psio
    dR_reqs = Float64[]
    dZ_reqs = Float64[]

    # Required cell widths at each LCFS vertex from bilinear interpolation error formula
    if bbox_curve !== nothing
        for (R, Z) in Ctr.vertices(bbox_curve)
            ψ_RR = abs(raw_profile.psi_in((R, Z); deriv=Val((2, 0))))
            ψ_ZZ = abs(raw_profile.psi_in((R, Z); deriv=Val((0, 2))))
            ψ_RR > 0 && push!(dR_reqs, sqrt(8 * Δψ / ψ_RR))
            ψ_ZZ > 0 && push!(dZ_reqs, sqrt(8 * Δψ / ψ_ZZ))
        end
    end

    # Axis constraint: global grid cell ≤ 0.2 × a_low (minimum semi-axis of innermost surface).
    # Using min(a_R, a_Z) for both directions matches the old scale_r/scale_z target_cell_frac
    # logic and ensures the constraint is the same in both directions.
    ψ_RR_ax = abs(raw_profile.psi_in((ro, zo); deriv=Val((2, 0))))
    ψ_ZZ_ax = abs(raw_profile.psi_in((ro, zo); deriv=Val((0, 2))))
    a_low_ax = min(sqrt(2 * psilow * psio / ψ_RR_ax), sqrt(2 * psilow * psio / ψ_ZZ_ax))
    push!(dR_reqs, 0.2 * a_low_ax)
    push!(dZ_reqs, 0.2 * a_low_ax)

    function _params_1d(reqs, span)
        h_min = max(minimum(reqs), span / 4000)
        h_max = min(max(maximum(reqs), h_min), span / 4)
        β = h_max > h_min * (1.0 + 1e-10) ? acosh(h_max / h_min) : 0.0
        # Cap at 3.0: allows aggressive concentration at x-points and axis while keeping
        # h_max/h_min = cosh(3)≈10 reasonable for intermediate flux surfaces.
        β = clamp(β, 0.0, 3.0)
        sinchβ = β < 1e-6 ? 1.0 : β / sinh(β)
        n = max(4, 1 + ceil(Int, span * sinchβ / h_min))
        return n, β
    end

    nr_fine, β_r = _params_1d(dR_reqs, r_hi - r_lo)
    nz_fine, β_z = _params_1d(dZ_reqs, z_hi - z_lo)
    return nr_fine, nz_fine, β_r, β_z
end

"""
    equilibrium_solver_by_inversion(raw_profile; resolution_factor=1.0,
                                    refine=nothing, β_r=nothing, β_z=nothing)

Driver for contour-tracing equilibrium reconstruction.

Grid parameters (n and β in each direction) are computed adaptively from the bilinear
interpolation error bound along the LCFS and an axis constraint. `resolution_factor`
scales both n values uniformly (higher → more points, same β).

Legacy keyword arguments `refine`, `β_r`, `β_z` override the adaptive values when
provided (for benchmark sweeps and backward compatibility).

Select via `eq_type = "efit_by_inversion"` in `gpec.toml`.
"""
function equilibrium_solver_by_inversion(
    raw_profile::DirectRunInput;
    resolution_factor::Float64 = 1.0,
    refine::Union{Nothing,Int} = nothing,
    β_r::Union{Nothing,Float64} = nothing,
    β_z::Union{Nothing,Float64} = nothing
)
    equil_params = raw_profile.config
    psio = raw_profile.psio
    mpsi = equil_params.mpsi
    mtheta = equil_params.mtheta
    psilow = equil_params.psilow
    psihigh = equil_params.psihigh

    # Build target psi_norm grid (same ldp scheme as direct solver)
    psi_nodes = [psilow + (psihigh - psilow) * sin((ipsi / mpsi) * (π / 2))^2 for ipsi in 0:mpsi]

    # Find magnetic axis and separatrix
    ro, zo, _, _ = direct_position!(raw_profile)

    # Detect plasma topology for sinh-stretching direction
    topology = classify_topology(raw_profile, psio)
    @info "efit_by_inversion: topology = $topology"

    # Clip fine grid to separatrix bounding box.
    # The EFIT nodal ψ values are free (already stored in the spline); a single coarse
    # Contour.jl call at ψ ≈ psihigh gives the plasma bounding box with no extra spline work.
    # A 5% margin on each side ensures small arc curvature outside the box is covered.
    nw = length(raw_profile.psi_in_xs)
    nh = length(raw_profile.psi_in_ys)
    ψ_coarse = raw_profile.psi_in.nodal_derivs.partials[1, :, :]
    ψ_bbox_target = psio * max(1.0 - psihigh, 1e-3)   # outermost needed surface, ≥ 0.001·psio
    cl_bbox = Ctr.contour(raw_profile.psi_in_xs, raw_profile.psi_in_ys, ψ_coarse, ψ_bbox_target)
    bbox_curve = select_plasma_contour(Ctr.lines(cl_bbox), ro, zo)
    if bbox_curve !== nothing
        bv = Ctr.vertices(bbox_curve)
        r_bbox_lo, r_bbox_hi = extrema(v[1] for v in bv)
        z_bbox_lo, z_bbox_hi = extrema(v[2] for v in bv)
        r_margin = (r_bbox_hi - r_bbox_lo) * 0.05
        z_margin = (z_bbox_hi - z_bbox_lo) * 0.05
        r_lo = max(raw_profile.rmin, r_bbox_lo - r_margin)
        r_hi = min(raw_profile.rmax, r_bbox_hi + r_margin)
        z_lo = max(raw_profile.zmin, z_bbox_lo - z_margin)
        z_hi = min(raw_profile.zmax, z_bbox_hi + z_margin)
    else
        r_lo, r_hi = raw_profile.rmin, raw_profile.rmax
        z_lo, z_hi = raw_profile.zmin, raw_profile.zmax
    end

    # ψ curvature at axis: needed for near_axis_threshold and zoomed core grid sizing.
    # a_low = min semi-axis of the innermost flux surface (at psilow) from ψ ≈ ½|ψ_RR|dR².
    ψ_RR_abs = abs(raw_profile.psi_in((ro, zo); deriv=Val((2, 0))))
    ψ_ZZ_abs = abs(raw_profile.psi_in((ro, zo); deriv=Val((0, 2))))
    a_low = min(sqrt(2 * psilow * psio / ψ_RR_abs), sqrt(2 * psilow * psio / ψ_ZZ_abs))

    # Compute physics-based (n, β) from LCFS curvature sampling and axis constraint.
    # resolution_factor scales n; β is determined by h_max/h_min curvature ratio alone.
    Δψ = psio * (psihigh - psilow) / mpsi
    nr_adp, nz_adp, β_r_adp, β_z_adp = adaptive_grid_params(
        raw_profile, ro, zo, r_lo, r_hi, z_lo, z_hi, psilow, Δψ, bbox_curve)
    β_r_use = β_r !== nothing ? β_r : β_r_adp
    β_z_use = β_z !== nothing ? β_z : β_z_adp
    if refine !== nothing
        nr_fine = max(4, round(Int, refine * nw * (r_hi - r_lo) / (raw_profile.rmax - raw_profile.rmin)))
        nz_fine = max(4, round(Int, refine * nh * (z_hi - z_lo) / (raw_profile.zmax - raw_profile.zmin)))
    else
        nr_fine = max(4, round(Int, nr_adp * resolution_factor))
        nz_fine = max(4, round(Int, nz_adp * resolution_factor))
    end
    r_fine = make_stretched_r_grid(r_lo, r_hi, ro, nr_fine, β_r_use)
    z_grid = make_stretched_z_grid(z_lo, z_hi, zo, nz_fine, topology, β_z_use)

    iro = clamp(searchsortedfirst(r_fine, ro), 2, length(r_fine))
    izo = clamp(searchsortedfirst(z_grid, zo), 2, length(z_grid))
    dR_axis = r_fine[iro] - r_fine[iro - 1]
    dZ_axis = z_grid[izo] - z_grid[izo - 1]

    # Near-axis threshold: surfaces spanning fewer than n_min_cells global-grid cells per
    # semi-axis are re-traced by the zoomed core grid.
    d_max = max(dR_axis, dZ_axis)
    n_min_cells = 10
    near_axis_threshold = (n_min_cells * d_max)^2 * max(ψ_RR_abs, ψ_ZZ_abs) / (2 * psio)

    @info "efit_by_inversion: a_low=$(@sprintf("%.1f", a_low*1000)) mm; grid=$(nr_fine)×$(nz_fine) β_r=$(@sprintf("%.2f", β_r_use)) β_z=$(@sprintf("%.2f", β_z_use)) (dR=$(@sprintf("%.2f", dR_axis*1000)) mm dZ=$(@sprintf("%.2f", dZ_axis*1000)) mm); zoom threshold ψ<$(@sprintf("%.4f", near_axis_threshold))"

    # Parallel grid evaluation: each column (fixed z) is independent.
    # One hint pair per thread (allocated once), reset at the start of each column.
    # Resetting R-hint is required (R starts over at rmin each column); resetting Z-hint
    # lets it settle on the first lookup (Z is fixed per column, so O(log n) once then O(1)).
    # Z-hint cannot be shared across threads for a non-uniform z_grid — each thread resets it.
    ψ_fine = Matrix{Float64}(undef, nr_fine, nz_fine)
    thread_hints = [(Ref(1), Ref(1)) for _ in 1:Threads.nthreads()]
    Threads.@threads for j in 1:nz_fine
        z = z_grid[j]
        h = thread_hints[Threads.threadid()]
        h[1][] = 1
        h[2][] = 1
        @inbounds for i in 1:nr_fine
            ψ_fine[i, j] = raw_profile.psi_in((r_fine[i], z); hint=h)
        end
    end

    # Theta grid for InverseRunInput: fractional turns in [0, 1].
    # InverseEquilibrium uses theta ∈ [0,1] and formula cos(2π*(theta + deta)),
    # so rz_in_ys must be in turns, not radians.
    theta_grid = collect(range(0.0, 1.0; length=mtheta + 1))

    # Build R(ψ, θ) and Z(ψ, θ) tables
    R_table = Matrix{Float64}(undef, mpsi + 1, mtheta + 1)
    Z_table = Matrix{Float64}(undef, mpsi + 1, mtheta + 1)

    # Parallel contour tracing: each surface is independent.
    # Ctr.contour has no global state (confirmed from source inspection) → thread-safe.
    # surface_status: 0=global Contour.jl success, 1=below threshold (zoomed pass), -1=error.
    surface_status = Vector{Int8}(undef, mpsi + 1)

    Threads.@threads for ipsi in 1:(mpsi+1)
        psifac = psi_nodes[ipsi]

        # Surface too small for the global grid to resolve with n_min=10: defer to zoomed pass.
        # Elliptic fill (quadratic ψ expansion — kept for reference):
        # a_R = sqrt(2 * psifac * psio / ψ_RR_abs)
        # a_Z = sqrt(2 * psifac * psio / ψ_ZZ_abs)
        # @inbounds for j in 1:(mtheta+1)
        #     θ = theta_grid[j]
        #     R_table[ipsi, j] = ro + a_R * cos(2π * θ)
        #     Z_table[ipsi, j] = zo + a_Z * sin(2π * θ)
        # end
        if psifac <= near_axis_threshold
            surface_status[ipsi] = 1
            continue
        end

        ψ_target = psio * (1.0 - psifac)
        cl = Ctr.contour(r_fine, z_grid, ψ_fine, ψ_target)
        curve = select_plasma_contour(Ctr.lines(cl), ro, zo)

        if curve === nothing
            surface_status[ipsi] = -1
            continue
        end

        resample_contour_to_theta_grid!(
            @view(R_table[ipsi, :]), @view(Z_table[ipsi, :]),
            curve, ro, zo, theta_grid
        )
        surface_status[ipsi] = 0
    end

    # Serial error check and summary (after all threads complete)
    for ipsi in 1:(mpsi+1)
        if surface_status[ipsi] == -1
            psifac = psi_nodes[ipsi]
            error("efit_by_inversion: No closed flux surface found at psifac = $(@sprintf("%.4f", psifac)) " *
                "— psihigh likely exceeds the separatrix or is outside the EFIT domain.")
        end
    end
    n_contour_success = count(==(0), surface_status)
    n_near_axis_fill  = count(==(1), surface_status)

    # Zoomed core pass: re-trace surfaces below the near-axis threshold on a dedicated
    # axis-centered grid. Both R and Z are sinh-stretched toward (ro, zo) — no x-point
    # stretching is needed inside this small box. The grid is sized so the cell at (ro, zo)
    # is ≤ a_low / n_min_cells, guaranteeing n_min_cells cells per semi-axis at psilow.
    #
    # To avoid cubic-spline transition artifacts at the zoom/global boundary, the zoomed
    # pass is extended to zoom_extend_factor × near_axis_threshold. At that higher ψ the
    # global grid has ~sqrt(zoom_extend_factor) × n_min_cells cells per semi-axis and agrees
    # well with the zoomed grid, so the spline sees no discontinuity at the handoff.
    if n_near_axis_fill > 0
        zoom_extend_factor = 9
        psi_zoom_max = zoom_extend_factor * near_axis_threshold
        n_zoom_surfaces = count(<=(psi_zoom_max), psi_nodes)

        # Zoomed bbox: semi-axes sized for psi_zoom_max (not just the threshold), +20%.
        a_zoom_r = sqrt(2 * psi_zoom_max * psio / ψ_RR_abs) * 1.2
        a_zoom_z = sqrt(2 * psi_zoom_max * psio / ψ_ZZ_abs) * 1.2
        r_zoom_lo = max(r_lo, ro - a_zoom_r)
        r_zoom_hi = min(r_hi, ro + a_zoom_r)
        z_zoom_lo = max(z_lo, zo - a_zoom_z)
        z_zoom_hi = min(z_hi, zo + a_zoom_z)

        # Grid sizing: n_min_cells at (ro, zo) required for the SMALLEST surface (psilow).
        # sinh-center cell ≈ (4β/sinh(β)) · a_zoom / (n/2) → n ≥ 8β·a_zoom·n_min / (sinh(β)·a_low)
        sinh_β_r = β_r_use > 1e-6 ? sinh(β_r_use) : 1.0
        nr_zoom = max(200, ceil(Int, 8 * β_r_use * a_zoom_r * n_min_cells / (sinh_β_r * a_low)))
        nz_zoom = max(200, ceil(Int, 8 * β_r_use * a_zoom_z * n_min_cells / (sinh_β_r * a_low)))
        r_zoom_grid = make_stretched_r_grid(r_zoom_lo, r_zoom_hi, ro, nr_zoom, β_r_use)
        z_zoom_grid = make_stretched_r_grid(z_zoom_lo, z_zoom_hi, zo, nz_zoom, β_r_use)

        iro_z = clamp(searchsortedfirst(r_zoom_grid, ro), 2, length(r_zoom_grid))
        izo_z = clamp(searchsortedfirst(z_zoom_grid, zo), 2, length(z_zoom_grid))
        dR_zoom = r_zoom_grid[iro_z] - r_zoom_grid[iro_z - 1]
        dZ_zoom = z_zoom_grid[izo_z] - z_zoom_grid[izo_z - 1]
        @info "efit_by_inversion: zoomed core grid $(nr_zoom)×$(nz_zoom) covering [ro±$(@sprintf("%.0f", a_zoom_r*1000)) mm, zo±$(@sprintf("%.0f", a_zoom_z*1000)) mm]; dR=$(@sprintf("%.2f", dR_zoom*1000)) mm, dZ=$(@sprintf("%.2f", dZ_zoom*1000)) mm (target ≤ $(@sprintf("%.2f", a_low/n_min_cells*1000)) mm); tracing $n_zoom_surfaces surfaces (threshold=$n_near_axis_fill + $zoom_extend_factor× overlap)"

        ψ_zoom = Matrix{Float64}(undef, nr_zoom, nz_zoom)
        Threads.@threads for j in 1:nz_zoom
            z = z_zoom_grid[j]
            h = thread_hints[Threads.threadid()]
            h[1][] = 1
            h[2][] = 1
            @inbounds for i in 1:nr_zoom
                ψ_zoom[i, j] = raw_profile.psi_in((r_zoom_grid[i], z); hint=h)
            end
        end

        Threads.@threads for ipsi in 1:n_zoom_surfaces
            psifac = psi_nodes[ipsi]
            ψ_target = psio * (1.0 - psifac)
            cl_z = Ctr.contour(r_zoom_grid, z_zoom_grid, ψ_zoom, ψ_target)
            curve_z = select_plasma_contour(Ctr.lines(cl_z), ro, zo)
            if curve_z === nothing
                surface_status[ipsi] = -1
                continue
            end
            resample_contour_to_theta_grid!(
                @view(R_table[ipsi, :]), @view(Z_table[ipsi, :]),
                curve_z, ro, zo, theta_grid
            )
            surface_status[ipsi] = 0
        end

        for ipsi in 1:n_zoom_surfaces
            if surface_status[ipsi] == -1
                psifac = psi_nodes[ipsi]
                error("efit_by_inversion: Zoomed core grid failed to find flux surface at ψ = $(@sprintf("%.4f", psifac)). " *
                    "Try increasing psilow or reducing the zoom bbox margin.")
            end
        end
        @info "efit_by_inversion: $(mpsi + 1 - n_zoom_surfaces) surfaces traced by global grid, $n_zoom_surfaces by zoomed core grid ($n_near_axis_fill below threshold + $(n_zoom_surfaces - n_near_axis_fill) overlap)"
    else
        @info "efit_by_inversion: $n_contour_success surfaces traced by global grid"
    end

    # Build InverseRunInput — same type consumed by equilibrium_solver(::InverseRunInput)
    rz_in_xs = psi_nodes
    rz_in_ys = theta_grid
    itp_opts2d = (search=LinearBinary(), bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))

    rz_in_R = cubic_interp((rz_in_xs, rz_in_ys), R_table; itp_opts2d...)
    rz_in_Z = cubic_interp((rz_in_xs, rz_in_ys), Z_table; itp_opts2d...)

    inv_input = InverseRunInput(raw_profile.config, raw_profile.sq_in,
        rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z, ro, zo, psio)

    pe = equilibrium_solver(inv_input)

    # Round-trip validation: (ψ,θ) → (R,Z) → ψ_spline − ψ_target.
    # Checks 4 angles at the outermost surface and at 75% of the radial grid.
    # A large error indicates that Contour.jl resolution was insufficient near the
    # x-point and the traced surface positions are inaccurate.
    max_rt_err = 0.0
    for ψ_check in (pe.rzphi_xs[end], pe.rzphi_xs[max(1, length(pe.rzphi_xs) * 3 ÷ 4)])
        for θ_check in (0.0, 0.25, 0.5, 0.75)
            r2  = pe.rzphi_rsquared((ψ_check, θ_check))
            off = pe.rzphi_offset((ψ_check, θ_check))
            rfac = sqrt(max(r2, 0.0))
            η    = 2π * (θ_check + off)
            R    = pe.ro + rfac * cos(η)
            Z    = pe.zo + rfac * sin(η)
            ψ_rt = 1.0 - raw_profile.psi_in((R, Z)) / psio
            max_rt_err = max(max_rt_err, abs(ψ_rt - ψ_check))
        end
    end
    if max_rt_err > 2e-3
        @warn "efit_by_inversion: round-trip error at edge = $(@sprintf("%.2e", max_rt_err)) > 2e-3; accuracy near psihigh may be limited. Consider reducing psihigh or increasing resolution_factor."
    else
        @info "efit_by_inversion: round-trip error at edge = $(@sprintf("%.2e", max_rt_err))"
    end

    return pe
end

