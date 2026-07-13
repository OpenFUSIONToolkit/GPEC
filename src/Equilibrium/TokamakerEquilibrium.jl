"""
TokaMaker equilibrium reader. Turns a live TokaMaker free-boundary Grad-Shafranov
solution into a GPEC `DirectRunInput`/`InverseRunInput`, reusing the existing
`equilibrium_solver`. Two interior routes plus a refine stub:

  - `eq_type = "tokamaker"`        — trace_surf inverse route (native flux-surface contours;
                                     also builds the vacuum-region exterior grid)
  - `eq_type = "tokamaker_direct"` — sample ψ(R,Z) onto a rectangular grid (efit-style direct
                                     route; used mainly for the GS-error comparison study)
  - `eq_type = "tokamaker_refine"` — STUB: re-solve a rough geqdsk/IMAS equilibrium (deferred)

TokaMaker is not a GPEC dependency. The equilibrium object is passed in as `additional_input`
(the caller's session already has TokaMaker loaded) and its API is resolved at runtime from the
object's own module, so core GPEC never links liboftpy.
"""

# Conversion of the poloidal flux between TokaMaker's FE field and GPEC's internal COCOS-2
# convention (ψ in Wb/rad). get_field_eval("psi") and get_refs().psi_bounds come from the same
# FE field, so they are mutually consistent; this factor only carries the overall 2π between the
# two conventions. VERIFY against a known case (compare psio here to the g-file psio for the same
# TokaMaker solution) before trusting absolute ψ magnitudes.
const TOKAMAKER_PSI_TWOPI = 1.0

"""
    _tokamaker_module(eq) -> Module

Resolve the TokaMaker top-level module from an equilibrium object so its exported API
(`trace_surf`, `get_profiles`, `get_q`, `get_globals`, `get_refs`, `get_field_eval`) can be
called without a compile-time dependency. Errors if `eq` is not a TokaMaker equilibrium.
"""
function _tokamaker_module(eq)
    root = Base.moduleroot(parentmodule(typeof(eq)))
    isdefined(root, :trace_surf) || error(
        "read_tokamaker: object of type $(typeof(eq)) is not a TokaMaker equilibrium " *
        "(module $root exports no `trace_surf`). Load the TokaMaker package and pass a " *
        "TokaMakerEquilibrium as additional_input.")
    return root
end

"""
    _resample_contour_to_theta(R, Z, ro, zo, theta_grid) -> (Rθ, Zθ)

Resample one flux-surface contour (`R`, `Z` points in any order/orientation) onto the uniform
poloidal grid `theta_grid` ∈ [0,1], parameterizing by normalized geometric angle about the
magnetic axis `(ro, zo)`. The outboard midplane (geometric angle 0) is θ=0. Assumes the surface
is star-shaped about the axis (monotone geometric angle), which holds for closed flux surfaces
away from an X-point. Pure — no TokaMaker calls; unit-testable with synthetic contours.
"""
function _resample_contour_to_theta(R::AbstractVector, Z::AbstractVector,
    ro::Real, zo::Real, theta_grid::AbstractVector)
    n = length(R)
    α = Vector{Float64}(undef, n)
    for k in 1:n
        α[k] = mod(atan(Z[k] - zo, R[k] - ro) / (2π), 1.0)
    end
    perm = sortperm(α)
    αs = α[perm]
    Rs = collect(float.(R[perm]))
    Zs = collect(float.(Z[perm]))

    # Drop duplicate angles (trace_surf can repeat the closing point) to keep knots strictly increasing.
    keep = [true; diff(αs) .> 1e-12]
    αs = αs[keep]
    Rs = Rs[keep]
    Zs = Zs[keep]

    # Periodic wrap: repeat the first knot at α+1 so interpolation covers the full circle.
    α_wrap = [αs; αs[1] + 1.0]
    R_wrap = [Rs; Rs[1]]
    Z_wrap = [Zs; Zs[1]]

    Rθ = similar(theta_grid, Float64)
    Zθ = similar(theta_grid, Float64)
    for (j, θ) in enumerate(theta_grid)
        Rθ[j] = _linear_periodic(α_wrap, R_wrap, θ)
        Zθ[j] = _linear_periodic(α_wrap, Z_wrap, θ)
    end
    return Rθ, Zθ
end

# Linear interpolation on a strictly increasing knot vector `x` covering the query `q`.
function _linear_periodic(x::AbstractVector, y::AbstractVector, q::Real)
    q < x[1] && (q += 1.0)  # wrap into [x[1], x[1]+1)
    i = searchsortedlast(x, q)
    i = clamp(i, 1, length(x) - 1)
    t = (q - x[i]) / (x[i+1] - x[i])
    return y[i] + t * (y[i+1] - y[i])
end

"""
    _tokamaker_sq_nodes(F, mu0P, q, psi_norm) -> Matrix

Assemble the 4-column `sq_in` node matrix `[F, μ₀P, q, √ψ_norm]` shared by both routes.
`F` is signed (COCOS handled downstream); `mu0P` is clamped non-negative.
"""
function _tokamaker_sq_nodes(F::AbstractVector, mu0P::AbstractVector,
    q::AbstractVector, psi_norm::AbstractVector)
    n = length(F)
    sq = zeros(Float64, n, 4)
    @. sq[:, 1] = F
    @. sq[:, 2] = max(mu0P, 0.0)
    @. sq[:, 3] = q
    @. sq[:, 4] = sqrt(clamp(psi_norm, 0.0, 1.0))
    return sq
end

"""
    read_tokamaker(config::EquilibriumConfig, eq) -> InverseRunInput

Trace-surf (inverse) route. Traces closed flux-surface contours from the TokaMaker mesh on a
normalized-ψ grid, resamples each onto a common θ grid, and assembles an `InverseRunInput` for
`equilibrium_solver`. The vacuum-region exterior grid is built separately by
[`build_tokamaker_exterior`](@ref) and attached to the resulting `PlasmaEquilibrium`.

Co-registration: profiles (`get_profiles`/`get_q`) and geometry (`trace_surf`) are sampled at the
SAME user-convention ψ values so they refer to the same surfaces.
"""
function read_tokamaker(config::EquilibriumConfig, eq)
    tm = _tokamaker_module(eq)
    @info "Reading TokaMaker equilibrium via trace_surf (inverse route)"

    refs = tm.get_refs(eq)
    ro, zo = refs.o_point[1], refs.o_point[2]
    psi_bounds = collect(float.(refs.psi_bounds))
    psio = abs(psi_bounds[2] - psi_bounds[1]) * TOKAMAKER_PSI_TWOPI

    # Interior ψ grid in TokaMaker "user" convention (0 = axis, 1 = edge). A modest input grid;
    # equilibrium_solver resamples onto its own radial grid from config.grid_type. The upper edge
    # The upper edge is limited by where surfaces stop closing (graceful truncation below), not by a
    # fixed margin: the tiny 1e-10 cap only guards against placing a grid point exactly on the
    # separatrix, letting psihigh drive the reach and capturing the steep near-separatrix q-shear
    # that a truncated efit/inverse grid misses.
    npsi = config.mpsi > 0 ? config.mpsi + 1 : 129
    pad = max(config.psilow, 1e-3)
    psi_user = collect(range(pad, min(config.psihigh, 1.0 - 1e-10); length=npsi))

    # Poloidal grid shared by every surface (matches the inverse solver's expectation).
    mtheta = config.mtheta > 0 ? config.mtheta : 512
    theta_grid = collect(range(0.0, 1.0; length=mtheta + 1))

    # Trace each surface. A diverted separatrix may stop closing as ψ̂ → 1, so degrade gracefully:
    # keep the surfaces that traced and cap the interior grid at the last closed one.
    R_rows = Vector{Vector{Float64}}()
    Z_rows = Vector{Vector{Float64}}()
    for ψ in psi_user
        contour = tm.trace_surf(eq, ψ)
        (contour === nothing || size(contour, 1) < 4) && break
        Rθ, Zθ = _resample_contour_to_theta(@view(contour[:, 1]), @view(contour[:, 2]), ro, zo, theta_grid)
        push!(R_rows, Rθ)
        push!(Z_rows, Zθ)
    end
    ngood = length(R_rows)
    ngood >= 4 || error("read_tokamaker: trace_surf closed on only $ngood surface(s); check psilow/psihigh.")
    if ngood < npsi
        @warn "read_tokamaker: trace_surf stopped closing past ψ̂=$(round(psi_user[ngood]; sigdigits=6)); " *
              "capping interior grid there (requested psihigh=$(config.psihigh))."
    end
    psi_user = psi_user[1:ngood]
    npsi = ngood

    prof = tm.get_profiles(eq; psi=psi_user)
    qres = tm.get_q(eq; psi=psi_user)
    mu0 = 4π * 1e-7
    sq_fs = _tokamaker_sq_nodes(prof.F, prof.P .* mu0, qres.q, psi_user)
    sq_xs = copy(psi_user)
    sq_in = cubic_interp(sq_xs, Series(sq_fs); extrap=ExtendExtrap())

    R_nodes = zeros(Float64, npsi, mtheta + 1)
    Z_nodes = zeros(Float64, npsi, mtheta + 1)
    for i in 1:npsi
        R_nodes[i, :] .= R_rows[i]
        Z_nodes[i, :] .= Z_rows[i]
    end
    @views R_nodes[:, end] .= R_nodes[:, 1]
    @views Z_nodes[:, end] .= Z_nodes[:, 1]

    opts2d = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    rz_in_R = cubic_interp((sq_xs, theta_grid), R_nodes; opts2d...)
    rz_in_Z = cubic_interp((sq_xs, theta_grid), Z_nodes; opts2d...)

    ingest = InverseIngest(copy(sq_xs), copy(sq_fs), copy(sq_xs), copy(theta_grid),
        Matrix(R_nodes), Matrix(Z_nodes), ro, zo, psio)

    @info "TokaMaker trace_surf: axis (ro=$(@sprintf("%.3f", ro)), zo=$(@sprintf("%.3f", zo))), " *
          "psio=$(@sprintf("%.3e", psio)), diverted=$(refs.diverted)"
    return InverseRunInput(config, sq_in, sq_xs, theta_grid, rz_in_R, rz_in_Z, ro, zo, psio, ingest)
end

"""
    build_tokamaker_exterior(config, eq, theta_grid) -> TokamakerExterior

March normalized flux ψ̂ outward from 1 tracing closed vacuum flux surfaces until traces stop
closing (X-point for a diverted plasma, limiter/wall for a limited one). Returns the traced
exterior flux-coordinate grid. The real-space field-map fields of `TokamakerExterior` are left
as stubs (diverted-SOL TODO).
"""
function build_tokamaker_exterior(config::EquilibriumConfig, eq, theta_grid::AbstractVector)
    tm = _tokamaker_module(eq)
    refs = tm.get_refs(eq)
    ro, zo = refs.o_point[1], refs.o_point[2]
    psi_bounds = collect(float.(refs.psi_bounds))

    # ψ̂ > 1 levels to attempt; stop at the first trace that fails to close.
    psi_ext_max = 1.20
    n_try = 20
    psi_levels = range(1.0 + (psi_ext_max - 1.0) / n_try, psi_ext_max; length=n_try)

    ψs = Float64[]
    Rrows = Vector{Vector{Float64}}()
    Zrows = Vector{Vector{Float64}}()
    for ψ in psi_levels
        contour = tm.trace_surf(eq, ψ)
        (contour === nothing || size(contour, 1) < 4) && break  # surfaces no longer close
        Rθ, Zθ = _resample_contour_to_theta(@view(contour[:, 1]), @view(contour[:, 2]), ro, zo, theta_grid)
        push!(ψs, ψ)
        push!(Rrows, Rθ)
        push!(Zrows, Zθ)
    end

    n_ext = length(ψs)
    R_ext = zeros(Float64, n_ext, length(theta_grid))
    Z_ext = zeros(Float64, n_ext, length(theta_grid))
    for i in 1:n_ext
        R_ext[i, :] .= Rrows[i]
        Z_ext[i, :] .= Zrows[i]
    end
    if n_ext == 0
        @info "TokaMaker exterior: no closed vacuum surfaces past the LCFS (diverted); " *
              "real-space field map deferred (stub)."
    else
        @info "TokaMaker exterior: traced $n_ext closed vacuum surfaces out to ψ̂=$(round(ψs[end]; sigdigits=4))."
    end
    return TokamakerExterior(ψs, collect(float.(theta_grid)), R_ext, Z_ext, refs.diverted, psi_bounds)
end

"""
    read_tokamaker_direct(config::EquilibriumConfig, eq) -> DirectRunInput

Direct route. Samples ψ(R,Z) from the TokaMaker mesh onto a rectangular grid (efit-style) and
assembles a `DirectRunInput`. Mainly for the three-way GS-error comparison against the trace and
geqdsk routes; also the natural home for the `DirectIngest` rerun snapshot.
"""
function read_tokamaker_direct(config::EquilibriumConfig, eq)
    tm = _tokamaker_module(eq)
    @info "Reading TokaMaker equilibrium via rectangular ψ(R,Z) sample (direct route)"

    refs = tm.get_refs(eq)
    ro, zo = refs.o_point[1], refs.o_point[2]

    # Rectangular sampling grid from the traced LCFS bounding box (padded to include vacuum).
    lcfs = tm.trace_surf(eq, 1.0 - 1e-3)
    lcfs === nothing && error("read_tokamaker_direct: could not trace LCFS for grid bounds.")
    rmin_p, rmax_p = extrema(@view lcfs[:, 1])
    zmin_p, zmax_p = extrema(@view lcfs[:, 2])
    padR = 0.15 * (rmax_p - rmin_p)
    padZ = 0.15 * (zmax_p - zmin_p)
    rmin, rmax = rmin_p - padR, rmax_p + padR
    zmin, zmax = zmin_p - padZ, zmax_p + padZ

    nw = config.mpsi > 0 ? config.mpsi + 1 : 129
    nh = nw
    r_grid = collect(range(rmin, rmax; length=nw))
    z_grid = collect(range(zmin, zmax; length=nh))

    psi_fi = tm.get_field_eval(eq, "psi")
    # Axis/edge ψ from the field itself (convention-free): TokaMaker's `psi_bounds` ordering and
    # the `get_field_eval("psi")` sign do NOT agree, so evaluate ψ at the o-point (axis) and an
    # LCFS point (boundary) directly, mirroring read_efit's simag/sibry.
    psi_axis = psi_fi([ro, zo])[1] * TOKAMAKER_PSI_TWOPI
    psi_bdry = psi_fi([lcfs[1, 1], lcfs[1, 2]])[1] * TOKAMAKER_PSI_TWOPI
    psio_signed = psi_bdry - psi_axis
    psio = abs(psio_signed)

    psi_rz = zeros(Float64, nw, nh)
    for (i, r) in enumerate(r_grid), (j, z) in enumerate(z_grid)
        psi_rz[i, j] = psi_fi([r, z])[1] * TOKAMAKER_PSI_TWOPI
    end

    # Match read_efit: ψ measured from the boundary, positive toward the axis.
    psi_proc = psi_bdry .- psi_rz
    psio_signed < 0.0 && (psi_proc .*= -1.0)

    # Profiles on a uniform normalized-ψ grid (0 = axis, 1 = edge).
    psi_norm = collect(range(0.0, 1.0; length=nw))
    psi_user = clamp.(psi_norm, 1e-6, 1.0 - 1e-6)
    prof = tm.get_profiles(eq; psi=psi_user)
    qres = tm.get_q(eq; psi=psi_user)
    mu0 = 4π * 1e-7
    fpol_sign = Int(sign(prof.F[end] == 0 ? prof.F[1] : prof.F[end]))
    sq_fs = _tokamaker_sq_nodes(abs.(prof.F), prof.P .* mu0, qres.q, psi_norm)
    sq_xs = copy(psi_norm)
    sq_in = cubic_interp(sq_xs, Series(sq_fs); extrap=ExtendExtrap())

    psi_in = cubic_interp((r_grid, z_grid), psi_proc; extrap=ExtendExtrap())
    ingest = DirectIngest(sq_xs, sq_fs, r_grid, z_grid, psi_proc, rmin, rmax, zmin, zmax, psio, fpol_sign)

    @info "TokaMaker direct: grid $(nw)×$(nh) over R∈[$(round(rmin; sigdigits=3)),$(round(rmax; sigdigits=3))], " *
          "psio=$(@sprintf("%.3e", psio)), bt_sign=$fpol_sign"
    return DirectRunInput(config, sq_in, psi_in, r_grid, z_grid, rmin, rmax, zmin, zmax, psio, fpol_sign, ingest)
end

"""
    read_tokamaker_refine(config::EquilibriumConfig, source)

STUB. Re-solve a rough geqdsk/IMAS equilibrium with TokaMaker (higher resolution, clean o-point/
separatrix) and consume the refined solution. Deferred: the refine core needs mesh generation
from the LCFS and careful profile/boundary handling, and will share one path for both geqdsk and
IMAS front-ends (both carry F′, P′, and a boundary).
"""
function read_tokamaker_refine(config::EquilibriumConfig, source)
    error("eq_type=\"tokamaker_refine\" is not implemented yet. Refine (re-solve a rough " *
          "geqdsk/IMAS equilibrium with TokaMaker) is a planned feature; for now build/solve the " *
          "equilibrium in TokaMaker and use eq_type=\"tokamaker\" (trace) or \"tokamaker_direct\".")
end
