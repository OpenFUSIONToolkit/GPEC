# Separatrix.jl
#
# Safety factor between the outermost grid surface and the separatrix of a diverted equilibrium,
# taken from the X-point(s) of the equilibrium's own poloidal flux rather than fitted, and the
# separatrix's midplane and extremal points (`equilibrium_separatrix_find!`).
#
# Near a separatrix a field line stalls poloidally at each X-point it passes, so
#
#     q(ψ) = c + Σ_i A_i ln[1 / (ψ_Xi − ψ)],    A_i = F / (2π R_Xi sqrt(−det H_i)),
#
# with ψ_Xi the flux at saddle i of ψ(R, Z), H_i its Hessian there, and F = R B_φ. The constant is
# fixed by the traced q on the outermost grid surface, so nothing is fitted. One term describes a
# single null, two coincident terms a connected double null, and a second term at larger flux a
# disconnected one.
#
# The saddles are selected from the last closed flux surface itself: each minimum of the poloidal
# field along the outermost traced surface is followed to the critical point it leads to. On a
# single-null shape this usually also reaches the far secondary X-point; its term is small. A
# limited plasma with an X-point nearby is not told apart from a diverted one. Divertors with a
# second null beside the primary one (snowflake) or on the leg (x-point target) are not modelled;
# the shear check below is the backstop.

# Accepted disagreement between the law's dq/dψ and the traced profile's over the outer knots. A
# law missing one of two equal X-points is low by a factor of two, so the bound sits well inside it.
const SEPARATRIX_SHEAR_RATIO = 1.5
# Outer knots the shear check is taken over.
const SEPARATRIX_CHECK_KNOTS = 4
# Closest approach to the separatrix flux a surface is placed at, in normalized flux.
const SEPARATRIX_PSI_GAP = 1e-9

# Gradient and Hessian of the 2D flux at (r, z).
function _flux_derivatives(psi_in, r::Float64, z::Float64)
    pr = Float64(psi_in((r, z); deriv=DerivOp(1, 0)))
    pz = Float64(psi_in((r, z); deriv=DerivOp(0, 1)))
    prr = Float64(psi_in((r, z); deriv=DerivOp(2, 0)))
    prz = Float64(psi_in((r, z); deriv=DerivOp(1, 1)))
    pzz = Float64(psi_in((r, z); deriv=DerivOp(0, 2)))
    return (; pr, pz, prr, prz, pzz, det=prr * pzz - prz^2)
end

# Follow |∇ψ|² downhill from a seed with damped Newton steps on ∇ψ = 0. Returns the critical point,
# or `nothing` when the descent leaves the flux grid or stalls short of one.
function _descend_to_critical_point(psi_in, ingest::DirectIngest, r::Float64, z::Float64, grad_tol::Float64)
    for _ in 1:100
        d = _flux_derivatives(psi_in, r, z)
        g = d.pr^2 + d.pz^2
        sqrt(g) <= grad_tol && return (; r, z, det=d.det)
        d.det == 0.0 && return nothing
        dr = -(d.pzz * d.pr - d.prz * d.pz) / d.det
        dz = -(d.prr * d.pz - d.prz * d.pr) / d.det
        t = 1.0
        stepped = false
        while t >= 1e-6
            rn, zn = r + t * dr, z + t * dz
            if ingest.rmin <= rn <= ingest.rmax && ingest.zmin <= zn <= ingest.zmax
                dn = _flux_derivatives(psi_in, rn, zn)
                if dn.pr^2 + dn.pz^2 < g
                    r, z, stepped = rn, zn, true
                    break
                end
            end
            t /= 2
        end
        stepped || return nothing
    end
    return nothing
end

"""
    separatrix_q_law(equil; psi_max=equil.profiles.xs[end], shear_ratio=SEPARATRIX_SHEAR_RATIO)
        -> nothing | (; psi_x, A, r_x, z_x, psi_anchor, q_anchor, shear_ratio)

Safety factor beyond the equilibrium's ψ grid from the X-point(s) on its last closed flux surface:

    q(ψ) = q_anchor + Σ_i A_i ln[(psi_x_i − psi_anchor) / (psi_x_i − ψ)]

`psi_x` are the normalized fluxes of the selected saddles (ascending, so `psi_x[1]` is where q
diverges), `A` their logarithmic coefficients `F / (2π R_X sqrt(−det H))`, `r_x`/`z_x` their
positions, and the anchor is the traced q at `psi_max`. `psi_x[1]` need not equal 1: a
reconstruction may place its boundary slightly inside the X-point flux.

Returns `nothing` when the law does not apply or cannot be trusted:

  - the equilibrium is not a direct (2D flux) one — an inverse equilibrium carries its own q;
  - no minimum of the poloidal field on the outermost surface leads to a saddle outside the grid
    (a plasma with no X-point nearby; a limited plasma that has one still gets a law);
  - q is not rising at the anchor;
  - the law's dq/dψ and the traced profile's differ by more than `shear_ratio` over the outer
    knots (returned as `shear_ratio`, law over traced).
"""
function separatrix_q_law(equil::PlasmaEquilibrium;
    psi_max::Real=Float64(equil.profiles.xs[end]),
    shear_ratio::Real=SEPARATRIX_SHEAR_RATIO)
    ingest = equil.ingest
    ingest isa DirectIngest || return nothing
    psi_in = cubic_interp((ingest.psi_xs, ingest.psi_ys), ingest.psi_rz; extrap=ExtendExtrap())
    xs = equil.profiles.xs
    psi_anchor = min(Float64(psi_max), Float64(xs[end]))
    q_anchor = Float64(equil.profiles.q_spline(psi_anchor))
    q_slope = Float64(equil.profiles.q_deriv(psi_anchor))
    (isfinite(q_anchor) && q_anchor * q_slope > 0) || return nothing

    # Poloidal field along the outermost traced surface.
    psi_surface = Float64(equil.rzphi_xs[end])
    thetas = equil.rzphi_ys[1:(end-1)]
    hint2d = (Ref(1), Ref(1))
    rs, zs, bp = similar(thetas), similar(thetas), similar(thetas)
    for (i, theta) in enumerate(thetas)
        r_minor = sqrt(equil.rzphi_rsquared((psi_surface, theta); hint=hint2d))
        theta_cyl = 2π * (theta + equil.rzphi_offset((psi_surface, theta); hint=hint2d))
        rs[i] = equil.ro + r_minor * cos(theta_cyl)
        zs[i] = equil.zo + r_minor * sin(theta_cyl)
        d = _flux_derivatives(psi_in, rs[i], zs[i])
        bp[i] = hypot(d.pr, d.pz) / rs[i]
    end

    # Each minimum of the poloidal field is where the surface passes a null; follow it there.
    n = length(bp)
    grad_tol = 1e-10 * maximum(bp .* rs)
    merge_dist = 1e-6 * (ingest.rmax - ingest.rmin)
    f_edge = abs(Float64(ingest.sq_fs[end, 1]))
    saddles = NamedTuple{(:psi, :A, :r, :z),NTuple{4,Float64}}[]
    for i in 1:n
        (bp[i] < bp[mod1(i - 1, n)] && bp[i] <= bp[mod1(i + 1, n)]) || continue
        cp = _descend_to_critical_point(psi_in, ingest, rs[i], zs[i], grad_tol)
        (cp === nothing || cp.det >= 0) && continue
        psi_x = 1.0 - Float64(psi_in((cp.r, cp.z))) / ingest.psio
        psi_x > psi_anchor || continue
        any(s -> hypot(s.r - cp.r, s.z - cp.z) < merge_dist, saddles) && continue
        push!(saddles, (psi=psi_x, A=f_edge / (2π * cp.r * sqrt(-cp.det)), r=cp.r, z=cp.z))
    end
    isempty(saddles) && return nothing
    sort!(saddles; by=s -> s.psi)

    law = (psi_x=[s.psi for s in saddles], A=[s.A for s in saddles], r_x=[s.r for s in saddles],
        z_x=[s.z for s in saddles], psi_anchor=psi_anchor, q_anchor=q_anchor)

    # The law must reproduce the traced profile's own shear where the two overlap.
    knots = filter(i -> xs[i] <= psi_anchor, eachindex(xs))
    isempty(knots) && return nothing
    band = knots[max(1, end - SEPARATRIX_CHECK_KNOTS + 1):end]
    hint = Ref(1)
    ratio = median([separatrix_q_dqdpsi(law, xs[i]) / Float64(equil.profiles.q_deriv(xs[i]; hint=hint)) for i in band])
    (isfinite(ratio) && inv(shear_ratio) <= ratio <= shear_ratio) || return nothing
    return merge(law, (; shear_ratio=ratio))
end

"""
dq/dψ on the separatrix law.
"""
separatrix_q_dqdpsi(law, psi::Real) = sign(law.q_anchor) * sum(A / (px - psi) for (A, px) in zip(law.A, law.psi_x))

"""
q on the separatrix law.
"""
separatrix_q(law, psi::Real) =
    law.q_anchor + sign(law.q_anchor) * sum(A * log((px - law.psi_anchor) / (px - psi)) for (A, px) in zip(law.A, law.psi_x))

"""
ψ at which the separatrix law reaches `q_target`. Returns `nothing` when the target is not beyond
the anchor, or lies closer to the separatrix flux than `psi_gap`.
"""
function separatrix_q_psi(law, q_target::Real; psi_gap::Real=SEPARATRIX_PSI_GAP)
    s = sign(law.q_anchor)
    s * (q_target - law.q_anchor) > 0 || return nothing
    psi_sep = law.psi_x[1]
    psi_gap < psi_sep - law.psi_anchor || return nothing
    # Solve in the logarithmic distance to the separatrix, where q is close to linear.
    residual(u) = s * (separatrix_q(law, psi_sep - exp(u)) - q_target)
    u_near, u_far = log(psi_gap), log(psi_sep - law.psi_anchor)
    residual(u_near) > 0 || return nothing
    return psi_sep - exp(find_zero(residual, (u_near, u_far), Roots.Brent()))
end

"""
    equilibrium_separatrix_find!(pe::PlasmaEquilibrium)

Find the separatrix midplane radii `rsep` and the top/bottom extremal points (`rext`, `zext`, with
`zsep = zext`), store them in `pe.params`, and return them. Port of Fortran `equil_out_sep_find`.
"""
function equilibrium_separatrix_find!(pe::PlasmaEquilibrium)
    mpsi = length(pe.rzphi_xs) - 1
    edge_idx = mpsi + 1  # Edge flux surface index
    psi_edge = pe.rzphi_xs[edge_idx]
    rsep = zeros(2)

    # Outboard and inboard midplane R via bracketed Brent on θ + η(θ) - η₀ = 0.
    # iside=1 → outboard (η₀=0.0, θ near 0),  iside=2 → inboard (η₀=0.5, θ near 0.5).
    for iside in 1:2
        eta0 = (iside == 1) ? 0.0 : 0.5
        hint2d = (Ref(1), Ref(1))
        theta_lo, theta_hi = (iside == 1) ? (-0.25, 0.25) : (0.25, 0.75)
        side_label = (iside == 1) ? "outboard" : "inboard"
        theta = try
            find_zero(
                theta -> theta + pe.rzphi_offset((psi_edge, theta); hint=hint2d) - eta0,
                (theta_lo, theta_hi), Roots.Brent();
                atol=1e-12, rtol=1e-12)
        catch e
            error("Separatrix $side_label midplane root not found in bracket " *
                  "[$(theta_lo), $(theta_hi)]: $(e.msg)")
        end
        r2 = pe.rzphi_rsquared((psi_edge, theta))
        offset = pe.rzphi_offset((psi_edge, theta))
        rsep[iside] = pe.ro + sqrt(r2) * cos(2π * (theta + offset))
    end

    # Top and bottom separatrix Z extrema via bracketed Brent on ∂z/∂θ = 0.
    # iside=1 → top (θ ∈ [0.0, 0.5]),  iside=2 → bottom (θ ∈ [0.5, 1.0]), as in Fortran.
    zsep = zeros(2)
    rext = zeros(2)
    zext = zeros(2)

    for iside in 1:2
        hint2d = (Ref(1), Ref(1))

        # Cache variables populated by z_deriv, read after convergence
        rfac = Ref(0.0)
        cos_phase = Ref(0.0)
        z_val = Ref(0.0)

        # ∂z/∂θ where z(θ) = zo + √r²(θ) · sin(2π(θ + η(θ)))
        function z_deriv(theta_inner)
            r2 = pe.rzphi_rsquared((psi_edge, theta_inner); hint=hint2d)
            r2y = pe.rzphi_rsquared((psi_edge, theta_inner); deriv=DerivOp(0, 1), hint=hint2d)
            η = pe.rzphi_offset((psi_edge, theta_inner); hint=hint2d)
            η1 = pe.rzphi_offset((psi_edge, theta_inner); deriv=DerivOp(0, 1), hint=hint2d)
            rfac_local = sqrt(max(0.0, r2))
            rfac1 = (rfac_local > 0) ? r2y / (2 * rfac_local) : 0.0
            phase1 = 2π * (1 + η1)
            sin_phase = sin(2π * (theta_inner + η))
            cos_phase_local = cos(2π * (theta_inner + η))

            rfac[] = rfac_local
            cos_phase[] = cos_phase_local
            z_val[] = pe.zo + rfac_local * sin_phase

            return rfac_local * phase1 * cos_phase_local + rfac1 * sin_phase
        end

        theta_lo, theta_hi = (iside == 1) ? (0.0, 0.5) : (0.5, 1.0)
        side_label = (iside == 1) ? "top" : "bottom"
        theta = try
            find_zero(z_deriv, (theta_lo, theta_hi), Roots.Brent();
                atol=1e-12, rtol=1e-12)
        catch e
            error("Separatrix $side_label Z-extremum root not found in bracket " *
                  "[$(theta_lo), $(theta_hi)]: $(e.msg)")
        end

        rext[iside] = pe.ro + rfac[] * cos_phase[]
        zsep[iside] = zext[iside] = z_val[]
    end

    pe.params.rsep = rsep
    pe.params.zsep = zsep
    pe.params.rext = rext
    pe.params.zext = zext
    return (rsep, zsep, rext, zext)
end
