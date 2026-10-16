"""
Cerfon-Freidberg analytic Solov'ev equilibria with a magnetic null on the plasma boundary.

The Solov'ev choice of constant `p'` and `FF'` makes the Grad-Shafranov equation linear.
In normalized coordinates `x = R/R₀`, `y = Z/R₀` it reduces to

    Δ*ψ̂ ≡ ψ̂ₓₓ − ψ̂ₓ/x + ψ̂_yy = (1 − A) x² + A

whose general solution is a particular solution plus a twelve-term homogeneous basis,

    ψ̂(x, y) = ψ_p(x) + Σᵢ cᵢ ψᵢ(x, y),    i = 1…12

with `ψ₁…ψ₇` up-down symmetric and `ψ₈…ψ₁₂` antisymmetric. The twelve coefficients are set
by point-wise conditions on the `ψ̂ = 0` surface, which becomes the plasma boundary. Placing
a null (∇ψ̂ = 0) on that surface makes `q` diverge as ψ_N → 1, so the edge is genuinely
unbounded and no choice of `psihigh` converges — the property this equilibrium exists to
exercise.

Reference: A. J. Cerfon and J. P. Freidberg, "One size fits all analytic solutions to the
Grad-Shafranov equation", Phys. Plasmas **17**, 032502 (2010).
"""

"""
Particular solution of Δ*ψ̂ = (1 − A)x² + A.
"""
cerfon_psi_p(x::Float64, A::Float64) = x^4 / 8 + A * (x^2 * log(x) / 2 - x^4 / 8)
cerfon_dpsi_p_dx(x::Float64, A::Float64) = x^3 / 2 + A * (x * log(x) + x / 2 - x^3 / 2)
cerfon_d2psi_p_dx2(x::Float64, A::Float64) = 3x^2 / 2 + A * (log(x) + 1.5 - 3x^2 / 2)

"""
    cerfon_basis(x, y)

The twelve homogeneous solutions of Δ*ψᵢ = 0. Entries 1–7 are up-down symmetric and 8–12
antisymmetric; a double null uses only the symmetric ones. Verified in
`test/runtests_cerfon.jl` by checking Δ*ψᵢ numerically.
"""
function cerfon_basis(x::Float64, y::Float64)
    L = log(x)
    x2, x4, x6 = x^2, x^4, x^6
    y2, y3, y4, y5, y6 = y^2, y^3, y^4, y^5, y^6
    return (1.0,
        x2,
        y2 - x2 * L,
        x4 - 4x2 * y2,
        2y4 - 9x2 * y2 + 3x4 * L - 12x2 * y2 * L,
        x6 - 12x4 * y2 + 8x2 * y4,
        8y6 - 140x2 * y4 + 75x4 * y2 - 15x6 * L + 180x4 * y2 * L - 120x2 * y4 * L,
        y,
        x2 * y,
        y3 - 3x2 * y * L,
        3x4 * y - 4x2 * y3,
        8y5 - 45x4 * y - 80x2 * y3 * L + 60x4 * y * L)
end

"""
∂ψᵢ/∂x for the twelve homogeneous solutions.
"""
function cerfon_basis_dx(x::Float64, y::Float64)
    L = log(x)
    x2, x3, x4, x5 = x^2, x^3, x^4, x^5
    y2, y3, y4 = y^2, y^3, y^4
    return (0.0,
        2x,
        -2x * L - x,
        4x3 - 8x * y2,
        -30x * y2 + 12x3 * L + 3x3 - 24x * y2 * L,
        6x5 - 48x3 * y2 + 16x * y4,
        -400x * y4 + 480x3 * y2 - 90x5 * L - 15x5 + 720x3 * y2 * L - 240x * y4 * L,
        0.0,
        2x * y,
        -6x * y * L - 3x * y,
        12x3 * y - 8x * y3,
        -120x3 * y - 160x * y3 * L - 80x * y3 + 240x3 * y * L)
end

"""
∂ψᵢ/∂y for the twelve homogeneous solutions.
"""
function cerfon_basis_dy(x::Float64, y::Float64)
    L = log(x)
    x2, x4 = x^2, x^4
    y2, y3, y4, y5 = y^2, y^3, y^4, y^5
    return (0.0,
        0.0,
        2y,
        -8x2 * y,
        8y3 - 18x2 * y - 24x2 * y * L,
        -24x4 * y + 32x2 * y3,
        48y5 - 560x2 * y3 + 150x4 * y + 360x4 * y * L - 480x2 * y3 * L,
        1.0,
        x2,
        3y2 - 3x2 * L,
        3x4 - 12x2 * y2,
        40y4 - 45x4 - 240x2 * y2 * L + 60x4 * L)
end

"""
∂²ψᵢ/∂x² for the twelve homogeneous solutions.
"""
function cerfon_basis_dxx(x::Float64, y::Float64)
    L = log(x)
    x2, x4 = x^2, x^4
    y2, y3, y4 = y^2, y^3, y^4
    return (0.0,
        2.0,
        -2L - 3,
        12x2 - 8y2,
        -54y2 + 36x2 * L + 21x2 - 24y2 * L,
        30x4 - 144x2 * y2 + 16y4,
        -640y4 + 2160x2 * y2 - 450x4 * L - 165x4 + 2160x2 * y2 * L - 240y4 * L,
        0.0,
        2y,
        -6y * L - 9y,
        36x2 * y - 8y3,
        -120x2 * y - 160y3 * L - 240y3 + 720x2 * y * L)
end

"""
∂²ψᵢ/∂y² for the twelve homogeneous solutions.
"""
function cerfon_basis_dyy(x::Float64, y::Float64)
    L = log(x)
    x2, x4 = x^2, x^4
    y2, y3, y4 = y^2, y^3, y^4
    return (0.0,
        0.0,
        2.0,
        -8x2,
        24y2 - 18x2 - 24x2 * L,
        -24x4 + 96x2 * y2,
        240y4 - 1680x2 * y2 + 150x4 + 360x4 * L - 1440x2 * y2 * L,
        0.0,
        0.0,
        6y,
        -24x2 * y,
        160y3 - 480x2 * y * L)
end

"""
∂²ψᵢ/∂x∂y for the twelve homogeneous solutions; needed for the Hessian at the axis.
"""
function cerfon_basis_dxy(x::Float64, y::Float64)
    L = log(x)
    x2, x3 = x^2, x^3
    y2, y3 = y^2, y^3
    return (0.0,
        0.0,
        0.0,
        -16x * y,
        -60x * y - 48x * y * L,
        -96x3 * y + 64x * y3,
        -1600x * y3 + 960x3 * y + 1440x3 * y * L - 960x * y3 * L,
        0.0,
        2x,
        -6x * L - 3x,
        12x3 - 24x * y2,
        -120x3 - 480x * y2 * L - 240x * y2 + 240x3 * L)
end

"""
Evaluate ψ̂ = ψ_p + Σ cᵢψᵢ at normalized coordinates.
"""
function cerfon_psihat(x, y, c, A)
    b = cerfon_basis(x, y)
    return cerfon_psi_p(x, A) + sum(c[i] * b[i] for i in 1:12)
end

"""
∇ψ̂ = (∂ψ̂/∂x, ∂ψ̂/∂y) at normalized coordinates.
"""
function cerfon_grad(x, y, c, A)
    bx, by = cerfon_basis_dx(x, y), cerfon_basis_dy(x, y)
    return (cerfon_dpsi_p_dx(x, A) + sum(c[i] * bx[i] for i in 1:12),
        sum(c[i] * by[i] for i in 1:12))
end

"""
Hessian (ψ̂ₓₓ, ψ̂_yy, ψ̂ₓ_y) of ψ̂ at normalized coordinates.
"""
function cerfon_hessian(x, y, c, A)
    bxx, byy, bxy = cerfon_basis_dxx(x, y), cerfon_basis_dyy(x, y), cerfon_basis_dxy(x, y)
    return (cerfon_d2psi_p_dx2(x, A) + sum(c[i] * bxx[i] for i in 1:12),
        sum(c[i] * byy[i] for i in 1:12),
        sum(c[i] * bxy[i] for i in 1:12))
end

"""
    cerfon_shape_points(cfg)

Boundary reference points in normalized coordinates: the outer and inner equatorial points
and the high point. The null position comes from `cerfon_null_point`.
"""
function cerfon_shape_points(cfg::CerfonConfig)
    ε, κ, δ = cfg.epsilon, cfg.kappa, cfg.delta
    return (xout=1 + ε, xin=1 - ε, xhigh=1 - δ * ε, yhigh=κ * ε)
end

"""
    cerfon_null_point(cfg)

Position `(x, y) = (1 − xsep·δ·ε, ±xsep·κ·ε)` of the magnetic null placed on the boundary,
`xsep` beyond the high point: below the midplane for `"lsn"`, above it for `"dn"`, whose lower
null follows from up-down symmetry.
"""
function cerfon_null_point(cfg::CerfonConfig)
    ε, κ, δ, s = cfg.epsilon, cfg.kappa, cfg.delta, cfg.xsep
    return (1 - s * δ * ε, (cfg.null == "dn" ? 1 : -1) * s * κ * ε)
end

"""
Curvature coefficients N₁, N₂, N₃ that impose elongation and triangularity.
"""
function cerfon_curvature_coeffs(cfg::CerfonConfig)
    ε, κ, δ = cfg.epsilon, cfg.kappa, cfg.delta
    α = asin(δ)
    return (-(1 + α)^2 / (ε * κ^2), (1 - α)^2 / (ε * κ^2), -κ / (ε * cos(α)^2))
end

const CERFON_BC_COND_MAX = 1e12  # condition number of the boundary-condition matrix above which the solve is refused

"""
    cerfon_solve_coeffs(cfg)

Assemble and solve the 12×12 boundary-condition system for the requested `null` type,
returning `(coefficients, null_points)`.

Every topology imposes ψ̂ = 0 at the inner and outer equatorial points and ψ̂ = ∇ψ̂ = 0 at the
primary null. The remaining conditions differ:

  - `"lsn"` — lower single null. Adds the high point (ψ̂ = 0 and ∂ψ̂/∂x = 0), all three
    N₁/N₂/N₃ curvature conditions, and ∂ψ̂/∂y = 0 at both equatorial points so they are the
    widest extent of the boundary. Shape is imposed exactly.
  - `"dn"` — double null. Up-down symmetric, so the antisymmetric coefficients c₈…c₁₂ are
    driven to zero and the shape follows from the equatorial points, the upper null and
    N₁/N₂. ∂ψ̂/∂y vanishes identically on the midplane for the symmetric basis, so the
    equatorial tangent conditions are dropped — imposing them here would make the system
    singular. The boundary reaches the nulls at ±`xsep`·κ·ε, so the realized elongation is
    `xsep`·κ rather than κ.

No snowflake is offered: an exact second-order null detaches the closed surfaces from it, and a
snowflake-minus is not star-shaped about the axis, which the direct flux-surface tracer requires.
"""
function cerfon_solve_coeffs(cfg::CerfonConfig)
    cfg.null in ("lsn", "dn") || error("Unknown Cerfon null type \"$(cfg.null)\"; expected \"lsn\" or \"dn\".")
    p = cerfon_shape_points(cfg)
    N1, N2, N3 = cerfon_curvature_coeffs(cfg)
    xn, yn = cerfon_null_point(cfg)
    A = cfg.A

    M = zeros(12, 12)
    b = zeros(12)
    row = 0
    set!(vals, rhs) = (row += 1; M[row, :] .= vals; b[row] = rhs)

    # Shared by every topology: the boundary passes through the equatorial points, and the
    # primary null lies on it and is a true magnetic null.
    set!(cerfon_basis(p.xout, 0.0), -cerfon_psi_p(p.xout, A))
    set!(cerfon_basis(p.xin, 0.0), -cerfon_psi_p(p.xin, A))
    set!(cerfon_basis(xn, yn), -cerfon_psi_p(xn, A))
    set!(cerfon_basis_dx(xn, yn), -cerfon_dpsi_p_dx(xn, A))
    set!(cerfon_basis_dy(xn, yn), 0.0)

    if cfg.null == "dn"
        set!(cerfon_basis_dyy(p.xout, 0.0) .+ N1 .* cerfon_basis_dx(p.xout, 0.0), -N1 * cerfon_dpsi_p_dx(p.xout, A))
        set!(cerfon_basis_dyy(p.xin, 0.0) .+ N2 .* cerfon_basis_dx(p.xin, 0.0), -N2 * cerfon_dpsi_p_dx(p.xin, A))
        for k in 8:12
            set!(ntuple(j -> j == k ? 1.0 : 0.0, 12), 0.0)
        end
    else
        # The boundary passes through the high point with a horizontal tangent, and the
        # equatorial points are its widest extent.
        set!(cerfon_basis(p.xhigh, p.yhigh), -cerfon_psi_p(p.xhigh, A))
        set!(cerfon_basis_dx(p.xhigh, p.yhigh), -cerfon_dpsi_p_dx(p.xhigh, A))
        set!(cerfon_basis_dy(p.xout, 0.0), 0.0)
        set!(cerfon_basis_dy(p.xin, 0.0), 0.0)
        set!(cerfon_basis_dxx(p.xhigh, p.yhigh) .+ N3 .* cerfon_basis_dy(p.xhigh, p.yhigh), -cerfon_d2psi_p_dx2(p.xhigh, A))
        set!(cerfon_basis_dyy(p.xout, 0.0) .+ N1 .* cerfon_basis_dx(p.xout, 0.0), -N1 * cerfon_dpsi_p_dx(p.xout, A))
        set!(cerfon_basis_dyy(p.xin, 0.0) .+ N2 .* cerfon_basis_dx(p.xin, 0.0), -N2 * cerfon_dpsi_p_dx(p.xin, A))
    end

    @assert row == 12 "Cerfon boundary-condition system has $row rows, expected 12"
    κM = cond(M)
    κM > CERFON_BC_COND_MAX &&
        error("Cerfon boundary-condition matrix is ill-conditioned (cond = $(@sprintf("%.2e", κM)) > $(@sprintf("%.0e", CERFON_BC_COND_MAX))); check the requested shape.")
    return M \ b, (xn, yn)
end

const CERFON_AXIS_SCAN_NX = 81            # x points in the coarse ψ̂-minimum scan
const CERFON_AXIS_SCAN_NY = 121           # y points in the coarse ψ̂-minimum scan
const CERFON_AXIS_NEWTON_MAXITER = 100    # Newton iteration cap for the axis polish
const CERFON_AXIS_NEWTON_TOL = 1e-13      # Newton step size, in normalized units, below which the axis is converged
const CERFON_AXIS_DET_FLOOR = 1e-20       # |det Hessian| below which the Newton step is treated as singular

"""
    cerfon_find_axis(cfg, c, A)

Locate the magnetic axis: a coarse scan for the minimum of ψ̂ over the shape box, then Newton
polishing on ∇ψ̂ = 0. Returns `(x_axis, y_axis)`.

The scan is not optional. Newton started from the geometric centre converges for the single
and double null, but strongly shaped cases push the axis well off the midplane and a centred
start can walk out of the domain (ψ̂ contains `ln x`, so `x ≤ 0` is not evaluable).
"""
function cerfon_find_axis(cfg::CerfonConfig, c, A)
    p = cerfon_shape_points(cfg)
    ynull = abs(cerfon_null_point(cfg)[2])
    ylo = min(-ynull, -p.yhigh)
    yhi = max(ynull, p.yhigh)

    x, y, best = 1.0, 0.0, Inf
    for xs in range(p.xin, p.xout; length=CERFON_AXIS_SCAN_NX), ys in range(ylo, yhi; length=CERFON_AXIS_SCAN_NY)
        v = cerfon_psihat(xs, ys, c, A)
        if v < best
            best, x, y = v, xs, ys
        end
    end

    converged = false
    for _ in 1:CERFON_AXIS_NEWTON_MAXITER
        gx, gy = cerfon_grad(x, y, c, A)
        hxx, hyy, hxy = cerfon_hessian(x, y, c, A)
        det = hxx * hyy - hxy^2
        abs(det) < CERFON_AXIS_DET_FLOOR && error("Singular Hessian while locating the Cerfon magnetic axis at ($x, $y).")
        dx = -(hyy * gx - hxy * gy) / det
        dy = -(hxx * gy - hxy * gx) / det
        # Cap the step at one scan cell so a bad Hessian cannot throw the iterate out of the domain.
        scale = min(1.0, (p.xout - p.xin) / (CERFON_AXIS_SCAN_NX - 1) / max(abs(dx), abs(dy), eps()))
        x += scale * dx
        y += scale * dy
        if abs(scale * dx) < CERFON_AXIS_NEWTON_TOL && abs(scale * dy) < CERFON_AXIS_NEWTON_TOL
            converged = true
            break
        end
    end
    converged || error("Failed to locate the Cerfon magnetic axis after $CERFON_AXIS_NEWTON_MAXITER Newton iterations.")
    return x, y
end

"""
    cerfon_flux_scale(cfg, c, A, xa, ya)

Flux scale `P > 0` such that ψ = −P·ψ̂ has the requested on-axis safety factor.

For elliptical surfaces about the axis, `q₀ = F₀/(R_a √(αβ))` where α, β are the curvatures
of ψ there, so `q₀ = F R₀ / (P x_a √(det Ĥ))`. Combining with the Solov'ev toroidal field
`F² = R₀²B₀² − 2 C A ψ/R₀²` (here `C = −P`) gives a closed form for `P` — no iteration.
"""
function cerfon_flux_scale(cfg::CerfonConfig, c, A, xa, ya)
    hxx, hyy, hxy = cerfon_hessian(xa, ya, c, A)
    detH = hxx * hyy - hxy^2
    detH <= 0 && error("Cerfon magnetic axis is not an extremum of ψ (det Hessian = $detH); check the requested shape.")
    psihat_a = cerfon_psihat(xa, ya, c, A)
    psihat_a >= 0 && error("Expected ψ̂ < 0 on axis for the Cerfon solution, got $psihat_a.")

    denom = cfg.q0^2 * xa^2 * detH - 2 * A * abs(psihat_a)
    denom <= 0 && error("No positive flux scale satisfies q0 = $(cfg.q0) at A = $A; lower q0 or make A more negative.")
    return cfg.r0^2 * cfg.b0 / sqrt(denom)
end

"""
    cerfon_run(equil_inputs, cerfon_inputs)

Build a `DirectRunInput` for a Cerfon-Freidberg analytic equilibrium. The `ψ̂ = 0` surface is
the plasma boundary and carries a magnetic null, so `q` diverges as ψ_N → 1 and `psihigh`
always truncates a genuinely unbounded edge.

The flux map is tabulated on an `mr × mz` grid spanning the boundary plus `box_margin`, and
the profile spline carries the Solov'ev `F(ψ)` and `μ₀p(ψ)`; `q` is left to the downstream
field-line integration, as for the other direct-path equilibria.
"""
function cerfon_run(equil_inputs::EquilibriumConfig, cerfon_inputs::CerfonConfig)
    cfg = cerfon_inputs
    c, (xn, yn) = cerfon_solve_coeffs(cfg)
    A = cfg.A
    xa, ya = cerfon_find_axis(cfg, c, A)
    P = cerfon_flux_scale(cfg, c, A, xa, ya)

    # ψ = −P ψ̂ is positive inside, zero on the boundary, and maximal on axis.
    psi_of = (x, y) -> -P * cerfon_psihat(x, y, c, A)
    psio = psi_of(xa, ya)

    # Solov'ev profiles: μ₀p and F² are both linear in ψ, with ψ = psio(1 − ψ_N). Matching
    # Δ*ψ = −μ₀R²p′ − FF′ against Δ*ψ = (C/R₀²)[(1−A)x² + A] gives μ₀p′ = −C(1−A)/R₀⁴ and
    # FF′ = −CA/R₀², with C = −P the scale in ψ = C·ψ̂.
    psi_norm = [(ia / (cfg.ma + 1))^2 for ia in 1:(cfg.ma+1)]
    sqfs = zeros(cfg.ma + 1, 4)
    for (i, pn) in enumerate(psi_norm)
        psi = psio * (1 - pn)
        # F² = F_edge² + 2APψ/R₀² (FF′ = −CA/R₀², C = −P); A < 0 is diamagnetic.
        f2 = (cfg.r0 * cfg.b0)^2 + 2 * A * P * psi / cfg.r0^2
        f2 <= 0 && error("Cerfon toroidal field vanishes at ψ_N = $pn (F² = $f2); reduce q0 or |A|.")
        sqfs[i, 1] = sqrt(f2)
        sqfs[i, 2] = P * (1 - A) * psi / cfg.r0^4
        sqfs[i, 4] = sqrt(pn)
    end
    sq_in = cubic_interp(psi_norm, Series(sqfs); extrap=ExtendExtrap())

    # Box hugs the boundary: the ψ = 0 level set also contains the divertor legs, and a
    # generous box would let the midplane separatrix search latch onto one of them.
    p = cerfon_shape_points(cfg)
    m = cfg.box_margin * cfg.epsilon
    rmin = cfg.r0 * (p.xin - m)
    rmax = cfg.r0 * (p.xout + m)
    ztop = cfg.r0 * (cfg.null == "dn" ? abs(yn) + m : p.yhigh + m)
    zbot = cfg.r0 * (-abs(yn) - m)

    r = [rmin + i * (rmax - rmin) / cfg.mr for i in 0:cfg.mr]
    z = [zbot + j * (ztop - zbot) / cfg.mz for j in 0:cfg.mz]
    psifs = [psi_of(ri / cfg.r0, zj / cfg.r0) for ri in r, zj in z]
    psi_in = cubic_interp((r, z), psifs; extrap=ExtendExtrap())

    @info "Generating Cerfon-Freidberg equilibrium: null=$(cfg.null), ε=$(@sprintf("%.3f", cfg.epsilon)), " *
          "κ=$(@sprintf("%.3f", cfg.kappa)), δ=$(@sprintf("%.3f", cfg.delta)), A=$(@sprintf("%.4f", A)), " *
          "q0=$(@sprintf("%.3f", cfg.q0)), axis at R=$(@sprintf("%.4f", xa * cfg.r0)), Z=$(@sprintf("%+.4f", ya * cfg.r0)), " *
          "null at R=$(@sprintf("%.4f", xn * cfg.r0)), Z=$(@sprintf("%+.4f", yn * cfg.r0))"

    # 1 is bt_sign=+1: F = +sqrt(F²) by construction. Analytic: ingest=nothing — replay
    # regenerates from the [CERFON_INPUT] TOML section.
    return DirectRunInput(equil_inputs, sq_in, psi_in, r, z, rmin, rmax, zbot, ztop, psio, 1, 1, nothing)
end
