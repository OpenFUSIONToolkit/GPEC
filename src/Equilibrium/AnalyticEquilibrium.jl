
"""
    lar_init_conditions(rmin, sigma_type, params)

Initializes the starting radius and state vector for solving the LAR ODE system.
Also evaluates the initial derivative using the analytic model.

## Arguments:

  - `rmin`: Normalized starting radius (as a fraction of `lar_a`).
  - `lar_input`: A `LargeAspectRatioConfig` object containing equilibrium parameters.

## Returns:

  - `r`: Physical radius corresponding to `rmin * lar_a`.
  - `y`: Initial state vector of length 5.
"""
function lar_init_conditions(rmin::Float64, lar_input::LargeAspectRatioConfig)
    lar_a = lar_input.lar_a
    lar_r0 = lar_input.lar_r0
    q0 = lar_input.q0

    # Ensure rmin is not too small to avoid numerical issues in LAR equations
    # Use a physically meaningful minimum (0.001% of minor radius)
    if rmin < 1e-6
        @warn "Setting rmin to at least 1e-6 to avoid singularities in LAR equations."
        rmin_safe = 1e-6
    else
        rmin_safe = rmin
    end

    r = rmin_safe * lar_a
    y = zeros(5)

    y[1] = r^2 / (lar_r0 * q0)
    y[2] = 1.0
    y[3] = y[1] * lar_r0 / 2

    dy = zeros(5)
    lar_der(dy, r, y, lar_input)

    y[5] = dy[5] * r / 4

    q = r^2 * y[2] / (lar_r0 * y[1])
    y[4] = (q / r)^2 * y[5] / 2

    return r, y
end

"""
    lar_der(dy, r, y, lar_input)

Evaluates the spatial derivatives of the LAR (Large Aspect Ratio) equilibrium ODE system
at a given radius `r`, using the current state vector `y` and equilibrium parameters.

## Arguments:
- `dy`: A preallocated vector where the computed derivatives will be stored (in-place).
- `r`: The radial position at which the derivative is evaluated.
- `y`: The current state vector.
- `lar_input`: A `LargeAspectRatioConfig` object containing equilibrium shaping parameters.

## Returns:
- 0. The result is stored in-place in `dy`.
"""

function lar_der(dy::Vector{Float64}, r::Float64, y::Vector{Float64}, lar_input::LargeAspectRatioConfig)
    lar_a = lar_input.lar_a
    lar_r0 = lar_input.lar_r0

    q0 = lar_input.q0

    beta0 = lar_input.beta0
    p00 = beta0 / 2.0
    p_pres = lar_input.p_pres
    p_sig = lar_input.p_sig

    sigma_type = lar_input.sigma_type
    x = r / lar_a
    xfac = 1 - x^2
    pp = -2 * p_pres * p00 * x * xfac^(p_pres - 1)  # p'
    sigma0 = 2.0 / (q0 * lar_r0)

    sigma = if sigma_type == "wesson"
        sigma0 * xfac^p_sig
    else
        sigma0 / (1 + x^(2 * p_sig))^(1 + 1 / p_sig)
    end

    # LAR equations have singularities at r=0; ensure we never evaluate there
    @assert r > 0.0 "LAR equations require r > 0 (called with r=$r)"

    bsq = (y[1] / r)^2 + y[2]^2     # B²
    q = r^2 * y[2] / (lar_r0 * y[1])

    dy[1] = -pp / bsq * y[1] + sigma * y[2] * r
    dy[2] = -pp / bsq * y[2] - sigma * y[1] / r
    dy[3] = y[1] * lar_r0 / r
    dy[4] = (q / r)^2 * y[5] / r
    dy[5] = y[2] * (r / q)^2 * (r / lar_r0) * (1 - 2 * (lar_r0 * q / y[2])^2 * pp)

    return 0
end

"""
    lar_run(lar_input)

Solves the LAR (Large Aspect Ratio) plasma equilibrium ODE system using analytic profiles
defined by `lar_input`, and returns the full solution table including derived quantities.

## Arguments:
- `lar_input`: A `LargeAspectRatioConfig` object containing profile and geometric parameters.

## Returns:
 - working on it not yet implemented
"""

function lar_run(equil_input::EquilibriumConfig, lar_input::LargeAspectRatioConfig)
    rmin = 1e-4
    lar_a = lar_input.lar_a
    lar_r0 = lar_input.lar_r0
    q0 = lar_input.q0
    beta0 = lar_input.beta0
    sigma_type = lar_input.sigma_type

    p00 = beta0 / 2.0
    lar_input.p_pres = max(lar_input.p_pres, 1.001)

    sigma0 = 2.0 / (q0 * lar_r0)

    ma = lar_input.ma
    mtau = lar_input.mtau

    function dydr(du, u, p, r)
        lar_input = p
        return lar_der(du, r, u, lar_input)
    end

    r0, y0 = lar_init_conditions(rmin, lar_input)
    tspan = (r0, lar_a)
    p = lar_input

    prob = ODEProblem(dydr, y0, tspan, p)

    sol = solve(prob, Rosenbrock23(; autodiff=false); reltol=equil_input.etol, abstol=1e-8, maxiters=10000, dense=false)

    r_arr = sol.t
    y_mat = reduce(hcat, sol.u)'
    steps = length(r_arr)

    temp = zeros(steps, 9)
    for i in 1:steps
        r = r_arr[i]
        @views y = y_mat[i, :]
        x = r / lar_a
        xfac = 1 - x^2
        pval = p00 * xfac^lar_input.p_pres
        sigma = (sigma_type == "wesson") ?
                sigma0 * xfac^lar_input.p_sig :
                sigma0 / (1 + x^(2 * lar_input.p_sig))^(1 + 1 / lar_input.p_sig)
        q = r^2 * y[2] / (lar_r0 * y[1])
        temp[i, 1] = r
        temp[i, 2] = y[1]
        temp[i, 3] = y[2]
        temp[i, 4] = y[3]
        temp[i, 5] = y[4]
        temp[i, 6] = y[5]
        temp[i, 7] = pval
        temp[i, 8] = sigma
        temp[i, 9] = q
    end

    xs_r = temp[:, 1]
    fs_r = temp[:, 2:9]
    spl = cubic_interp(xs_r, Series(fs_r); extrap=ExtendExtrap())
    spl_deriv = deriv1(spl)

    dr = lar_a / (ma + 1)
    r = 0.0
    psio = temp[end, 4]  # ψ_edge

    sq_xs = zeros(ma + 1)
    sq_fs = zeros(ma + 1, 3)
    r_nodes = zeros(ma + 1)
    rzphi_y_nodes = range(0.0, 1.0; length=mtau + 1)
    rzphi_fs_nodes = zeros(ma + 1, mtau + 1, 2)

    hint = Ref(1)
    for ia in 1:(ma+1)
        r += dr
        r_nodes[ia] = r
        f = spl(r; hint=hint)
        f1 = spl_deriv(r; hint=hint)
        dψdr = f1[3]

        # Fill spline data for sq_in
        sq_xs[ia] = f[3] / psio  # ψ / psio
        sq_fs[ia, 1] = lar_r0 * f[2]  # F = R0 * Bphi
        sq_fs[ia, 2] = f[6]  # P
        sq_fs[ia, 3] = f[8]  # q

        # Compute Shafranov shift and fill rzphi grid
        r2 = -(f[4] * r / f[8]) / dψdr
        if lar_input.zeroth
            r2 = 0.0
        end
        for itau in 1:(mtau+1)
            θ = 2π * (itau - 1) / mtau
            cosθ, sinθ = cos(θ), sin(θ)
            rfac = r + r2 * cosθ
            rzphi_fs_nodes[ia, itau, 1] = lar_r0 + rfac * cosθ
            rzphi_fs_nodes[ia, itau, 2] = rfac * sinθ
        end
    end

    sq_in = cubic_interp(sq_xs, Series(sq_fs); extrap=ExtendExtrap())
    # rz_in_xs is ψ_N (see InverseRunInput struct docs).  Passing physical r
    # works only by accident when lar_a ≈ 1; otherwise the inverse solver
    # extrapolates the (R, Z) splines at outer surfaces.
    rz_in_xs = sq_xs
    rz_in_ys = collect(rzphi_y_nodes)

    itp_2d_opts = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    rz_in_R = cubic_interp((rz_in_xs, rz_in_ys), rzphi_fs_nodes[:, :, 1]; itp_2d_opts...)
    rz_in_Z = cubic_interp((rz_in_xs, rz_in_ys), rzphi_fs_nodes[:, :, 2]; itp_2d_opts...)

    # LAR equilibrium has midplane symmetry, so magnetic axis is at Z = 0
    return InverseRunInput(equil_input, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z, lar_r0, 0.0, psio)
end

"""
    tj_f1(x, nu, qc)

TJ's poloidal flux function f1(x) where x = r/a.
Uses Taylor expansion near axis for numerical stability.

Reference: R. Fitzpatrick, TJ code, LightEquilibrium.cpp
"""
function tj_f1(x::Float64, nu::Float64, qc::Float64)
    if x < 0.1
        x2 = x * x
        return x2 * (1 - (nu-1)*x2/2 + (nu-1)*(nu-2)*x2*x2/6 -
                      (nu-1)*(nu-2)*(nu-3)*x2*x2*x2/24) / qc
    else
        return (1 - (1 - x*x)^nu) / (nu * qc)
    end
end

"""
    tj_f1p(x, nu, qc)

Derivative of TJ's f1 with respect to x (= r/a).
"""
function tj_f1p(x::Float64, nu::Float64, qc::Float64)
    if x < 0.1
        x2 = x * x
        return 2*x * (1 - (nu-1)*x2 + (nu-1)*(nu-2)*x2*x2/2 -
                       (nu-1)*(nu-2)*(nu-3)*x2*x2*x2/6) / qc
    else
        return 2*x * (1 - x*x)^(nu-1) / qc
    end
end

"""
Internal parameter bundle for the TJ shape ODE (ψ, g₂, H₁, H₁', f₃).  Built
once per TJ call so both `tj_run` and `tj_run_direct` share the same numerics.

Fields:
  - physical: a, R0, qc, mu, pc, B0
  - derived:  epsa2 = (a/R0)²
  - near-axis BC constants: rmin, x0 = rmin, r0 = rmin·a, f1c = 1/qc,
                             p2ppc = d²p₂/dx²|_0 = −2·μ·pc
"""
struct TJShapeParams
    a::Float64
    R0::Float64
    qc::Float64
    mu::Float64
    pc::Float64
    B0::Float64
    epsa2::Float64
    rmin::Float64
    x0::Float64
    r0::Float64
    f1c::Float64
    p2ppc::Float64
end

function TJShapeParams(tj::TJConfig; rmin::Float64 = 1e-4)
    a, R0 = tj.lar_a, tj.lar_r0
    mu    = max(tj.mu, 1.001)
    return TJShapeParams(
        a, R0, tj.qc, mu, tj.pc, tj.B0,
        (a / R0)^2,
        rmin, rmin, rmin * a,
        1.0 / tj.qc,
        -2.0 * mu * tj.pc,
    )
end

"""
RHS for the TJ shape ODE (Equilibrium.cpp rhs_chooser=0 and rhs_chooser=1 dy[1]
combined).  State: y[1]=ψ, y[2]=g₂, y[3]=H₁, y[4]=H₁', y[5]=f₃.  TJ writes
derivatives in x=r/a; we advance in physical r=a·x so d/dr = (1/a)·d/dx.

The params argument carries TJShapeParams fields plus the current `nu`.
"""
function tj_shape_rhs!(dy, y, params, r)
    (; a, B0, qc, mu, pc, epsa2, nu) = params
    x    = r / a
    xfac = max(1 - x^2, 0.0)
    f1   = tj_f1(x, nu, qc)
    f1px = tj_f1p(x, nu, qc)
    p2px = -2 * mu * pc * x * xfac^(mu - 1)

    # TJ writes its physical ψ as εa²·B₀·R₀²·Psi_TJ_norm where
    # dPsi_TJ_norm/dr_TJ = (f1 + εa²·f3)/r_TJ (Equilibrium.cpp rhs_chooser=1 dy[1]).
    # Converting to physical r = a·r_TJ gives dψ/dr = a²·B₀·(f1+εa²·f3)/r.
    f3_cur = y[5]
    dy[1] = B0 * (f1 + epsa2 * f3_cur) * a^2 / r

    # g₂'(x) = −p2'(x) − f1·f1'(x)/x²
    dy[2] = (-p2px - f1 * f1px / (x * x)) / a

    # H₁''(x) = −(2f1'/f1 − 1/x)·H₁' − 1 + 2x³·p2'/f1²
    facf = 2 * f1px / f1 - 1 / x
    facp = 2 * x^3 * p2px / (f1 * f1)
    H1, H1p = y[3], y[4]
    dy[3] = H1p / a
    dy[4] = (-facf * H1p - 1 + facp) / a

    # f₃'(x) for Hₙ = Vₙ = 0 (n ≥ 2 harmonics rescaled to zero in TJ benchmark).
    g2, f3 = y[2], y[5]
    f3p_x = -f3 * f1px / f1 -
             f1 * (3 * x^2 / 2 - 2 * x * H1p + H1p^2) / x +
             f1px * (g2 - 3 * x^2 / 4 + H1 + 3 * H1p^2 / 2) +
             x^2 * p2px * (g2 + x^2 / 2 - 3 * x * H1p - 2 * H1) / f1
    dy[5] = f3p_x / a
    return nothing
end

"""Initial conditions at x = x0 (TJ Equilibrium.cpp lines 438-442)."""
function tj_shape_initial(p::TJShapeParams, nu::Float64)
    f1_0 = tj_f1(p.x0, nu, p.qc)
    y0 = zeros(5)
    y0[1] = p.B0 * f1_0 * p.a^2 / 2                                  # ψ(r0)
    y0[2] = -(p.f1c^2 + p.p2ppc / 2) * p.x0^2                         # g₂
    y0[3] = (2 * p.p2ppc / p.f1c^2 - 1) * p.x0^2 / 8                  # H₁
    y0[4] = (2 * p.p2ppc / p.f1c^2 - 1) * p.x0 / 4                    # H₁'
    y0[5] = 0.0                                                        # f₃
    return y0
end

"""
Integrate the TJ shape ODE for the given ν.  Pass `saveat` to collect output
on a prescribed dense grid (used by `tj_run_direct` so the downstream Hₙ / ψ
splines sit on uniform nodes); leave it nothing for the default adaptive
save pattern used by `tj_run`.
"""
function tj_shape_solve(p::TJShapeParams, nu::Float64;
                        reltol::Float64 = 1e-7, abstol::Float64 = 1e-8,
                        saveat = nothing)
    rhs_params = (; p.a, p.B0, p.qc, p.mu, p.pc, p.epsa2, nu = nu)
    prob = ODEProblem(tj_shape_rhs!, tj_shape_initial(p, nu), (p.r0, p.a), rhs_params)
    if saveat === nothing
        return solve(prob, Vern9(); reltol, abstol, maxiters = 10000, dense = false)
    else
        return solve(prob, Vern9(); reltol, abstol, maxiters = 10000, saveat = saveat)
    end
end

"""
TJ's `Setnu` / `GetNu`: root-find ν so that q₂(x=1) matches `qa_target`.

`q₂ = x²·(1+εa²·g₂)·exp(−εa²·f3/f1)/f1`; at x=1 and low β this picks up an
O(εa²) correction relative to the lowest-order guess ν = qa/qc, which matters
for the TJ benchmark at large ε.  Falls back to the lowest-order ν if the
bracket search diverges.
"""
function tj_find_nu(p::TJShapeParams, qa_target::Float64; reltol::Float64 = 1e-7)
    function q2_edge(nu::Float64)
        sol   = tj_shape_solve(p, nu; reltol)
        g2end = sol.u[end][2]
        f3end = sol.u[end][5]
        f1end = tj_f1(1.0, nu, p.qc)
        return (1 + p.epsa2 * g2end) * exp(-p.epsa2 * f3end / f1end) / f1end
    end
    nu_guess = qa_target / p.qc
    return try
        find_zero(nu -> q2_edge(nu) - qa_target, (0.5 * nu_guess, 2 * nu_guess);
                  atol = 1e-8, rtol = 1e-10)
    catch err
        @warn "ν root-find failed for TJ equilibrium; falling back to lowest-order ν = qa/qc" error = err
        nu_guess
    end
end

"""
    tj_run(equil_input, tj_input)

Construct a cylindrical tokamak equilibrium using the TJ analytic model.

Adapted from R. Fitzpatrick's TJ code (https://github.com/rfitzp/TJ).
Profiles are analytic:

    f1(x) = [1 - (1-x²)^ν] / (ν·qc),   p2(x) = pc·(1-x²)^μ,   x = r/a

with ν = qa/qc.  The 2D geometry is built from TJ's inverse-aspect-ratio
expansion.  With zero edge shaping (Hna = Vna = 0) — the TJ benchmark
configuration — flux surfaces are shifted circles

    R(r,θ) = R₀ + Δ(r) + α(r)·r·cos θ
    Z(r,θ) =            α(r)·r·sin θ

where Δ and α come from the shaping ODE for (g₂, H₁, H₁') (Equilibrium.cpp
rhs_chooser=0 in TJ):

    Δ(r)   = R₀·εa²·H₁(x)                             (Shafranov shift)
    α(r)   = 1 − εa²·(x²/8 − H₁/2)                    (from L(x) = x³/8 − x·H₁/2)
    εa     = a/R₀

The higher-order toroidal-flux correction g₂ enters the output F profile as
F = R₀·B₀·(1 + εa²·g₂), and the higher-order poloidal flux f₃ enters the
safety factor as q₂ = x²·(1 + εa²·g₂)·exp(−εa²·f₃/f1)/f1 (EFIT.cpp).

The (n ≥ 2) horizontal/vertical shaping harmonics Hₙ(r), Vₙ(r) are not yet
included; they are zero in the TJ benchmark scans.
"""
function tj_run(equil_input::EquilibriumConfig, tj::TJConfig)
    a, R0  = tj.lar_a, tj.lar_r0
    qc, mu = tj.qc, max(tj.mu, 1.001)
    pc, B0 = tj.pc, tj.B0
    ma, mtau = tj.ma, tj.mtau
    p = TJShapeParams(tj)
    epsa2     = p.epsa2
    p00_phys  = B0^2 * epsa2 * pc          # μ₀P = B₀²·εa²·p₂ at axis

    nu  = tj_find_nu(p, tj.qa; reltol = equil_input.etol)
    sol = tj_shape_solve(p, nu; reltol = equil_input.etol)

    r_arr = sol.t
    y_mat = reduce(hcat, sol.u)'
    steps = length(r_arr)

    # Profile table: columns [r, F, P, q, ψ, g₂, H₁].  H₁' and f₃ are only
    # needed inside the ODE; F, q folded via EFIT.cpp formulas.
    temp = zeros(steps, 7)
    for i in 1:steps
        r = r_arr[i]
        x = r / a
        xfac = max(1 - x^2, 0.0)
        f1 = tj_f1(x, nu, qc)

        ψ  = y_mat[i, 1]
        g2 = y_mat[i, 2]
        H1 = y_mat[i, 3]
        f3 = y_mat[i, 5]

        F = R0 * B0 * (1 + epsa2 * g2)
        P = p00_phys * xfac^mu
        q = x > 1e-10 ? x^2 * (1 + epsa2 * g2) * exp(-epsa2 * f3 / f1) / f1 : qc

        temp[i, 1] = r
        temp[i, 2] = F
        temp[i, 3] = P
        temp[i, 4] = q
        temp[i, 5] = ψ
        temp[i, 6] = g2
        temp[i, 7] = H1
    end

    xs_r = temp[:, 1]
    fs_r = temp[:, 2:7]
    spl = cubic_interp(xs_r, Series(fs_r); extrap=ExtendExtrap())

    dr = a / (ma + 1)
    r = 0.0
    psio = temp[end, 5]

    sq_xs = zeros(ma + 1)
    sq_fs = zeros(ma + 1, 3)
    r_nodes = zeros(ma + 1)
    rzphi_y_nodes = range(0.0, 1.0; length=mtau + 1)
    rzphi_fs_nodes = zeros(ma + 1, mtau + 1, 2)

    hint = Ref(1)
    for ia in 1:(ma+1)
        r += dr
        r_nodes[ia] = r
        f = spl(r; hint=hint)
        # f[1]=F, f[2]=P, f[3]=q, f[4]=ψ, f[5]=g₂, f[6]=H₁

        sq_xs[ia]    = f[4] / psio
        sq_fs[ia, 1] = f[1]           # F
        sq_fs[ia, 2] = f[2]           # P
        sq_fs[ia, 3] = f[3]           # q

        if tj.zeroth
            Δ = 0.0
            α = 1.0
        else
            x = r / a
            H1_r = f[6]
            Δ = R0 * epsa2 * H1_r
            α = 1 - epsa2 * (x^2 / 8 - H1_r / 2)
        end

        for itau in 1:(mtau+1)
            θ = 2π * (itau - 1) / mtau
            rzphi_fs_nodes[ia, itau, 1] = R0 + Δ + α * r * cos(θ)
            rzphi_fs_nodes[ia, itau, 2] =          α * r * sin(θ)
        end
    end

    sq_in = cubic_interp(sq_xs, Series(sq_fs); extrap=ExtendExtrap())
    # InverseRunInput's rz_in_xs is specified as ψ_N (see EquilibriumTypes.jl docs);
    # the inverse solver queries (R, Z) splines at ψ_N values from sq_xs.  Passing
    # physical r here happens to work when a ≈ 1 (r and ψ_N cover the same range)
    # but extrapolates the (R, Z) splines for any a < 1, corrupting outer surfaces.
    rz_in_xs = sq_xs
    rz_in_ys = collect(rzphi_y_nodes)

    itp_2d_opts = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    rz_in_R = cubic_interp((rz_in_xs, rz_in_ys), rzphi_fs_nodes[:, :, 1]; itp_2d_opts...)
    rz_in_Z = cubic_interp((rz_in_xs, rz_in_ys), rzphi_fs_nodes[:, :, 2]; itp_2d_opts...)

    return InverseRunInput(equil_input, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z, R0, 0.0, psio)
end

"""
    tj_run_direct(equil_input, tj_input; nrbox=257, nzbox=257, rc=1.2)

Option B pipeline: construct ψ(R, Z) on a 2D grid from the TJ analytic model
and return a `DirectRunInput` so the equilibrium is processed by the direct-GS
solver (same path as the TJ-geqdsk scans).

Using the inverse pipeline on just the first-order Shafranov-shifted-circle
geometry systematically under-drives the external kink at large ε because the
inverse solver consumes the prescribed q₂ profile and never recomputes q from
geometry.  The direct pipeline, in contrast, line-integrates F·∮dθ/(R²·Bp) on
the 2D ψ(R,Z) field, so higher-order geometric effects (buried in the shape of
ψ away from the axis) feed back into q and δW.  Reproducing TJ's full geqdsk
path therefore requires rebuilding ψ(R,Z) from the analytic model itself — not
just the flux-surface coordinates — including the vacuum region outside the
plasma.

The benchmark keeps edge shaping `Hna = Vna = 0`, so the ODE-integrated shape
harmonics Hₙ, Vₙ for n ≥ 2 are rescaled to zero; only the H₁ Shafranov shift
contributes.  ψ(R, Z) is constructed by:

  - for each grid point, iterating the map (R, Z) → (r, w) 10× per
    TJ Equilibrium.cpp EFIT::CalculateEFIT (handles the εa²·H₁ shift of the
    axis);
  - evaluating ψ_plasma(r) from the radial ψ-ODE when r < 1, TJ's analytic
    vacuum solution `GetPSIvac` when 1 ≤ r < rc, and the 1/r² far-field form
    when r ≥ rc.

References (TJ code, Fitzpatrick, https://github.com/rfitzp/TJ):
  - Equilibrium.cpp::CashKarp45Rhs (shape ODE, rhs_chooser = 0 and 1)
  - Equilibrium.cpp::GetPSIvac, GetHHvac
  - EFIT.cpp::CalculateEFIT
"""
function tj_run_direct(equil_input::EquilibriumConfig, tj::TJConfig;
                       nrbox::Int = 257, nzbox::Int = 257, rc::Float64 = 1.2)
    a, R0  = tj.lar_a, tj.lar_r0
    qc, mu = tj.qc, max(tj.mu, 1.001)
    pc, B0 = tj.pc, tj.B0
    p = TJShapeParams(tj)
    epsa, epsa2 = p.a / p.R0, p.epsa2
    p00_phys    = B0^2 * epsa2 * pc

    # ν root-find (TJ Setnu): q₂(1) = qa_target.
    nu = tj_find_nu(p, tj.qa; reltol = equil_input.etol)

    # Dense saveat so the downstream splines (H₁, g₂, f₃, ψ) are evaluated on
    # a fine uniform r grid rather than the ~30 adaptive Vern9 steps — otherwise
    # the (R, Z) → (r, w) Newton iteration hits spline interpolation artifacts.
    dense_r = collect(range(p.r0, p.a; length = 1024))
    sol     = tj_shape_solve(p, nu; reltol = equil_input.etol,
                              abstol = 1e-10, saveat = dense_r)
    r_arr   = sol.t
    y_mat   = reduce(hcat, sol.u)'

    # Radial splines in TJ's dimensionless x = r/a on a clean grid for H₁ etc.
    x_nodes = r_arr ./ a
    ψ_of_r   = cubic_interp(r_arr, y_mat[:, 1]; extrap=ExtendExtrap())
    H1_of_x  = cubic_interp(x_nodes, y_mat[:, 3]; extrap=ExtendExtrap())
    H1p_of_x = cubic_interp(x_nodes, y_mat[:, 4]; extrap=ExtendExtrap())
    g2_of_x  = cubic_interp(x_nodes, y_mat[:, 2]; extrap=ExtendExtrap())
    f3_of_x  = cubic_interp(x_nodes, y_mat[:, 5]; extrap=ExtendExtrap())

    # Edge values needed by GetPSIvac
    f1a  = tj_f1(1.0, nu, qc)
    f3a  = f3_of_x(1.0)
    H1a  = H1_of_x(1.0)
    H1ap = H1p_of_x(1.0)
    psio = ψ_of_r(a)   # ψ at r = a (boundary)

    # Psi scaling factor that matches TJ's EFIT writer: Psi_TJ_phys = εa²·B0·R0²·Psi_norm
    psi_scale = epsa2 * B0 * R0^2

    # GetHHvac for n = 1 (Equilibrium.cpp line 1792).  Hₙ vacuum for n ≥ 2
    # vanishes because H_n(1) = H_n'(1) = 0 after TJ's Hna/Vna rescaling.
    function H1_vac(r::Float64)
        return H1a - 0.5 * r^2 * log(r) + 0.25 * (2 * H1ap + 1) * (r^2 - 1)
    end

    # Getf_R, Getf_Z (Equilibrium.cpp lines 1915, 1965): full TJ shift of (R,Z)
    # from the nominal shifted circle.  With Hn = Vn = 0 for n ≥ 2 the residual
    # terms are:
    #   f_R = εa²·H₁(r) + εa³·L(r)·cos(w)
    #   f_Z =          −εa³·L(r)·sin(w)
    # L(r) = r³/8 − r·H₁(r)/2.  The εa³ terms were omitted in my first pass and
    # shifted the pole location of the ε-scan to ε ≈ 0.41 instead of ε ≈ 0.66.
    # Per TJ (Equilibrium.cpp lines 1917, 1967), freeze f_R, f_Z at r = rc and
    # scale the inner value by r²/rc² for r ≥ rc to prevent the Newton iteration
    # from diverging in the far vacuum.
    function L_of(r::Float64)
        rr = (r >= rc) ? (rc - 1e-8) : r
        H1 = (rr < 1.0) ? H1_of_x(rr) : H1_vac(rr)
        return rr^3 / 8 - rr * H1 / 2
    end
    function f_R_shift(r::Float64, w::Float64)
        if r >= rc
            # TJ's capping: f_R(r, w) = f_R(rc − ε, w) · r² / rc²
            return f_R_shift(rc - 1e-8, w) * r^2 / rc^2
        end
        H1 = (r < 1.0) ? H1_of_x(r) : H1_vac(r)
        L  = r^3 / 8 - r * H1 / 2
        return epsa2 * H1 + epsa2 * epsa * L * cos(w)
    end
    function f_Z_shift(r::Float64, w::Float64)
        if r >= rc
            return f_Z_shift(rc - 1e-8, w) * r^2 / rc^2
        end
        H1 = (r < 1.0) ? H1_of_x(r) : H1_vac(r)
        L  = r^3 / 8 - r * H1 / 2
        return -epsa2 * epsa * L * sin(w)
    end

    # (R_norm, Z_norm) → (r, w) by TJ's 10-step fixed-point iteration
    # (EFIT.cpp lines 213-228).  R_norm, Z_norm are normalized to R₀.
    function find_rw(R_norm::Float64, Z_norm::Float64)
        r = sqrt((R_norm - 1.0)^2 + Z_norm^2) / epsa
        w = atan(Z_norm, 1.0 - R_norm)
        for _ in 1:10
            RR = R_norm - f_R_shift(r, w)
            ZZ = Z_norm - f_Z_shift(r, w)
            r = sqrt((RR - 1.0)^2 + ZZ^2) / epsa
            w = atan(ZZ, 1.0 - RR)
        end
        return r, w
    end

    # GetPSIvac (Equilibrium.cpp line 1867) with Hn = Vn = 0 for n ≥ 2.
    # Returns the TJ-normalized vacuum ψ (in units where the plasma interior
    # ψ-ODE ran); multiplied by psi_scale for physical units.
    function psi_vac(r::Float64)
        logr = log(r)
        sum1 = 1.0 - H1ap + H1ap^2
        sum2 = -H1ap * r^2 * logr + 0.5 * r^2 * logr^2 +
               0.5 * (1.0 + H1ap^2) * (r^2 - 1.0)
        return f1a * logr + epsa2 * f3a * logr -
               0.5 * epsa2 * f1a * (-sum1 * logr + sum2)
    end

    # ψ(r) inside plasma, from my ODE.  ψ_ana(0) ≈ 0, ψ_ana(a) = psio.  The
    # clamp keeps the argument inside the spline's data range [p.r0, p.a].
    function psi_plasma_physical(r::Float64)
        r_phys = clamp(r * p.a, p.r0, p.a)
        return ψ_of_r(r_phys)
    end

    # Build psi_in in the direct-GS solver's expected convention:
    # positive at axis, zero at LCFS, negative outside (per DirectRunInput docs).
    # Inside plasma: psi = psio − ψ_plasma(r)  (axis ≈ psio, boundary = 0).
    # Outside: psi = −psi_scale · GetPSIvac(r)  (0 at LCFS, negative outside).
    #
    # Grid spans R₀ ± rc·a × ±rc·a (where rc is the vacuum-shell radius in
    # units of a), giving a comfortable margin for the separatrix finder.
    r_span = rc * a
    psi_in_xs = collect(range(R0 - r_span, R0 + r_span; length = nrbox))
    psi_in_ys = collect(range(-r_span, r_span; length = nzbox))
    psi_rz    = zeros(Float64, nrbox, nzbox)

    for i in 1:nrbox, j in 1:nzbox
        R_norm = psi_in_xs[i] / R0
        Z_norm = psi_in_ys[j] / R0
        r_lbl, _ = find_rw(R_norm, Z_norm)

        if r_lbl < 1.0
            ψ_p = psi_plasma_physical(r_lbl)
            psi_rz[i, j] = psio - ψ_p                         # plasma: +psio at axis, 0 at LCFS
        elseif r_lbl < rc
            psi_rz[i, j] = -psi_scale * psi_vac(r_lbl)        # vacuum: 0 at LCFS, neg. outside
        else
            psi_rz[i, j] = -psi_scale * psi_vac(rc) * r_lbl^2 / rc^2
        end
    end

    # 2D spline consumed by direct-GS
    psi_in = cubic_interp((psi_in_xs, psi_in_ys), psi_rz; extrap=ExtendExtrap())

    # 1D profile spline, same layout as read_efit (4 columns).  Use TJ's
    # analytic q₂ on the radial grid so that the prescribed q is consistent with
    # the ψ(R,Z) we just constructed.
    psi_norm_grid = range(0.0, 1.0; length = nrbox)
    F_nodes  = zeros(nrbox); P_nodes = zeros(nrbox); q_nodes = zeros(nrbox)
    for i in 1:nrbox
        ψN = psi_norm_grid[i]
        # Invert ψN = (ψ_plasma(r) - 0) / psio  ⇒  find r such that ψ_plasma(r) = ψN·psio.
        # ψ_plasma is monotonic in r so a Brent search on [p.r0, p.a] converges quickly.
        target = ψN * psio
        rlocal = if ψN ≤ 0.0
            p.r0
        elseif ψN ≥ 1.0
            p.a
        else
            find_zero(r -> ψ_of_r(r) - target, (p.r0, p.a); atol=1e-10, rtol=1e-12)
        end
        x = rlocal / p.a
        f1 = tj_f1(x, nu, qc)
        g2_val = g2_of_x(x)
        f3_val = f3_of_x(x)
        xfac = max(1 - x^2, 0.0)
        F_nodes[i] = R0 * B0 * (1 + epsa2 * g2_val)
        P_nodes[i] = p00_phys * xfac^mu
        q_nodes[i] = (x > 1e-10) ? x^2 * (1 + epsa2 * g2_val) *
                                    exp(-epsa2 * f3_val / f1) / f1 : qc
    end
    sq_fs_nodes = hcat(F_nodes, P_nodes, q_nodes, sqrt.(collect(psi_norm_grid)))
    sq_in = cubic_interp(collect(psi_norm_grid), Series(sq_fs_nodes); extrap=ExtendExtrap())

    rmin_grid, rmax_grid = extrema(psi_in_xs)
    zmin_grid, zmax_grid = extrema(psi_in_ys)

    return DirectRunInput(equil_input, sq_in, psi_in, psi_in_xs, psi_in_ys,
                          rmin_grid, rmax_grid, zmin_grid, zmax_grid, psio)
end

"""
This function handles the Solovev analytical equilibrium model, transforming the input parameters
into the necessary splines and scalar values for equilibrium construction. This is a Julia version
of the Fortran code in sol.f, with no major differences except for arrays going from 0:n to 1:n+1.

## Arguments:

  - `mr`: Number of radial grid zones
  - `mz`: Number of axial grid zones
  - `ma`: Number of flux grid zones
  - `e`: Elongation
  - `a`: Minor radius
  - `r0`: Major radius
  - `q0`: Safety factor at the o-point
  - `p0fac`: Scales on axis pressure (s*P. beta changes. Phi,q constant)
  - `b0fac`: Scales on toroidal field (s*Phi,s*f,s^2*P. bt changes. Shape,beta constant)
  - `f0fac`: Scales on toroidal field (s*f. bt,q changes. Phi,p,bp constant)

## Returns:

  - `DirectRunInput` object
"""
function sol_run(equil_inputs::EquilibriumConfig, sol_inputs::SolovevConfig)

    mr = sol_inputs.mr
    mz = sol_inputs.mz
    ma = sol_inputs.ma
    e = sol_inputs.e
    a = sol_inputs.a
    r0 = sol_inputs.r0
    q0 = sol_inputs.q0
    p0fac = sol_inputs.p0fac
    b0fac = sol_inputs.b0fac
    f0fac = sol_inputs.f0fac

    # Validate inputs
    if p0fac < 1.0
        @warn "Forcing p0fac ≥ 1 (no negative pressure)"
        p0fac = 1.0
    end

    # Compute scalar data
    f0 = r0 * b0fac
    psio = e * f0 * a * a / (2 * q0 * r0)
    psifac = psio / (a * r0)^2
    efac = 1 / (e * e)
    pfac = 2 * psio^2 * (e * e + 1) / (a * r0 * e)^2
    rmin = r0 - 1.5 * a
    rmax = r0 + 1.5 * a
    zmax = 1.5 * e * a
    zmin = -zmax

    # Compute 1D data and spline
    sqfs = Array{Float64}(undef, ma + 1, 4)
    psis = [(ia / (ma + 1))^2 for ia in 1:(ma+1)]
    sqfs[:, 1] .= f0 .* f0fac
    sqfs[:, 2] .= pfac .* (1 .* p0fac .- psis)
    sqfs[:, 3] .= 0.0
    sq_in = cubic_interp(psis, Series(sqfs); extrap=ExtendExtrap())

    # Compute 2D data and spline
    r = [rmin + i * (rmax - rmin) / mr for i in 0:mr]
    z = [zmin + j * (zmax - zmin) / mz for j in 0:mz]
    psifs = Array{Float64}(undef, mr + 1, mz + 1)
    for iz in 1:(mz+1)
        for ir in 1:(mr+1)
            psifs[ir, iz] = psio - psifac * (efac * (r[ir] * z[iz])^2 + (r[ir]^2 - r0^2)^2 / 4)
        end
    end
    psi_in_xs = r
    psi_in_ys = z
    psi_in = cubic_interp((psi_in_xs, psi_in_ys), psifs; extrap=ExtendExtrap())

    # Print out equilibrium info
    @info "Generating Solovev equilibrium: mr=$mr, mz=$mz, ma=$ma, e=$(@sprintf("%.3f", e)), a=$(@sprintf("%.3f", a)), r0=$(@sprintf("%.3f", r0)), q0=$(@sprintf("%.3f", q0))"

    return DirectRunInput(equil_inputs, sq_in, psi_in, psi_in_xs, psi_in_ys, rmin, rmax, zmin, zmax, psio)
end
