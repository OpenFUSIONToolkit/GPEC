"""
    TJAnalyticConfig(...)

Parameters for the TJ-analytic cylindrical large-aspect-ratio equilibrium, a GPEC adaptation of
the analytic profile family of R. Fitzpatrick's TJ code (https://github.com/rfitzp/TJ), not a
re-implementation of TJ. The q profile is

    f1(r) = [1 - (1-r²)^ν] / (ν·q0)
    q(r)  = r² / f1(r)

with the current peaking ν solved so q matches `qa` at the edge (ν = qa/q0 at lowest order). The
pressure is p₂(r) = p0·(1-r²)^μ with μ = `pressure_peaking`. Lengths are normalized to R₀, fields
to B₀. `solver` picks the pipeline: `:inverse` feeds the flux-surface geometry to the inverse
solver, `:direct` builds ψ(R, Z) for the direct solver.
"""
@kwdef struct TJAnalyticConfig <: AnalyticEquilibriumConfig
    r0::Float64 = 10.0         # Major radius R₀ [m]
    a::Float64 = 1.0           # Minor radius a [m] (ε = a/R₀)
    q0::Float64 = 1.5          # On-axis safety factor
    qa::Float64 = 3.6          # Edge safety factor
    p0::Float64 = 0.001        # Normalized on-axis pressure
    pressure_peaking::Float64 = 2.0 # Pressure peaking exponent μ: p₂ = p0·(1-r²)^μ
    B0::Float64 = 12.0         # On-axis toroidal field [T]
    ma::Int = 128              # Radial grid points
    mtheta::Int = 128          # Poloidal grid points
    zeroth::Bool = false       # If true, suppress Shafranov shift
    solver::Symbol = :inverse  # Equilibrium pipeline: :inverse or :direct
end

"""
    tj_analytic_f1(x, nu, qc)

TJ-analytic poloidal flux function f1(x) where x = r/a, following the
analytic-profile parameterization of R. Fitzpatrick's TJ code
(https://github.com/rfitzp/TJ). Uses a Taylor expansion near the axis
for numerical stability.
"""
function tj_analytic_f1(x::Float64, nu::Float64, qc::Float64)
    if x < 0.1
        x2 = x * x
        return x2 * (1 - (nu - 1) * x2 / 2 + (nu - 1) * (nu - 2) * x2 * x2 / 6 -
                     (nu - 1) * (nu - 2) * (nu - 3) * x2 * x2 * x2 / 24) / qc
    else
        return (1 - (1 - x * x)^nu) / (nu * qc)
    end
end

"""
    tj_analytic_f1p(x, nu, qc)

Derivative of the TJ-analytic f1 with respect to x (= r/a).  See
R. Fitzpatrick's TJ code (https://github.com/rfitzp/TJ) for the original
parameterization.
"""
function tj_analytic_f1p(x::Float64, nu::Float64, qc::Float64)
    if x < 0.1
        x2 = x * x
        return 2 * x * (1 - (nu - 1) * x2 + (nu - 1) * (nu - 2) * x2 * x2 / 2 -
                        (nu - 1) * (nu - 2) * (nu - 3) * x2 * x2 * x2 / 6) / qc
    else
        return 2 * x * (1 - x * x)^(nu - 1) / qc
    end
end

"""
Internal parameter bundle for the TJ-analytic shape ODE (ψ, g₂, H₁, H₁', f₃) —
GPEC adaptation of the analytic shape ODE used in R. Fitzpatrick's TJ code
(https://github.com/rfitzp/TJ), shared by the inverse and direct pipelines.

Fields:

  - physical: a, R0, qc, mu, pc, B0
  - derived:  epsa2 = (a/R0)²
  - near-axis BC constants: rmin, x0 = rmin, r0 = rmin·a, f1c = 1/qc,
    p2ppc = d²p₂/dx²|_0 = −2·μ·pc
"""
struct TJAnalyticShapeParams
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

function TJAnalyticShapeParams(tj::TJAnalyticConfig; rmin::Float64=1e-4)
    a, R0 = tj.a, tj.r0
    mu = max(tj.pressure_peaking, 1.001)
    return TJAnalyticShapeParams(
        a, R0, tj.q0, mu, tj.p0, tj.B0,
        (a / R0)^2,
        rmin, rmin, rmin * a,
        1.0 / tj.q0,
        -2.0 * mu * tj.p0
    )
end

"""
RHS for the TJ-analytic shape ODE (R. Fitzpatrick's TJ code parameterization,
https://github.com/rfitzp/TJ).  State: y[1]=ψ, y[2]=g₂, y[3]=H₁, y[4]=H₁',
y[5]=f₃.  The original derivation is written in x = r/a; we advance in
physical r = a·x so d/dr = (1/a)·d/dx.

The params argument carries TJAnalyticShapeParams fields plus the current `nu`.
"""
function tj_analytic_shape_rhs!(dy, y, params, r)
    (; a, B0, qc, mu, pc, epsa2, nu) = params
    x = r / a
    xfac = max(1 - x^2, 0.0)
    f1 = tj_analytic_f1(x, nu, qc)
    f1px = tj_analytic_f1p(x, nu, qc)
    p2px = -2 * mu * pc * x * xfac^(mu - 1)

    # The TJ-analytic model writes its physical ψ as εa²·B₀·R₀²·Psi_norm where
    # dPsi_norm/dr_norm = (f1 + εa²·f3)/r_norm (cf. Fitzpatrick's TJ code).
    # Converting to physical r = a·r_norm gives dψ/dr = a²·B₀·(f1+εa²·f3)/r.
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

    # f₃'(x) for Hₙ = Vₙ = 0 (n ≥ 2 harmonics rescaled to zero, as in the
    # TJ-analytic benchmark configuration of Fitzpatrick's TJ code).
    g2, f3 = y[2], y[5]
    f3p_x =
        -f3 * f1px / f1 -
        f1 * (3 * x^2 / 2 - 2 * x * H1p + H1p^2) / x +
        f1px * (g2 - 3 * x^2 / 4 + H1 + 3 * H1p^2 / 2) +
        x^2 * p2px * (g2 + x^2 / 2 - 3 * x * H1p - 2 * H1) / f1
    dy[5] = f3p_x / a
    return nothing
end

"""
Initial conditions at x = x0, matching the TJ-analytic model's near-axis
expansion (cf. R. Fitzpatrick's TJ code, https://github.com/rfitzp/TJ).
"""
function tj_analytic_shape_initial(p::TJAnalyticShapeParams, nu::Float64)
    f1_0 = tj_analytic_f1(p.x0, nu, p.qc)
    y0 = zeros(5)
    y0[1] = p.B0 * f1_0 * p.a^2 / 2                                  # ψ(r0)
    y0[2] = -(p.f1c^2 + p.p2ppc / 2) * p.x0^2                         # g₂
    y0[3] = (2 * p.p2ppc / p.f1c^2 - 1) * p.x0^2 / 8                  # H₁
    y0[4] = (2 * p.p2ppc / p.f1c^2 - 1) * p.x0 / 4                    # H₁'
    y0[5] = 0.0                                                        # f₃
    return y0
end

"""
Integrate the TJ-analytic shape ODE for the given ν.  Pass `saveat` to collect
output on a prescribed dense grid (used by `tj_analytic_run_direct` so the
downstream Hₙ / ψ splines sit on uniform nodes); leave it `nothing` for
the default adaptive save pattern used by `tj_analytic_run_inverse`. `reltol` and `abstol` go to the
Vern9 solve; `abstol` defaults to `equil_abstol(reltol)`.
"""
function tj_analytic_shape_solve(p::TJAnalyticShapeParams, nu::Float64;
    reltol::Float64=1e-7, abstol::Float64=equil_abstol(reltol),
    saveat=nothing)
    rhs_params = (; p.a, p.B0, p.qc, p.mu, p.pc, p.epsa2, nu=nu)
    prob = ODEProblem(tj_analytic_shape_rhs!, tj_analytic_shape_initial(p, nu), (p.r0, p.a), rhs_params)
    sol = if saveat === nothing
        solve(prob, Vern9(); reltol, abstol, dense=false)
    else
        solve(prob, Vern9(); reltol, abstol, saveat=saveat)
    end
    return check_equil_solve(sol, "TJ-analytic shaping functions at ν = $nu")
end

"""
TJ-analytic ν root-find (Fitzpatrick's `Setnu` / `GetNu` in
https://github.com/rfitzp/TJ): solve for ν so that q₂(x=1) matches
`qa_target`.

`q₂ = x²·(1+εa²·g₂)·exp(−εa²·f3/f1)/f1`; at x=1 and low β this picks up an
O(εa²) correction relative to the lowest-order guess ν = qa/qc, which
matters for the TJ-analytic benchmark at large ε.  Falls back to the
lowest-order ν if the root-find has no bracket or does not converge; an
error from a trial shaping solve propagates.
"""
function tj_analytic_find_nu(p::TJAnalyticShapeParams, qa_target::Float64; reltol::Float64=1e-7, abstol::Float64=equil_abstol(reltol))
    function q2_edge(nu::Float64)
        sol = tj_analytic_shape_solve(p, nu; reltol, abstol)
        g2end = sol.u[end][2]
        f3end = sol.u[end][5]
        f1end = tj_analytic_f1(1.0, nu, p.qc)
        return (1 + p.epsa2 * g2end) * exp(-p.epsa2 * f3end / f1end) / f1end
    end
    nu_guess = qa_target / p.qc
    return try
        find_zero(nu -> q2_edge(nu) - qa_target, (0.5 * nu_guess, 2 * nu_guess);
            atol=1e-8, rtol=1e-10)
    catch err
        # Only a failed root-find falls back; a shaping solve that stopped early must propagate.
        fallback = err isa Roots.ConvergenceFailed || (err isa ArgumentError && occursin("not a bracketing interval", err.msg))
        fallback || rethrow()
        @warn "ν root-find failed for TJ-analytic equilibrium; falling back to lowest-order ν = qa/qc" exception = (err, catch_backtrace())
        nu_guess
    end
end

"""
    tj_analytic_run(equil_input, tj_input)

Solve the TJ-analytic equilibrium with the pipeline `tj_input.solver` names, `:inverse`
([`tj_analytic_run_inverse`](@ref)) or `:direct` ([`tj_analytic_run_direct`](@ref)).
"""
function tj_analytic_run(equil_input::EquilibriumConfig, tj::TJAnalyticConfig)
    tj.solver === :inverse && return tj_analytic_run_inverse(equil_input, tj)
    tj.solver === :direct && return tj_analytic_run_direct(equil_input, tj)
    return error("TJAnalytic solver = \"$(tj.solver)\" is not one of \"inverse\", \"direct\"")
end

"""
    tj_analytic_run_inverse(equil_input, tj_input)

Build the TJ-analytic equilibrium of `tj` as inverse-solver input. With zero
edge shaping (Hna = Vna = 0, the TJ benchmark configuration) the flux surfaces are shifted circles

    R(r,θ) = R₀ + Δ(r) + α(r)·r·cos θ
    Z(r,θ) =            α(r)·r·sin θ

with Δ = R₀·εa²·H₁(x), α = 1 − εa²·(x²/8 − H₁/2), εa = a/R₀ and x = r/a, from the shaping ODE.
The output profiles are F = R₀·B₀·(1 + εa²·g₂) and q₂ = x²·(1 + εa²·g₂)·exp(−εa²·f₃/f1)/f1.
The n ≥ 2 shaping harmonics Hₙ, Vₙ are not included.
"""
function tj_analytic_run_inverse(equil_input::EquilibriumConfig, tj::TJAnalyticConfig)
    a, R0 = tj.a, tj.r0
    qc, mu = tj.q0, max(tj.pressure_peaking, 1.001)
    pc, B0 = tj.p0, tj.B0
    ma, mtau = tj.ma, tj.mtheta
    p = TJAnalyticShapeParams(tj)
    epsa2 = p.epsa2
    p00_phys = B0^2 * epsa2 * pc          # μ₀P = B₀²·εa²·p₂ at axis

    nu = tj_analytic_find_nu(p, tj.qa; reltol=equil_input.flux_surface_rtol)
    sol = tj_analytic_shape_solve(p, nu; reltol=equil_input.flux_surface_rtol)

    r_arr = sol.t
    y_mat = reduce(hcat, sol.u)'
    steps = length(r_arr)

    # Profile table columns: r, F, P, q, ψ, g₂, H₁ (F and q as in TJ's EFIT writer).
    temp = zeros(steps, 7)
    for i in 1:steps
        r = r_arr[i]
        x = r / a
        xfac = max(1 - x^2, 0.0)
        f1 = tj_analytic_f1(x, nu, qc)

        ψ = y_mat[i, 1]
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

        sq_xs[ia] = f[4] / psio
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
            rzphi_fs_nodes[ia, itau, 2] = α * r * sin(θ)
        end
    end

    sq_in = cubic_interp(sq_xs, Series(sq_fs); extrap=ExtendExtrap())
    # rz_in_xs is ψ_N (see InverseRunInput), not physical r; the two only coincide when a ≈ 1.
    rz_in_xs = sq_xs
    rz_in_ys = collect(rzphi_y_nodes)

    itp_2d_opts = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    rz_in_R = cubic_interp((rz_in_xs, rz_in_ys), rzphi_fs_nodes[:, :, 1]; itp_2d_opts...)
    rz_in_Z = cubic_interp((rz_in_xs, rz_in_ys), rzphi_fs_nodes[:, :, 2]; itp_2d_opts...)

    # Analytic: ingest=nothing — replay regenerates from the [TJAnalytic] deck section.
    return InverseRunInput(equil_input, sq_in, rz_in_xs, rz_in_ys, rz_in_R, rz_in_Z, R0, 0.0, psio, nothing)
end

"""
    tj_analytic_run_direct(equil_input, tj_input; nrbox=257, nzbox=257, rc=1.2)

Build ψ(R, Z) on an `nrbox × nzbox` grid from the TJ-analytic model and return it as direct-solver
input, so q and δW are recomputed from the 2D geometry instead of taken from the prescribed q₂ (the
inverse pipeline under-drives the external kink at large ε). Each grid point is mapped to (r, w) by
the fixed-point iteration of TJ's EFIT writer; ψ comes from the radial ψ-ODE for r < 1, TJ's vacuum
solution `GetPSIvac` for 1 ≤ r < `rc`, and that value at `rc` scaled by r²/rc² beyond (r, `rc` in
units of a). Edge shaping is `Hna = Vna = 0`, so only the H₁ Shafranov shift contributes.
"""
function tj_analytic_run_direct(equil_input::EquilibriumConfig, tj::TJAnalyticConfig;
    nrbox::Int=257, nzbox::Int=257, rc::Float64=1.2)
    a, R0 = tj.a, tj.r0
    qc, mu = tj.q0, max(tj.pressure_peaking, 1.001)
    pc, B0 = tj.p0, tj.B0
    p = TJAnalyticShapeParams(tj)
    epsa, epsa2 = p.a / p.R0, p.epsa2
    p00_phys = B0^2 * epsa2 * pc

    # ν root-find (cf. Fitzpatrick TJ's Setnu): q₂(1) = qa_target.
    nu = tj_analytic_find_nu(p, tj.qa; reltol=equil_input.flux_surface_rtol)

    # Dense uniform saveat: on the adaptive steps the (R, Z) → (r, w) iteration hits spline artifacts.
    # That inversion also needs a tighter abstol ceiling (1e-10) than the other equilibrium solves.
    dense_r = collect(range(p.r0, p.a; length=1024))
    sol = tj_analytic_shape_solve(p, nu; reltol=equil_input.flux_surface_rtol, abstol=equil_abstol(equil_input.flux_surface_rtol, 1e-10), saveat=dense_r)
    r_arr = sol.t
    y_mat = reduce(hcat, sol.u)'

    # Radial splines in the TJ-analytic dimensionless x = r/a on a clean grid for H₁ etc.
    x_nodes = r_arr ./ a
    ψ_of_r = cubic_interp(r_arr, y_mat[:, 1]; extrap=ExtendExtrap())
    H1_of_x = cubic_interp(x_nodes, y_mat[:, 3]; extrap=ExtendExtrap())
    H1p_of_x = cubic_interp(x_nodes, y_mat[:, 4]; extrap=ExtendExtrap())
    g2_of_x = cubic_interp(x_nodes, y_mat[:, 2]; extrap=ExtendExtrap())
    f3_of_x = cubic_interp(x_nodes, y_mat[:, 5]; extrap=ExtendExtrap())

    # Edge values needed by GetPSIvac
    f1a = tj_analytic_f1(1.0, nu, qc)
    f3a = f3_of_x(1.0)
    H1a = H1_of_x(1.0)
    H1ap = H1p_of_x(1.0)
    psio = ψ_of_r(a)   # ψ at r = a (boundary)

    # Psi scaling factor matching the TJ-analytic EFIT writer: Psi_phys = εa²·B0·R0²·Psi_norm
    psi_scale = epsa2 * B0 * R0^2

    # TJ-analytic GetHHvac for n = 1 (cf. Fitzpatrick's TJ).  The n ≥ 2 vacuum
    # Hₙ vanishes because H_n(1) = H_n'(1) = 0 after the Hna/Vna rescaling.
    function H1_vac(r::Float64)
        return H1a - 0.5 * r^2 * log(r) + 0.25 * (2 * H1ap + 1) * (r^2 - 1)
    end

    # TJ's shifts of (R, Z) from the nominal circle with Hₙ = Vₙ = 0 for n ≥ 2, L(r) = r³/8 − r·H₁(r)/2:
    #   f_R = εa²·H₁(r) + εa³·L(r)·cos(w)
    #   f_Z =          −εa³·L(r)·sin(w)
    # As in TJ, frozen at r = rc and scaled by r²/rc² beyond, so the iteration does not diverge in the far vacuum.
    function f_R_shift(r::Float64, w::Float64)
        if r >= rc
            return f_R_shift(rc - 1e-8, w) * r^2 / rc^2
        end
        H1 = (r < 1.0) ? H1_of_x(r) : H1_vac(r)
        L = r^3 / 8 - r * H1 / 2
        return epsa2 * H1 + epsa2 * epsa * L * cos(w)
    end
    function f_Z_shift(r::Float64, w::Float64)
        if r >= rc
            return f_Z_shift(rc - 1e-8, w) * r^2 / rc^2
        end
        H1 = (r < 1.0) ? H1_of_x(r) : H1_vac(r)
        L = r^3 / 8 - r * H1 / 2
        return -epsa2 * epsa * L * sin(w)
    end

    # (R_norm, Z_norm) → (r, w) by the 10-step fixed-point iteration of TJ's EFIT writer; R_norm, Z_norm are normalized to R₀.
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

    # TJ's GetPSIvac with Hₙ = Vₙ = 0 for n ≥ 2, in normalized units (callers multiply by psi_scale).
    function psi_vac(r::Float64)
        logr = log(r)
        sum1 = 1.0 - H1ap + H1ap^2
        sum2 = -H1ap * r^2 * logr + 0.5 * r^2 * logr^2 +
               0.5 * (1.0 + H1ap^2) * (r^2 - 1.0)
        return f1a * logr + epsa2 * f3a * logr -
               0.5 * epsa2 * f1a * (-sum1 * logr + sum2)
    end

    # ψ(r) inside the plasma from the ψ-ODE (≈ 0 on axis, psio at r = a), clamped to the data range [p.r0, p.a].
    function psi_plasma_physical(r::Float64)
        r_phys = clamp(r * p.a, p.r0, p.a)
        return ψ_of_r(r_phys)
    end

    # Direct-solver convention: ψ positive at the axis, zero at the LCFS, negative outside.
    # The grid spans R₀ ± rc·a by ±rc·a, a margin for the separatrix finder.
    r_span = rc * a
    psi_in_xs = collect(range(R0 - r_span, R0 + r_span; length=nrbox))
    psi_in_ys = collect(range(-r_span, r_span; length=nzbox))
    psi_rz = zeros(Float64, nrbox, nzbox)

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

    # 1D profile spline in read_efit's 4-column layout, with q₂ consistent with the ψ(R, Z) above.
    psi_norm_grid = range(0.0, 1.0; length=nrbox)
    F_nodes = zeros(nrbox)
    P_nodes = zeros(nrbox)
    q_nodes = zeros(nrbox)
    for i in 1:nrbox
        ψN = psi_norm_grid[i]
        # Solve ψ_plasma(r) = ψN·psio for r (ψ_plasma is monotonic in r).
        target = ψN * psio
        rlocal = if ψN ≤ 0.0
            p.r0
        elseif ψN ≥ 1.0
            p.a
        else
            find_zero(r -> ψ_of_r(r) - target, (p.r0, p.a); atol=1e-10, rtol=1e-12)
        end
        x = rlocal / p.a
        f1 = tj_analytic_f1(x, nu, qc)
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

    # Analytic: ingest=nothing — replay regenerates from the [TJAnalytic] deck section.
    return DirectRunInput(equil_input, sq_in, psi_in, psi_in_xs, psi_in_ys,
        rmin_grid, rmax_grid, zmin_grid, zmax_grid, psio, 1, 1, nothing)
end
