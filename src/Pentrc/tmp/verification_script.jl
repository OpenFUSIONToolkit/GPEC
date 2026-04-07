include(joinpath(@__DIR__, "..", "..", "JPEC.jl"))

using .JPEC
using LinearAlgebra

const P = JPEC.PENTRC

function reference_xintgrnd(x; wn, wt, we, wd, wb, nuk, leff, n,
                            ximag, energy_imaxis, xnufac, xnutype, xf0type, qt)
    cx = energy_imaxis ? im * x : x + im * ximag

    nux =
        if xnutype == "zero"
            0.0 + 0.0im
        elseif xnutype == "small"
            1e-5 * we + 0.0im
        elseif xnutype == "krook"
            nuk + 0.0im
        elseif xnutype == "harmonic"
            x == 0 ? complex(floatmax(Float64), 0.0) :
                     nuk * (1 + 0.25 * leff^2) * cx^(-1.5)
        else
            error("bad xnutype")
        end
    nux *= xnufac

    denom = im * (leff * wb * sqrt(cx) + n * (we + wd * cx)) - nux

    fx =
        if xf0type == "maxwellian"
            (we + wn + wt * (cx - 1.5)) * cx^2.5 * exp(-cx) / denom
        elseif xf0type == "jkp"
            (we + wn + 2 * wt) * cx^2.5 * exp(-cx) / denom
        elseif xf0type == "cgl"
            cx^2.5 * exp(-cx) / (im * n)
        else
            error("bad xf0type")
        end

    if qt
        fx = (cx - 2.5) * fx
    end

    return fx
end

function reference_xintgrl(; wn, wt, we, wd, wb, nuk, ell, leff, n, ximag, xmax,
                            xnufac, xnutype, xf0type, qt, npts=200_000)
    real_part = 0.0 + 0.0im
    imag_part = 0.0 + 0.0im

    if ximag != 0.0
        xs = range(1e-15, ximag; length=npts)
        vals = [reference_xintgrnd(x; wn, wt, we, wd, wb, nuk, leff, n,
                                   ximag=ximag, energy_imaxis=true,
                                   xnufac=xnufac, xnutype=xnutype, xf0type=xf0type, qt=qt) for x in xs]
        for i in 1:length(xs)-1
            dx = xs[i + 1] - xs[i]
            imag_part += 0.5 * dx * (vals[i] + vals[i + 1])
        end
    end

    xs = range(1e-15, xmax; length=npts)
    vals = [reference_xintgrnd(x; wn, wt, we, wd, wb, nuk, leff, n,
                               ximag=ximag, energy_imaxis=false,
                               xnufac=xnufac, xnutype=xnutype, xf0type=xf0type, qt=qt) for x in xs]
    for i in 1:length(xs)-1
        dx = xs[i + 1] - xs[i]
        real_part += 0.5 * dx * (vals[i] + vals[i + 1])
    end

    return real_part + imag_part
end

function eval_current_xintgrnd(x)
    ydot = zeros(2)
    P.xintgrnd!(ydot, zeros(2), nothing, x)
    return complex(ydot[1], ydot[2])
end

function relerr(a, b)
    denom = max(abs(b), 1e-30)
    return abs(a - b) / denom
end

function make_test_flambda()
    λ = collect(range(0.25, 0.85; length=41))
    fs = zeros(ComplexF64, length(λ), 5)
    for (i, x) in enumerate(λ)
        fs[i, 1] = 0.8 + 0.4 * cos(2π * x)
        fs[i, 2] = -0.2 + 0.6 * x
        fs[i, 3] = (1.0 + 0.3im) * (1.2 - x)
        fs[i, 4] = (0.5 - 0.4im) * x^2
        fs[i, 5] = (0.2 + 0.1im) * (1 - x)^2
    end
    return P.Spl.CubicSpline(λ, fs; bctype="extrap")
end

function make_test_turns()
    λ = collect(range(0.25, 0.85; length=41))
    fs = zeros(Float64, length(λ), 6)
    for (i, x) in enumerate(λ)
        fs[i, :] .= (
            x,
            x^2,
            sinpi(x),
            cospi(x),
            1.0 - x,
            x * (1.0 - x),
        )
    end
    return P.Spl.CubicSpline(λ, fs; bctype="extrap")
end

function reference_lintgrnd(lmda, flambda; wn, wt, we, nuk, bobmax, epsr, q, ell, n, rex, imx, psi, method)
    flambda_f = P.Spl.spline_eval!(flambda, lmda)
    wb = real(flambda_f[1])
    wd = real(flambda_f[2])

    xint =
        if lmda <= bobmax
            nueff = nuk
            lnq = ell + n * q
            P.xintgrl_lsode(wn, wt, we, wd, wb, nueff, ell, lnq, n, psi, lmda, method; op_record=false) +
            P.xintgrl_lsode(wn, wt, we, wd, -wb, nueff, ell, lnq, n, psi, lmda, method; op_record=false)
        else
            nueff = nuk / (2 * epsr)
            lnq = Float64(ell)
            P.xintgrl_lsode(wn, wt, we, wd, wb, nueff, ell, lnq, n, psi, lmda, method; op_record=false)
        end

    ydot = zeros(Float64, 2 * (flambda.nqty - 2))
    for i in 3:flambda.nqty
        fres = flambda_f[i] * (rex * real(xint) + 1im * imx * imag(xint))
        ydot[1 + 2 * (i - 3)] = real(fres)
        ydot[2 + 2 * (i - 3)] = imag(fres)
    end
    return ydot
end

function reference_lambdaintgrl(flambda; wn, wt, we, nuk, bobmax, epsr, q, ell, n, rex, imx, psi, method, npts=400)
    neq = 2 * (flambda.nqty - 2)
    npts = iseven(npts) ? npts + 1 : npts
    λs = collect(range(flambda.xs[1], flambda.xs[end]; length=npts))
    vals = [reference_lintgrnd(λ, flambda; wn, wt, we, nuk, bobmax, epsr, q, ell, n, rex, imx, psi, method) for λ in λs]
    y = zeros(Float64, neq)
    h = λs[2] - λs[1]
    for i in 1:2:length(λs)-2
        y .+= (h / 3.0) .* (vals[i] .+ 4 .* vals[i + 1] .+ vals[i + 2])
    end
    return ComplexF64.(y[1:2:end], y[2:2:end])
end

function synthetic_surface_fun(psi, n, l, zi, mi, wdfac, divxfac, electron, method)
    sgn = electron ? -1.0 : 1.0
    amp = sgn * (1.0 + 0.05 * zi + 0.01 * mi) * (1.0 + 0.12 * abs(l) + 0.03 * n)
    realp = amp * ((1.0 + wdfac * psi) * cospi((l + n + 1) * psi) + 0.1 * length(method))
    imagp = amp * (divxfac * (1.0 - psi + 0.15 * l) + 0.05 * sinpi((l + 1) * psi))
    return complex(realp, imagp)
end

function reference_tintgrl(psilims; nn, nl, zi, mi, wdfac, divxfac, electron, method, npts=20001)
    ψ0, ψ1 = P._effective_psi_interval(psilims)
    if ψ0 == ψ1
        return zeros(ComplexF64, 2 * nl + 1)
    end

    ψs = collect(range(ψ0, ψ1; length=npts))
    out = zeros(ComplexF64, 2 * nl + 1)
    for (idx, l) in enumerate(-nl:nl)
        vals = [synthetic_surface_fun(ψ, nn, l, zi, mi, wdfac, divxfac, electron, method) for ψ in ψs]
        for i in 1:length(ψs)-1
            dψ = ψs[i + 1] - ψs[i]
            out[idx] += 0.5 * dψ * (vals[i] + vals[i + 1])
        end
    end
    return out
end

function main()
    outdir = @__DIR__

    cases = [
        (
            name = "case1_maxwellian_harmonic",
            wn = 1.2e3, wt = -4.5e2, we = 2.1e3, wd = -3.0e1, wb = 7.5e2,
            nuk = 2.5, ell = 1, leff = 1.75, n = 3, ximag = 0.0,
            xnutype = "harmonic", xf0type = "maxwellian", qt = false
        ),
        (
            name = "case2_jkp_krook_imag",
            wn = -8.0e2, wt = 3.0e2, we = 1.1e3, wd = 2.5e1, wb = -6.0e2,
            nuk = 1.8, ell = 2, leff = -0.5, n = 2, ximag = 0.35,
            xnutype = "krook", xf0type = "jkp", qt = true
        ),
        (
            name = "case3_cgl_zero",
            wn = 0.0, wt = 0.0, we = 6.0e2, wd = 4.0, wb = 2.0e2,
            nuk = 0.0, ell = 0, leff = 0.0, n = 1, ximag = 0.0,
            xnutype = "zero", xf0type = "cgl", qt = false
        ),
    ]

    report_lines = String[
        "# Xint Verification",
        "",
        "Reference model: line-by-line translation of `src/Pentrc/energy.f90` for `xintgrnd` and contour-integral form of `xintgrl_lsode`.",
        "",
        "Julia solver used for `xintgrl_lsode`: `OrdinaryDiffEq.Tsit5()`",
        "",
        "This file is the single bundled artifact for the `xintgrnd!` / `xintgrl_lsode` comparison.",
        "",
        "## Summary",
        "",
        "| case | xintgrnd max abs err | xintgrnd mean abs err | xintgrl current | xintgrl ref | rel err |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for case in cases
        P.ximag = case.ximag
        P.xnufac = 1.0
        P.xnutype = case.xnutype
        P.xf0type = case.xf0type
        global P.qt = case.qt
        P.set_energy_var!(:energy_wn, case.wn)
        P.set_energy_var!(:energy_wt, case.wt)
        P.set_energy_var!(:energy_we, case.we)
        P.set_energy_var!(:energy_wd, case.wd)
        P.set_energy_var!(:energy_wb, case.wb)
        P.set_energy_var!(:energy_nuk, case.nuk)
        P.set_energy_var!(:energy_leff, case.leff)
        P.set_energy_var!(:energy_n, case.n)

        xs = collect(range(1e-6, 12.0; length=200))
        max_abs_err = 0.0
        mean_abs_err = 0.0
        sample_rows = String[
            "",
            "### $(case.name)",
            "",
            "| x | current(real) | current(imag) | ref(real) | ref(imag) | abs err |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for x in xs
            P.energy_imaxis = false
            current = eval_current_xintgrnd(x)
            ref = reference_xintgrnd(x; wn=case.wn, wt=case.wt, we=case.we, wd=case.wd, wb=case.wb,
                                     nuk=case.nuk, leff=case.leff, n=case.n, ximag=case.ximag,
                                     energy_imaxis=false, xnufac=P.xnufac, xnutype=case.xnutype,
                                     xf0type=case.xf0type, qt=case.qt)
            err = abs(current - ref)
            max_abs_err = max(max_abs_err, err)
            mean_abs_err += err

            if length(sample_rows) < 11
                push!(sample_rows,
                      "| $(x) | $(real(current)) | $(imag(current)) | $(real(ref)) | $(imag(ref)) | $(err) |")
            end
        end
        mean_abs_err /= length(xs)

        current_int = P.xintgrl_lsode(case.wn, case.wt, case.we, case.wd, case.wb, case.nuk,
                                      case.ell, case.leff, case.n, 0.4, 0.3, "rlar")
        ref_int = reference_xintgrl(wn=case.wn, wt=case.wt, we=case.we, wd=case.wd, wb=case.wb,
                                    nuk=case.nuk, ell=case.ell, leff=case.leff, n=case.n,
                                    ximag=case.ximag, xmax=P.xmax, xnufac=P.xnufac,
                                    xnutype=case.xnutype, xf0type=case.xf0type, qt=case.qt)

        rel = relerr(current_int, ref_int)
        push!(report_lines, "| $(case.name) | $(max_abs_err) | $(mean_abs_err) | $(current_int) | $(ref_int) | $(rel) |")
        append!(report_lines, sample_rows)
        push!(report_lines, "")
        push!(report_lines, "- parameters:")
        push!(report_lines, "  - `xnutype=$(case.xnutype)`, `xf0type=$(case.xf0type)`, `qt=$(case.qt)`, `ximag=$(case.ximag)`")
        push!(report_lines, "  - `wn=$(case.wn)`, `wt=$(case.wt)`, `we=$(case.we)`, `wd=$(case.wd)`, `wb=$(case.wb)`, `nuk=$(case.nuk)`")
        push!(report_lines, "  - `ell=$(case.ell)`, `leff=$(case.leff)`, `n=$(case.n)`")
        push!(report_lines, "")
    end

    open(joinpath(outdir, "xint_verification.md"), "w") do io
        println(io, join(report_lines, "\n"))
    end

    flambda = make_test_flambda()
    turns = make_test_turns()
    lambda_case = (
        wn = 8.0e2, wt = -1.5e2, we = 1.3e3, nuk = 0.9,
        bobmax = 0.53, epsr = 0.18, q = 2.4, ell = 1, n = 2,
        rex = 1.0, imx = 1.0, psi = 0.42, method = "clar"
    )

    P.ximag = 0.15
    P.xnufac = 1.0
    P.xnutype = "harmonic"
    P.xf0type = "maxwellian"
    global P.qt = false

    P.set_pitch_var!(:pitch_wn, lambda_case.wn)
    P.set_pitch_var!(:pitch_wt, lambda_case.wt)
    P.set_pitch_var!(:pitch_we, lambda_case.we)
    P.set_pitch_var!(:pitch_nuk, lambda_case.nuk)
    P.set_pitch_var!(:pitch_bobmax, lambda_case.bobmax)
    P.set_pitch_var!(:pitch_epsr, lambda_case.epsr)
    P.set_pitch_var!(:pitch_q, lambda_case.q)
    P.set_pitch_var!(:pitch_ell, lambda_case.ell)
    P.set_pitch_var!(:pitch_n, lambda_case.n)
    P.set_pitch_var!(:pitch_rex, lambda_case.rex)
    P.set_pitch_var!(:pitch_imx, lambda_case.imx)
    P.set_pitch_var!(:pitch_psi, lambda_case.psi)
    P.set_pitch_var!(:pitch_method, lambda_case.method)
    P.set_pitch_var!(:pitch_flambda, flambda)
    P.set_pitch_var!(:pitch_turns, turns)

    sample_λ = collect(range(flambda.xs[1], flambda.xs[end]; length=8))
    max_lint_err = 0.0
    mean_lint_err = 0.0
    lambda_lines = String[
        "# Lambda Verification",
        "",
        "Reference model: direct source translation of `lintgrnd` from `src/Pentrc/pitch.f90`, using the already-verified Julia `xintgrl_lsode` for the energy-space sub-integral.",
        "",
        "Julia solver used for `lambdaintgrl_lsode`: `OrdinaryDiffEq.Tsit5()`",
        "",
        "| lambda | lint current norm | lint ref norm | lint abs err |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for λ in sample_λ
        cur = zeros(Float64, 2 * (flambda.nqty - 2))
        P.lintgrnd!(cur, zeros(length(cur)), nothing, λ)
        ref = reference_lintgrnd(λ, flambda;
                                 wn=lambda_case.wn, wt=lambda_case.wt, we=lambda_case.we,
                                 nuk=lambda_case.nuk, bobmax=lambda_case.bobmax, epsr=lambda_case.epsr,
                                 q=lambda_case.q, ell=lambda_case.ell, n=lambda_case.n,
                                 rex=lambda_case.rex, imx=lambda_case.imx, psi=lambda_case.psi,
                                 method=lambda_case.method)
        err = norm(cur .- ref)
        max_lint_err = max(max_lint_err, err)
        mean_lint_err += err
        push!(lambda_lines, "| $(λ) | $(norm(cur)) | $(norm(ref)) | $(err) |")
    end
    mean_lint_err /= length(sample_λ)

    current_lambda = P.lambdaintgrl_lsode(lambda_case.wn, lambda_case.wt, lambda_case.we, lambda_case.nuk,
                                          lambda_case.bobmax, lambda_case.epsr, lambda_case.q,
                                          flambda, lambda_case.ell, lambda_case.n,
                                          lambda_case.rex, lambda_case.imx, lambda_case.psi,
                                          turns, lambda_case.method; op_record=false)
    ref_lambda = reference_lambdaintgrl(flambda;
                                        wn=lambda_case.wn, wt=lambda_case.wt, we=lambda_case.we,
                                        nuk=lambda_case.nuk, bobmax=lambda_case.bobmax, epsr=lambda_case.epsr,
                                        q=lambda_case.q, ell=lambda_case.ell, n=lambda_case.n,
                                        rex=lambda_case.rex, imx=lambda_case.imx, psi=lambda_case.psi,
                                        method=lambda_case.method)

    push!(lambda_lines, "")
    push!(lambda_lines, "## Integral Comparison")
    push!(lambda_lines, "")
    push!(lambda_lines, "- `lintgrnd!` max sample error: `$(max_lint_err)`")
    push!(lambda_lines, "- `lintgrnd!` mean sample error: `$(mean_lint_err)`")
    push!(lambda_lines, "- `lambdaintgrl_lsode` current: `$(current_lambda)`")
    push!(lambda_lines, "- `lambdaintgrl_lsode` ref: `$(ref_lambda)`")
    push!(lambda_lines, "- vector norm abs err: `$(norm(current_lambda .- ref_lambda))`")
    push!(lambda_lines, "- vector norm rel err: `$(norm(current_lambda .- ref_lambda) / max(norm(ref_lambda), 1e-30))`")
    push!(lambda_lines, "")
    push!(lambda_lines, "## Parameters")
    push!(lambda_lines, "")
    push!(lambda_lines, "- `wn=$(lambda_case.wn)`, `wt=$(lambda_case.wt)`, `we=$(lambda_case.we)`, `nuk=$(lambda_case.nuk)`")
    push!(lambda_lines, "- `bobmax=$(lambda_case.bobmax)`, `epsr=$(lambda_case.epsr)`, `q=$(lambda_case.q)`")
    push!(lambda_lines, "- `ell=$(lambda_case.ell)`, `n=$(lambda_case.n)`, `rex=$(lambda_case.rex)`, `imx=$(lambda_case.imx)`")
    push!(lambda_lines, "- `psi=$(lambda_case.psi)`, `method=$(lambda_case.method)`")

    open(joinpath(outdir, "lambda_verification.md"), "w") do io
        println(io, join(lambda_lines, "\n"))
    end

    old_sq = P.sq
    old_xs_m = copy(P.xs_m)
    try
        P.sq = P.Spl.CubicSpline([0.18, 0.46, 0.92], [1.0, 1.0, 1.0]; bctype="extrap")
        P.xs_m = Any[
            P.Spl.CubicSpline([0.12, 0.44, 0.84], [0.0, 0.0, 0.0]; bctype="extrap"),
            nothing,
            nothing,
        ]

        tint_case = (
            psilims = [0.08, 0.95],
            nn = 3, nl = 2, zi = 1, mi = 2,
            wdfac = 0.35, divxfac = -0.22, electron = false, method = "clar",
        )

        current_components = P._tintgrl_lsode_impl(tint_case.psilims, tint_case.nn, tint_case.nl,
                                                   tint_case.zi, tint_case.mi, tint_case.wdfac,
                                                   tint_case.divxfac, tint_case.electron, tint_case.method;
                                                   surface_fun=synthetic_surface_fun,
                                                   check_inputs=false)
        ref_components = reference_tintgrl(tint_case.psilims;
                                           nn=tint_case.nn, nl=tint_case.nl,
                                           zi=tint_case.zi, mi=tint_case.mi,
                                           wdfac=tint_case.wdfac, divxfac=tint_case.divxfac,
                                           electron=tint_case.electron, method=tint_case.method)

        tint_lines = String[
            "# Tint Verification",
            "",
            "Reference model: direct translation of the `tintgrl_lsode` wrapper behavior in `src/Pentrc/torque.F90`, with a synthetic smooth `tpsi` surrogate and the same clipped psi interval logic used by the Julia port.",
            "",
            "Julia solver used for `tintgrl_lsode`: `OrdinaryDiffEq.Tsit5()`",
            "",
            "## Effective Interval",
            "",
            "- requested `psilims=$(tint_case.psilims)`",
            "- clipped interval `$(P._effective_psi_interval(tint_case.psilims))`",
            "- `sq` interval `($(P.sq.xs[1]), $(P.sq.xs[end]))`",
            "- `xs_m[1]` interval `($(P.xs_m[1].xs[1]), $(P.xs_m[1].xs[end]))`",
            "",
            "## Harmonic Comparison",
            "",
            "| l | current | ref | abs err | rel err |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]

        for (idx, l) in enumerate(-tint_case.nl:tint_case.nl)
            cur = current_components[idx]
            ref = ref_components[idx]
            push!(tint_lines, "| $(l) | $(cur) | $(ref) | $(abs(cur - ref)) | $(relerr(cur, ref)) |")
        end

        total_current = sum(current_components)
        total_ref = sum(ref_components)
        push!(tint_lines, "")
        push!(tint_lines, "## Total Comparison")
        push!(tint_lines, "")
        push!(tint_lines, "- current total: `$(total_current)`")
        push!(tint_lines, "- ref total: `$(total_ref)`")
        push!(tint_lines, "- component vector abs err: `$(norm(current_components .- ref_components))`")
        push!(tint_lines, "- component vector rel err: `$(norm(current_components .- ref_components) / max(norm(ref_components), 1e-30))`")
        push!(tint_lines, "- total rel err: `$(relerr(total_current, total_ref))`")
        push!(tint_lines, "")
        push!(tint_lines, "## Parameters")
        push!(tint_lines, "")
        push!(tint_lines, "- `nn=$(tint_case.nn)`, `nl=$(tint_case.nl)`, `zi=$(tint_case.zi)`, `mi=$(tint_case.mi)`")
        push!(tint_lines, "- `wdfac=$(tint_case.wdfac)`, `divxfac=$(tint_case.divxfac)`, `electron=$(tint_case.electron)`")
        push!(tint_lines, "- `method=$(tint_case.method)`")
        push!(tint_lines, "- tolerances: `abstol=$(P.atol_psi)`, `reltol=$(P.rtol_psi)`")

        open(joinpath(outdir, "tint_verification.md"), "w") do io
            println(io, join(tint_lines, "\n"))
        end
    finally
        P.sq = old_sq
        P.xs_m = old_xs_m
    end
end

main()
