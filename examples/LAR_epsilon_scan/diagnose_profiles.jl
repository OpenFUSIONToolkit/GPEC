#!/usr/bin/env julia
"""
Diagnose LAR equilibrium profiles: P, P', FF', q, dV/dpsi vs psi_N.

Generates overlay plots comparing Julia LAR analytic equilibria against
TJ geqdsk-based equilibria (from the archive branch) at several epsilon values.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium: LargeAspectRatioConfig, EquilibriumConfig, setup_equilibrium
using Printf
using Plots

# ============================================================================
# Generate LAR equilibria at several epsilon values
# ============================================================================

function make_lar_equil(epsilon; p_sig=1.5, beta0=1e-3)
    lar = LargeAspectRatioConfig(;
        lar_r0=1.0/epsilon, lar_a=1.0, beta0=beta0,
        q0=1.5, p_pres=2.0, p_sig=p_sig,
        sigma_type="wesson", ma=128, mtau=128,
    )
    eq = EquilibriumConfig(; eq_type="lar", psilow=0.01, psihigh=0.995, mpsi=128, mtheta=512)
    return setup_equilibrium(eq, lar)
end

function make_tj_equil(epsilon)
    # Extract geqdsk from archive branch
    fname = "TJ_epsilon_scan_$(epsilon).geqdsk"
    tmpfile = joinpath(tempdir(), fname)
    run(pipeline(`git show perf/riccati-full-geqdsk-scans:examples/LAR_epsilon_scan/equilibria/$fname`, stdout=tmpfile))
    eq = EquilibriumConfig(; eq_type="efit", eq_filename=tmpfile,
        psilow=0.01, psihigh=0.995, mpsi=128, mtheta=512)
    equil = setup_equilibrium(eq)
    rm(tmpfile; force=true)
    return equil
end

function extract_profiles(equil)
    xs = equil.profiles.xs
    n = length(xs)
    q = [equil.profiles.q_spline(x) for x in xs]
    F = [equil.profiles.F_spline(x) for x in xs]
    P = [equil.profiles.P_spline(x) for x in xs]
    dVdpsi = [equil.profiles.dVdpsi_spline(x) for x in xs]
    q_deriv = [equil.profiles.q_deriv(x) for x in xs]
    F_deriv = [equil.profiles.F_deriv(x) for x in xs]
    P_deriv = [equil.profiles.P_deriv(x) for x in xs]

    # FF' = F * dF/dpsi (toroidal field function derivative)
    FFp = F .* F_deriv

    return (xs=xs, q=q, F=F, P=P, dVdpsi=dVdpsi,
            q_deriv=q_deriv, F_deriv=F_deriv, P_deriv=P_deriv, FFp=FFp)
end

# ============================================================================
# Main: generate profile comparison figures
# ============================================================================

function main()
    epsilons = [0.2495, 0.4072, 0.5510]
    p_sigs = Dict{Float64,Float64}()

    # First, find p_sig for each epsilon
    @info "Finding p_sig for each epsilon..."
    for eps in epsilons
        for p_sig in range(0.5, 5.0; length=20)
            equil = make_lar_equil(eps; p_sig=p_sig)
            if abs(equil.params.qmax - 3.6) < 0.1
                p_sigs[eps] = p_sig
                @printf("  ε=%.4f: p_sig=%.3f → qmax=%.3f\n", eps, p_sig, equil.params.qmax)
                break
            end
        end
    end

    # Generate profiles for each epsilon
    fig_q = plot(; xlabel="ψ_N", ylabel="q", title="Safety Factor Profile", legend=:topleft, left_margin=12Plots.mm)
    fig_P = plot(; xlabel="ψ_N", ylabel="P (μ₀P)", title="Pressure Profile", legend=:topright, left_margin=12Plots.mm)
    fig_Pp = plot(; xlabel="ψ_N", ylabel="P' = dP/dψ", title="Pressure Gradient", legend=:bottomright, left_margin=12Plots.mm)
    fig_FFp = plot(; xlabel="ψ_N", ylabel="FF'", title="FF' Profile", legend=:topleft, left_margin=12Plots.mm)
    fig_dV = plot(; xlabel="ψ_N", ylabel="dV/dψ", title="Volume Element", legend=:topleft, left_margin=12Plots.mm)
    fig_F = plot(; xlabel="ψ_N", ylabel="F = R·Bφ", title="Toroidal Field Function", legend=:topleft, left_margin=12Plots.mm)

    colors = [:blue, :red, :green]

    for (i, eps) in enumerate(epsilons)
        p_sig = get(p_sigs, eps, 1.5)
        lar_equil = make_lar_equil(eps; p_sig=p_sig)
        lar = extract_profiles(lar_equil)

        # Try to load TJ geqdsk
        tj = nothing
        try
            tj_equil = make_tj_equil(eps)
            tj = extract_profiles(tj_equil)
        catch e
            @warn "Could not load TJ geqdsk for ε=$eps: $e"
        end

        c = colors[i]
        label_lar = "LAR ε=$(eps)"
        label_tj = "TJ ε=$(eps)"

        plot!(fig_q, lar.xs, lar.q; label=label_lar, lw=2, color=c)
        plot!(fig_P, lar.xs, lar.P; label=label_lar, lw=2, color=c)
        plot!(fig_Pp, lar.xs, lar.P_deriv; label=label_lar, lw=2, color=c)
        plot!(fig_FFp, lar.xs, lar.FFp; label=label_lar, lw=2, color=c)
        plot!(fig_dV, lar.xs, lar.dVdpsi; label=label_lar, lw=2, color=c)
        plot!(fig_F, lar.xs, lar.F; label=label_lar, lw=2, color=c)

        if tj !== nothing
            plot!(fig_q, tj.xs, tj.q; label=label_tj, lw=1.5, ls=:dash, color=c)
            plot!(fig_P, tj.xs, tj.P; label=label_tj, lw=1.5, ls=:dash, color=c)
            plot!(fig_Pp, tj.xs, tj.P_deriv; label=label_tj, lw=1.5, ls=:dash, color=c)
            plot!(fig_FFp, tj.xs, tj.FFp; label=label_tj, lw=1.5, ls=:dash, color=c)
            plot!(fig_dV, tj.xs, tj.dVdpsi; label=label_tj, lw=1.5, ls=:dash, color=c)
            plot!(fig_F, tj.xs, tj.F; label=label_tj, lw=1.5, ls=:dash, color=c)
        end
    end

    # Combine into a single figure
    fig = plot(fig_q, fig_P, fig_Pp, fig_FFp, fig_dV, fig_F;
        layout=(2, 3), size=(1500, 800),
        plot_title="LAR Equilibrium Profiles: Julia (solid) vs TJ (dashed)")

    outfile = joinpath(@__DIR__, "profile_diagnostics.png")
    savefig(fig, outfile)
    @info "Figure saved to $outfile"
    println(outfile)
end

main()
