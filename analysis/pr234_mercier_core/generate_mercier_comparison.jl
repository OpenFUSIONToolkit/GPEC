using GeneralizedPerturbedEquilibrium
using TOML
using Printf
using FastInterpolations

const GPEC = GeneralizedPerturbedEquilibrium

include("restored_Mercier.jl")

repo_root = normpath(joinpath(@__DIR__, "..", ".."))
example_dir = joinpath(repo_root, "examples", "DIIID-like_ideal_example")
inputs = TOML.parsefile(joinpath(example_dir, "gpec.toml"))
eq_config = GPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], example_dir)
equil = GPEC.Equilibrium.setup_equilibrium(eq_config)

xs = equil.profiles.xs
old_locstab = zeros(Float64, length(xs), 5)
restored_mercier_scan!(old_locstab, equil)

out_csv = joinpath(@__DIR__, "mercier_comparison.csv")
open(out_csv, "w") do io
    println(io, "i,psi,q,qprime,pprime,old_mercier_di,old_mercier_dr,old_mercier_h,current_resistive_di,current_resistive_dr,current_resistive_h,det_d0bar,ballooning_delta_prime,old_current_di_diff,old_current_di_rel_diff,det_old_di_diff,det_old_di_rel_diff")
    for i in eachindex(xs)
        psi = xs[i]
        psi <= 1.0 || continue

        old_di = old_locstab[i, 1] / psi
        old_dr = old_locstab[i, 2] / psi
        old_h = old_locstab[i, 3]

        current = GPEC.ForceFreeStates.resistive_interchange(i, equil)
        coeff = GPEC.ForceFreeStates.prepare_ballooning_coefficients(i, equil)
        ballooning_delta_prime = try
            GPEC.ForceFreeStates.ballooning_delta_prime(i, equil).delta_prime
        catch
            NaN
        end

        old_current_diff = old_di - current.di
        old_current_rel = abs(old_current_diff) / (abs(old_di) + 1e-14)
        det_old_diff = coeff.di - old_di
        det_old_rel = abs(det_old_diff) / (abs(old_di) + 1e-14)

        @printf(
            io,
            "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            i,
            psi,
            equil.profiles.q_spline.y[i],
            equil.profiles.q_deriv(psi),
            equil.profiles.P_deriv(psi),
            old_di,
            old_dr,
            old_h,
            current.di,
            current.dr,
            current.h,
            coeff.di,
            ballooning_delta_prime,
            old_current_diff,
            old_current_rel,
            det_old_diff,
            det_old_rel,
        )
    end
end

println(out_csv)
