using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
# The figures need a Makie backend from your global environment: CairoMakie saves files, GLMakie opens windows.
using GeneralizedPerturbedEquilibrium, CairoMakie
using GeneralizedPerturbedEquilibrium: Analysis

# Analyze the multi-n run
h5path = joinpath(@__DIR__, "gpec.h5")
display(Analysis.Equilibrium.plot_equilibrium_summary(h5path; save_path=joinpath(@__DIR__, "equilibrium_summary.png")))
display(Analysis.ForceFreeStates.plot_ffs_summary(h5path; save_path=joinpath(@__DIR__, "ffs_summary.png")))
display(Analysis.PerturbedEquilibrium.plot_perturbed_equilibrium_summary(h5path; save_path=joinpath(@__DIR__, "pe_summary.png")))

# Analyze the single-n runs
for n in 1:2
    dir = joinpath(@__DIR__, "single_n_$n")
    display(Analysis.ForceFreeStates.plot_ffs_summary(joinpath(dir, "euler_n$n.h5"); save_path=joinpath(dir, "ffs_summary.png")))
end
