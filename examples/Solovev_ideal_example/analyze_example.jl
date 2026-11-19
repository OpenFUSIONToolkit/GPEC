using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
# The figures need a Makie backend from your global environment: CairoMakie saves files, GLMakie opens windows.
using GeneralizedPerturbedEquilibrium, CairoMakie
using GeneralizedPerturbedEquilibrium: Analysis

# Run the deck and plot its results in memory; every plot also takes the path of the gpec.h5 the run wrote.
ret = GeneralizedPerturbedEquilibrium.main([@__DIR__])
display(available_plots(ret))

display(Analysis.Equilibrium.plot_equilibrium_summary(ret.equil; save_path=joinpath(@__DIR__, "equilibrium_summary.png")))
display(Analysis.ForceFreeStates.plot_ffs_summary(ret.ffs; save_path=joinpath(@__DIR__, "ffs_summary.png")))
