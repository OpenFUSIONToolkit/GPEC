using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium, Plots
using GeneralizedPerturbedEquilibrium: Analysis
isinteractive() ? plotlyjs() : gr()

h5path = joinpath(@__DIR__, "gpec.h5")

# Summary plots
p_eq = Analysis.Equilibrium.plot_equilibrium_summary(h5path)
p_ffs = Analysis.ForceFreeStates.plot_ffs_summary(h5path)
p_pe  = Analysis.PerturbedEquilibrium.plot_perturbed_equilibrium_summary(h5path)

display(p_eq);  Plots.savefig(p_eq,  joinpath(@__DIR__, "equilibrium_summary.png"))
display(p_ffs); Plots.savefig(p_ffs, joinpath(@__DIR__, "ffs_summary.png"))
display(p_pe);  Plots.savefig(p_pe,  joinpath(@__DIR__, "pe_summary.png"))