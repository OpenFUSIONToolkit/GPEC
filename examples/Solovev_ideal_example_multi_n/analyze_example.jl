using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium, Plots
using GeneralizedPerturbedEquilibrium: Analysis
isinteractive() ? plotlyjs() : gr()

# Analyze the multi-n run
h5path = joinpath(@__DIR__, "gpec.h5")
# Summary plots
p_eq = Analysis.Equilibrium.plot_equilibrium_summary(h5path)
p_ffs = Analysis.ForceFreeStates.plot_ffs_summary(h5path)
p_pe = Analysis.PerturbedEquilibrium.plot_perturbed_equilibrium_summary(h5path)

display(p_eq);
Plots.savefig(p_eq, joinpath(@__DIR__, "equilibrium_summary.png"))
display(p_ffs);
Plots.savefig(p_ffs, joinpath(@__DIR__, "ffs_summary.png"))
display(p_pe);
Plots.savefig(p_pe, joinpath(@__DIR__, "pe_summary.png"))

# Analyze the single-n runs
h5path_n1 = joinpath(@__DIR__, "single_n_1", "euler_n1.h5")
h5path_n2 = joinpath(@__DIR__, "single_n_2", "euler_n2.h5")
p_n1 = Analysis.ForceFreeStates.plot_ffs_summary(h5path_n1)
p_n2 = Analysis.ForceFreeStates.plot_ffs_summary(h5path_n2)

display(p_n1);
Plots.savefig(p_n1, joinpath(@__DIR__, "single_n_1", "ffs_summary.png"))
display(p_n2);
Plots.savefig(p_n2, joinpath(@__DIR__, "single_n_2", "ffs_summary.png"))
