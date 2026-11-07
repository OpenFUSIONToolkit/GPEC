using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium, Plots
using GeneralizedPerturbedEquilibrium: Analysis
isinteractive() ? plotlyjs() : gr()

# Run the deck and plot its results in memory; every plot also takes the path of the gpec.h5 the run wrote.
ret = GeneralizedPerturbedEquilibrium.main([@__DIR__])
display(available_plots(ret))

p_eq = Analysis.Equilibrium.plot_equilibrium_summary(ret.equil)
p_ffs = Analysis.ForceFreeStates.plot_ffs_summary(ret.ffs)

display(p_eq);
Plots.savefig(p_eq, joinpath(@__DIR__, "equilibrium_summary.png"));
display(p_ffs);
Plots.savefig(p_ffs, joinpath(@__DIR__, "ffs_summary.png"));
