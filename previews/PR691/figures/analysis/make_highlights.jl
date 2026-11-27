# make_highlights.jl
#
# Figures: the highlight Analysis figures on the Analysis docs page, from the
# DIII-D-like ideal example run with the Forward integrator and the C-coil
# perturbed equilibrium (its kinetic and error-field stages left out). About two
# minutes.
#
# Run manually, with CairoMakie in the global environment:
#     julia --project=. docs/src/figures/analysis/make_highlights.jl

include(joinpath(@__DIR__, "..", "..", "..", "figure_tools.jl"))

using GeneralizedPerturbedEquilibrium

const G = GeneralizedPerturbedEquilibrium
const A = G.Analysis

ret = mktempdir() do dir
    src = joinpath(pkgdir(G), "examples", "DIIID-like_ideal_example")
    for name in readdir(src)
        endswith(name, ".jl") || name == "benchmark_inputs.jld2" || cp(joinpath(src, name), joinpath(dir, name))
    end
    deck = TOML.parsefile(joinpath(dir, "gpec.toml"))
    delete!(deck, "KineticForces")
    delete!(deck, "ErrorFields")
    open(io -> TOML.print(io, deck), joinpath(dir, "gpec.toml"), "w")
    G.main([dir])
end

depends = ["src/Analysis/Equilibrium.jl", "src/Analysis/ForceFreeStates.jl", "src/Analysis/PerturbedEquilibrium.jl", "src/Analysis/CoilForcing.jl"]
for (fig, name) in (
    (A.Equilibrium.plot_equilibrium_summary(ret.equil), "equilibrium_summary"),
    (A.ForceFreeStates.plot_ffs_summary(ret.ffs), "ffs_summary"),
    (A.PerturbedEquilibrium.plot_perturbed_equilibrium_summary(ret.pe), "pe_summary"),
    (A.CoilForcing.plot_coil_geometry_3d(ret.pe), "coils_3d"))
    save_doc_figure(fig, "analysis", name; script=basename(@__FILE__), depends)
end
