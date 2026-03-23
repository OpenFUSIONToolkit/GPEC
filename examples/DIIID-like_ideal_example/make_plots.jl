using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium, Plots, GeneralizedPerturbedEquilibrium.Analysis
gr()

h5path = joinpath(@__DIR__, "gpec.h5")
outdir = joinpath(@__DIR__, "plots")
mkpath(outdir)

# --- Equilibrium ---
p = Analysis.Equilibrium.plot_equilibrium_summary(h5path)
savefig(p, joinpath(outdir, "equil_summary.png"))

p = Analysis.Equilibrium.plot_qprofile(h5path)
savefig(p, joinpath(outdir, "equil_qprofile.png"))

p = Analysis.Equilibrium.plot_pressure_profile(h5path)
savefig(p, joinpath(outdir, "equil_pressure.png"))

p = Analysis.Equilibrium.plot_f_profile(h5path)
savefig(p, joinpath(outdir, "equil_F.png"))

# --- ForceFreeStates ---
p = Analysis.ForceFreeStates.plot_mode_displacement(h5path; modes=1:5)
savefig(p, joinpath(outdir, "ffs_mode_displacement.png"))

p = Analysis.ForceFreeStates.plot_eigenmode_summary(h5path)
savefig(p, joinpath(outdir, "ffs_eigenmode_summary.png"))

p = Analysis.ForceFreeStates.plot_stability_criterion(h5path)
savefig(p, joinpath(outdir, "ffs_stability_criterion.png"))

p = Analysis.ForceFreeStates.plot_energy_eigenvectors(h5path)
savefig(p, joinpath(outdir, "ffs_energy_eigenvectors.png"))

p = Analysis.ForceFreeStates.plot_eigenvalue_spectrum(h5path)
savefig(p, joinpath(outdir, "ffs_eigenvalue_spectrum.png"))

p = Analysis.ForceFreeStates.plot_delta_prime(h5path)
savefig(p, joinpath(outdir, "ffs_delta_prime.png"))

p = Analysis.ForceFreeStates.plot_dcon_summary(h5path)
savefig(p, joinpath(outdir, "ffs_dcon_summary.png"))

p = Analysis.ForceFreeStates.plot_singular_surfaces(h5path)
savefig(p, joinpath(outdir, "ffs_singular_surfaces.png"))

# --- PerturbedEquilibrium ---
p = Analysis.PerturbedEquilibrium.plot_resonant_flux(h5path)
savefig(p, joinpath(outdir, "pe_resonant_flux.png"))

p = Analysis.PerturbedEquilibrium.plot_island_widths(h5path)
savefig(p, joinpath(outdir, "pe_island_widths.png"))

p = Analysis.PerturbedEquilibrium.plot_chirikov_parameter(h5path)
savefig(p, joinpath(outdir, "pe_chirikov.png"))

p = Analysis.PerturbedEquilibrium.plot_pe_delta_prime(h5path)
savefig(p, joinpath(outdir, "pe_delta_prime.png"))

p = Analysis.PerturbedEquilibrium.plot_resonant_field(h5path)
savefig(p, joinpath(outdir, "pe_resonant_field.png"))

p = Analysis.PerturbedEquilibrium.plot_mode_spectrogram(h5path; component=:xi_psi)
savefig(p, joinpath(outdir, "pe_spectrogram_xi_psi.png"))

p = Analysis.PerturbedEquilibrium.plot_mode_spectrogram(h5path; component=:b_psi)
savefig(p, joinpath(outdir, "pe_spectrogram_b_psi.png"))

p = Analysis.PerturbedEquilibrium.plot_perturbed_equilibrium_summary(h5path)
savefig(p, joinpath(outdir, "pe_summary.png"))

println("All plots saved to: $outdir")
