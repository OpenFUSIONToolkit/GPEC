using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
# The figures need a Makie backend from your global environment: CairoMakie saves files, GLMakie opens windows.
using GeneralizedPerturbedEquilibrium, CairoMakie, Printf, HDF5
using GeneralizedPerturbedEquilibrium: PerturbedEquilibrium, ErrorFields
AEF = GeneralizedPerturbedEquilibrium.Analysis.ErrorFields

h5path = joinpath(@__DIR__, "gpec.h5")
# Show a figure and save it beside this script.
function show_and_save(fig, name)
    display(fig)
    path = joinpath(@__DIR__, name * ".png")
    save(path, fig)
    println("Saved: ", path)
    return fig
end

# The run stored every coil set's spectrum linearization. Project it onto the dominant
# resonant-coupling mode over the core window (ψ_N ≤ 0.9, the default) to see which coils matter.
rc = PerturbedEquilibrium.ResonantCoupling(h5path)
sens = ErrorFields.CoilSensitivities(h5path)
full = ErrorFields.sensitivity_table(sens, PerturbedEquilibrium.dominant_coupling(rc))

names = full.coil_names
fcoils = findall(startswith("F"), names)

println("coil       |δ_nom|    per mm of shift   per mm of rim    curvature")
for (j, nm) in enumerate(names)
    @printf("%-8s %10.3e   %13.3e   %13.3e   %9.1e\n", nm, abs(full.delta_as_designed[j]), full.abs_delta_shift_per_mm[j],
        full.abs_delta_rim_per_mm[j], max(maximum(sens.shift_linearity_residual[:, j]), maximum(sens.tilt_linearity_residual[:, j])))
end

# Error field per millimetre of in-plane shift and per millimetre of rim displacement from tilt,
# F coils only, both in the units mechanical tolerances arrive in. Axisymmetric hoops have no
# nominal n=1 drive, so the sensitivities are their whole error-field story.
show_and_save(AEF.plot_coil_sensitivities(full; quantity=:shift, coils=names[fcoils]), "fcoil_shift_sensitivity")
show_and_save(AEF.plot_coil_sensitivities(full; quantity=:rim, coils=names[fcoils]), "fcoil_rim_sensitivity")

# Where the sensitivity lives in mode space: the spectrum derivative of the most and least
# sensitive F coils under a 1 mm shift along x, against the C-coil's nominal spectrum.
step_series(m, a) = (vcat(m[1] - 1, m, m[end] + 1), vcat(0.0, a, 0.0))
m = sens.m_modes
order = sortperm(full.abs_delta_shift_per_mm[fcoils]; rev=true)
picks = [fcoils[order[1]], fcoils[order[end]]]
c = findfirst(==("d3d_c"), names)   # file-based sets are named machine_name
spectra = vcat([("C-coil nominal (20 A)", abs.(sens.field_as_designed[:, c]))],
    [("$(names[j]) ∂b̃/∂Δx · 1 mm", 1e-3 .* abs.(sens.shift_sensitivity_per_m[:, 1, j])) for j in picks])
floor = minimum(minimum(filter(>(0), a)) for (_, a) in spectra) / 3
top = maximum(maximum(a) for (_, a) in spectra)
fig = Figure(; size=(1100, 420))
ax = Axis(fig[1, 1]; xlabel="poloidal mode m", ylabel="|b̃| (T)", yscale=log10,
    title="Root-area-weighted spectra: C-coil as built vs F coils shifted 1 mm in x")
for (label, a) in spectra
    stairs!(ax, step_series(m, max.(a, floor))...; step=:center, linewidth=2, label)
end
ylims!(ax, floor, 2top)
Legend(fig[1, 2], ax; framevisible=false)
show_and_save(fig, "fcoil_spectra")

# The tolerance Monte Carlo: the intrinsic and corrected |δ| distributions the run wrote (full
# window, dominant mode), and a post-hoc re-run over the edge window with the same tolerances.
mc = ErrorFields.MonteCarloResult(h5path)
mc_edge = ErrorFields.run_monte_carlo(h5path; psi_low=0.7, nsample=200_000, nbatch=4, seed=1)
show_and_save(AEF.plot_tolerance_pdf(["all rational surfaces" => mc, "ψ_N ≥ 0.7" => mc_edge]), "tolerance_pdf")
centers(m) = (m.abs_delta_bin_edges[1:end-1] .+ m.abs_delta_bin_edges[2:end]) ./ 2
@printf("⟨|δ|⟩ = %.3e intrinsic, %.3e corrected; as designed %.3e; batch spread of ⟨|δ|⟩ ≈ %.1e\n",
    mc.abs_delta_sampled_mean, mc.abs_delta_efc_sampled_mean, mc.abs_delta_total_as_designed,
    maximum(abs.(vec(sum(mc.abs_delta_pdf_batches .* centers(mc) .* diff(mc.abs_delta_bin_edges); dims=1)) .- mc.abs_delta_sampled_mean)))

# Locking risk: the overlap distribution against the threshold's distribution, and the risk against
# tolerance scale with the allowable tolerance for a 1 % target read off the scan.
scan = ErrorFields.ToleranceScan(h5path)
locking_probability_percent, locking_probability_efc_percent, thr_nom = h5open(
    f -> (read(f["ErrorFields/Risk/locking_probability_percent"]), read(f["ErrorFields/Risk/locking_probability_efc_percent"]),
        read(f["ErrorFields/Risk/threshold_fit"])), h5path, "r")
show_and_save(AEF.plot_threshold_scaling(h5path), "threshold_scaling")
show_and_save(AEF.plot_locking_risk(h5path; target_percent=1.0), "locking_risk")
@printf("P_lock = %.2f %% intrinsic, %.2f %% corrected at the design tolerances; allowable scale for 1 %%: %.2f (intrinsic), %.2f (corrected)\n",
    locking_probability_percent, locking_probability_efc_percent, ErrorFields.allowable_tolerance(scan, 1.0), ErrorFields.allowable_tolerance(scan, 1.0; corrected=true))

# The run's assessment on one page. The single-run figures above overplot design revisions when
# given several `label => gpec.h5` pairs.
show_and_save(AEF.plot_error_field_summary(h5path), "error_field_summary")

# How much intrinsic error field the C-coil could correct before its own NTV torque, at a
# 4 N·m budget, costs the rotation that holds the penetration threshold up.
show_and_save(AEF.plot_efc_ntv_limits(h5path; torque_budget=4.0), "efc_ntv_limits")
for c in ErrorFields.read_efc_couplings(h5path)
    lim = ErrorFields.max_correctable_overlap(c; delta_threshold=thr_nom, torque_budget=4.0)
    @printf("%s: |δ| = %.3e per kAt, resonant fraction %.1f %%, torque %.3e (full) / %.3e (residual) N·m per kAt²; correctable up to %.2f × threshold\n",
        c.coil_name, c.delta_per_kat, c.resonant_fraction_percent, c.torque_full_per_kat2, c.torque_residual_per_kat2, lim.with_ntv / thr_nom)
end

# ---------------------------------------------------------------------------------------------
# Comparing a coil revision against this run, without re-solving the plasma.
#
# The context holds everything that does not depend on which coils are being judged, so any
# geometry can be swept against it. Here the example's own coils, shifted a millimetre, stand in
# for a second revision.
FT = GeneralizedPerturbedEquilibrium.ForcingTerms

ctx = ErrorFields.ResonantDriveContext(h5path)
sets = FT.load_coil_sets(ctx.cfg, 1; equil=ctx.equil)      # the deck's own coils, rebuilt from the file
as_built = ErrorFields.coil_overlaps(ctx, sets)
moved = ErrorFields.coil_overlaps(ctx, [FT.apply_transforms(cs, FT.CoilSetConfig(; shiftx=fill(1e-3, cs.ncoil)); n_tilt=0) for cs in sets])

println("\ncoil            |δ| as built    |δ| shifted 1 mm   resonant fraction")
for (a, b) in zip(as_built, moved)
    @printf("%-14s %12.4e   %14.4e   %14.1f %%\n", a.coil_name, abs(a.delta), abs(b.delta), a.resonant_fraction_percent)
end

# Currents are applied afterwards by scaling, so a pair driven in opposition is one call. An
# unmatched name raises rather than contributing zero, which is what a renamed coil would do.
pair = ErrorFields.combine_overlaps(as_built, as_built[1].coil_name => 1.0, as_built[end].coil_name => -1.0; name="opposed pair")
@printf("opposed pair: |δ| = %.4e, resonant fraction %.1f %%\n", abs(pair.delta), pair.resonant_fraction_percent)

# Why a coil couples as it does, which the per-coil bars above cannot answer. The first says
# whether it drives a different part of the spectrum or simply drives less; the second decomposes
# the overlap into the harmonics that produced it; the third puts both on the control surface
# against the dominant mode.
for (name, fn) in (("applied_spectra", AEF.plot_applied_spectra), ("overlap_contributions", AEF.plot_overlap_contributions),
    ("surface_overlay", AEF.plot_surface_overlay))
    show_and_save(fn(ctx, as_built), name)
end
