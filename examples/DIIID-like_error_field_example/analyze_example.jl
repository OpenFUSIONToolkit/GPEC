using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium, Plots, Printf, HDF5
using GeneralizedPerturbedEquilibrium: PerturbedEquilibrium, ErrorFields
isinteractive() ? plotlyjs() : gr()

h5path = joinpath(@__DIR__, "gpec.h5")

# The run stored every coil set's spectrum linearization. Project it onto the dominant
# resonant-coupling mode over the core window (ψ_N ≤ 0.9, the default) to see which coils matter.
rc = PerturbedEquilibrium.ResonantCoupling(h5path)
sens = ErrorFields.CoilSensitivities(h5path)
full = ErrorFields.sensitivity_table(sens, PerturbedEquilibrium.dominant_coupling(rc))

names = full.coil_names
fcoils = findall(startswith("F"), names)

println("coil       |δ_nom|    |∂δ/∂Δ| per mm   |∂δ/∂θ| per 0.1°   curvature")
for (j, nm) in enumerate(names)
    @printf("%-8s %10.3e   %12.3e   %14.3e   %9.1e\n", nm, abs(full.delta_nominal[j]), 1e-3 * full.shift_rms[j],
        0.1 * full.tilt_rms[j], max(maximum(sens.shift_linearity_residual[:, j]), maximum(sens.tilt_linearity_residual[:, j])))
end

# Error field per millimetre of in-plane shift and per 0.1° of tilt, F coils only. Axisymmetric
# hoops have no nominal n=1 drive, so the sensitivities are their whole error-field story.
p_shift = bar(names[fcoils], 1e-3 .* full.shift_rms[fcoils]; label="core window ψ_N ≤ 0.9", alpha=0.75,
    ylabel="|∂δ/∂Δ| per mm", title="Dominant-mode error field per mm of F-coil shift", xrotation=45, xticks=(1:length(fcoils), names[fcoils]),
    left_margin=12Plots.mm, bottom_margin=8Plots.mm, legend=:topright)
p_tilt = bar(names[fcoils], 0.1 .* full.tilt_rms[fcoils]; label="core window ψ_N ≤ 0.9", alpha=0.75,
    ylabel="|∂δ/∂θ| per 0.1°", title="Dominant-mode error field per 0.1° of F-coil tilt", xrotation=45, xticks=(1:length(fcoils), names[fcoils]),
    left_margin=12Plots.mm, bottom_margin=8Plots.mm, legend=:topright)
p_sens = plot(p_shift, p_tilt; layout=(2, 1), size=(900, 700))
display(p_sens)
sens_path = joinpath(@__DIR__, "fcoil_sensitivities.png")
Plots.savefig(p_sens, sens_path)
println("Saved: ", abspath(sens_path))

# Where the sensitivity lives in mode space: the spectrum derivative of the most and least
# sensitive F coils under a 1 mm shift along x, against the C-coil's nominal spectrum.
step_series(m, a) = (vcat(m[1] - 1, m, m[end] + 1), vcat(0.0, a, 0.0))
m = sens.m_modes
order = sortperm(full.shift_rms[fcoils]; rev=true)
picks = [fcoils[order[1]], fcoils[order[end]]]
p_spec = plot(; xlabel="poloidal mode m", ylabel="|b̃| [T]", yscale=:log10, legend=:topleft,
    title="Root-area-weighted spectra: C-coil as built vs F coils shifted 1 mm in x",
    left_margin=12Plots.mm, bottom_margin=6Plots.mm, size=(900, 420))
c = findfirst(==("d3d_c"), names)   # file-based sets are named machine_name
me, ae = step_series(m, max.(abs.(sens.nominal_field[:, c]), 1e-30))
plot!(p_spec, me, ae; seriestype=:steppre, lw=2, label="C-coil nominal (20 A)")
for j in picks
    me, ae = step_series(m, max.(1e-3 .* abs.(sens.shift_sensitivity[:, 1, j]), 1e-30))
    plot!(p_spec, me, ae; seriestype=:steppre, lw=2, label="$(names[j]) ∂b̃/∂Δx · 1 mm")
end
display(p_spec)
spec_path = joinpath(@__DIR__, "fcoil_spectra.png")
Plots.savefig(p_spec, spec_path)
println("Saved: ", abspath(spec_path))

# The tolerance Monte Carlo: the intrinsic and corrected |δ| distributions the run wrote (full
# window, dominant mode), and a post-hoc re-run over the edge window with the same tolerances.
mc = ErrorFields.MonteCarloResult(h5path)
mc_edge = ErrorFields.run_monte_carlo(h5path; psi_low=0.7, nsample=200_000, nbatch=4, seed=1)
centers(m) = (m.bin_edges[1:end-1] .+ m.bin_edges[2:end]) ./ 2
p_pdf = plot(; xlabel="dominant-mode overlap |δ|", ylabel="probability density", legend=:topright,
    title="Tolerance Monte Carlo: |δ| over sampled misalignments", left_margin=12Plots.mm, bottom_margin=6Plots.mm, size=(900, 420))
plot!(p_pdf, centers(mc), mc.pdf; lw=2, label="intrinsic, all rational surfaces")
plot!(p_pdf, centers(mc), mc.pdf_efc; lw=2, label="corrected (efc_factor = 2)")
plot!(p_pdf, centers(mc_edge), mc_edge.pdf; lw=2, ls=:dash, label="intrinsic, ψ_N ≥ 0.7")
vline!(p_pdf, [mc.delta_nominal]; ls=:dot, c=:black, label="as designed")
display(p_pdf)
pdf_path = joinpath(@__DIR__, "tolerance_pdf.png")
Plots.savefig(p_pdf, pdf_path)
println("Saved: ", abspath(pdf_path))
@printf("⟨|δ|⟩ = %.3e intrinsic, %.3e corrected; as designed %.3e; batch spread of ⟨|δ|⟩ ≈ %.1e\n",
    mc.mean_abs_delta, mc.mean_abs_delta_efc, mc.delta_nominal,
    maximum(abs.(vec(sum(mc.pdf_batches .* centers(mc) .* diff(mc.bin_edges); dims=1)) .- mc.mean_abs_delta)))

# Locking risk: the threshold distribution against the overlap distribution, and the risk against
# tolerance scale with the allowable tolerance for a 1 % target read off the scan.
scan = ErrorFields.ToleranceScan(h5path)
risk_nominal = h5open(
    f -> (read(f["ErrorFields/Risk/plock_percent"]), read(f["ErrorFields/Risk/plock_efc_percent"]),
        read(f["ErrorFields/Risk/threshold_nominal"]), read(f["ErrorFields/Risk/threshold_pdf"]), read(f["ErrorFields/Risk/p_lock_given_delta"])), h5path, "r")
plock, plock_efc, thr_nom, thr_pdf, p_given = risk_nominal
p_thr = plot(; xlabel="dominant-mode overlap |δ|", ylabel="probability density", legend=:topright, xscale=:log10,
    title="Overlap distribution vs ITPA penetration threshold", left_margin=12Plots.mm, bottom_margin=6Plots.mm, size=(900, 420))
c = centers(mc)
keep = c .> 0
plot!(p_thr, c[keep], mc.pdf[keep] ./ maximum(mc.pdf); lw=2, label="intrinsic |δ| (normalized)")
plot!(p_thr, c[keep], mc.pdf_efc[keep] ./ maximum(mc.pdf_efc); lw=2, label="corrected |δ| (normalized)")
plot!(p_thr, c[keep], thr_pdf[keep] ./ maximum(thr_pdf); lw=2, c=:black, label="threshold density (normalized)")
plot!(p_thr, mc.bin_edges[2:end], p_given[2:end]; lw=1.5, ls=:dash, c=:red, label="P(lock | δ)")
vline!(p_thr, [thr_nom]; ls=:dot, c=:black, label="nominal threshold")
p_scan = plot(; xlabel="tolerance scale (× tolerances.toml)", ylabel="locking probability [%]", xscale=:log10, yscale=:log10,
    legend=:topleft, title="Risk vs tolerance scale", left_margin=12Plots.mm, bottom_margin=6Plots.mm)
plot!(p_scan, scan.scale, max.(scan.plock, 1e-4); marker=:circle, lw=2, label="intrinsic")
plot!(p_scan, scan.scale, max.(scan.plock_efc, 1e-4); marker=:square, lw=2, label="corrected")
hline!(p_scan, [1.0]; ls=:dash, c=:gray, label="1 % target")
p_risk = plot(p_thr, p_scan; layout=(2, 1), size=(900, 760))
display(p_risk)
risk_path = joinpath(@__DIR__, "locking_risk.png")
Plots.savefig(p_risk, risk_path)
println("Saved: ", abspath(risk_path))
@printf("P_lock = %.2f %% intrinsic, %.2f %% corrected at the design tolerances; allowable scale for 1 %%: %.2f (intrinsic), %.2f (corrected)\n",
    plock, plock_efc, ErrorFields.allowable_tolerance(scan, 1.0), ErrorFields.allowable_tolerance(scan, 1.0; corrected=true))

# The same figures through the Analysis module, which overplots design revisions when given
# several `label => gpec.h5` pairs.
p_summary = GeneralizedPerturbedEquilibrium.Analysis.ErrorFields.plot_error_field_summary(h5path; save_path=joinpath(@__DIR__, "error_field_summary.png"))
display(p_summary)
