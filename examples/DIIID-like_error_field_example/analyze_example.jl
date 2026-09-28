using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using GeneralizedPerturbedEquilibrium, Plots, Printf
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

println("coil       |δ_nom|    per mm of shift   per mm of rim    curvature")
for (j, nm) in enumerate(names)
    @printf("%-8s %10.3e   %13.3e   %13.3e   %9.1e\n", nm, abs(full.delta_nominal[j]), full.delta_per_mm_shift[j],
        full.delta_per_mm_rim[j], max(maximum(sens.shift_linearity_residual[:, j]), maximum(sens.tilt_linearity_residual[:, j])))
end

# Error field per millimetre of in-plane shift and per millimetre of rim displacement from tilt,
# F coils only, both in the units mechanical tolerances arrive in. Axisymmetric hoops have no
# nominal n=1 drive, so the sensitivities are their whole error-field story.
p_shift = bar(names[fcoils], full.delta_per_mm_shift[fcoils]; label="core window ψ_N ≤ 0.9", alpha=0.75,
    ylabel="|δ| per mm", title="Dominant-mode error field per mm of F-coil shift", xrotation=45, xticks=(1:length(fcoils), names[fcoils]),
    left_margin=12Plots.mm, bottom_margin=8Plots.mm, legend=:topright)
p_tilt = bar(names[fcoils], full.delta_per_mm_rim[fcoils]; label="core window ψ_N ≤ 0.9", alpha=0.75,
    ylabel="|δ| per mm of rim", title="Dominant-mode error field per mm of F-coil rim displacement", xrotation=45,
    xticks=(1:length(fcoils), names[fcoils]), left_margin=12Plots.mm, bottom_margin=8Plots.mm, legend=:topright)
p_sens = plot(p_shift, p_tilt; layout=(2, 1), size=(900, 700))
display(p_sens)
sens_path = joinpath(@__DIR__, "fcoil_sensitivities.png")
Plots.savefig(p_sens, sens_path)
println("Saved: ", abspath(sens_path))

# Where the sensitivity lives in mode space: the spectrum derivative of the most and least
# sensitive F coils under a 1 mm shift along x, against the C-coil's nominal spectrum.
step_series(m, a) = (vcat(m[1] - 1, m, m[end] + 1), vcat(0.0, a, 0.0))
m = sens.m_modes
order = sortperm(full.delta_per_mm_shift[fcoils]; rev=true)
picks = [fcoils[order[1]], fcoils[order[end]]]
p_spec = plot(; xlabel="poloidal mode m", ylabel="|b̃| [T]", yscale=:log10, legend=:topleft,
    title="Root-area-weighted spectra: C-coil as built vs F coils shifted 1 mm in x",
    left_margin=12Plots.mm, bottom_margin=6Plots.mm, size=(900, 420))
c = findfirst(==("d3d_c"), names)   # file-based sets are named machine_name
me, ae = step_series(m, max.(abs.(sens.nominal_field[:, c]), 1e-30))
plot!(p_spec, me, ae; seriestype=:steppre, lw=2, label="C-coil nominal (1 kA)")
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
