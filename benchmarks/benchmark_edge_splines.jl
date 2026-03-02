"""
Benchmark and visualization script for edge inverse splines (near-separatrix extension).

This script demonstrates the edge inverse spline construction introduced for diverted
plasmas, allowing JPEC to integrate beyond psihigh toward the separatrix.

For a diverted plasma, it produces the following plots saved alongside the input toml:
  1. iota (= 1/q) vs psin: shows the iota inner spline descending smoothly to 0 at psin=1
  2. q vs psin: compares the direct spline (ExtendExtrap) vs the edge inverse spline
  3. dV/dψ vs psin: same comparison
  4. R(psin) and Z(psin) for fixed poloidal angles toward the edge
  5. Flux surface cross-section with x-point marked
  6. Rational surface density in the edge zone
  7. et[1] vs psi in the edge zone (stability diagnostic, requires a prior jpec.h5 output)

For a limited plasma (no x-point), the script prints a summary and exits since no
edge inverse splines are built.

Usage:
    julia --project=. benchmarks/benchmark_edge_splines.jl [path/to/jpec.toml]

If no path is given, defaults to examples/DIIID-like_ideal_example/jpec.toml.
"""

using JPEC
using JPEC.Equilibrium: InverseCubicSpline
using HDF5
using Roots
using Plots
using TOML
using Printf

# --- Configuration ---
toml_path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example", "jpec.toml")
toml_path = abspath(toml_path)
output_dir = dirname(toml_path)

println("="^70)
println("Edge Spline Benchmark")
println("="^70)
println("Input: $toml_path")
println()

# --- Load equilibrium ---
println("Loading equilibrium...")
inputs = TOML.parsefile(toml_path)
eq_config = JPEC.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], output_dir)
pe = JPEC.Equilibrium.setup_equilibrium(eq_config)
profiles = pe.profiles
params = pe.params
psihigh = pe.config.psihigh

println(@sprintf("  psihigh  = %.4f", psihigh))
println(@sprintf("  is_diverted = %s", string(params.is_diverted)))
if !isnothing(params.r_xpoint)
    println(@sprintf("  X-point:  R = %.4f m, Z = %.4f m", params.r_xpoint, params.z_xpoint))
end
println()

# --- Check diverted topology ---
if isnothing(params.is_diverted) || !params.is_diverted
    println("Plasma is LIMITED (no x-point detected). No edge inverse splines were built.")
    println("Re-run with a diverted equilibrium to see edge spline diagnostics.")
    exit(0)
end

if isnothing(profiles.q_spline_iota_inverse)
    println("ERROR: is_diverted=true but q_spline_iota_inverse is nothing. Check equilibrium_solver.")
    exit(1)
end

# --- Access spline internals ---
# Inner iota spline: stores iota(psin) with anchor at psin=1
iota_inner = profiles.q_spline_iota_inverse.inner
dVdpsi_inv_inner = profiles.dVdpsi_spline_inv.inner

# Grid from the direct spline
psi_core = profiles.xs

# --- Build dense evaluation grids ---
psi_plot_core = range(psi_core[1], psihigh, length=200)
psi_plot_edge = range(psihigh, 0.9999, length=300)
psi_plot_all  = vcat(collect(psi_plot_core), collect(psi_plot_edge))

# Direct spline evaluated everywhere (extrapolates linearly beyond psihigh)
q_direct  = profiles.q_spline_direct.(psi_plot_all)
dV_direct = profiles.dVdpsi_spline.(psi_plot_all)

# Edge inverse spline (only valid in edge zone)
hint_q  = Ref(1)
hint_dV = Ref(1)
q_edge_vals  = [1.0 / iota_inner(psi; hint=hint_q)  for psi in psi_plot_edge]
dV_edge_vals = [1.0 / dVdpsi_inv_inner(psi; hint=hint_dV) for psi in psi_plot_edge]
iota_edge_plot = [iota_inner(psi; hint=hint_q)  for psi in psi_plot_edge]

# Key reference values
q_at_psihigh_direct  = profiles.q_spline_direct(psihigh)
q_at_psihigh_edge    = profiles.q_spline_iota_inverse(psihigh)
dV_at_psihigh_direct = profiles.dVdpsi_spline(psihigh)
dV_at_psihigh_edge   = profiles.dVdpsi_spline_inv(psihigh)

# --- Print key diagnostics ---
println("Continuity check at psihigh:")
println(@sprintf("  q:    direct = %.5f,  edge = %.5f,  |Δ| = %.2e",
    q_at_psihigh_direct, q_at_psihigh_edge, abs(q_at_psihigh_direct - q_at_psihigh_edge)))
println(@sprintf("  dV/dψ: direct = %.5f,  edge = %.5f,  |Δ| = %.2e",
    dV_at_psihigh_direct, dV_at_psihigh_edge, abs(dV_at_psihigh_direct - dV_at_psihigh_edge)))
println()

# --- Plot 1: iota vs psin ---
p1 = plot(
    collect(psi_plot_edge), iota_edge_plot;
    xlabel = "ψₙ",
    ylabel = "ι = 1/q",
    title  = "Rotational transform ι = 1/q toward separatrix",
    label  = "Edge iota spline",
    lw = 2, color = :blue,
    legend = :topright
)
vline!(p1, [psihigh]; label = @sprintf("psihigh = %.3f", psihigh), ls = :dash, color = :gray)
hline!(p1, [0.0]; label = "ι = 0 (separatrix)", ls = :dot, color = :red, lw = 1.5)
savefig(p1, joinpath(output_dir, "edge_spline_iota.png"))
println("Saved: edge_spline_iota.png")

# --- Plot 2: q vs psin ---
# Cap display range so the diverging ExtendExtrap doesn't crush the interesting region
q_cap = 2.5 * q_at_psihigh_direct
q_direct_plot = min.(q_direct, q_cap)
q_edge_plot   = min.(q_edge_vals, q_cap)

p2 = plot(
    psi_plot_all, q_direct_plot;
    xlabel = "ψₙ",
    ylabel = "q (safety factor)",
    title  = "Safety factor q: direct vs edge inverse spline",
    label  = "Direct spline (ExtendExtrap beyond psihigh)",
    lw = 2, color = :orange, ls = :dash,
    xlims = (0.8, 1.0),
    ylims = (q_at_psihigh_direct * 0.5, q_cap * 1.05)
)
plot!(p2, collect(psi_plot_edge), q_edge_plot;
    label = "Edge inverse spline (1/ι)", lw = 2, color = :blue)
vline!(p2, [psihigh]; label = @sprintf("psihigh = %.3f", psihigh), ls = :dash, color = :gray)
savefig(p2, joinpath(output_dir, "edge_spline_q.png"))
println("Saved: edge_spline_q.png")

# --- Plot 3: dV/dψ vs psin ---
dV_cap = 2.5 * dV_at_psihigh_direct
dV_direct_plot = min.(dV_direct, dV_cap)
dV_edge_plot   = min.(dV_edge_vals, dV_cap)

p3 = plot(
    psi_plot_all, dV_direct_plot;
    xlabel = "ψₙ",
    ylabel = "dV/dψ",
    title  = "Volume gradient dV/dψ: direct vs edge inverse spline",
    label  = "Direct spline (ExtendExtrap beyond psihigh)",
    lw = 2, color = :orange, ls = :dash,
    ylims = (dV_at_psihigh_direct * 0.5, dV_cap * 1.05)
)
plot!(p3, collect(psi_plot_edge), dV_edge_plot;
    label = "Edge inverse spline", lw = 2, color = :blue)
vline!(p3, [psihigh]; label = @sprintf("psihigh = %.3f", psihigh), ls = :dash, color = :gray)
savefig(p3, joinpath(output_dir, "edge_spline_dVdpsi.png"))
println("Saved: edge_spline_dVdpsi.png")

# --- Helper: reconstruct (R, Z) from stored rzphi 2D splines ---
# rzphi_rsquared stores rfac² = (R-ro)² + (Z-zo)²
# rzphi_offset stores η/(2π) - θ, where η is the geometric poloidal angle
# R = ro + rfac * cos(η),  Z = zo + rfac * sin(η)
function RZ_at(pe, psi, theta)
    rsq    = pe.rzphi_rsquared((psi, theta))
    offset = pe.rzphi_offset((psi, theta))
    rfac   = sqrt(max(0.0, rsq))
    eta    = 2π * (theta + offset)
    R = pe.ro + rfac * cos(eta)
    Z = pe.zo + rfac * sin(eta)
    return R, Z
end

# --- Plot 4: R(psin) and Z(psin) for several fixed theta values toward the edge ---
theta_vals  = [0.0, 0.25, 0.5, 0.75]                # outboard, top, inboard, bottom
theta_names = ["θ=0 (outboard)", "θ=0.25 (top)", "θ=0.5 (inboard)", "θ=0.75 (bottom)"]
psi_rz_plot = range(psi_core[1], psihigh, length=150)  # only in valid spline domain

p4a = plot(; xlabel="ψₙ", ylabel="R [m]",
           title="Major radius R(ψₙ) for fixed poloidal angles", legend=:topright)
p4b = plot(; xlabel="ψₙ", ylabel="Z [m]",
           title="Vertical position Z(ψₙ) for fixed poloidal angles", legend=:topright)

for (theta, name) in zip(theta_vals, theta_names)
    R_vals = [RZ_at(pe, psi, theta)[1] for psi in psi_rz_plot]
    Z_vals = [RZ_at(pe, psi, theta)[2] for psi in psi_rz_plot]
    plot!(p4a, collect(psi_rz_plot), R_vals; label=name, lw=2)
    plot!(p4b, collect(psi_rz_plot), Z_vals; label=name, lw=2)
end
vline!(p4a, [psihigh]; label=@sprintf("psihigh=%.3f", psihigh), ls=:dash, color=:gray)
vline!(p4b, [psihigh]; label=@sprintf("psihigh=%.3f", psihigh), ls=:dash, color=:gray)

p4 = plot(p4a, p4b; layout=(1, 2), size=(900, 400))
savefig(p4, joinpath(output_dir, "edge_spline_RZ_vs_psin.png"))
println("Saved: edge_spline_RZ_vs_psin.png")

# --- Plot 5: Flux surface cross-section with x-point ---
theta_fs   = range(0.0, 1.0, length=200)   # full poloidal circuit
psin_surfaces = [0.5, 0.7, 0.85, 0.93, 0.97, psihigh]  # core → edge

p5 = plot(; xlabel="R [m]", ylabel="Z [m]",
           title="Flux surface cross-section",
           aspect_ratio=:equal, legend=:topright)

colors = cgrad(:blues, length(psin_surfaces); rev=false)
for (i, psin_val) in enumerate(psin_surfaces)
    R_fs = [RZ_at(pe, psin_val, th)[1] for th in theta_fs]
    Z_fs = [RZ_at(pe, psin_val, th)[2] for th in theta_fs]
    lbl  = @sprintf("ψₙ = %.3f%s", psin_val, psin_val == psihigh ? " (LCFS)" : "")
    plot!(p5, R_fs, Z_fs; label=lbl, lw=1.5, color=colors[i])
end

# Mark magnetic axis and x-point
scatter!(p5, [pe.ro], [pe.zo]; label="Axis", marker=:cross, ms=8, color=:black)
scatter!(p5, [params.r_xpoint], [params.z_xpoint];
         label=@sprintf("X-point (R=%.3f, Z=%.3f)", params.r_xpoint, params.z_xpoint),
         marker=:x, ms=10, color=:red, markerstrokewidth=2)

savefig(p5, joinpath(output_dir, "edge_spline_cross_section.png"))
println("Saved: edge_spline_cross_section.png")

# --- Plot 6: Rational surface density in edge zone ---
# Scan for q = m/n rational surfaces using the iota inverse spline
println("Scanning rational surfaces in edge zone [psihigh, 0.9999]...")
nn = 1
m_start = trunc(Int, q_at_psihigh_direct) + 1
m_end   = trunc(Int, min(q_at_psihigh_direct * 5, 500.0))

edge_surfaces = Float64[]
hint = Ref(1)
for m in m_start:m_end
    iota_target = nn / m
    # Skip if this iota is above the edge zone (i.e., q < q(psihigh))
    iota_at_start = iota_inner(psihigh; hint=hint)
    if iota_target >= iota_at_start
        continue
    end
    # iota_target < iota(psihigh), so the surface is in the edge zone
    # The anchor gives iota(1.0)=0 < iota_target, so a root exists in (psihigh, 1.0)
    psi_bracket_hi = 1.0 - 1e-8
    try
        psi_surf = find_zero(
            psi -> iota_inner(psi; hint=hint) - iota_target,
            (psihigh, psi_bracket_hi), Roots.Brent(); xatol=1e-8
        )
        push!(edge_surfaces, psi_surf)
        # Stop when surfaces become too dense (< edge_layer_width = 1e-4)
        if length(edge_surfaces) >= 2 && edge_surfaces[end] - edge_surfaces[end-1] < 1e-4
            break
        end
    catch
        # No root in bracket for this m/n — skip
    end
end
println(@sprintf("  Found %d rational surfaces (n=%d) in edge zone before density cutoff",
    length(edge_surfaces), nn))

p6 = plot(
    collect(psi_plot_edge), q_edge_vals;
    xlabel = "ψₙ",
    ylabel = "q",
    title  = @sprintf("Rational surfaces in edge zone (n=%d)", nn),
    label  = "q(ψ) [edge inverse spline]",
    lw = 2, color = :blue,
    ylims = (q_at_psihigh_direct * 0.9, min(q_at_psihigh_direct * 4, maximum(q_edge_vals, init=q_at_psihigh_direct * 2.0)))
)
for (i, psi_s) in enumerate(edge_surfaces)
    lbl = i == 1 ? "Edge rational surfaces" : ""
    vline!(p6, [psi_s]; label = lbl, color = :red, alpha = 0.5, lw = 0.8)
end
vline!(p6, [psihigh]; label = @sprintf("psihigh = %.3f", psihigh), ls = :dash, color = :gray)
savefig(p6, joinpath(output_dir, "edge_spline_rational_surfaces.png"))
println("Saved: edge_spline_rational_surfaces.png")

# --- Plot 7: et[1] vs q — edge stability diagnostic (from jpec.h5) ---
# x-axis is q (safety factor), converted from psi via the iota inverse spline.
# A stability boundary line at et=0 and a reference value from the develop branch are overlaid.
h5file = joinpath(output_dir, "jpec.h5")
if isfile(h5file)
    h5open(h5file, "r") do f
        if haskey(f, "integration/psi_edge_scan") && haskey(f, "integration/et_edge_scan")
            psi_es  = read(f["integration/psi_edge_scan"])
            et_es   = read(f["integration/et_edge_scan"])
            et_real = real.(et_es)

            # Convert psi → q using iota inverse spline (above psihigh) or direct spline (below)
            hint_q7 = Ref(1)
            q_es = map(psi_es) do psi
                if psi > psihigh && !isnothing(profiles.q_spline_iota_inverse)
                    profiles.q_spline_iota_inverse(psi)
                else
                    profiles.q_spline_direct(psi; hint=hint_q7)
                end
            end

            q_at_psihigh = profiles.q_spline_direct(psihigh)
            et_peak_idx = argmax(et_real)
            q_peak      = q_es[et_peak_idx]

            p7 = plot(
                q_es, et_real;
                xlabel = "q (safety factor)",
                ylabel = "Re(et[1])",
                title  = "Edge stability: Re(et[1]) vs q from psiedge to psilim",
                label  = "Re(et[1])",
                lw = 2, color = :blue, legend = :topright,
                xlims = (4.0, q_es[end])
            )
            hline!(p7, [0.0]; label = "stability boundary (et = 0)",
                   ls = :dot, color = :black, lw = 1.5)
            vline!(p7, [q_at_psihigh]; label = @sprintf("q(psihigh) = %.3f", q_at_psihigh),
                   ls = :dash, color = :gray)
            hline!(p7, [1.707]; label = "develop branch ref: q≈5.2, et=+1.707",
                   ls = :dash, color = :green, lw = 1.5)
            scatter!(p7, [q_peak], [et_real[et_peak_idx]];
                     label = @sprintf("peak (q=%.4f, et=%.3e)", q_peak, et_real[et_peak_idx]),
                     marker = :star5, ms = 10, color = :red)
            savefig(p7, joinpath(output_dir, "edge_spline_stability.png"))
            println("Saved: edge_spline_stability.png")
        else
            println("Note: integration/psi_edge_scan not found in jpec.h5 — run JPEC with psiedge < psilim first")
        end
    end
else
    println("Note: jpec.h5 not found in $output_dir — run JPEC first to generate edge stability plot")
end

# --- Plot 8: F(ψ) direct vs F_iota_matched in the far edge ---
if !isnothing(profiles.F_spline_iota_matched)
    psi_far_plot = range(psihigh, 1.0 - 1e-4, length=200)
    F_direct_vals  = [profiles.F_spline_direct(psi)        for psi in psi_far_plot]
    F_matched_vals = [profiles.F_spline_iota_matched(psi)  for psi in psi_far_plot]
    dF_direct_vals  = [profiles.F_deriv_direct(psi)         for psi in psi_far_plot]
    dF_matched_vals = [profiles.F_deriv_iota_matched(psi)   for psi in psi_far_plot]

    p8a = plot(collect(psi_far_plot), F_direct_vals;
        xlabel="ψₙ", ylabel="F = 2π·R·Bₜ",
        title="F(ψ): direct vs iota-matched in far edge",
        label="F_direct (ExtendExtrap)", lw=2, color=:orange, ls=:dash)
    plot!(p8a, collect(psi_far_plot), F_matched_vals;
        label="F_iota_matched", lw=2, color=:blue)
    vline!(p8a, [psihigh]; label=@sprintf("psihigh=%.3f", psihigh), ls=:dash, color=:gray)

    p8b = plot(collect(psi_far_plot), dF_direct_vals;
        xlabel="ψₙ", ylabel="dF/dψ",
        title="dF/dψ: direct vs iota-matched",
        label="dF_direct", lw=2, color=:orange, ls=:dash)
    plot!(p8b, collect(psi_far_plot), dF_matched_vals;
        label="dF_iota_matched", lw=2, color=:blue)
    vline!(p8b, [psihigh]; label=@sprintf("psihigh=%.3f", psihigh), ls=:dash, color=:gray)

    p8 = plot(p8a, p8b; layout=(1,2), size=(950, 400))
    savefig(p8, joinpath(output_dir, "edge_spline_F_comparison.png"))
    println("Saved: edge_spline_F_comparison.png")
else
    println("Note: F_spline_iota_matched not built (limited plasma or old run) — skipping Plot 8")
end

# --- Plots 9 and 10: GS residual diagnostics ---
# Re-run equilibrium_gse! with diagnose_src=true to write gsei.h5 and gsec.h5,
# then read and plot the θ-integrated GS error per flux surface.
println("Running GSE diagnostics (diagnose_src=true)...")
pe.params.diagnose_src = true
JPEC.Equilibrium.equilibrium_gse!(pe)
pe.params.diagnose_src = false

gsec_file = joinpath(output_dir, "gsec.h5")
gsei_file = joinpath(output_dir, "gsei.h5")

if isfile(gsec_file) && isfile(gsei_file)
    # --- Plot 9: GS residual by poloidal angle from gsec.h5 ---
    h5open(gsec_file, "r") do f
        psi_xs_gse = read(f["mpsi"]) + 1  # number of ψ grid points
        flux_fsx_data = read(f["flux_fsx"])  # gs_div_dpsi[:,:,1]
        source_data   = read(f["source"])
        total_data    = read(f["total"])

        # Per-θ residual: |flux_fsx + source| summed over θ (proxy for residual)
        mt_gse = size(flux_fsx_data, 2)
        theta_fracs = [0.0, 0.25, 0.5, 0.75]
        theta_labels = ["θ=0.00 (outboard)", "θ=0.25 (top)", "θ=0.50 (inboard)", "θ=0.75 (bottom)"]
        theta_idxs = [max(1, round(Int, th * (mt_gse-1) + 1)) for th in theta_fracs]

        psi_gse_vals = pe.rzphi_xs

        p9 = plot(; xlabel="ψₙ", ylabel="|GS residual (per θ)|",
            title="GS residual by poloidal angle", yscale=:log10, legend=:topleft)
        for (idx, lbl) in zip(theta_idxs, theta_labels)
            res_col = abs.(flux_fsx_data[:, idx] .+ source_data[:, idx])
            plot!(p9, psi_gse_vals, max.(res_col, 1e-15); label=lbl, lw=1.5)
        end
        hline!(p9, [1e-2]; label="warning threshold", ls=:dash, color=:red, lw=1.5)
        vline!(p9, [psihigh]; label=@sprintf("psihigh=%.3f", psihigh), ls=:dash, color=:gray)
        savefig(p9, joinpath(output_dir, "edge_spline_gse_by_theta.png"))
        println("Saved: edge_spline_gse_by_theta.png")
    end

    # --- Plot 10: θ-integrated GS error from gsei.h5 ---
    h5open(gsei_file, "r") do f
        errori_data = read(f["errori"])  # gse_abs_error (θ-integrated)
        xs_gse      = read(f["xs"])

        errori_vec = vec(errori_data)
        max_err = maximum(errori_vec)
        if max_err > 1e-2
            @warn @sprintf("GS residual exceeds threshold: max |gse_integrated| = %.2e at psin=%.4f",
                max_err, xs_gse[argmax(errori_vec)])
        else
            println(@sprintf("  GS residual OK: max |gse_integrated| = %.2e", max_err))
        end

        p10 = plot(
            xs_gse, max.(errori_vec, 1e-15);
            xlabel="ψₙ", ylabel="|ΔΨ + source| (θ-integrated)",
            title="θ-integrated GS error per flux surface",
            label="|gse_integrated|", lw=2, color=:blue,
            yscale=:log10, legend=:topleft
        )
        hline!(p10, [1e-2]; label="warning threshold (1e-2)", ls=:dash, color=:red, lw=1.5)
        vline!(p10, [psihigh]; label=@sprintf("psihigh=%.3f", psihigh), ls=:dash, color=:gray)
        savefig(p10, joinpath(output_dir, "edge_spline_gse_integrated.png"))
        println("Saved: edge_spline_gse_integrated.png")
    end
else
    println("Note: GSE HDF5 files not found — skipping Plots 9 and 10")
end

# --- Final summary ---
println()
println("="^70)
println("Summary")
println("="^70)
println(@sprintf("  X-point location:   R = %.4f m,  Z = %.4f m",
    params.r_xpoint, params.z_xpoint))
println(@sprintf("  psihigh = %.4f,  qa = %s",
    psihigh, isinf(params.qa) ? "Inf (diverted, separatrix)" : @sprintf("%.4f", params.qa)))
println(@sprintf("  q(psihigh) = %.4f,  iota(psihigh) = %.6f",
    q_at_psihigh_direct, iota_inner(psihigh)))
println(@sprintf("  iota anchor at psin=1: iota(1.0) = %.2e (target: ~0)",
    iota_inner(1.0)))
println(@sprintf("  q match at psihigh:    |q_direct - q_edge| = %.2e",
    abs(q_at_psihigh_direct - q_at_psihigh_edge)))
println(@sprintf("  dV/dψ match at psihigh: |dV_direct - dV_edge| = %.2e",
    abs(dV_at_psihigh_direct - dV_at_psihigh_edge)))
println()
println("Plots saved to: $output_dir")
