"""
Benchmark and visualization script for edge inverse splines (near-separatrix extension).

This script demonstrates the edge inverse spline construction introduced for diverted
plasmas, allowing GPEC to integrate beyond psihigh toward the separatrix.

For a diverted plasma, it produces the following plots saved alongside the input toml:
  1. iota (= 1/q) vs psin: shows the iota inner spline descending smoothly to 0 at psin=1
  2. q vs psin: compares the direct spline (ExtendExtrap) vs the edge inverse spline
  3. dV/dψ vs psin: same comparison
  4. R(psin) and Z(psin) for fixed poloidal angles toward the edge
  5. Flux surface cross-section with x-point marked
  6. Rational surface density in the edge zone
  7. et[1] vs psi in the edge zone (stability diagnostic, requires a prior gpec.h5 output)

For a limited plasma (no x-point), the script prints a summary and exits since no
edge inverse splines are built.

Usage:
    julia --project=. benchmarks/benchmark_edge_splines.jl [path/to/gpec.toml]

If no path is given, defaults to examples/DIIID-like_ideal_example/gpec.toml.
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium: eval_q
using FastInterpolations
using HDF5
using Roots
using Plots
using TOML
using Printf

# --- Configuration ---
toml_path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example", "gpec.toml")
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
eq_config = GeneralizedPerturbedEquilibrium.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], output_dir)
pe = GeneralizedPerturbedEquilibrium.Equilibrium.setup_equilibrium(eq_config)
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

if isnothing(profiles.iota_spline)
    println("ERROR: is_diverted=true but iota_spline is nothing. Check equilibrium_solver.")
    exit(1)
end

# --- Access spline internals ---
# Inner iota spline: stores iota(psin) with anchor at psin=1
iota_inner = profiles.iota_spline
dVdpsi_inv_inner = profiles.dVdpsi_inv_spline

# Grid from the direct spline
psi_core = profiles.xs

# --- Build dense evaluation grids ---
psi_plot_core = range(psi_core[1], psihigh, length=200)
psi_plot_edge = range(psihigh, 0.9999, length=300)
psi_plot_all  = vcat(collect(psi_plot_core), collect(psi_plot_edge))

# Direct spline evaluated everywhere (extrapolates linearly beyond psihigh)
q_direct  = profiles.q_spline.(psi_plot_all)
dV_direct = profiles.dVdpsi_spline.(psi_plot_all)

# Edge inverse spline (only valid in edge zone)
hint_q  = Ref(1)
hint_dV = Ref(1)
q_edge_vals  = [1.0 / iota_inner(psi; hint=hint_q)  for psi in psi_plot_edge]
dV_edge_vals = [1.0 / dVdpsi_inv_inner(psi; hint=hint_dV) for psi in psi_plot_edge]
iota_edge_plot = [iota_inner(psi; hint=hint_q)  for psi in psi_plot_edge]

# Key reference values
q_at_psihigh_direct  = profiles.q_spline(psihigh)
q_at_psihigh_edge    = eval_q(profiles, psihigh)
dV_at_psihigh_direct = profiles.dVdpsi_spline(psihigh)
dV_at_psihigh_edge   = 1.0 / profiles.dVdpsi_inv_spline(psihigh)

# --- Print key diagnostics ---
println("Continuity check at psihigh:")
println(@sprintf("  q:    direct = %.5f,  edge = %.5f,  |Δ| = %.2e",
    q_at_psihigh_direct, q_at_psihigh_edge, abs(q_at_psihigh_direct - q_at_psihigh_edge)))
println(@sprintf("  dV/dψ: direct = %.5f,  edge = %.5f,  |Δ| = %.2e",
    dV_at_psihigh_direct, dV_at_psihigh_edge, abs(dV_at_psihigh_direct - dV_at_psihigh_edge)))
println()

# --- Plot 1: iota vs psin (built into edge_spline_profiles.png later; kept as p1 for reuse) ---
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

# --- Plot 2: q vs psin ---
# Cap display range so the diverging ExtendExtrap doesn't crush the interesting region
q_cap = 2.5 * q_at_psihigh_direct
q_direct_plot = min.(q_direct, q_cap)
q_edge_plot   = min.(q_edge_vals, q_cap)

# Compute q derivatives for the junction check.
# Focus on a window around psihigh to show continuity (or lack thereof).
psi_junc = range(psihigh - 0.02, min(0.9999, psihigh + (1.0 - psihigh) * 0.7); length=300)

# Direct spline: q, q', q'', q''' (only valid up to psihigh; ExtendExtrap beyond)
q_d0_dir = profiles.q_spline.(psi_junc)
q_d1_dir = profiles.q_deriv.(psi_junc)
_deriv2_q = deriv2(profiles.q_spline)
_deriv3_q = deriv3(profiles.q_spline)
q_d2_dir = _deriv2_q.(psi_junc)
q_d3_dir = _deriv3_q.(psi_junc)

# Edge inverse spline: compute via chain rule from iota inner spline
# q = 1/iota, q' = -iota'/iota², q'' = (2iota'²-iota·iota'')/iota³, etc.
_iota_inner  = profiles.iota_spline
_d1_iota = deriv1(_iota_inner)
_d2_iota = deriv2(_iota_inner)
_d3_iota = deriv3(_iota_inner)
q_d0_edge = [1.0 / _iota_inner(p)   for p in psi_junc]
q_d1_edge = map(psi_junc) do p
    ι = _iota_inner(p); ι1 = _d1_iota(p)
    -ι1 / ι^2
end
q_d2_edge = map(psi_junc) do p
    ι = _iota_inner(p); ι1 = _d1_iota(p); ι2 = _d2_iota(p)
    (2*ι1^2 - ι*ι2) / ι^3
end
q_d3_edge = map(psi_junc) do p
    ι = _iota_inner(p); ι1 = _d1_iota(p); ι2 = _d2_iota(p); ι3 = _d3_iota(p)
    (-6*ι1^3 + 6*ι*ι1*ι2 - ι^2*ι3) / ι^4
end

# Cap large values so plots are readable (derivatives blow up near separatrix)
d1_cap = abs(profiles.q_deriv(psihigh)) * 5
d2_cap = abs(_deriv2_q(psihigh)) * 10
d3_cap = abs(_deriv3_q(psihigh)) * 20

function make_deriv_panel(ylabel_str, title_str, dir_vals, edge_vals, cap_val)
    p = plot(; xlabel="ψₙ", ylabel=ylabel_str, title=title_str, legend=:topleft)
    plot!(p, collect(psi_junc), clamp.(dir_vals, -cap_val, cap_val);
          label="Direct", lw=2, color=:orange, ls=:dash)
    plot!(p, collect(psi_junc), clamp.(edge_vals, -cap_val, cap_val);
          label="Edge inverse", lw=2, color=:blue)
    vline!(p, [psihigh]; label=@sprintf("psihigh=%.4f", psihigh), ls=:dash, color=:gray, lw=1)
    return p
end

p2a = make_deriv_panel("q",      "q: direct vs edge inverse",        q_d0_dir, q_d0_edge, q_cap)
p2b = make_deriv_panel("dq/dψ",  "q' (1st deriv): junction match",   q_d1_dir, q_d1_edge, d1_cap)
p2c = make_deriv_panel("d²q/dψ²","q'' (2nd deriv): junction match",  q_d2_dir, q_d2_edge, d2_cap)
p2d = make_deriv_panel("d³q/dψ³","q''' (3rd deriv): junction match", q_d3_dir, q_d3_edge, d3_cap)
p2 = plot(p2a, p2b, p2c, p2d; layout=(4,1), size=(900, 1100), bottom_margin=4Plots.mm)
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

# psilim for this run = last edge rational surface + edge_layer_width (default 1e-4) buffer.
# qlim is used to cap q-axis plots so far-edge data near the separatrix (q→∞) doesn't compress
# the interesting region.
psilim_bench = isempty(edge_surfaces) ? psihigh : edge_surfaces[end]
qlim_bench   = (!isnothing(profiles.iota_spline) && psilim_bench > psihigh) ?
               eval_q(profiles, psilim_bench) :
               profiles.q_spline(psilim_bench)

# Helper: psi → q using iota inverse above psihigh (available to all subsequent plots)
psi_to_q_outer = psi_arr -> map(psi -> eval_q(profiles, psi), psi_arr)

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
vline!(p6, [psihigh];      label = @sprintf("psihigh = %.3f", psihigh), ls = :dash, color = :gray)
vline!(p6, [psilim_bench]; label = @sprintf("psilim = %.4f  (q = %.1f)", psilim_bench, qlim_bench),
       ls = :dashdot, color = :purple, lw = 1.5)
savefig(p6, joinpath(output_dir, "edge_spline_rational_surfaces.png"))
println("Saved: edge_spline_rational_surfaces.png")

# --- Plot 7: edge stability scan — total, plasma, vacuum energy components vs q ---
#
# Top panel: et, ep, ev on a linear scale clipped to ±display_lim. The "physically reliable"
# window (where wp = U₂·U₁⁻¹ is well-conditioned) is narrow: only the first ~6 steps after
# psiedge, before the accumulated Gaussian reductions make U₁ ill-conditioned. Outside that
# window all three curves blow up (|et|, |ep| >> 1000) and are clipped by ylims.
# ev (vacuum, computed from the wv spline rather than U₁) stays bounded at ~O(1000) for most
# of the scan and is shown separately on the log-scale bottom panel to reveal its full range.
#
# Above psihigh: wv is frozen at psihigh (ExtendExtrap) and resonant spikes near
# rational surfaces (singfac → 0) are subsampled for rendering performance.
h5file = joinpath(output_dir, "gpec.h5")
if isfile(h5file)
    h5open(h5file, "r") do f
        if haskey(f, "edge_scan/psi") && haskey(f, "edge_scan/total_energy")
            psi_es  = read(f["edge_scan/psi"])
            et_es   = read(f["edge_scan/total_energy"])
            et_real = real.(et_es)

            # Convert psi → q
            hint_q7 = Ref(1)
            q_es = map(psi -> eval_q(profiles, psi; hint=hint_q7), psi_es)

            q_at_psihigh = profiles.q_spline(psihigh)
            q_max_scan   = maximum(q_es)
            core_mask    = psi_es .<= psihigh
            above_mask   = psi_es .>  psihigh

            # Peak: ignore NaN steps (where U₁ was singular)
            et_for_peak = replace(et_real, NaN => -Inf)
            et_peak_idx = argmax(et_for_peak)
            q_peak      = q_es[et_peak_idx]
            et_peak     = et_real[et_peak_idx]

            n_valid = count(!isnan, et_real)
            println(@sprintf("  Scan: q = %.3f → %.3f  (ψ = %.4f → %.4f, %d/%d valid steps)",
                             q_es[1], q_max_scan, psi_es[1], psi_es[end], n_valid, length(psi_es)))
            println(@sprintf("  Peak: q = %.3f, et = %+.3f", q_peak, et_peak))

            # Load ep/ev/vacuum_eigenvalue if available
            has_pv     = haskey(f, "edge_scan/plasma_energy") && haskey(f, "edge_scan/vacuum_energy")
            has_evonly = haskey(f, "edge_scan/vacuum_eigenvalue")
            ep_real    = has_pv     ? real.(read(f["edge_scan/plasma_energy"]))    : nothing
            ev_real    = has_pv     ? real.(read(f["edge_scan/vacuum_energy"]))    : nothing
            evonly_real = has_evonly ? read(f["edge_scan/vacuum_eigenvalue"])       : nothing

            # --- Panel A: linear-scale clip showing valid window + full et range ---
            # Clip at ±display_lim so the well-conditioned window is visible.
            display_lim = 10.0
            # Subsample above-psihigh for rendering; separate filters for et vs ev; exclude NaN
            valid_above_et = findall(above_mask .& .!isnan.(et_real) .& (abs.(et_real) .<= display_lim))
            stride_et = max(1, length(valid_above_et) ÷ 2000)
            above_ss_et = valid_above_et[1:stride_et:end]

            pA = plot(;
                xlabel = "q (safety factor)",
                ylabel = "Energy (raw eigenvalue)",
                title  = @sprintf("Edge stability: energy components vs q  [ψ = %.3f → %.3f]",
                                  psi_es[1], psi_es[end]),
                legend = :topright,
                xlims  = (4.0, qlim_bench * 1.02),
                ylims  = (-display_lim, max(3.0, et_peak * 1.2)),
                size   = (900, 400)
            )
            # et: core (solid) + above-psihigh filtered (dashed)
            plot!(pA, q_es[core_mask], et_real[core_mask];
                  label = "Total et  [ψ ≤ psihigh]", lw = 2, color = :black)
            if !isempty(above_ss_et)
                plot!(pA, q_es[above_ss_et], et_real[above_ss_et];
                      label = @sprintf("Total et  [ψ > psihigh, |et|≤%.0f]", display_lim),
                      lw = 1, color = :black, ls = :dash, alpha = 0.6)
            end
            if has_pv
                plot!(pA, q_es[core_mask], ep_real[core_mask];
                      label = "Plasma ep  [ψ ≤ psihigh]", lw = 2, color = :blue)
                plot!(pA, q_es[core_mask], ev_real[core_mask];
                      label = "Vacuum ev  [ψ ≤ psihigh]", lw = 2, color = :red)
                valid_above_pv = findall(above_mask .& .!isnan.(ep_real) .& .!isnan.(ev_real) .& (abs.(ep_real) .<= display_lim) .& (abs.(ev_real) .<= display_lim))
                stride_pv = max(1, length(valid_above_pv) ÷ 2000)
                above_ss_pv = valid_above_pv[1:stride_pv:end]
                if !isempty(above_ss_pv)
                    plot!(pA, q_es[above_ss_pv], ep_real[above_ss_pv];
                          label = "", lw = 1, color = :blue, ls = :dash, alpha = 0.5)
                    plot!(pA, q_es[above_ss_pv], ev_real[above_ss_pv];
                          label = "", lw = 1, color = :red, ls = :dash, alpha = 0.5)
                end
            end
            hline!(pA, [0.0]; label = "et = 0", ls = :dot, color = :gray, lw = 1.5)
            vline!(pA, [q_at_psihigh]; label = @sprintf("psihigh: q = %.3f", q_at_psihigh),
                   ls = :dash, color = :gray)
            hline!(pA, [1.707]; label = "develop ref: et = +1.707",
                   ls = :dash, color = :green, lw = 1.5)
            scatter!(pA, [q_peak], [clamp(et_peak, -display_lim, display_lim)];
                     label = @sprintf("peak (q=%.3f, et=%+.3f)", q_peak, et_peak),
                     marker = :star5, ms = 10, color = :red)

            # --- Panel B: vacuum_eigenvalue — min eigenvalue of wv alone (EL-independent wv quality check) ---
            # Between rational surfaces singfac ≈ 1 so vacuum_eigenvalue ≈ O(1);
            # at rational surfaces singfac → 0 so vacuum_eigenvalue → 0.
            # Physically vacuum_eigenvalue >= 0; negative values are artifacts from beyond
            # rzphi geometry (wv computed via ExtendExtrap).  Filter to [0, clip_ev].
            pB = nothing
            if has_evonly
                clip_ev   = 5.0
                in_range  = map(x -> !isnan(x) && x >= 0.0 && x <= clip_ev, evonly_real)
                n_clipped = count(!isnan, evonly_real) - count(in_range)
                valid_ev2 = findall(in_range)
                stride_all = max(1, length(valid_ev2) ÷ 4000)
                ss_all = valid_ev2[1:stride_all:end]
                ev_max = isempty(valid_ev2) ? 1.0 : maximum(evonly_real[valid_ev2])

                pB = plot(;
                    xlabel = "q (safety factor)",
                    ylabel = "vacuum_eigenvalue  (min eigval of wv)",
                    title  = @sprintf("wv quality: vacuum_eigenvalue (EL-independent)  [%d steps outside [0,%.0f]]",
                                      n_clipped, clip_ev),
                    legend = :topright,
                    xlims  = (minimum(q_es) * 0.98, qlim_bench * 1.02),
                    ylims  = (0.0, ev_max * 1.15 + 0.01),
                    size   = (900, 300)
                )
                plot!(pB, q_es[ss_all], evonly_real[ss_all];
                      label = "vacuum_eigenvalue (min eigval wv)", lw = 1, color = :purple, alpha = 0.8)
                vline!(pB, [q_at_psihigh]; label = @sprintf("psihigh: q=%.3f", q_at_psihigh),
                       ls = :dash, color = :gray)
                if in_range[et_peak_idx]
                    scatter!(pB, [q_peak], [evonly_real[et_peak_idx]];
                             label = @sprintf("vacuum_eigenvalue at peak (%.4f)", evonly_real[et_peak_idx]),
                             marker = :star5, ms = 8, color = :purple)
                end
            end

            panels = filter(!isnothing, [pA, pB])
            p7 = length(panels) == 1 ? panels[1] :
                 plot(panels...; layout=(length(panels), 1), size=(900, 350 * length(panels)))
            savefig(p7, joinpath(output_dir, "edge_spline_stability.png"))
            println("Saved: edge_spline_stability.png")
        else
            println("Note: edge_scan/psi not found in gpec.h5 — run GPEC with psiedge < psilim first")
        end
    end
else
    println("Note: gpec.h5 not found in $output_dir — run GPEC first to generate edge stability plot")
end

# --- Plot 8: X-point asymptotic geometry — rfac(ψ) at selected θ values ---
# Shows how the rzphi spline extends into the far edge (psin > psihigh) via X-point scaling.
# rfac(ψ, θ) = √((R-Ro)² + (Z-Zo)²) should shrink toward zero as ψ → 1 (X-point approach).
if !isnothing(pe.params.r_xpoint)
    psi_far_plot = range(psihigh - 0.005, pe.rzphi_xs[end]; length=300)
    theta_plot_vals = [0.0, 0.25, 0.5, 0.75]
    theta_plot_labels = ["θ=0.00 (outboard)", "θ=0.25 (top)", "θ=0.50 (inboard)", "θ=0.75 (bottom)"]
    theta_plot_colors = [:blue, :orange, :green, :red]

    # rfac from the extended rzphi spline
    rfac_vals = [sqrt(max(0.0, pe.rzphi_rsquared((psi, theta)))) for psi in psi_far_plot, theta in theta_plot_vals]

    # Expected X-point asymptotic limit: rfac at ψ → 1 should approach rfac_xpoint(θ) = 0
    # (the X-point is also at the magnetic axis distance from ro, zo)
    r_x, z_x = pe.params.r_xpoint, pe.params.z_xpoint
    rfac_xpoint = sqrt((r_x - pe.ro)^2 + (z_x - pe.zo)^2)

    p8 = plot(; xlabel="ψₙ", ylabel="rfac = √((R-R₀)² + (Z-Z₀)²) [m]",
        title="X-point asymptotic geometry: rfac(ψ) in far edge",
        legend=:topleft)
    for (k, (theta, lbl)) in enumerate(zip(theta_plot_vals, theta_plot_labels))
        plot!(p8, collect(psi_far_plot), rfac_vals[:, k]; label=lbl, lw=2, color=theta_plot_colors[k])
    end
    hline!(p8, [rfac_xpoint]; label=@sprintf("rfac at X-point = %.4f m", rfac_xpoint), ls=:dot, color=:black, lw=1.5)
    vline!(p8, [psihigh]; label=@sprintf("psihigh=%.3f", psihigh), ls=:dash, color=:gray)

    savefig(p8, joinpath(output_dir, "edge_spline_geometry.png"))
    println("Saved: edge_spline_geometry.png")
else
    println("Note: X-point not detected (limited plasma) — skipping Plot 8")
end

# Extend the rzphi spline into the far-edge zone so the GSE diagnostic can evaluate
# GS residuals beyond psihigh.  Normally this extension is triggered by ForceFreeStates
# after sing_find!, but the benchmark only calls setup_equilibrium.
if !isnothing(profiles.iota_spline) && !isempty(edge_surfaces)
    GeneralizedPerturbedEquilibrium.Equilibrium.equilibrium_extend_rzphi!(pe, edge_surfaces)
end

# --- Plots 9 and 10: GS residual diagnostics ---
# Re-run equilibrium_gse! with diagnose_src=true to write gsei.h5 and gsec.h5,
# then read and plot the θ-integrated GS error per flux surface.
println("Running GSE diagnostics (diagnose_src=true)...")
pe.params.diagnose_src = true
GeneralizedPerturbedEquilibrium.Equilibrium.equilibrium_gse!(pe)
pe.params.diagnose_src = false

# --- Compute far-edge GSE using X-point asymptotic geometry with constant F ---
# With F ≈ const (dF/dψ ≈ 0) and P' ≈ 0 near the separatrix, the GS source term is ≈ 0.
# The far-edge rzphi splines now use X-point asymptotic geometry (shrinking toward X-point),
# which is GS-consistent in the Δ*ψ = 0 limit. This diagnostic confirms the improvement.
# Evaluated at both spline NODES and inter-node MIDPOINTS to check cubic spline quality.
far_edge_gse_available = false
psi_far_eval  = Float64[]
is_node_far   = Bool[]
gse_far_per_theta_mat = Matrix{Float64}(undef, 0, 4)
gse_far_int   = Float64[]
theta_labels_gse = ["θ=0.00 (outboard)", "θ=0.25 (top)", "θ=0.50 (inboard)", "θ=0.75 (bottom)"]
theta_colors_gse = [:blue, :orange, :green, :red]

if !isnothing(pe.params.r_xpoint)
    theta_all  = pe.rzphi_ys
    psio_gse   = pe.psio
    mtheta_bc  = length(theta_all) - 1
    dtheta_bc  = (theta_all[end] - theta_all[1]) / mtheta_bc
    theta_fracs_gse = [0.0, 0.25, 0.5, 0.75]
    theta_idxs_gse  = [max(1, round(Int, th * (length(theta_all)-1) + 1)) for th in theta_fracs_gse]

    # Far-edge psi grid: the extended rzphi spline covers [psilow, rzphi_xs[end]].
    # Use the far-edge portion (above psihigh) of the extended spline grid.
    psi_far_nodes = filter(psi -> psi > psihigh, pe.rzphi_xs)

    # Interleave midpoints with nodes
    for i in eachindex(psi_far_nodes)
        push!(psi_far_eval, psi_far_nodes[i])
        push!(is_node_far, true)
        if i < lastindex(psi_far_nodes)
            push!(psi_far_eval, (psi_far_nodes[i] + psi_far_nodes[i+1]) / 2)
            push!(is_node_far, false)
        end
    end

    gse_far_per_theta_mat = zeros(Float64, length(psi_far_eval), 4)
    gse_far_int = zeros(Float64, length(psi_far_eval))

    for (ipsi, psi) in enumerate(psi_far_eval)
        # F and P from the profiles (F is constant via ExtendExtrap above psihigh; F'≈0, P'≈0)
        F_val   = profiles.F_spline(psi)
        dF_dpsi = profiles.F_deriv(psi)
        dP_dpsi = profiles.P_deriv(psi)

        source_theta = zeros(Float64, length(theta_all))
        for (itheta, theta) in enumerate(theta_all)
            f4   = pe.rzphi_jac((psi, theta))   # J_coord from extended spline
            R, _ = RZ_at(pe, psi, theta)         # R from extended spline
            source_theta[itheta] = f4 / (2π * psio_gse * π^2) *
                                   (F_val * dF_dpsi / (2π * R)^2 + dP_dpsi)
        end

        for (k, itheta) in enumerate(theta_idxs_gse)
            gse_far_per_theta_mat[ipsi, k] = abs(source_theta[itheta])
        end

        gse_far_int[ipsi] = abs(sum(source_theta[1:mtheta_bc]) * dtheta_bc)
    end

    if !isempty(gse_far_int)
        max_far_gse = maximum(gse_far_int)
        println(@sprintf("  Far-edge GSE (X-point asymptotic): max θ-integrated |source| = %.2e", max_far_gse))
        far_edge_gse_available = true
    end
end

gsec_file = joinpath(output_dir, "gsec.h5")
gsei_file = joinpath(output_dir, "gsei.h5")

if isfile(gsec_file) && isfile(gsei_file)
    # --- Plot 9: GS residual by poloidal angle from gsec.h5 ---
    h5open(gsec_file, "r") do f
        flux_fsx_data = read(f["flux_fsx"])  # gs_div_dpsi[:,:,1]
        source_data   = read(f["source"])

        mt_gse = size(flux_fsx_data, 2)
        theta_fracs = [0.0, 0.25, 0.5, 0.75]
        theta_labels = ["θ=0.00 (outboard)", "θ=0.25 (top)", "θ=0.50 (inboard)", "θ=0.75 (bottom)"]
        theta_idxs = [max(1, round(Int, th * (mt_gse-1) + 1)) for th in theta_fracs]
        # Use core profile grid (not the extended rzphi_xs which includes far-edge nodes)
        psi_gse_vals = profiles.xs
        q_gse_vals   = psi_to_q_outer(psi_gse_vals)

        # Build residual curves once for reuse in both panels
        res_curves = [(abs.(flux_fsx_data[:, idx] .+ source_data[:, idx]), lbl, theta_colors_gse[k])
                      for (k, (idx, lbl)) in enumerate(zip(theta_idxs, theta_labels))]

        function make_p9_panel(xvals_core, xfar, xlabel_str, vline_val; xlims_val=nothing)
            p = plot(; xlabel=xlabel_str, ylabel="|GS residual (per θ)|",
                title="GS residual by poloidal angle (core + far edge)", yscale=:log10, legend=:topleft)
            for (res_col, lbl, col) in res_curves
                plot!(p, xvals_core, max.(res_col, 1e-15); label=lbl, lw=1.5, color=col)
            end
            if far_edge_gse_available
                for (k, _) in enumerate(theta_labels_gse)
                    far_lbl = k == 1 ? "far edge, X-point geometry (dashed)" : ""
                    plot!(p, xfar, max.(gse_far_per_theta_mat[:, k], 1e-15);
                          label=far_lbl, lw=1.5, ls=:dash, color=theta_colors_gse[k], alpha=0.8)
                end
            end
            hline!(p, [1e-2]; label="warning threshold", ls=:dot, color=:black, lw=1.5)
            vline!(p, [vline_val]; label=@sprintf("psihigh: %s=%.3f", xlabel_str, vline_val),
                   ls=:dash, color=:gray)
            if !isnothing(xlims_val)
                xlims!(p, xlims_val)
                vline!(p, [qlim_bench]; label=@sprintf("psilim: q=%.1f", qlim_bench),
                       ls=:dashdot, color=:purple, lw=1.5)
            end
            return p
        end

        q_far_eval = far_edge_gse_available ? psi_to_q_outer(psi_far_eval) : Float64[]
        p9a = make_p9_panel(psi_gse_vals, psi_far_eval, "ψₙ", psihigh)
        p9b = make_p9_panel(q_gse_vals,   q_far_eval,   "q",  q_at_psihigh_direct;
                            xlims_val=(q_at_psihigh_direct * 0.9, qlim_bench * 1.05))
        p9 = plot(p9a, p9b; layout=(1,2), size=(1400, 500), bottom_margin=8Plots.mm)
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
        q_xs_gse = psi_to_q_outer(xs_gse)

        function make_p10_panel(xvals_core, xfar, xlabel_str, vline_val; xlims_val=nothing)
            leg_pos = isnothing(xlims_val) ? :topleft : :bottomright
            p = plot(xvals_core, max.(errori_vec, 1e-15);
                xlabel=xlabel_str, ylabel="|θ-integrated GS residual|",
                title="θ-integrated GS error: core + far-edge X-point geometry",
                label="core |gse_integrated|", lw=2, color=:blue,
                yscale=:log10, legend=leg_pos)
            if far_edge_gse_available
                node_mask_far = is_node_far
                mid_mask_far  = .!is_node_far
                scatter!(p, xfar[node_mask_far], max.(gse_far_int[node_mask_far], 1e-15);
                         label="far edge nodes (spline knots)", marker=:circle, ms=5,
                         color=:red, markerstrokewidth=0)
                scatter!(p, xfar[mid_mask_far], max.(gse_far_int[mid_mask_far], 1e-15);
                         label="far edge midpoints (interpolated)", marker=:diamond, ms=4,
                         color=:orange, alpha=0.8, markerstrokewidth=0)
            end
            hline!(p, [1e-2]; label="warning threshold (1e-2)", ls=:dot, color=:black, lw=1.5)
            vline!(p, [vline_val]; label=@sprintf("psihigh: %s=%.3f", xlabel_str, vline_val),
                   ls=:dash, color=:gray)
            if !isnothing(xlims_val)
                xlims!(p, xlims_val)
                vline!(p, [qlim_bench]; label=@sprintf("psilim: q=%.1f", qlim_bench),
                       ls=:dashdot, color=:purple, lw=1.5)
            end
            return p
        end

        q_far_eval = far_edge_gse_available ? psi_to_q_outer(psi_far_eval) : Float64[]
        p10a = make_p10_panel(xs_gse,   psi_far_eval, "ψₙ", psihigh)
        p10b = make_p10_panel(q_xs_gse, q_far_eval,   "q",  q_at_psihigh_direct;
                              xlims_val=(q_at_psihigh_direct * 0.9, qlim_bench * 1.05))
        p10 = plot(p10a, p10b; layout=(1,2), size=(1400, 500), bottom_margin=8Plots.mm)
        savefig(p10, joinpath(output_dir, "edge_spline_gse_integrated.png"))
        println("Saved: edge_spline_gse_integrated.png")
    end
else
    println("Note: GSE HDF5 files not found — skipping Plots 9 and 10")
end

# --- Plot 11: Far-edge q profile — EFIT direct vs iota inverse spline ---
# The X-point geometry approach uses the iota inverse spline for q in the far edge (via
# q_spline_iota_inverse), while F is held constant (ExtendExtrap). This plot shows how
# q from the two sources compares across the edge region.
if !isnothing(profiles.iota_spline)
    println("Computing far-edge q profile comparison...")

    psi_edge_gse = vcat(
        collect(range(psihigh - 0.005, psihigh, length=20)),
        collect(range(psihigh, pe.rzphi_xs[end]; length=280))
    )
    unique!(sort!(psi_edge_gse))

    # q from direct spline (constant ExtendExtrap beyond psihigh)
    q_direct_edge = profiles.q_spline.(psi_edge_gse)

    # q from iota inverse spline (correct q behavior toward separatrix)
    above_mask_gse = psi_edge_gse .>= psihigh
    q_iota_edge = map(psi_edge_gse) do psi
        eval_q(profiles, psi)
    end

    # Panel A: q_direct vs q_iota in the far edge
    p11a = plot(psi_edge_gse, q_direct_edge;
        xlabel="ψₙ", ylabel="q",
        title="Far-edge q: EFIT ExtendExtrap vs iota inverse",
        label="q_direct (EFIT, ExtendExtrap beyond psihigh)", lw=2, color=:orange, ls=:dash,
        legend=:topleft, ylims=(0.0, qlim_bench))
    plot!(p11a, psi_edge_gse[above_mask_gse], q_iota_edge[above_mask_gse];
        label="q = 1/ι(ψ)  [iota inverse spline, used in ODE]", lw=2, color=:blue)
    vline!(p11a, [psihigh]; label=@sprintf("psihigh=%.4f", psihigh), ls=:dash, color=:gray)

    # Panel B: F(ψ) in the far edge (should be approximately constant)
    F_edge = profiles.F_spline.(psi_edge_gse)
    dF_edge = profiles.F_deriv.(psi_edge_gse)
    p11b = plot(psi_edge_gse, F_edge;
        xlabel="ψₙ", ylabel="F = 2π·R·Bₜ",
        title="F(ψ) in far edge (constant via ExtendExtrap → F'≈0 → GS source ≈ 0)",
        label="F_spline", lw=2, color=:blue, legend=:topleft)
    vline!(p11b, [psihigh]; label=@sprintf("psihigh=%.4f", psihigh), ls=:dash, color=:gray)

    # Panel C: dF/dψ (should be near zero above psihigh confirming GS source ≈ 0)
    p11c = plot(psi_edge_gse, abs.(dF_edge);
        xlabel="ψₙ", ylabel="|dF/dψ|",
        title="|dF/dψ| in far edge (proxy for GS source = F·F'/R²)",
        label="|dF/dψ|", lw=2, color=:red, yscale=:log10, legend=:topleft)
    vline!(p11c, [psihigh]; label=@sprintf("psihigh=%.4f", psihigh), ls=:dash, color=:gray)

    # p1 (iota) is included here so all spline quality plots are in one file
    p11 = plot(p11a, p11b, p11c, p1; layout=(4,1), size=(800, 1100))
    savefig(p11, joinpath(output_dir, "edge_spline_profiles.png"))
    println("Saved: edge_spline_profiles.png")
else
    println("Note: iota inverse spline not available (limited plasma) — skipping Plot 11")
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
