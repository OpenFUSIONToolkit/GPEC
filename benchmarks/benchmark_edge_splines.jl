"""
Benchmark and visualization script for edge inverse splines (near-separatrix extension).

This script demonstrates the edge inverse spline construction introduced for diverted
plasmas, allowing JPEC to integrate beyond psihigh toward the separatrix.

For a diverted plasma, it produces the following plots saved alongside the input toml:
  1. iota (= 1/q) vs psin: shows the iota inner spline descending smoothly to 0 at psin=1
  2. q vs psin: compares the direct spline (ExtendExtrap) vs the edge inverse spline
  3. dV/dψ vs psin: same comparison
  4. Rational surface density in the edge zone

For a limited plasma (no x-point), the script prints a summary and exits since no
edge inverse splines are built.

Usage:
    julia --project=. benchmarks/benchmark_edge_splines.jl [path/to/jpec.toml]

If no path is given, defaults to examples/DIIID-like_ideal_example/jpec.toml.
"""

using JPEC
using JPEC.Equilibrium: InverseCubicSpline
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

# --- Plot 4: Rational surface density in edge zone ---
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

p4 = plot(
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
    vline!(p4, [psi_s]; label = lbl, color = :red, alpha = 0.5, lw = 0.8)
end
vline!(p4, [psihigh]; label = @sprintf("psihigh = %.3f", psihigh), ls = :dash, color = :gray)
savefig(p4, joinpath(output_dir, "edge_spline_rational_surfaces.png"))
println("Saved: edge_spline_rational_surfaces.png")

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
