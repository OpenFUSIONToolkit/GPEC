"""
equil_spline_comparison.jl

Detailed comparison of equilibrium descriptions produced by the EFIT equilibrium
construction approaches ("efit" direct solver vs "efit_by_inversion" inverse solver),
focusing on understanding why efit_by_inversion gives erroneous stability (et[1]) values
for certain psihigh choices.

Produces (saved to benchmarks/equil_spline_comparison_psihigh<value>/):
  1. 1D profile comparisons: F, P, dV/dψ, q with full-domain, core zoom, edge zoom, and diff subplots
  2. Flux surface contour overlays: psi(R,Z) with full, core zoom, and x-point zoom panels
  3. 2D rzphi spline profiles at θ = 0, 0.25, 0.5, 0.75 for each of the 4 geometric splines
  4. 2D eqfun physics spline profiles at the same theta slices
  5. Printed numerical summaries of deep-core and far-edge differences for every spline

Usage:
  julia --project=. benchmarks/equil_spline_comparison.jl [example_path] [psihigh]

psihigh values of interest:
  0.996  → poor et[1] agreement
  0.997  → ok   et[1] agreement
  0.999  → poor et[1] agreement again
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using TOML, Printf, Statistics, Plots

# ─── CLI arguments ─────────────────────────────────────────────────────────────
example_path = length(ARGS) > 0 ? ARGS[1] :
    joinpath(@__DIR__, "../examples/DIIID-like_ideal_example")
psihigh_arg = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 0.997
config_path  = joinpath(example_path, "gpec.toml")

println("=" ^ 70)
println("Equilibrium Spline Comparison: efit vs efit_by_inversion")
println("Example  : $example_path")
println("psihigh  : $psihigh_arg")
println("=" ^ 70)

# ─── Output directory ──────────────────────────────────────────────────────────
psihigh_tag = replace(string(psihigh_arg), "." => "p")
outdir = joinpath(@__DIR__, "equil_spline_comparison_psihigh$(psihigh_tag)")
mkpath(outdir)
println("Output   : $outdir\n")

# ─── Config factory ────────────────────────────────────────────────────────────
function make_config(path::String, eq_type::String, psihigh::Float64)
    raw = TOML.parsefile(path)
    raw["Equilibrium"]["eq_type"]  = eq_type
    raw["Equilibrium"]["psihigh"] = psihigh
    return Equilibrium.EquilibriumConfig(raw["Equilibrium"], dirname(path))
end

# ─── Load both equilibria ──────────────────────────────────────────────────────
methods = ["efit", "efit_by_inversion"]
pes     = Dict{String, Any}()
for m in methods
    println("--- Running: $m ---")
    cfg = make_config(config_path, m, psihigh_arg)
    pe  = setup_equilibrium(cfg)
    pes[m] = pe
    println("  Done.\n")
end

pe_dir = pes["efit"]
pe_inv = pes["efit_by_inversion"]

# ─── Helpers ───────────────────────────────────────────────────────────────────
"Convert (ψ, θ) straight-field-line coordinates → (R, Z) physical coordinates."
function psi_theta_to_RZ(pe, psi, theta)
    r2   = pe.rzphi_rsquared((psi, theta))
    off  = pe.rzphi_offset((psi, theta))
    rfac = sqrt(max(r2, 0.0))
    η    = 2π * (theta + off)
    R    = pe.ro + rfac * cos(η)
    Z    = pe.zo + rfac * sin(η)
    return R, Z
end

"Compute the full (R, Z) trace of each flux surface in psi_vals."
function flux_surface_RZ(pe, psi_vals, theta_range)
    R_all = [Float64[] for _ in 1:length(psi_vals)]
    Z_all = [Float64[] for _ in 1:length(psi_vals)]
    for (k, psi) in enumerate(psi_vals)
        for θ in theta_range
            R, Z = psi_theta_to_RZ(pe, psi, θ)
            push!(R_all[k], R)
            push!(Z_all[k], Z)
        end
        # Close each contour
        push!(R_all[k], R_all[k][1])
        push!(Z_all[k], Z_all[k][1])
    end
    return R_all, Z_all
end

# Common psi evaluation grids
psi_lo = pe_dir.rzphi_xs[1]
psi_hi = pe_dir.rzphi_xs[end]
psi_full = collect(range(psi_lo, psi_hi; length=500))
psi_core = collect(range(psi_lo, min(0.10, 0.5 * psi_hi); length=100))
psi_edge = collect(range(max(0.85, psi_lo + 0.5 * (psi_hi - psi_lo)), psi_hi; length=200))

mask_core = psi_full .< 0.10
mask_edge = psi_full .> 0.85

theta_select = [0.0, 0.25, 0.5, 0.75]
theta_colors = [:blue, :green, :darkorange, :purple]

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  1D PROFILE COMPARISONS
# ═══════════════════════════════════════════════════════════════════════════════
println("=" ^ 70)
println("1. 1D Profile Comparisons")
println("=" ^ 70)

profile_specs = [
    ("F  (2π·R·Bₜ)",  pe -> pe.profiles.F_spline),
    ("μ₀·P",          pe -> pe.profiles.P_spline),
    ("dV/dψ",         pe -> pe.profiles.dVdpsi_spline),
    ("q",             pe -> pe.profiles.q_spline),
]

for (idx, (pname, spl_getter)) in enumerate(profile_specs)
    spl_d = spl_getter(pe_dir)
    spl_i = spl_getter(pe_inv)

    y_d = [spl_d(ψ) for ψ in psi_full]
    y_i = [spl_i(ψ) for ψ in psi_full]
    Δy  = y_d .- y_i

    p_full = plot(psi_full, y_d; lw=2, color=:blue, label="efit",
        xlabel="ψ (normalized)", ylabel=pname,
        title="$pname — full domain  (psihigh=$psihigh_arg)")
    plot!(p_full, psi_full, y_i; lw=2, color=:red, ls=:dash, label="efit_by_inversion")

    p_core = plot(psi_full[mask_core], y_d[mask_core]; lw=2, color=:blue, label="efit",
        xlabel="ψ", ylabel=pname, title="Deep core  (ψ < 0.10)")
    plot!(p_core, psi_full[mask_core], y_i[mask_core]; lw=2, color=:red, ls=:dash, label="efit_by_inversion")

    p_edge = plot(psi_full[mask_edge], y_d[mask_edge]; lw=2, color=:blue, label="efit",
        xlabel="ψ", ylabel=pname, title="Far edge  (ψ > 0.85)")
    plot!(p_edge, psi_full[mask_edge], y_i[mask_edge]; lw=2, color=:red, ls=:dash, label="efit_by_inversion")

    p_diff = plot(psi_full, Δy; lw=1.5, color=:purple, label="direct − inverse",
        xlabel="ψ", ylabel="Δ$pname", title="Difference")
    hline!(p_diff, [0.0]; color=:black, lw=1, ls=:dot, label="")

    p_combo = plot(p_full, p_core, p_edge, p_diff;
        layout=(2, 2), size=(1400, 900))
    tag = replace(pname, r"[^A-Za-z0-9_]" => "_")
    savefig(p_combo, joinpath(outdir, "1d_profile_$(idx)_$(tag).png"))

    @printf("  %-14s  max|Δ|=%.3e  rms|Δ|=%.3e  edge(ψ>0.85) max|Δ|=%.3e\n",
        pname, maximum(abs.(Δy)), sqrt(mean(Δy .^ 2)),
        any(mask_edge) ? maximum(abs.(Δy[mask_edge])) : NaN)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FLUX SURFACE CONTOUR OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("2. Flux Surface Contour Overlays")
println("=" ^ 70)

psi_contours  = collect(range(psi_lo, psi_hi; length=20))
theta_contour = range(0.0, 1.0; length=256)

R_dir, Z_dir = flux_surface_RZ(pe_dir, psi_contours, theta_contour)
R_inv, Z_inv = flux_surface_RZ(pe_inv, psi_contours, theta_contour)

# Full-domain overlay
p_ctr_full = plot(; aspect_ratio=:equal,
    xlabel="R [m]", ylabel="Z [m]",
    title="Flux surfaces: efit vs efit_by_inversion  (psihigh=$psihigh_arg)")
for k in 1:length(psi_contours)
    ld = k == 1 ? "efit" : ""
    li = k == 1 ? "efit_by_inversion" : ""
    plot!(p_ctr_full, R_dir[k], Z_dir[k]; color=:blue, lw=0.9, alpha=0.7, label=ld)
    plot!(p_ctr_full, R_inv[k], Z_inv[k]; color=:red,  lw=0.9, alpha=0.7, ls=:dash, label=li)
end

# Deep core zoom (innermost 4 surfaces)
n_core = 4
p_ctr_core = plot(; aspect_ratio=:equal,
    xlabel="R [m]", ylabel="Z [m]",
    title="Deep core zoom (innermost $(n_core) surfaces)")
for k in 1:min(n_core, length(psi_contours))
    ld = k == 1 ? "efit" : ""
    li = k == 1 ? "efit_by_inversion" : ""
    plot!(p_ctr_core, R_dir[k], Z_dir[k]; color=:blue, lw=1.5, label=ld)
    plot!(p_ctr_core, R_inv[k], Z_inv[k]; color=:red,  lw=1.5, ls=:dash, label=li)
end
all_R_c = vcat(R_dir[1:n_core]..., R_inv[1:n_core]...)
all_Z_c = vcat(Z_dir[1:n_core]..., Z_inv[1:n_core]...)
pad = 0.03
xlims!(p_ctr_core, minimum(all_R_c) - pad, maximum(all_R_c) + pad)
ylims!(p_ctr_core, minimum(all_Z_c) - pad, maximum(all_Z_c) + pad)

# X-point (far edge) zoom — outermost 3 surfaces, centred on lowest-Z point
n_outer = length(psi_contours)
n_edge  = min(3, n_outer)
idx_edge = (n_outer - n_edge + 1):n_outer
p_ctr_xpt = plot(; aspect_ratio=:equal,
    xlabel="R [m]", ylabel="Z [m]",
    title="Far edge / x-point zoom (outermost $(n_edge) surfaces)")
for (cnt, k) in enumerate(idx_edge)
    ld = cnt == 1 ? "efit" : ""
    li = cnt == 1 ? "efit_by_inversion" : ""
    plot!(p_ctr_xpt, R_dir[k], Z_dir[k]; color=:blue, lw=1.5, label=ld)
    plot!(p_ctr_xpt, R_inv[k], Z_inv[k]; color=:red,  lw=1.5, ls=:dash, label=li)
end
all_R_e = vcat(R_dir[idx_edge]..., R_inv[idx_edge]...)
all_Z_e = vcat(Z_dir[idx_edge]..., Z_inv[idx_edge]...)
xpt_idx = argmin(all_Z_e)
xpt_R   = all_R_e[xpt_idx]
xpt_Z   = all_Z_e[xpt_idx]
xlims!(p_ctr_xpt, xpt_R - 0.20, xpt_R + 0.20)
ylims!(p_ctr_xpt, xpt_Z - 0.05, xpt_Z + 0.40)

p_ctr_combo = plot(p_ctr_full, p_ctr_core, p_ctr_xpt;
    layout=(1, 3), size=(2100, 800))
savefig(p_ctr_combo, joinpath(outdir, "flux_contours_combined.png"))
println("  Saved flux contour figures.")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  2D RZPHI GEOMETRIC SPLINES AT SELECT THETA SLICES
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("3. 2D rzphi Geometric Spline Profiles at Select θ Values")
println("=" ^ 70)

rzphi_specs = [
    ("r²=(R-R₀)²+(Z-Z₀)²",  pe -> ((ψ, θ) -> pe.rzphi_rsquared((ψ, θ)))),
    ("angle offset  η/2π−θ", pe -> ((ψ, θ) -> pe.rzphi_offset((ψ, θ)))),
    ("ν  (toroidal shift)",  pe -> ((ψ, θ) -> pe.rzphi_nu((ψ, θ)))),
    ("Jacobian",             pe -> ((ψ, θ) -> pe.rzphi_jac((ψ, θ)))),
]

for (sidx, (sname, fn_getter)) in enumerate(rzphi_specs)
    fn_d = fn_getter(pe_dir)
    fn_i = fn_getter(pe_inv)

    yd_all = [[fn_d(ψ, θ) for ψ in psi_full] for θ in theta_select]
    yi_all = [[fn_i(ψ, θ) for ψ in psi_full] for θ in theta_select]

    p_s_full = plot(; xlabel="ψ", ylabel=sname,
        title="$sname — full domain  (psihigh=$psihigh_arg)")
    p_s_core = plot(; xlabel="ψ", ylabel=sname, title="Deep core  (ψ < 0.10)")
    p_s_edge = plot(; xlabel="ψ", ylabel=sname, title="Far edge  (ψ > 0.85)")
    p_s_diff = plot(; xlabel="ψ", ylabel="Δ$sname", title="direct − inverse")

    for (tidx, θ) in enumerate(theta_select)
        c  = theta_colors[tidx]
        yd = yd_all[tidx]
        yi = yi_all[tidx]
        Δ  = yd .- yi

        plot!(p_s_full, psi_full, yd; color=c, lw=2,   label="efit  θ=$(θ)")
        plot!(p_s_full, psi_full, yi; color=c, lw=1.5, ls=:dash, label="inv. θ=$(θ)")

        plot!(p_s_core, psi_full[mask_core], yd[mask_core]; color=c, lw=2,   label="efit θ=$(θ)")
        plot!(p_s_core, psi_full[mask_core], yi[mask_core]; color=c, lw=1.5, ls=:dash, label="inv. θ=$(θ)")

        plot!(p_s_edge, psi_full[mask_edge], yd[mask_edge]; color=c, lw=2,   label="efit θ=$(θ)")
        plot!(p_s_edge, psi_full[mask_edge], yi[mask_edge]; color=c, lw=1.5, ls=:dash, label="inv. θ=$(θ)")

        plot!(p_s_diff, psi_full, Δ; color=c, lw=1.5, label="Δ θ=$(θ)")
    end
    hline!(p_s_diff, [0.0]; color=:black, lw=1, ls=:dot, label="")

    p_combo_s = plot(p_s_full, p_s_core, p_s_edge, p_s_diff;
        layout=(2, 2), size=(1400, 900))
    tag = replace(sname, r"[^A-Za-z0-9_]" => "_")[1:min(30, end)]
    savefig(p_combo_s, joinpath(outdir, "rzphi_spline_$(sidx)_$(tag).png"))

    println("  $sname:")
    for (tidx, θ) in enumerate(theta_select)
        Δ = yd_all[tidx] .- yi_all[tidx]
        Δc = any(mask_core) ? Δ[mask_core] : [0.0]
        Δe = any(mask_edge) ? Δ[mask_edge] : [0.0]
        @printf("    θ=%.2f  core max|Δ|=%.3e  edge max|Δ|=%.3e  edge rms=%.3e\n",
            θ, maximum(abs.(Δc)), maximum(abs.(Δe)), sqrt(mean(Δe .^ 2)))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  EQFUN PHYSICS SPLINES AT SELECT THETA SLICES
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("4. Physics (eqfun) Spline Profiles at Select θ Values")
println("=" ^ 70)

eqfun_specs = [
    ("|B| (total field [T])",        pe -> ((ψ, θ) -> pe.eqfun_B((ψ, θ)))),
    ("metric1  (e₁·e₂+q·e₃·e₁)/JB²", pe -> ((ψ, θ) -> pe.eqfun_metric1((ψ, θ)))),
    ("metric2  (e₂·e₃+q·e₃²)/JB²",   pe -> ((ψ, θ) -> pe.eqfun_metric2((ψ, θ)))),
]

for (eidx, (ename, fn_getter)) in enumerate(eqfun_specs)
    fn_d = fn_getter(pe_dir)
    fn_i = fn_getter(pe_inv)

    yd_all = [[fn_d(ψ, θ) for ψ in psi_full] for θ in theta_select]
    yi_all = [[fn_i(ψ, θ) for ψ in psi_full] for θ in theta_select]

    p_e_full = plot(; xlabel="ψ", ylabel=ename,
        title="$ename — full domain  (psihigh=$psihigh_arg)")
    p_e_core = plot(; xlabel="ψ", ylabel=ename, title="Deep core  (ψ < 0.10)")
    p_e_edge = plot(; xlabel="ψ", ylabel=ename, title="Far edge  (ψ > 0.85)")
    p_e_diff = plot(; xlabel="ψ", ylabel="Δ$ename", title="direct − inverse")

    for (tidx, θ) in enumerate(theta_select)
        c  = theta_colors[tidx]
        yd = yd_all[tidx]
        yi = yi_all[tidx]
        Δ  = yd .- yi

        plot!(p_e_full, psi_full, yd; color=c, lw=2,   label="efit  θ=$(θ)")
        plot!(p_e_full, psi_full, yi; color=c, lw=1.5, ls=:dash, label="inv. θ=$(θ)")

        plot!(p_e_core, psi_full[mask_core], yd[mask_core]; color=c, lw=2,   label="efit θ=$(θ)")
        plot!(p_e_core, psi_full[mask_core], yi[mask_core]; color=c, lw=1.5, ls=:dash, label="inv. θ=$(θ)")

        plot!(p_e_edge, psi_full[mask_edge], yd[mask_edge]; color=c, lw=2,   label="efit θ=$(θ)")
        plot!(p_e_edge, psi_full[mask_edge], yi[mask_edge]; color=c, lw=1.5, ls=:dash, label="inv. θ=$(θ)")

        plot!(p_e_diff, psi_full, Δ; color=c, lw=1.5, label="Δ θ=$(θ)")
    end
    hline!(p_e_diff, [0.0]; color=:black, lw=1, ls=:dot, label="")

    p_combo_e = plot(p_e_full, p_e_core, p_e_edge, p_e_diff;
        layout=(2, 2), size=(1400, 900))
    tag = replace(ename, r"[^A-Za-z0-9_]" => "_")[1:min(30, end)]
    savefig(p_combo_e, joinpath(outdir, "eqfun_$(eidx)_$(tag).png"))

    println("  $ename:")
    for (tidx, θ) in enumerate(theta_select)
        Δ = yd_all[tidx] .- yi_all[tidx]
        Δc = any(mask_core) ? Δ[mask_core] : [0.0]
        Δe = any(mask_edge) ? Δ[mask_edge] : [0.0]
        @printf("    θ=%.2f  core max|Δ|=%.3e  edge max|Δ|=%.3e  edge rms=%.3e\n",
            θ, maximum(abs.(Δc)), maximum(abs.(Δe)), sqrt(mean(Δe .^ 2)))
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
println("\n" * "=" ^ 70)
println("Summary: axis / edge parameter comparison")
println("=" ^ 70)

@printf("  %-22s  %12s  %12s\n", "Parameter", "efit (direct)", "efit_by_inv")
@printf("  %-22s  %12.6f  %12.6f\n", "ro [m]", pe_dir.ro, pe_inv.ro)
@printf("  %-22s  %12.6f  %12.6f\n", "zo [m]", pe_dir.zo, pe_inv.zo)
@printf("  %-22s  %12.6f  %12.6f\n", "psio [Wb/rad]", pe_dir.psio, pe_inv.psio)
@printf("  %-22s  %12.6f  %12.6f\n", "psilow", pe_dir.rzphi_xs[1], pe_inv.rzphi_xs[1])
@printf("  %-22s  %12.6f  %12.6f\n", "psihigh", pe_dir.rzphi_xs[end], pe_inv.rzphi_xs[end])
@printf("  %-22s  %12.6f  %12.6f\n", "q(psilow)",
    pe_dir.profiles.q_spline(pe_dir.rzphi_xs[1]),
    pe_inv.profiles.q_spline(pe_inv.rzphi_xs[1]))
@printf("  %-22s  %12.6f  %12.6f\n", "q(psihigh)",
    pe_dir.profiles.q_spline(pe_dir.rzphi_xs[end]),
    pe_inv.profiles.q_spline(pe_inv.rzphi_xs[end]))

# Q-monotonicity check
q_dir = [pe_dir.profiles.q_spline(ψ) for ψ in psi_full]
q_inv = [pe_inv.profiles.q_spline(ψ) for ψ in psi_full]
dq_dir = diff(q_dir)
dq_inv = diff(q_inv)
n_nonmono_dir_edge = count(dq_dir[mask_edge[2:end]] .< 0)
n_nonmono_inv_edge = count(dq_inv[mask_edge[2:end]] .< 0)
@printf("  %-22s  %12d  %12d\n", "q non-mono steps (edge)", n_nonmono_dir_edge, n_nonmono_inv_edge)

println("\nAll figures saved to: $outdir")
