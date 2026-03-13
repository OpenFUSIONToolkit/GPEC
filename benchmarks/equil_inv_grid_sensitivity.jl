"""
equil_inv_grid_sensitivity.jl

Convergence study for the efit_by_inversion Cartesian grid parameters.

Sweeps:
  A. refine ∈ {2, 3, 4, 5, 6, 8}  at β_z = 2.0  (legacy refine parameter)
  B. β_z   ∈ {0, 0.5, 1, 1.5, 2, 3, 4, 6}  at refine = 5  (legacy β_z sweep)
  C. resolution_factor ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}  (adaptive grid)

Metrics (evaluated in the outer 20% of the radial domain, ψ > 0.80):
  - max |Δq(ψ)|      vs efit and vs high-res inversion reference
  - max |ΔdV/dψ(ψ)|  vs efit and vs high-res inversion reference
  - max roundtrip error (ψ, θ) → (R, Z) → ψ_spline at edge
  - max |Δr²|, |Δoffset|, |ΔJac| of 2D splines at θ = 0, 0.25, 0.5, 0.75

Outputs:
  conv_refine.png    — metric vs refine (3 rows × 2 cols)
  conv_bz.png        — metric vs β_z    (3 rows × 2 cols)
  conv_rf.png        — metric vs resolution_factor (adaptive, 3 rows × 2 cols)
  pareto.png         — roundtrip error vs runtime (Pareto view, all sweeps)
  surfaces_xpt.png   — R(θ), Z(θ) near x-point for each refine value
  hreq_profile.png   — required cell widths dR_req(R), dZ_req(Z) along LCFS

Usage:
  julia --project=. benchmarks/equil_inv_grid_sensitivity.jl [example_path]
"""

using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium
using TOML, Printf, Statistics

example_path = length(ARGS) > 0 ? ARGS[1] : joinpath(@__DIR__, "../examples/DIIID-like_ideal_example")
config_path  = joinpath(example_path, "gpec.toml")

outdir = joinpath(@__DIR__, "equil_inv_grid_sensitivity")
mkpath(outdir)
println("Output   : $outdir\n")

# ── helpers ───────────────────────────────────────────────────────────────────

function make_config(path, eq_type; mpsi=128, mtheta=256)
    raw = TOML.parsefile(path)
    raw["Equilibrium"]["eq_type"]  = eq_type
    raw["Equilibrium"]["mpsi"]     = mpsi
    raw["Equilibrium"]["mtheta"]   = mtheta
    return Equilibrium.EquilibriumConfig(raw["Equilibrium"], dirname(path))
end

"""Run efit_by_inversion with legacy refine/β_z override kwargs, return (pe, runtime)."""
function run_inv(cfg, refine::Int, β_z::Float64)
    raw = Equilibrium.read_efit(cfg)
    t = @elapsed begin
        pe = Equilibrium.equilibrium_solver_by_inversion(raw; refine=refine, β_z=β_z)
        Equilibrium.equilibrium_global_parameters!(pe)
        Equilibrium.equilibrium_qfind!(pe)
    end
    return pe, t
end

"""Run efit_by_inversion with adaptive grid (resolution_factor only), return (pe, runtime)."""
function run_inv_adaptive(cfg; resolution_factor::Float64=1.0)
    raw = Equilibrium.read_efit(cfg)
    t = @elapsed begin
        pe = Equilibrium.equilibrium_solver_by_inversion(raw; resolution_factor=resolution_factor)
        Equilibrium.equilibrium_global_parameters!(pe)
        Equilibrium.equilibrium_qfind!(pe)
    end
    return pe, t
end

"""Evaluate 1D profile errors on psi_eval grid (relative to reference arrays)."""
function profile_errors(pe, ref_q, ref_dvdpsi, psi_eval)
    q_errs   = Float64[]
    dv_errs  = Float64[]
    for ψ in psi_eval
        push!(q_errs,  abs(pe.profiles.q_spline(ψ)      - ref_q(ψ)))
        push!(dv_errs, abs(pe.profiles.dVdpsi_spline(ψ) - ref_dvdpsi(ψ)))
    end
    return maximum(q_errs), maximum(dv_errs)
end

"""Evaluate 2D spline errors at (psi_eval, θ_vals) vs reference pe."""
function spline2d_errors(pe, ref_pe, psi_eval, θ_vals)
    r2_err  = Dict(θ => Float64[] for θ in θ_vals)
    off_err = Dict(θ => Float64[] for θ in θ_vals)
    jac_err = Dict(θ => Float64[] for θ in θ_vals)
    for ψ in psi_eval, θ in θ_vals
        push!(r2_err[θ],  abs(pe.rzphi_rsquared((ψ, θ)) - ref_pe.rzphi_rsquared((ψ, θ))))
        push!(off_err[θ], abs(pe.rzphi_offset((ψ, θ))   - ref_pe.rzphi_offset((ψ, θ))))
        push!(jac_err[θ], abs(pe.rzphi_jac((ψ, θ)) - ref_pe.rzphi_jac((ψ, θ))))
    end
    max_r2  = Dict(θ => maximum(r2_err[θ])  for θ in θ_vals)
    max_off = Dict(θ => maximum(off_err[θ]) for θ in θ_vals)
    max_jac = Dict(θ => maximum(jac_err[θ]) for θ in θ_vals)
    return max_r2, max_off, max_jac
end

"""Max roundtrip error over a sparse (ψ, θ) sample grid in edge region."""
function roundtrip_error_edge(pe, raw_profile; psi_min=0.80)
    psi_xs = pe.rzphi_xs
    psio   = pe.psio
    mtheta = length(pe.rzphi_ys) - 1
    errors = Float64[]
    for ipsi in 1:4:length(psi_xs)
        psi_xs[ipsi] < psi_min && continue
        ψ = psi_xs[ipsi]
        for itheta in 1:8:mtheta+1
            θ = pe.rzphi_ys[itheta]
            r2   = pe.rzphi_rsquared((ψ, θ))
            off  = pe.rzphi_offset((ψ, θ))
            rfac = sqrt(max(r2, 0.0))
            η    = 2π * (θ + off)
            R    = pe.ro + rfac * cos(η)
            Z    = pe.zo + rfac * sin(η)
            ψ_rt = 1.0 - raw_profile.psi_in((R, Z)) / psio
            push!(errors, abs(ψ_rt - ψ))
        end
    end
    return isempty(errors) ? NaN : maximum(errors)
end

# ── pre-computed references ────────────────────────────────────────────────────

println("Setting up references…")

cfg_base  = make_config(config_path, "efit_by_inversion")
raw_input = Equilibrium.read_efit(cfg_base)

# efit reference: standard direct equilibrium (absolute ground truth for profiles)
cfg_efit = make_config(config_path, "efit")
pe_efit  = Equilibrium.setup_equilibrium(cfg_efit)

# High-res inversion reference for self-convergence (refine=8, β_z=2)
println("  Computing high-res inversion reference (refine=8, β_z=2.0)…")
pe_hires, _ = run_inv(cfg_base, 8, 2.0)

# Adaptive reference (resolution_factor=1.0 — physics-based defaults)
println("  Computing adaptive reference (resolution_factor=1.0)…")
pe_adaptive, t_adaptive = run_inv_adaptive(cfg_base; resolution_factor=1.0)
println("  Adaptive run: $(t_adaptive)s")

# Evaluation grids
psi_eval = range(0.80, cfg_base.psihigh - 0.002, length=40) |> collect
θ_vals   = [0.0, 0.25, 0.5, 0.75]

# Spline accessors for efit reference
ref_q_efit     = pe_efit.profiles.q_spline
ref_dvdpsi_efit = pe_efit.profiles.dVdpsi_spline
ref_q_hires     = pe_hires.profiles.q_spline
ref_dvdpsi_hires = pe_hires.profiles.dVdpsi_spline

# JIT warmup
println("  JIT warmup…")
run_inv(cfg_base, 3, 2.0)
run_inv_adaptive(cfg_base; resolution_factor=1.0)
println("References ready.\n")

# ── Sweep A: refine ────────────────────────────────────────────────────────────

refine_vals = [2, 3, 4, 5, 6, 8]
β_z_fixed   = 2.0

println("=" ^ 65)
println("Sweep A: refine ∈ $refine_vals at β_z = $β_z_fixed")
println("=" ^ 65)

rows_A = []
for refine in refine_vals
    pe, t = run_inv(cfg_base, refine, β_z_fixed)

    q_efit,  dv_efit  = profile_errors(pe, ref_q_efit,  ref_dvdpsi_efit,  psi_eval)
    q_hires, dv_hires = profile_errors(pe, ref_q_hires, ref_dvdpsi_hires, psi_eval)
    rt_err = roundtrip_error_edge(pe, raw_input)
    r2_e, off_e, jac_e = spline2d_errors(pe, pe_efit,  psi_eval, θ_vals)
    r2_h, off_h, jac_h = spline2d_errors(pe, pe_hires, psi_eval, θ_vals)

    @printf("  refine=%d  t=%.2fs  rt=%.2e  q_efit=%.2e  q_hires=%.2e  dv_efit=%.2e  dv_hires=%.2e\n",
        refine, t, rt_err, q_efit, q_hires, dv_efit, dv_hires)

    push!(rows_A, (
        refine=refine, β_z=β_z_fixed, runtime=t,
        q_efit=q_efit, dv_efit=dv_efit,
        q_hires=q_hires, dv_hires=dv_hires,
        rt_err=rt_err,
        r2_efit  =maximum(values(r2_e)),
        off_efit =maximum(values(off_e)),
        jac_efit =maximum(values(jac_e)),
        r2_hires =maximum(values(r2_h)),
        off_hires=maximum(values(off_h)),
        jac_hires=maximum(values(jac_h)),
        r2_efit_per_theta  =r2_e,
        off_efit_per_theta =off_e,
        r2_hires_per_theta =r2_h,
        off_hires_per_theta=off_h,
    ))
end

# ── Sweep B: β_z ──────────────────────────────────────────────────────────────

bz_vals    = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
refine_fixed = 5

println("\n" * "=" ^ 65)
println("Sweep B: β_z ∈ $bz_vals at refine = $refine_fixed")
println("=" ^ 65)

rows_B = []
for β_z in bz_vals
    pe, t = run_inv(cfg_base, refine_fixed, β_z)

    q_efit,  dv_efit  = profile_errors(pe, ref_q_efit,  ref_dvdpsi_efit,  psi_eval)
    q_hires, dv_hires = profile_errors(pe, ref_q_hires, ref_dvdpsi_hires, psi_eval)
    rt_err = roundtrip_error_edge(pe, raw_input)
    r2_e, off_e, jac_e = spline2d_errors(pe, pe_efit,  psi_eval, θ_vals)
    r2_h, off_h, jac_h = spline2d_errors(pe, pe_hires, psi_eval, θ_vals)

    @printf("  β_z=%.2f  t=%.2fs  rt=%.2e  q_efit=%.2e  q_hires=%.2e  dv_efit=%.2e  dv_hires=%.2e\n",
        β_z, t, rt_err, q_efit, q_hires, dv_efit, dv_hires)

    push!(rows_B, (
        refine=refine_fixed, β_z=β_z, runtime=t,
        q_efit=q_efit, dv_efit=dv_efit,
        q_hires=q_hires, dv_hires=dv_hires,
        rt_err=rt_err,
        r2_efit  =maximum(values(r2_e)),
        off_efit =maximum(values(off_e)),
        jac_efit =maximum(values(jac_e)),
        r2_hires =maximum(values(r2_h)),
        off_hires=maximum(values(off_h)),
        jac_hires=maximum(values(jac_h)),
        r2_efit_per_theta  =r2_e,
        off_efit_per_theta =off_e,
        r2_hires_per_theta =r2_h,
        off_hires_per_theta=off_h,
    ))
end

# ── Sweep C: resolution_factor (adaptive grid) ────────────────────────────────

rf_vals = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

println("\n" * "=" ^ 65)
println("Sweep C: resolution_factor ∈ $rf_vals (adaptive grid)")
println("=" ^ 65)

rows_C = []
for rf in rf_vals
    pe, t = run_inv_adaptive(cfg_base; resolution_factor=rf)

    q_efit,  dv_efit  = profile_errors(pe, ref_q_efit,  ref_dvdpsi_efit,  psi_eval)
    q_hires, dv_hires = profile_errors(pe, ref_q_hires, ref_dvdpsi_hires, psi_eval)
    rt_err = roundtrip_error_edge(pe, raw_input)
    r2_e, off_e, jac_e = spline2d_errors(pe, pe_efit,  psi_eval, θ_vals)
    r2_h, off_h, jac_h = spline2d_errors(pe, pe_hires, psi_eval, θ_vals)

    marker = isapprox(rf, 1.0) ? " ← default" : ""
    @printf("  rf=%.2f  t=%.2fs  rt=%.2e  q_efit=%.2e  q_hires=%.2e  dv_efit=%.2e  dv_hires=%.2e%s\n",
        rf, t, rt_err, q_efit, q_hires, dv_efit, dv_hires, marker)

    push!(rows_C, (
        resolution_factor=rf, runtime=t,
        q_efit=q_efit, dv_efit=dv_efit,
        q_hires=q_hires, dv_hires=dv_hires,
        rt_err=rt_err,
        r2_efit  =maximum(values(r2_e)),
        off_efit =maximum(values(off_e)),
        jac_efit =maximum(values(jac_e)),
        r2_hires =maximum(values(r2_h)),
        off_hires=maximum(values(off_h)),
        jac_hires=maximum(values(jac_h)),
        r2_hires_per_theta=r2_h,
    ))
end

# ── CSV output ────────────────────────────────────────────────────────────────

function write_csv(rows, path)
    scalar_keys = [:refine, :β_z, :runtime, :q_efit, :dv_efit, :q_hires, :dv_hires,
                   :rt_err, :r2_efit, :off_efit, :jac_efit, :r2_hires, :off_hires, :jac_hires]
    open(path, "w") do io
        println(io, join(string.(scalar_keys), ","))
        for row in rows
            println(io, join([string(row[k]) for k in scalar_keys], ","))
        end
    end
end

function write_csv_C(rows, path)
    scalar_keys = [:resolution_factor, :runtime, :q_efit, :dv_efit, :q_hires, :dv_hires,
                   :rt_err, :r2_efit, :off_efit, :jac_efit, :r2_hires, :off_hires, :jac_hires]
    open(path, "w") do io
        println(io, join(string.(scalar_keys), ","))
        for row in rows
            println(io, join([string(row[k]) for k in scalar_keys], ","))
        end
    end
end

write_csv(rows_A, joinpath(outdir, "equil_inv_sweep_refine.csv"))
write_csv(rows_B, joinpath(outdir, "equil_inv_sweep_bz.csv"))
write_csv_C(rows_C, joinpath(outdir, "equil_inv_sweep_rf.csv"))
println("\nCSV results saved.")

# ── Plots ─────────────────────────────────────────────────────────────────────

try
    using Plots
    gr()

    θ_colors = [:royalblue, :crimson, :forestgreen, :darkorange]
    θ_labels = ["θ=0" "θ=0.25" "θ=0.5" "θ=0.75"]   # matrix row for Plots legend

    # ── Figure 1: conv_refine.png ─────────────────────────────────────────────
    xv = [r.refine for r in rows_A]

    p1 = plot(xv, [r.q_efit for r in rows_A];  yscale=:log10, marker=:circle,
              title="q vs efit", xlabel="refine", ylabel="|Δq| max (ψ>0.80)", legend=false)
    p2 = plot(xv, [r.q_hires for r in rows_A]; yscale=:log10, marker=:circle,
              title="q vs hi-res", xlabel="refine", ylabel="|Δq| max", legend=false)
    p3 = plot(xv, [r.dv_efit for r in rows_A]; yscale=:log10, marker=:circle,
              title="dV/dψ vs efit", xlabel="refine", ylabel="|ΔdV/dψ| max", legend=false)
    p4 = plot(xv, [r.dv_hires for r in rows_A]; yscale=:log10, marker=:circle,
              title="dV/dψ vs hi-res", xlabel="refine", ylabel="|ΔdV/dψ| max", legend=false)
    p5 = plot(xv, [r.rt_err for r in rows_A];  yscale=:log10, marker=:circle,
              title="roundtrip error (edge)", xlabel="refine", ylabel="max |Δψ|", legend=false)
    p6 = plot(; yscale=:log10, title="r² vs hi-res per θ", xlabel="refine", ylabel="|Δr²| max")
    for (iθ, θ) in enumerate(θ_vals)
        plot!(p6, xv, [r.r2_hires_per_theta[θ] for r in rows_A];
              marker=:circle, label=θ_labels[iθ], color=θ_colors[iθ])
    end

    savefig(plot(p1, p2, p3, p4, p5, p6; layout=(3,2), size=(900,800)),
            joinpath(outdir, "conv_refine.png"))
    println("Saved conv_refine.png")

    # ── Figure 2: conv_bz.png ─────────────────────────────────────────────────
    xv2 = [r.β_z for r in rows_B]

    q2a = plot(xv2, [r.q_efit for r in rows_B];  yscale=:log10, marker=:circle,
               title="q vs efit", xlabel="β_z", legend=false)
    q2b = plot(xv2, [r.q_hires for r in rows_B]; yscale=:log10, marker=:circle,
               title="q vs hi-res", xlabel="β_z", legend=false)
    d2a = plot(xv2, [r.dv_efit for r in rows_B];  yscale=:log10, marker=:circle,
               title="dV/dψ vs efit", xlabel="β_z", legend=false)
    d2b = plot(xv2, [r.dv_hires for r in rows_B]; yscale=:log10, marker=:circle,
               title="dV/dψ vs hi-res", xlabel="β_z", legend=false)
    rt2 = plot(xv2, [r.rt_err for r in rows_B];   yscale=:log10, marker=:circle,
               title="roundtrip error (edge)", xlabel="β_z", legend=false)
    r2b = plot(; yscale=:log10, title="r² vs hi-res per θ", xlabel="β_z", ylabel="|Δr²| max")
    for (iθ, θ) in enumerate(θ_vals)
        plot!(r2b, xv2, [r.r2_hires_per_theta[θ] for r in rows_B];
              marker=:circle, label=θ_labels[iθ], color=θ_colors[iθ])
    end

    savefig(plot(q2a, q2b, d2a, d2b, rt2, r2b; layout=(3,2), size=(900,800)),
            joinpath(outdir, "conv_bz.png"))
    println("Saved conv_bz.png")

    # ── Figure 2b: conv_rf.png (adaptive Sweep C) ─────────────────────────────
    xv3 = [r.resolution_factor for r in rows_C]

    q3a = plot(xv3, [r.q_efit for r in rows_C];  yscale=:log10, marker=:circle,
               title="q vs efit", xlabel="resolution_factor", legend=false)
    vline!(q3a, [1.0]; color=:gray, ls=:dash)
    q3b = plot(xv3, [r.q_hires for r in rows_C]; yscale=:log10, marker=:circle,
               title="q vs hi-res", xlabel="resolution_factor", legend=false)
    vline!(q3b, [1.0]; color=:gray, ls=:dash)
    d3a = plot(xv3, [r.dv_efit for r in rows_C];  yscale=:log10, marker=:circle,
               title="dV/dψ vs efit", xlabel="resolution_factor", legend=false)
    d3b = plot(xv3, [r.dv_hires for r in rows_C]; yscale=:log10, marker=:circle,
               title="dV/dψ vs hi-res", xlabel="resolution_factor", legend=false)
    rt3 = plot(xv3, [r.rt_err for r in rows_C];   yscale=:log10, marker=:circle,
               title="roundtrip error (edge)", xlabel="resolution_factor", legend=false)
    vline!(rt3, [1.0]; color=:gray, ls=:dash, label="default rf=1")
    r3b = plot(; yscale=:log10, title="r² vs hi-res per θ", xlabel="resolution_factor", ylabel="|Δr²| max")
    for (iθ, θ) in enumerate(θ_vals)
        plot!(r3b, xv3, [r.r2_hires_per_theta[θ] for r in rows_C];
              marker=:circle, label=θ_labels[iθ], color=θ_colors[iθ])
    end

    savefig(plot(q3a, q3b, d3a, d3b, rt3, r3b; layout=(3,2), size=(900,800)),
            joinpath(outdir, "conv_rf.png"))
    println("Saved conv_rf.png")

    # ── Figure 3: pareto.png ──────────────────────────────────────────────────
    par = plot(; title="Pareto: accuracy vs cost (ψ>0.80)",
               xlabel="runtime (s)", ylabel="max roundtrip |Δψ|", yscale=:log10)
    scatter!(par, [r.runtime for r in rows_A], [r.rt_err for r in rows_A];
             color=:royalblue, markershape=:circle, ms=8, label="refine sweep")
    for r in rows_A
        annotate!(par, r.runtime, r.rt_err, text("r=$(r.refine)", 7, :left))
    end
    scatter!(par, [r.runtime for r in rows_B], [r.rt_err for r in rows_B];
             color=:crimson, markershape=:diamond, ms=8, label="β_z sweep")
    for r in rows_B
        annotate!(par, r.runtime, r.rt_err, text("β=$(r.β_z)", 7, :left))
    end
    scatter!(par, [r.runtime for r in rows_C], [r.rt_err for r in rows_C];
             color=:forestgreen, markershape=:star5, ms=10, label="adaptive rf sweep")
    for r in rows_C
        annotate!(par, r.runtime, r.rt_err, text("rf=$(r.resolution_factor)", 7, :left))
    end
    savefig(par, joinpath(outdir, "pareto.png"))
    println("Saved pareto.png")

    # ── Figure 5: hreq_profile.png ────────────────────────────────────────────
    # Show dR_req and dZ_req sampled around the LCFS to show what drives the adaptive grid.
    try
        import Contour as Ctr_bench
        psio_b = raw_input.psio
        psihigh_b = cfg_base.psihigh
        mpsi_b = cfg_base.mpsi
        psilow_b = cfg_base.psilow
        Δψ_b = psio_b * (psihigh_b - psilow_b) / mpsi_b
        ψ_coarse_b = raw_input.psi_in.nodal_derivs.partials[1, :, :]
        ψ_bbox_b = psio_b * max(1.0 - psihigh_b, 1e-3)
        cl_b = Ctr_bench.contour(raw_input.psi_in_xs, raw_input.psi_in_ys, ψ_coarse_b, ψ_bbox_b)
        lcfs_b = Equilibrium.select_plasma_contour(Ctr_bench.lines(cl_b), pe_efit.ro, pe_efit.zo)
        if lcfs_b !== nothing
            verts_b = Ctr_bench.vertices(lcfs_b)
            Rs_b = [v[1] for v in verts_b]
            Zs_b = [v[2] for v in verts_b]
            dR_req_b = Float64[]
            dZ_req_b = Float64[]
            for (R, Z) in verts_b
                ψ_RR = abs(raw_input.psi_in((R, Z); deriv=Val((2, 0))))
                ψ_ZZ = abs(raw_input.psi_in((R, Z); deriv=Val((0, 2))))
                push!(dR_req_b, ψ_RR > 0 ? sqrt(8 * Δψ_b / ψ_RR) * 1000 : NaN)
                push!(dZ_req_b, ψ_ZZ > 0 ? sqrt(8 * Δψ_b / ψ_ZZ) * 1000 : NaN)
            end
            # Angle along LCFS for x-axis
            ro_b, zo_b = pe_efit.ro, pe_efit.zo
            angles_b = [mod(atan(Z - zo_b, R - ro_b), 2π) * 180/π for (R, Z) in verts_b]

            hp1 = plot(angles_b, dR_req_b; marker=:none, lw=1.5, color=:royalblue,
                       xlabel="geometric angle (°)", ylabel="dR_req (mm)",
                       title="Required R cell width along LCFS", legend=false)
            hp2 = plot(angles_b, dZ_req_b; marker=:none, lw=1.5, color=:crimson,
                       xlabel="geometric angle (°)", ylabel="dZ_req (mm)",
                       title="Required Z cell width along LCFS", legend=false)
            hp3 = plot(Rs_b, Zs_b; marker=:none, lw=1.5, color=:black,
                       xlabel="R (m)", ylabel="Z (m)", title="LCFS shape", aspect_ratio=:equal, legend=false)
            savefig(plot(hp1, hp2, hp3; layout=(1,3), size=(1200,350)),
                    joinpath(outdir, "hreq_profile.png"))
            println("Saved hreq_profile.png")
        end
    catch e
        @warn "hreq_profile.png skipped: $e"
    end

    # ── Figure 4: surfaces_xpt.png ────────────────────────────────────────────
    function surface_RZ(pe, ψ_val, θ_dense)
        Rv = Float64[]; Zv = Float64[]
        for θ in θ_dense
            r2  = pe.rzphi_rsquared((ψ_val, θ))
            off = pe.rzphi_offset((ψ_val, θ))
            rfac = sqrt(max(r2, 0.0))
            η = 2π * (θ + off)
            push!(Rv, pe.ro + rfac * cos(η))
            push!(Zv, pe.zo + rfac * sin(η))
        end
        return Rv, Zv
    end

    θ_dense = range(0, 1; length=256)
    ψ_show  = cfg_base.psihigh - 0.002
    Rv_ref, Zv_ref = surface_RZ(pe_efit, ψ_show, θ_dense)

    colors_A = palette(:viridis, length(rows_A))
    colors_B = palette(:plasma,  length(rows_B))

    sR_A = plot(collect(θ_dense), Rv_ref; color=:black, ls=:dash, lw=2,
                title="R(θ) — refine sweep", xlabel="θ", ylabel="R (m)", label="efit ref")
    sZ_A = plot(collect(θ_dense), Zv_ref; color=:black, ls=:dash, lw=2,
                title="Z(θ) — refine sweep", xlabel="θ", ylabel="Z (m)", label="efit ref")
    for (i, r) in enumerate(rows_A)
        pe_tmp, _ = run_inv(cfg_base, r.refine, r.β_z)
        Rv, Zv = surface_RZ(pe_tmp, ψ_show, θ_dense)
        plot!(sR_A, collect(θ_dense), Rv; color=colors_A[i], lw=1.5, label="r=$(r.refine)")
        plot!(sZ_A, collect(θ_dense), Zv; color=colors_A[i], lw=1.5, label=false)
    end

    sR_B = plot(collect(θ_dense), Rv_ref; color=:black, ls=:dash, lw=2,
                title="R(θ) — β_z sweep", xlabel="θ", ylabel="R (m)", label="efit ref")
    sZ_B = plot(collect(θ_dense), Zv_ref; color=:black, ls=:dash, lw=2,
                title="Z(θ) — β_z sweep", xlabel="θ", ylabel="Z (m)", label="efit ref")
    for (i, r) in enumerate(rows_B)
        pe_tmp, _ = run_inv(cfg_base, r.refine, r.β_z)
        Rv, Zv = surface_RZ(pe_tmp, ψ_show, θ_dense)
        plot!(sR_B, collect(θ_dense), Rv; color=colors_B[i], lw=1.5, label="β=$(r.β_z)")
        plot!(sZ_B, collect(θ_dense), Zv; color=colors_B[i], lw=1.5, label=false)
    end

    savefig(plot(sR_A, sZ_A, sR_B, sZ_B; layout=(2,2), size=(900,600)),
            joinpath(outdir, "surfaces_xpt.png"))
    println("Saved surfaces_xpt.png")

catch e
    @warn "Plotting skipped (Plots not available or error): $e"
end

println("\nAll figures saved to: $outdir")
println("Done.")
