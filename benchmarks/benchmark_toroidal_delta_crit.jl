# Verification and figures for the r_s-referenced Connor et al. 2015 Eq. 59 toroidal
# critical-Δ geometric factor (`toroidal_dgeo`):
#   1. large-aspect-ratio convergence on TJ-analytic circular equilibria, dgeo/√(n s r_s/R₀)
#      vs ψ_N for several ε, for the correct Λ = ψ_t'² ι'/2π and for the Fortran STRIDE form
#      that carries one power of ψ_t' (dimensional, off by ψ_t'^(-1/2));
#   2. scale invariance under B₀ → B₀/2 and (a, R₀) → 2(a, R₀) at fixed ε;
#   3. an independent recomputation of ⟨B²⟩, ⟨|∇ψ_N|²⟩ and V' from R(ψ,θ), Z(ψ,θ), F(ψ), psio
#      alone (no DCON metric elements), compared with ResistGeometry;
#   4. Δ_crit per rational surface on the DIII-D-like SLAYER example, rfitzp vs toroidal.
#
# Usage: julia --project=. benchmarks/benchmark_toroidal_delta_crit.jl
# Outputs go to benchmarks/toroidal_delta_crit/ (not committed).
using Printf, TOML, Plots
using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium: TJAnalyticConfig, EquilibriumConfig, setup_equilibrium, read_kinetic_file
using GeneralizedPerturbedEquilibrium.ForceFreeStates: resist_geometry, ForceFreeStatesInternal, ForceFreeStatesControl,
    sing_lim!, sing_find!, resist_eval_all!
using GeneralizedPerturbedEquilibrium.InnerLayer: toroidal_dgeo, r_based_shear, surface_minor_radius, surface_da_dpsi,
    build_slayer_inputs
using GeneralizedPerturbedEquilibrium.Utilities: KineticProfiles
using FastInterpolations
using FastInterpolations: cubic_interp, Series, PeriodicBC, integrate

const EXAMPLE_D3D = joinpath(@__DIR__, "..", "examples", "DIIID-like_SLAYER_example")
const OUT = joinpath(@__DIR__, "toroidal_delta_crit")
mkpath(OUT)

function tj_equil(; eps=0.2, a=1.0, B0=12.0, mpsi=128, mtheta=256)
    tj = TJAnalyticConfig(lar_r0=a / eps, lar_a=a, qc=1.5, qa=3.6, pc=0.001, mu=2.0, B0=B0, ma=128, mtau=128)
    eq = EquilibriumConfig(eq_type="tj_analytic", psilow=0.01, psihigh=0.995, mpsi=mpsi, mtheta=mtheta, etol=1e-7)
    return setup_equilibrium(eq, tj)
end

# (dgeo, Fortran-form dgeo, rfitzp factor √(n s r_s/R₀), ResistGeometry) at one surface.
function dgeo_at(pe, psi, n)
    chi1 = 2π * pe.psio
    q = pe.profiles.q_spline(psi)
    q1 = pe.profiles.q_deriv(psi)
    rg = resist_geometry(pe, psi, q1)
    rs = surface_minor_radius(pe, psi)
    da = surface_da_dpsi(pe, psi)
    dg = toroidal_dgeo(; chi1=chi1, v1=rg.v1_local, q=q, q1=q1, n=n,
        avg_bsq=rg.avg_bsq, avg_dpsisq=rg.avg_dpsisq, k_ref=rs / da)
    dg_fortran = dg / sqrt(q * chi1 / rg.v1_local)
    lar = sqrt(n * r_based_shear(rs, q, q1, da) * rs / pe.ro)
    return dg, dg_fortran, lar, rg
end

# 1. Large-aspect-ratio convergence
println("1. Large-aspect-ratio convergence (n = 1)")
psis = collect(0.05:0.025:0.95)
p1 = plot(; xlabel="ψ_N", ylabel="dgeo / √(n s r_s/R₀)", title="Connor Eq. 59, r_s reference (Λ = ψ_t'² ι'/2π)",
    legend=:topleft, xlims=(0, 1), left_margin=8Plots.mm, bottom_margin=5Plots.mm, titlefontsize=10)
p2 = plot(; xlabel="ψ_N", ylabel="dgeo / √(n s r_s/R₀)", title="Fortran STRIDE form (Λ with ψ_t'¹)",
    legend=:topleft, xlims=(0, 1), left_margin=8Plots.mm, bottom_margin=5Plots.mm, titlefontsize=10)
hline!(p1, [1.0]; color=:black, ls=:dash, label="rfitzp")
hline!(p2, [1.0]; color=:black, ls=:dash, label="rfitzp")
for eps in (0.05, 0.1, 0.2, 0.3)
    pe = tj_equil(; eps=eps)
    rc = Float64[]
    rf = Float64[]
    for psi in psis
        dg, dgf, lar, _ = dgeo_at(pe, psi, 1)
        push!(rc, dg / lar)
        push!(rf, dgf / lar)
    end
    @printf("  ε = %.2f  ratio(correct) min/max = %.4f / %.4f   ratio(Fortran) min/max = %.3f / %.3f\n",
        eps, minimum(rc), maximum(rc), minimum(rf), maximum(rf))
    plot!(p1, psis, rc; lw=2, label=@sprintf("ε = %.2f", eps))
    plot!(p2, psis, rf; lw=2, label=@sprintf("ε = %.2f", eps))
end
for ext in ("png", "pdf")
    f = joinpath(OUT, "lar_convergence.$ext")
    savefig(plot(p1, p2; layout=(1, 2), size=(1100, 420)), f)
    println("saved: ", abspath(f))
end

# 2. Scale invariance
println("\n2. Scale invariance at ε = 0.2 (dgeo must be identical; the Fortran form scales as ψ_t'^(-1/2))")
ref, halfB, twice = tj_equil(), tj_equil(; B0=6.0), tj_equil(; a=2.0)
@printf("%6s %12s %12s %12s | %12s %12s %12s\n", "psi", "dgeo(ref)", "dgeo(B0/2)", "dgeo(2a,2R)", "fort(ref)", "fort(B0/2)", "fort(2a,2R)")
for psi in (0.3, 0.6, 0.9)
    d0, f0 = dgeo_at(ref, psi, 1)
    d1, f1 = dgeo_at(halfB, psi, 1)
    d2, f2 = dgeo_at(twice, psi, 1)
    @printf("%6.2f %12.6f %12.6f %12.6f | %12.6f %12.6f %12.6f\n", psi, d0, d1, d2, f0, f1, f2)
end

# 3. Independent metric: ψ_N is axisymmetric, so |∇ψ_N|² = |∂_θ(R,Z)|²/J₂² with J₂ = R_ψ Z_θ − R_θ Z_ψ;
# B² = (F/R)² + psio²|∇ψ_N|²/R² with F = F_spline/(2π); volume element 2πR|J₂| dψ dθ.
function independent_averages(pe, psi)
    ys = pe.rzphi_ys
    F = pe.profiles.F_spline(psi) / (2π)
    w = zeros(length(ys))
    b2 = zeros(length(ys))
    g2 = zeros(length(ys))
    for (i, th) in enumerate(ys)
        f1 = pe.rzphi_rsquared((psi, th))
        f2 = pe.rzphi_offset((psi, th))
        f1p = pe.rzphi_rsquared((psi, th); deriv=DerivOp(1, 0))
        f1t = pe.rzphi_rsquared((psi, th); deriv=DerivOp(0, 1))
        f2p = pe.rzphi_offset((psi, th); deriv=DerivOp(1, 0))
        f2t = pe.rzphi_offset((psi, th); deriv=DerivOp(0, 1))
        rfac = sqrt(f1)
        eta = 2π * (th + f2)
        R = pe.ro + rfac * cos(eta)
        rp, rt = f1p / (2rfac), f1t / (2rfac)
        ep, et = 2π * f2p, 2π * (1 + f2t)
        Rp = rp * cos(eta) - rfac * sin(eta) * ep
        Zp = rp * sin(eta) + rfac * cos(eta) * ep
        Rt = rt * cos(eta) - rfac * sin(eta) * et
        Zt = rt * sin(eta) + rfac * cos(eta) * et
        J2 = Rp * Zt - Rt * Zp
        gpsi2 = (Rt^2 + Zt^2) / J2^2
        w[i] = 2π * R * abs(J2)
        g2[i] = gpsi2
        b2[i] = (F / R)^2 + pe.psio^2 * gpsi2 / R^2
    end
    s = integrate(cubic_interp(ys, Series(hcat(w, w .* b2, w .* g2)); bc=PeriodicBC()))
    return (; v1=s[1], avg_bsq=s[2] / s[1], avg_dpsisq=s[3] / s[1])
end

function metric_table(pe, psis, label)
    println("\n3. Independent metric check: ", label)
    @printf("%6s %14s %14s %10s | %14s %14s %10s | %12s %12s %10s\n",
        "psi", "<B2> RG", "<B2> indep", "rel", "<|dpsi|2> RG", "indep", "rel", "v1 RG", "v1 indep", "rel")
    for psi in psis
        rg = resist_geometry(pe, psi, pe.profiles.q_deriv(psi))
        ia = independent_averages(pe, psi)
        @printf("%6.4f %14.6e %14.6e %10.2e | %14.6e %14.6e %10.2e | %12.5e %12.5e %10.2e\n",
            psi, rg.avg_bsq, ia.avg_bsq, abs(ia.avg_bsq / rg.avg_bsq - 1),
            rg.avg_dpsisq, ia.avg_dpsisq, abs(ia.avg_dpsisq / rg.avg_dpsisq - 1),
            rg.v1_local, ia.v1, abs(ia.v1 / rg.v1_local - 1))
    end
end
metric_table(ref, (0.3, 0.6, 0.9), "TJ circular ε = 0.2")

# 4. DIII-D-like example: Δ_crit per rational surface
inputs = TOML.parsefile(joinpath(EXAMPLE_D3D, "gpec.toml"))
equil = setup_equilibrium(EquilibriumConfig(inputs["Equilibrium"], EXAMPLE_D3D), nothing)
ctrl = ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in inputs["ForceFreeStates"])...)
intr = ForceFreeStatesInternal(; dir_path=EXAMPLE_D3D)
intr.nlow = ctrl.nn_low
intr.nhigh = ctrl.nn_high
intr.npert = 1
sing_lim!(intr, ctrl, equil)
sing_find!(intr, equil)
resist_eval_all!(intr, equil)
sings = intr.sing
metric_table(equil, [s.psifac for s in sings], "DIII-D-like rational surfaces")

kin = read_kinetic_file(joinpath(EXAMPLE_D3D, inputs["SLAYER"]["profile_file"]))
npsi = length(kin.psi)
profiles = KineticProfiles(; psi=kin.psi, n_e=kin.n_e, T_e=kin.T_e, T_i=kin.T_i,
    omega=(kin.omega_E === nothing ? zeros(npsi) : kin.omega_E), omega_e=zeros(npsi), omega_i=zeros(npsi))
_chi(v) = (v !== nothing && any(!=(0.0), v)) ? (let itp = cubic_interp(kin.psi, v); ψ -> Float64(itp(ψ)) end) : 1.0
kw = (; chi_perp=_chi(kin.chi_e), chi_tor=_chi(kin.chi_phi))
p_rf = build_slayer_inputs(equil, sings, profiles; dc_type=:rfitzp, kw...)
p_tor = build_slayer_inputs(equil, sings, profiles; dc_type=:toroidal, kw...)
println("\n4. DIII-D-like SLAYER example, per rational surface (midplane label)")
@printf("%5s %7s %7s %8s %8s %10s %9s %11s %11s %11s %8s\n",
    "m/n", "psi_N", "q", "r_s", "s_r", "D_R", "dgeo", "√(nsr/R)", "dc_rfitzp", "dc_toroid", "ratio")
for (s, a, b) in zip(sings, p_rf, p_tor)
    lar = sqrt(a.n * a.sval_r * a.rs / a.R0)
    @printf("%5s %7.4f %7.3f %8.4f %8.4f %10.4f %9.4f %11.4f %11.4f %11.4f %8.4f\n",
        "$(a.m)/$(a.n)", s.psifac, s.q, a.rs, a.sval_r, a.dr_val, b.dgeo_val, lar, a.dc_tmp, b.dc_tmp, b.dc_tmp / a.dc_tmp)
end
keep = [i for (i, a) in enumerate(p_rf) if a.m in (2, 3, 4) && a.n == 1]
labels = ["$(p_rf[i].m)/$(p_rf[i].n)" for i in keep]
x = 1:length(keep)
fig = plot(; xticks=(x, labels), xlabel="rational surface m/n", ylabel="Δ_crit (r_s reference)",
    title="DIII-D-like, Δ_crit at the 2/1, 3/1, 4/1 surfaces (midplane label)", legend=:topleft,
    left_margin=8Plots.mm, bottom_margin=5Plots.mm, titlefontsize=10, size=(560, 420), xlims=(0.5, length(keep) + 0.5))
scatter!(fig, x .- 0.08, [p_rf[i].dc_tmp for i in keep]; ms=7, label="rfitzp")
scatter!(fig, x .+ 0.08, [p_tor[i].dc_tmp for i in keep]; ms=7, marker=:diamond, label="toroidal (Eq. 59, r_s ref.)")
for (xx, i) in zip(x, keep)
    annotate!(fig, xx + 0.08, p_tor[i].dc_tmp, text(@sprintf("  ×%.2f", p_tor[i].dc_tmp / p_rf[i].dc_tmp), 8, :left))
end
for ext in ("png", "pdf")
    f = joinpath(OUT, "diiid_delta_crit_234.$ext")
    savefig(fig, f)
    println("saved: ", abspath(f))
end
