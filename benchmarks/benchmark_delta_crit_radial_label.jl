# Radial-label sensitivity of the SLAYER critical-Δ threshold.
#
# The tearing threshold is Δ'_rs > Δ_crit, with Δ'_rs the outer Δ' converted to the r_s
# reference through K^(2μ), K = r_s·dψ_N/dr. The Connor et al. 2015 Eq. 59 (`:toroidal`)
# critical-Δ scales with the radial label through the same K, so its threshold margin should
# be nearly label-invariant; the cylindrical `:rfitzp` formula is not covariant under a
# relabeling. Two cases:
#   - the DIII-D-like SLAYER example (shaped): margin Δ'_rs/Δ_crit per label and branch;
#   - the TJ-analytic circular ε scan: on a circular equilibrium every label coincides to
#     O(ε²), so the label spread and the toroidal/rfitzp ratio must both approach their
#     large-aspect-ratio limits (0 and 1) as ε → 0; a residual is a bug, not a convention.
# Each case computes a fresh Riccati Δ' matrix (no HDF5 output, no PE, no SLAYER stage).
#
# Usage: julia --project=. benchmarks/benchmark_delta_crit_radial_label.jl [--no-diiid] [--no-tj]
# Outputs go to benchmarks/delta_crit_radial_label/ (not committed).
using Printf, TOML, Plots
using GeneralizedPerturbedEquilibrium
const GPE = GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.Equilibrium: read_kinetic_file
using GeneralizedPerturbedEquilibrium.InnerLayer: build_slayer_inputs
using GeneralizedPerturbedEquilibrium.Utilities: KineticProfiles
using FastInterpolations: cubic_interp

const EXAMPLE_D3D = joinpath(@__DIR__, "..", "examples", "DIIID-like_SLAYER_example")
const EXAMPLE_LAR = joinpath(@__DIR__, "..", "examples", "LAR_epsilon_scan")
const OUT = joinpath(@__DIR__, "delta_crit_radial_label")
const LABELS = (:midplane, :flux, :volume)
mkpath(OUT)

# Outer-region solve only, from a run directory holding gpec.toml.
function outer_solve(dir)
    inputs, eq_config, additional_input = GPE.build_inputs_from_toml(dir)
    delete!(inputs, "SLAYER")
    inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = false
    inputs["ForceFreeStates"]["force_termination"] = true
    ffs = GPE.main_from_inputs(inputs, eq_config, additional_input, dir, "benchmark").ffs
    return ffs.equil, ffs.surfaces, ffs.delta_prime.matrix
end

# Per-surface rows (K, μ, Δ'_rs, Δ_crit for both branches) for one label.
function label_rows(equil, sings, dp, profiles, lab; kw...)
    p_rf = build_slayer_inputs(equil, sings, profiles; dc_type=:rfitzp, rs_method=lab, kw...)
    p_tor = build_slayer_inputs(equil, sings, profiles; dc_type=:toroidal, rs_method=lab, kw...)
    dp_rs = real.(GPE.Tearing.Runner.delta_prime_to_rs_reference(dp, p_tor))
    return [(; label=lab, m=a.m, n=a.n, mn="$(a.m)/$(a.n)", K=a.k_ref, mu=a.mu_mercier,
        dp_rs=dp_rs[k, k], dc_rf=a.dc_tmp, dc_tor=b.dc_tmp) for (k, (a, b)) in enumerate(zip(p_rf, p_tor))]
end

function print_rows(rows)
    @printf("%-9s %5s %7s %7s %8s %9s %9s %10s %10s %9s %9s\n",
        "label", "m/n", "K", "mu", "K^2mu", "dp_rs", "dc_rf", "dc_tor", "dp/dc_rf", "dp/dc_tor", "tor/rf")
    for r in rows
        @printf("%-9s %5s %7.4f %7.4f %8.4f %9.3f %9.3f %10.3f %10.4f %9.4f %9.4f\n",
            r.label, r.mn, r.K, r.mu, r.K^(2r.mu), r.dp_rs, r.dc_rf, r.dc_tor, r.dp_rs / r.dc_rf, r.dp_rs / r.dc_tor, r.dc_tor / r.dc_rf)
    end
end

# Label spread (max/min − 1) of the margin Δ'_rs/Δ_crit per surface, for both branches.
function margin_spread(rows, mn)
    rs = filter(r -> r.mn == mn, rows)
    m_rf = [abs(r.dp_rs / r.dc_rf) for r in rs]
    m_tor = [abs(r.dp_rs / r.dc_tor) for r in rs]
    return maximum(m_rf) / minimum(m_rf) - 1, maximum(m_tor) / minimum(m_tor) - 1
end

# ---------------------------------------------------------------------------
# DIII-D-like SLAYER example (shaped)
# ---------------------------------------------------------------------------
if !("--no-diiid" in ARGS)
    println("=== DIII-D-like SLAYER example ===")
    profile_file = TOML.parsefile(joinpath(EXAMPLE_D3D, "gpec.toml"))["SLAYER"]["profile_file"]
    equil, sings, dp = outer_solve(EXAMPLE_D3D)
    kin = read_kinetic_file(joinpath(EXAMPLE_D3D, profile_file))
    npsi = length(kin.psi)
    profiles = KineticProfiles(; psi=kin.psi, n_e=kin.n_e, T_e=kin.T_e, T_i=kin.T_i,
        omega=(kin.omega_E === nothing ? zeros(npsi) : kin.omega_E), omega_e=zeros(npsi), omega_i=zeros(npsi))
    _chi(v) = (v !== nothing && any(!=(0.0), v)) ? (let itp = cubic_interp(kin.psi, v); ψ -> Float64(itp(ψ)) end) : 1.0
    kw = (; chi_perp=_chi(kin.chi_e), chi_tor=_chi(kin.chi_phi))
    println("Δ' (ψ_N reference) diagonal: ", [round(real(dp[i, i]); digits=3) for i in 1:length(sings)])
    rows = reduce(vcat, [label_rows(equil, sings, dp, profiles, lab; kw...) for lab in LABELS])
    rows = filter(r -> r.m in (2, 3, 4), rows)
    print_rows(rows)
    mns = unique([r.mn for r in rows])
    println("\nLabel spread of the threshold margin |Δ'_rs/Δ_crit| (max/min over labels − 1):")
    @printf("%5s %10s %10s\n", "m/n", "rfitzp", "toroidal")
    for mn in mns
        s_rf, s_tor = margin_spread(rows, mn)
        @printf("%5s %10.4f %10.4f\n", mn, s_rf, s_tor)
    end
    x = 1:length(mns)
    p = plot(; layout=(1, 2), size=(1000, 400), left_margin=8Plots.mm, bottom_margin=5Plots.mm, titlefontsize=10)
    for (i, (dc, ttl)) in enumerate(((:dc_rf, "rfitzp"), (:dc_tor, "toroidal (Eq. 59, r_s ref.)")))
        plot!(p[i]; xticks=(x, mns), xlabel="rational surface m/n", ylabel="Δ'_rs / Δ_crit",
            title="DIII-D-like threshold margin, $ttl", legend=(i == 1 ? :topright : false))
        for (j, lab) in enumerate(LABELS)
            vals = [let r = only(filter(r -> r.mn == mn && r.label == lab, rows)); r.dp_rs / getfield(r, dc) end for mn in mns]
            scatter!(p[i], x .+ 0.12 * (j - 2), vals; ms=6, label=String(lab))
        end
        hline!(p[i], [1.0]; color=:black, ls=:dash, label="")
    end
    for ext in ("png", "pdf")
        f = joinpath(OUT, "diiid_threshold_margin_by_label.$ext")
        savefig(p, f)
        println("saved: ", abspath(f))
    end
end

# ---------------------------------------------------------------------------
# TJ-analytic circular ε scan
# ---------------------------------------------------------------------------
if !("--no-tj" in ARGS)
    println("\n=== TJ-analytic circular ε scan ===")
    # Baseline from the LAR ε-scan example; only lar_r0 (ε), the grid, and the on-axis
    # pressure are overridden. pc = 0.01 (the LAR_resistive_match_test value) gives a
    # finite D_R so Δ_crit is not numerically tiny; the margin ratios are D_R-independent.
    base = TOML.parsefile(joinpath(EXAMPLE_LAR, "gpec.toml"))
    epsilons = [0.05, 0.1, 0.2, 0.3]
    # Synthetic flat-density, parabolic-temperature kinetic profiles (no kinetic file ships
    # with the analytic equilibrium); they set χ∥ through the W_d loop but cancel in the ratios.
    psi_k = collect(0.0:0.1:1.0)
    nk = length(psi_k)
    profiles = KineticProfiles(; psi=psi_k, n_e=fill(3.0e19, nk), T_e=2000.0 .* (1 .- 0.8 .* psi_k),
        T_i=2000.0 .* (1 .- 0.8 .* psi_k), omega=zeros(nk), omega_e=zeros(nk), omega_i=zeros(nk))
    scan = []
    for eps in epsilons
        run_dir = mktempdir(; prefix="gpec_dcrit_label_tj_")
        cfg = deepcopy(base)
        cfg["TJ_ANALYTIC_INPUT"]["lar_r0"] = cfg["TJ_ANALYTIC_INPUT"]["lar_a"] / eps
        cfg["TJ_ANALYTIC_INPUT"]["pc"] = 0.01
        cfg["Equilibrium"]["mpsi"] = 128
        cfg["Equilibrium"]["mtheta"] = 256
        open(joinpath(run_dir, "gpec.toml"), "w") do io
            TOML.print(io, cfg)
        end
        equil, sings, dp = outer_solve(run_dir)
        rows = reduce(vcat, [label_rows(equil, sings, dp, profiles, lab; chi_perp=1.0, chi_tor=1.0) for lab in LABELS])
        println("\nε = $eps   Δ' (ψ_N reference) diagonal: ", [round(real(dp[i, i]); digits=3) for i in 1:length(sings)])
        print_rows(rows)
        for mn in unique([r.mn for r in rows])
            s_rf, s_tor = margin_spread(rows, mn)
            r_mid = only(filter(r -> r.mn == mn && r.label == :midplane, rows))
            r_flx = only(filter(r -> r.mn == mn && r.label == :flux, rows))
            push!(scan, (; eps, mn, s_rf, s_tor, tor_rf_mid=r_mid.dc_tor / r_mid.dc_rf, tor_rf_flux=r_flx.dc_tor / r_flx.dc_rf))
        end
    end
    println("\nε scan summary (spread = max/min over labels − 1 of |Δ'_rs/Δ_crit|):")
    @printf("%6s %5s %12s %12s %14s %14s\n", "eps", "m/n", "spread rf", "spread tor", "tor/rf mid", "tor/rf flux")
    for s in scan
        @printf("%6.3f %5s %12.4f %12.4f %14.4f %14.4f\n", s.eps, s.mn, s.s_rf, s.s_tor, s.tor_rf_mid, s.tor_rf_flux)
    end
    mns = unique([s.mn for s in scan])
    p = plot(; layout=(1, 2), size=(1000, 400), left_margin=8Plots.mm, bottom_margin=5Plots.mm, titlefontsize=10)
    plot!(p[1]; xlabel="ε = a/R₀", ylabel="label spread of |Δ'_rs/Δ_crit|", title="TJ circular: label spread of the margin", legend=:topleft)
    plot!(p[2]; xlabel="ε = a/R₀", ylabel="Δ_crit,toroidal / Δ_crit,rfitzp", title="TJ circular: toroidal / rfitzp", legend=:topleft)
    for mn in mns
        ss = filter(s -> s.mn == mn, scan)
        plot!(p[1], [s.eps for s in ss], [s.s_rf for s in ss]; lw=2, marker=:circle, label="rfitzp $mn")
        plot!(p[1], [s.eps for s in ss], [s.s_tor for s in ss]; lw=2, marker=:diamond, ls=:dash, label="toroidal $mn")
        plot!(p[2], [s.eps for s in ss], [s.tor_rf_mid for s in ss]; lw=2, marker=:circle, label="midplane $mn")
        plot!(p[2], [s.eps for s in ss], [s.tor_rf_flux for s in ss]; lw=2, marker=:diamond, ls=:dash, label="flux $mn")
    end
    hline!(p[2], [1.0]; color=:black, ls=:dot, label="")
    for ext in ("png", "pdf")
        f = joinpath(OUT, "tj_epsilon_scan_by_label.$ext")
        savefig(p, f)
        println("saved: ", abspath(f))
    end
end
