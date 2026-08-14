# Quantifies whether splining iota = 1/q instead of q would reduce the knot count needed
# for a given edge accuracy (GitHub discussion around the auto-grid redesign; PR #179
# lineage owns any production iota work). For a grid ending at psihigh < 1, interpolation
# error transforms as err_q ≈ err_iota · q², cancelling the flattening of the profile, so
# no leading-order gain is expected — this script documents that with numbers on the
# DIII-D-like example equilibrium.
#
# Usage: julia --project=. benchmarks/benchmark_q_vs_iota_edge.jl
# Outputs (not committed): benchmarks/q_vs_iota_edge.csv, benchmarks/q_vs_iota_edge.png

using Pkg;
Pkg.activate(joinpath(@__DIR__, ".."))
using GeneralizedPerturbedEquilibrium
using FastInterpolations
using Printf
using Plots

const GPE = GeneralizedPerturbedEquilibrium
const EXAMPLE_DIR = joinpath(@__DIR__, "..", "examples", "DIIID-like_ideal_example")

# Dense ldp reference equilibrium: treat its q(ψ) as ground truth
function reference_q()
    inputs, _, additional_input = GPE.build_inputs_from_toml(EXAMPLE_DIR)
    equil_dict = merge(inputs["Equilibrium"], Dict{String,Any}("grid_type" => "ldp", "mpsi" => 1024))
    eq_config = GPE.Equilibrium.EquilibriumConfig(equil_dict, EXAMPLE_DIR)
    equil = GPE.Equilibrium.setup_equilibrium(eq_config, additional_input)
    return equil, eq_config
end

# Rational surface ψ_s(q = m/n) and q'(ψ_s) recovered from a fitted spline by bisection
function rational_locations(q_itp, dq_itp, psis, q_targets)
    out = Dict{Float64,Tuple{Float64,Float64}}()
    for qt in q_targets
        lo, hi = psis[1], psis[end]
        (q_itp(lo) - qt) * (q_itp(hi) - qt) > 0 && continue
        for _ in 1:200
            mid = 0.5 * (lo + hi)
            (q_itp(lo) - qt) * (q_itp(mid) - qt) <= 0 ? (hi = mid) : (lo = mid)
        end
        ps = 0.5 * (lo + hi)
        out[qt] = (ps, dq_itp(ps))
    end
    return out
end

function main()
    equil, eq_config = reference_q()
    xs_ref = equil.profiles.xs
    q_ref_itp = equil.profiles.q_spline
    dq_ref_itp = equil.profiles.q_deriv
    psilow, psihigh = xs_ref[1], xs_ref[end]

    q_lo = ceil(q_ref_itp(psilow))
    q_hi = floor(q_ref_itp(psihigh))
    q_targets = collect(q_lo:q_hi)  # n=1 rationals
    ref_rats = rational_locations(q_ref_itp, dq_ref_itp, xs_ref, q_targets)

    psi_eval = collect(range(psilow, psihigh; length=4001))
    edge_band = psi_eval .> 0.9

    rows = String[]
    push!(rows, "N,repr,max_rel_q_err,edge_rel_q_err,edge_rel_qprime_err,max_psi_s_err,max_rel_q1_err")
    results = Dict{String,Vector{NTuple{2,Float64}}}("q" => [], "iota" => [])

    for N in (16, 32, 64, 128, 256)
        # Same log_asymptotic knot layout the auto grid uses for fixed mpsi
        cfg = deepcopy(eq_config)
        cfg.mpsi = N
        knots = GPE.Equilibrium._build_psi_grid(cfg, psilow, psihigh)
        q_nodes = [q_ref_itp(p) for p in knots]

        for repr in ("q", "iota")
            itp = repr == "q" ? cubic_interp(knots, q_nodes; extrap=ExtendExtrap()) :
                  cubic_interp(knots, 1.0 ./ q_nodes; extrap=ExtendExtrap())
            ditp = deriv1(itp)
            qf = repr == "q" ? (p -> itp(p)) : (p -> 1.0 / itp(p))
            dqf = repr == "q" ? (p -> ditp(p)) : (p -> -ditp(p) / itp(p)^2)  # q' = -ι'/ι²

            q_err = [abs(qf(p) - q_ref_itp(p)) / abs(q_ref_itp(p)) for p in psi_eval]
            dq_err = [abs(dqf(p) - dq_ref_itp(p)) / max(abs(dq_ref_itp(p)), 1e-10) for p in psi_eval]
            rats = rational_locations(qf, dqf, knots, q_targets)
            psi_s_err = maximum(abs(rats[qt][1] - ref_rats[qt][1]) for qt in keys(ref_rats); init=0.0)
            q1_err = maximum(abs(rats[qt][2] - ref_rats[qt][2]) / abs(ref_rats[qt][2]) for qt in keys(ref_rats); init=0.0)

            push!(rows, @sprintf("%d,%s,%.3e,%.3e,%.3e,%.3e,%.3e",
                N, repr, maximum(q_err), maximum(q_err[edge_band]), maximum(dq_err[edge_band]), psi_s_err, q1_err))
            push!(results[repr], (Float64(N), maximum(q_err[edge_band])))
        end
    end

    csv_path = joinpath(@__DIR__, "q_vs_iota_edge.csv")
    open(csv_path, "w") do io
        foreach(r -> println(io, r), rows)
    end
    println("Wrote ", abspath(csv_path))
    foreach(println, rows)

    p = plot(; xscale=:log10, yscale=:log10, xlabel="knots N", ylabel="max relative q error, ψ > 0.9",
        title="q-spline vs ι-spline edge accuracy", legend=:topright, left_margin=12Plots.mm, bottom_margin=4Plots.mm)
    for (repr, pts) in results
        plot!(p, first.(pts), last.(pts); marker=:circle, lw=2, label="$repr-spline")
    end
    png_path = joinpath(@__DIR__, "q_vs_iota_edge.png")
    savefig(p, png_path)
    println("Wrote ", abspath(png_path))
end

main()
