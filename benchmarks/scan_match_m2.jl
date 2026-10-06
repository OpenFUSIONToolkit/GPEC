# Inner-layer scan on ONE outer solve: build the DIIID-like gal-resistive case once with
# the Galerkin integrator (rpec + cut solution), then sweep the layer rotation (or η) with
# cheap post-solve MatchProblem solves, overlaying the matched m-target |ξ_ψ| profile and
# reporting the penetrated resonant field per point. Replaces the one-full-deck-run-per-
# scan-point loop behind scan_rotation_m2.jl / scan_resistivity_m2.jl; prints the outer-solve
# and per-point timings plus a one-point equivalence check against the deck-style path.
# (The PE response leg of the old scans needs free-boundary energies the standalone gal
# solve does not produce yet — pending the gal δW / surface-current response port.)
# Usage: julia --project=. benchmarks/scan_match_m2.jl [rotation|eta] [out.png] [m]

using GeneralizedPerturbedEquilibrium
using Plots, Printf, TOML

GPEC = GeneralizedPerturbedEquilibrium

scanvar = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :rotation
outpng = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "scan_match_m2_$(scanvar).png")
mtarget = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2
scanvar in (:rotation, :eta) || error("scan variable must be rotation or eta (got $scanvar)")

deckdir = joinpath(@__DIR__, "..", "examples", "DIIID-like_gal_resistive_pe_example")
deck = TOML.parsefile(joinpath(deckdir, "gpec.toml"))
ffs_table = deck["ForceFreeStates"]

# The deck's layer parameters are the scan baseline; the swept variable replaces its vector.
eta0 = Vector{Float64}(ffs_table["gal_eta"])
rho0 = Vector{Float64}(ffs_table["gal_rho"])
rot0 = Vector{Float64}(ffs_table["gal_rotation"])
msing = length(eta0)
scanvals = scanvar === :rotation ? [1.0, 2.0, 4.0, 8.0, 16.0] : [2e-8, 4e-8, 8e-8, 1.6e-7, 3.2e-7]

# Deck → API: the gal_* solver knobs become the Galerkin alg (plus the basis retention the
# post-solve match needs); every other [ForceFreeStates] key is a solve keyword. The match
# family and the write flag are owned by the scan itself.
match_keys = ("gal_match_flag", "gal_ideal_flag", "gal_eta", "gal_rho", "gal_rotation", "gal_gamma",
    "gal_inner_solver", "gal_inner_xfac", "gal_inner_nx", "gal_inner_nq", "gal_inner_cutoff", "gal_inner_kmax")
alg_fields = Dict(String(f) => f for f in fieldnames(Galerkin))
alg_kwargs = Dict{Symbol,Any}(alg_fields[k[5:end]] => v for (k, v) in ffs_table
                              if startswith(k, "gal_") && !(k in match_keys) && haskey(alg_fields, k[5:end]))
alg = Galerkin(; alg_kwargs..., rpec_flag=true, cut_solution=true)
ffs_kwargs = Dict(Symbol(k) => v for (k, v) in ffs_table
                  if !(startswith(k, "gal_") || k in ("integrator", "nn_low", "nn_high", "write_outputs_to_HDF5")))
ffs_kwargs[:write_outputs_to_HDF5] = false

equil = GPEC.Equilibrium.setup_equilibrium(GPEC.Equilibrium.EquilibriumConfig(deck["Equilibrium"], deckdir))
wall = GPEC.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in get(deck, "Wall", Dict{String,Any}()))...)

t_outer = @elapsed ffs = solve(equil, alg; nn=ffs_table["nn_low"], wall=wall, dir_path=deckdir, ffs_kwargs...)
@printf("outer Galerkin solve: %.1f s (msing=%d)\n", t_outer, msing)

model = GGJ(; solver=Symbol(get(ffs_table, "gal_inner_solver", "ray")))
col = mtarget - ffs.mlow + 1
curves = Tuple{Float64,Vector{Float64},Vector{Float64}}[]
t_match = 0.0
matched1 = nothing
for v in scanvals
    eta = scanvar === :eta ? fill(v, msing) : eta0
    rot = scanvar === :rotation ? fill(v, msing) : rot0
    global t_match += @elapsed matched = solve(MatchProblem(ffs; eta=eta, rho=rho0, rotation=rot, gamma=ffs_table["gal_gamma"]), model)
    v == scanvals[1] && (global matched1 = matched)
    # The matched identity-at-edge profiles: mode row and drive column of the m-target.
    sol = matched.solution
    push!(curves, (v, sol.psi_store, abs.(vec(sol.u_store[col, col, 1, :]))))
    isurf = findfirst(==(mtarget), matched.galerkin.sing_m)
    isurf === nothing || @printf("  %s = %-8g |bpen(m=%d)| = %.4e\n", scanvar, v, mtarget,
        maximum(abs.(matched.galerkin.match.bpen[isurf, :])))
end
@printf("match per point: %.2f s avg over %d points (outer solve amortized once)\n",
    t_match / length(scanvals), length(scanvals))

# One-point equivalence: the deck-style path (match keys as solve keywords, routed through
# the same post-solve MatchProblem by the driver) must reproduce the scan's first point.
v1 = scanvals[1]
t_full = @elapsed ffs_deckstyle = solve(equil, alg; nn=ffs_table["nn_low"], wall=wall, dir_path=deckdir, ffs_kwargs...,
    gal_match_flag=true,
    gal_eta=scanvar === :eta ? fill(v1, msing) : eta0, gal_rho=rho0,
    gal_rotation=scanvar === :rotation ? fill(v1, msing) : rot0, gal_gamma=ffs_table["gal_gamma"],
    gal_inner_solver=get(ffs_table, "gal_inner_solver", "ray"))
maxdiff = maximum(abs.(ffs_deckstyle.bpen .- matched1.bpen))
@printf("one-point equivalence: max |Δbpen| = %.3e (deck-style full re-solve: %.1f s → scan speedup ≈ %.0fx/point)\n",
    maxdiff, t_full, t_full / (t_match / length(scanvals)))

sing_m = ffs.galerkin.sing_m
psi_res = mtarget in sing_m ? ffs.galerkin.sing_psi[findfirst(==(mtarget), sing_m)] : NaN

cols = cgrad(:plasma, max(length(scanvals), 2); categorical=true)
unitlab = scanvar === :rotation ? "Hz" : "Ω·m"
plt = plot(; size=(1000, 640), xlabel="ψ_N", ylabel="|ξ_ψ(m=$mtarget)|  (m=$mtarget unit-edge drive)",
    title="m=$mtarget matched displacement — $(scanvar) scan (one outer solve, post-solve match)",
    legend=:topleft, left_margin=13Plots.mm, bottom_margin=5Plots.mm, right_margin=4Plots.mm)
for (i, (v, psi, prof)) in enumerate(curves)
    plot!(plt, psi, prof; color=cols[i], lw=2, label=@sprintf("%s = %g %s", scanvar, v, unitlab))
end
isnan(psi_res) || vline!(plt, [psi_res]; color=:red, ls=:dot, lw=1.6, label="q=$mtarget surface")

savefig(plt, outpng)
println("saved: ", abspath(outpng))
