#!/usr/bin/env julia
"""
    profile_kf.jl

Capture a ProfileView flamegraph of KineticForces fgar on the DIIID case at
the chosen outer-ψ tolerance (see `kf_tolerance_scan.jl`). A narrow warmup
call triggers JIT compilation first so the profile reflects steady-state
allocation / time costs.

Run interactively:
    julia --project=. benchmarks/profile_kf.jl

Outputs:
- Flamegraph opens in ProfileView window (run from a session with a display).
- An HTML dump is written to `benchmarks/kf_flamegraph.html` for headless runs.

Requires ProfileView.jl in the local env (not added to root Project.toml —
benchmarks-only dep per CLAUDE.md's don't-bloat rule). If missing, install
into the local depot with:
    julia --project=. -e 'using Pkg; Pkg.add("ProfileView")'

Note: ProfileView is optional. If unavailable, this script falls back to
writing a text-format profile report so the numbers are still captured.
"""

using Printf
using Profile

using GeneralizedPerturbedEquilibrium
const GPE = GeneralizedPerturbedEquilibrium
const KF  = GPE.KineticForces

include(joinpath(@__DIR__, "benchmark_diiid_kinetic.jl"))
include(joinpath(@__DIR__, "kf_tolerance_scan.jl"))  # build_kf_state, time_fgar

# Chosen tolerance — update after running kf_tolerance_scan.jl.
const PROFILE_TOL = parse(Float64, get(ENV, "KF_PROFILE_TOL", "1e-3"))

function main()
    equil, intr, kf_intr, kinetic_profiles = build_kf_state()

    println(stderr, "\n--- JIT warmup (narrow ψ window) ---")
    warmup = time_fgar(PROFILE_TOL, equil, kf_intr, kinetic_profiles;
                       psilims=[0.5, 0.6])
    println(stderr, "  warmup: $(warmup.wall) s, $(warmup.nsteps) ψ-steps")

    Profile.clear()
    Profile.init(; n=10^7, delay=0.001)

    println(stderr, "\n--- Profiling fgar (full ψ range) ---")
    t0 = time()
    @profile time_fgar(PROFILE_TOL, equil, kf_intr, kinetic_profiles)
    wall = time() - t0
    println(stderr, "  profiled run: $wall s")

    # Text report — always written
    txt_path = joinpath(@__DIR__, "kf_profile_report.txt")
    open(txt_path, "w") do io
        Profile.print(io; format=:flat, sortedby=:count, mincount=20, maxdepth=50)
    end
    println(stderr, "  Text report → $(abspath(txt_path))")

    # Try ProfileView / FlameGraphs if available
    try
        @eval using ProfileView
        fv = Base.invokelatest(ProfileView.view)
        # If we're in a headless environment, save HTML instead
        try
            @eval using ProfileSVG
            svg_path = joinpath(@__DIR__, "kf_flamegraph.svg")
            Base.invokelatest(ProfileSVG.save, svg_path)
            println(stderr, "  Flamegraph SVG → $(abspath(svg_path))")
        catch
            println(stderr, "  ProfileView window opened (run headed for SVG/HTML export).")
        end
    catch err
        println(stderr, "  ProfileView unavailable ($err) — text report only.")
    end

    return nothing
end

main()
