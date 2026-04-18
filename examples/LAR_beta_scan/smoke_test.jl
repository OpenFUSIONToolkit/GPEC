#!/usr/bin/env julia
# Quick smoke test: run a single beta scan point (pf=0.1) and check delta prime BVP output.
# Usage: julia --project=../.. smoke_test.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))

using TOML

# Set up config for pf=0.1
config_path = joinpath(@__DIR__, "gpec.toml")
config = TOML.parsefile(config_path)
config["Equilibrium"]["eq_filename"] = joinpath(@__DIR__, "equilibria", "TJ_betascan_0.1.geqdsk")
config["ForceFreeStates"]["verbose"] = true

# Write working config
work_dir = mktempdir()
open(joinpath(work_dir, "gpec.toml"), "w") do io
    TOML.print(io, config)
end

@info "Working directory: $work_dir"
@info "Running GPEC on TJ_betascan_0.1.geqdsk..."

using GeneralizedPerturbedEquilibrium
GeneralizedPerturbedEquilibrium.main([work_dir])

# Read results
using HDF5
h5file = joinpath(work_dir, "gpec.h5")
h5open(h5file, "r") do f
    msing = read(f["info/msing"])
    @info "Number of singular surfaces: $msing"

    if haskey(f, "singular")
        for s in 1:msing
            grp = "singular/surface_$s"
            if haskey(f, grp)
                m = read(f["$grp/m"])
                q = read(f["$grp/q_rational"])
                psi = read(f["$grp/psi"])
                @info "  Surface $s: m=$m, q=$q, ψ=$psi"
            end
        end
    end

    if haskey(f, "integration/delta_prime_matrix")
        dp = read(f["integration/delta_prime_matrix"])
        @info "Delta prime matrix (msing×msing):"
        @info "  Shape: $(size(dp))"
        for i in axes(dp, 1)
            @info "  dp[$i,$i] = $(dp[i,i])"
        end
        @info "Fortran reference: dp_21 ≈ 16.445, dp_31 ≈ 8.341"
    else
        @warn "No delta_prime_matrix found in output"
    end

    # Energy diagnostics
    if haskey(f, "integration/et")
        et = read(f["integration/et"])
        @info "Total energy et[1] = $(et[1])"
    end
    if haskey(f, "integration/ep")
        ep = read(f["integration/ep"])
        @info "Plasma energy ep[1] = $(ep[1])"
    end
    if haskey(f, "integration/ev")
        ev = read(f["integration/ev"])
        @info "Vacuum energy ev[1] = $(ev[1])"
    end
end

# Cleanup
rm(work_dir; recursive=true, force=true)
