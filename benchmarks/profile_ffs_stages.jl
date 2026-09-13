#!/usr/bin/env julia
"""
profile_ffs_stages.jl - Wall-clock breakdown of the Equilibrium and Force-Free States stages of one case.

Replays the calls `main` makes for one case directory and times each stage separately: equilibrium
setup and re-forming, local stability, matrix splines, the Euler-Lagrange integration, the
free-boundary energies, the Δ′ BVP, result assembly and the HDF5 write. Use it to see where a run's
time actually goes before touching any single kernel.

# Usage

```bash
julia --project=. -t 4 benchmarks/profile_ffs_stages.jl --case <dir> [--verbose true|false] [--blas-threads k] [--no-hdf5] [--repeat n]
```

`--verbose` overrides the case's `verbose` (default: as in the TOML). `--repeat n` replays the whole
sequence n times in one process, so the first pass shows cold (JIT-inclusive) timings and later passes
the warm compute cost. Inputs come from the case directory; the HDF5 file is written to a temporary
directory and deleted.
"""

using LinearAlgebra
using Printf
using TOML
using GeneralizedPerturbedEquilibrium

const GPE = GeneralizedPerturbedEquilibrium
const FFS = GPE.ForceFreeStates

function parse_args(args)
    opts = Dict{String,Any}("case" => nothing, "verbose" => nothing, "blas" => nothing, "hdf5" => true, "repeat" => 1)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--case"
            opts["case"] = abspath(args[i+1])
            i += 2
        elseif a == "--verbose"
            opts["verbose"] = parse(Bool, args[i+1])
            i += 2
        elseif a == "--blas-threads"
            opts["blas"] = parse(Int, args[i+1])
            i += 2
        elseif a == "--no-hdf5"
            opts["hdf5"] = false
            i += 1
        elseif a == "--repeat"
            opts["repeat"] = parse(Int, args[i+1])
            i += 2
        else
            error("unknown argument $a")
        end
    end
    opts["case"] === nothing && error("--case <dir> is required")
    return opts
end

function main(args)
    opts = parse_args(args)
    case = opts["case"]
    opts["blas"] !== nothing && BLAS.set_num_threads(opts["blas"])
    inputs = TOML.parsefile(joinpath(case, "gpec.toml"))
    ffs_table = inputs["ForceFreeStates"]
    opts["verbose"] !== nothing && (ffs_table["verbose"] = opts["verbose"])
    ffs_table["write_outputs_to_HDF5"] = false
    for pass in 1:opts["repeat"]
        workdir = mktempdir()
        ffs_table["HDF5_filename"] = joinpath(workdir, "gpec.h5")
        timings = Pair{String,Float64}[]
        stage(name, f) = (t = @elapsed(r = f()); push!(timings, name => t); r)

        eq_config = GPE.Equilibrium.EquilibriumConfig(inputs["Equilibrium"], case)
        ctrl = FFS.ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in ffs_table)...)
        intr = FFS.ForceFreeStatesInternal(; dir_path=case)
        intr.wall_settings = GPE.Vacuum.WallShapeSettings(; (Symbol(k) => v for (k, v) in inputs["Wall"])...)
        GPE.resolve_mode_space!(intr, ctrl)

        equil = stage("equilibrium setup", () -> GPE.Equilibrium.setup_equilibrium(eq_config, nothing))
        equil = stage("equilibrium re-form (two-pass grid)", () -> GPE.maybe_reform_equilibrium(equil, eq_config, nothing, intr, ctrl, nothing))
        locstab, ballooning_boundary = stage("local stability (Mercier/ballooning)", () -> GPE.run_local_stability(ctrl, equil))
        kf_ctrl = GPE.KineticForces.KineticForcesControl()
        metric, mats = stage("metric + F/G/K matrix splines", () -> GPE.prepare_force_free_states!(intr, ctrl, equil, kf_ctrl, nothing))
        odet, fm_propagators, fm_chunks, fm_S_left = stage("Euler-Lagrange integration", () -> FFS.eulerlagrange_integration(ctrl, equil, mats, intr))
        free_energies = nothing
        if ctrl.vac_flag
            free_energies = stage("free-boundary energies (vacuum + eigen)", () -> FFS.free_run(odet, ctrl, equil, mats, intr))
            stage("normalize eigenfunctions", () -> FFS.normalize_eigenfunctions!(odet, free_energies.wt, equil.psio))
            if ctrl.kinetic_factor == 0 && intr.msing > 0 && fm_propagators !== nothing
                stage(
                    "Δ′ BVP (debug=$(ctrl.verbose))",
                    () -> FFS.compute_delta_prime_matrix!(intr, fm_propagators, fm_chunks;
                        wv=free_energies.wv, psio=equil.psio, debug=ctrl.verbose, S_at_surface_left=fm_S_left, ctrl=ctrl, equil=equil, mats=mats)
                )
            end
        end
        result = stage("build_result", () -> FFS.build_result(Symbol(ctrl.integrator), ctrl, equil, intr, metric, mats, odet, free_energies, nothing, nothing))
        if opts["hdf5"]
            stage("HDF5 write", () -> GPE.write_outputs_to_HDF5(result; locstab=locstab, ballooning_boundary=ballooning_boundary))
        end
        rm(workdir; recursive=true, force=true)

        total = sum(last, timings)
        println()
        @printf("pass %d/%d (%s)  case %s  N=%d msing=%d integrator=%s threads=%d blas=%d verbose=%s ODE steps=%d\n",
            pass, opts["repeat"], pass == 1 ? "cold, includes JIT" : "warm", basename(case), intr.numpert_total, intr.msing, ctrl.integrator, Threads.nthreads(),
            BLAS.get_num_threads(), ctrl.verbose, odet.total_steps)
        for (name, t) in timings
            @printf("  %-42s %8.2f s  %5.1f%%\n", name, t, 100t / total)
        end
        @printf("  %-42s %8.2f s\n", "total", total)
    end
end

main(ARGS)
