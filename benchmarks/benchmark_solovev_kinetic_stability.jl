#!/usr/bin/env julia
"""
    benchmark_solovev_kinetic_stability.jl

Benchmark the self-consistent kinetic-DCON path (`kinetic_source="calculated"`)
for the analytic **Solovev** equilibrium against Fortran GPEC's kinetic DCON.

Unlike the DIIID benchmark (which reads a pre-existing `dcon_output_n1.nc`),
no committed Fortran reference exists for this Solovev case, so this script
*generates a matched Fortran input deck* from the Julia regression fixture and
runs Fortran `dcon` itself. The Julia and Fortran cases are made physically
identical: same Solovev parameters, same kinetic profile, same poloidal-mode
band (`delta_mlow=delta_mhigh=0`), same `nl`, collision operator, grid, and
edge truncation (`sas_flag`/`dmlim`).

The comparison is the least-stable total-energy eigenvalue:
  - Julia:   `FreeBoundaryStability/XiNorm/eigenmode_energies[1]` in `gpec.h5`
  - Fortran: the "Energies: ... real = …, imaginary = …" line printed by `dcon`
             (== `W_t_eigenvalue[1]`; parsing the log avoids a NetCDF dependency).

Acceptance (same thresholds as the DIIID kinetic benchmark):
  - Re(et[1]) within 5 %  of Fortran
  - Im(et[1]) within 20 % of Fortran (the damping rate — smaller, more FP-sensitive)

Reference result (2026-06, develop, mpert=8, nl=1, nutype="harmonic"):
    Re: Julia 15.888  vs Fortran 15.619   → 1.7 %
    Im: Julia -0.711  vs Fortran -0.660   → 7.6 %
This confirms the PV+residue energy integral (issue #227) is physically correct;
the pre-rewrite (`ximag`) value of 34.176 was not.

Usage:
    julia --project=. benchmarks/benchmark_solovev_kinetic_stability.jl

Environment (both required):
    GPEC_FORTRAN_DCON   path to the Fortran `dcon` executable
    GPEC_FORTRAN_SOLDIR Fortran solovev_kinetic_example dir used as deck template
"""

using Printf
using TOML
using HDF5

using GeneralizedPerturbedEquilibrium
const GPE = GeneralizedPerturbedEquilibrium

const REPO            = normpath(joinpath(@__DIR__, ".."))
const JULIA_FIXTURE   = joinpath(REPO, "test", "test_data", "regression_solovev_kinetic_calculated")
const FORTRAN_DCON    = get(ENV, "GPEC_FORTRAN_DCON", "")
const FORTRAN_SOLDIR  = get(ENV, "GPEC_FORTRAN_SOLDIR", "")

_p(args...) = (println(stderr, args...); flush(stderr))

"""
    replace_namelist_value!(path, key, value)

Overwrite `key = …` (Fortran namelist, `=`-separated, value up to a trailing
comment `!` or line end) with `value`, preserving the rest of the line.
"""
function replace_namelist_value!(path::String, key::String, value::AbstractString)
    lines = readlines(path)
    pat = Regex("^(\\s*" * key * "\\s*=\\s*)([^!]*)(!?.*)\$", "i")
    found = false
    for (i, ln) in enumerate(lines)
        m = match(pat, ln)
        if m !== nothing
            lines[i] = m.captures[1] * value * " " *
                       (isempty(m.captures[3]) ? "" : m.captures[3])
            found = true
        end
    end
    found || error("key '$key' not found in $path")
    open(path, "w") do io
        for ln in lines
            println(io, ln)
        end
    end
end

"""
    build_matched_fortran_deck(workdir) → workdir

Copy the Fortran solovev_kinetic_example deck into `workdir` and override the
fields that must match the Julia regression fixture (read from its `gpec.toml`
and `sol.toml`). Converts the fixture's `kinetic.dat` into Fortran `kinetic.txt`.
"""
function build_matched_fortran_deck(workdir::String)
    isdir(FORTRAN_SOLDIR) || error("Fortran deck template not found: '$FORTRAN_SOLDIR' (set GPEC_FORTRAN_SOLDIR)")
    for f in ("sol.in", "equil.in", "dcon.in", "pentrc.in", "vac.in")
        cp(joinpath(FORTRAN_SOLDIR, f), joinpath(workdir, f); force=true)
    end

    gpec = TOML.parsefile(joinpath(JULIA_FIXTURE, "gpec.toml"))
    eq   = gpec["Equilibrium"]
    ffs  = gpec["ForceFreeStates"]
    kf   = get(gpec, "KineticForces", Dict{String,Any}())

    # equil.in — grid + coordinate settings to match the Julia [Equilibrium] block.
    eqin = joinpath(workdir, "equil.in")
    replace_namelist_value!(eqin, "psilow",  string(eq["psilow"]))
    replace_namelist_value!(eqin, "psihigh", string(eq["psihigh"]))
    replace_namelist_value!(eqin, "mpsi",    string(eq["mpsi"]))
    replace_namelist_value!(eqin, "mtheta",  string(eq["mtheta"]))

    # dcon.in — kinetic flags already match; align mode band, edge truncation, n.
    dconin = joinpath(workdir, "dcon.in")
    replace_namelist_value!(dconin, "nn",          string(ffs["nn_low"]))
    replace_namelist_value!(dconin, "delta_mlow",  string(get(ffs, "delta_mlow", 0)))
    replace_namelist_value!(dconin, "delta_mhigh", string(get(ffs, "delta_mhigh", 0)))
    replace_namelist_value!(dconin, "qlow",        string(ffs["qlow"]))
    replace_namelist_value!(dconin, "qhigh",       string(ffs["qhigh"]))
    replace_namelist_value!(dconin, "singfac_min", string(ffs["singfac_min"]))
    # Julia set_psilim_via_dmlim defaults true (sas_flag) with dmlim 0.2.
    replace_namelist_value!(dconin, "sas_flag", "t")
    replace_namelist_value!(dconin, "dmlim",    "0.2")

    # pentrc.in — bounce harmonics + species; defaults mirror KineticForcesControl.
    pentin = joinpath(workdir, "pentrc.in")
    replace_namelist_value!(pentin, "nl",     string(get(kf, "nl", 1)))
    replace_namelist_value!(pentin, "nutype", "\"" * string(get(kf, "nutype", "harmonic")) * "\"")
    replace_namelist_value!(pentin, "f0type", "\"" * string(get(kf, "f0type", "maxwellian")) * "\"")
    replace_namelist_value!(pentin, "nufac",  string(get(kf, "nufac", 1)))
    replace_namelist_value!(pentin, "kinetic_file", "\"kinetic.txt\"")

    # kinetic.txt — Fortran readtable wants a title line then numeric rows; the
    # fixture's kinetic.dat is already the same 6-column ASCII layout.
    cp(joinpath(JULIA_FIXTURE, "kinetic.dat"), joinpath(workdir, "kinetic.txt"); force=true)

    return workdir
end

"""
    run_fortran_reference(workdir) → (re, im)

Run Fortran `dcon` in `workdir` and parse the least-stable total-energy
eigenvalue from its "Energies: ... real = …, imaginary = …" line.
"""
function run_fortran_reference(workdir::String)
    isfile(FORTRAN_DCON) || error("Fortran dcon not found: $FORTRAN_DCON (set GPEC_FORTRAN_DCON)")
    logpath = joinpath(workdir, "dcon_run.log")
    open(logpath, "w") do io
        run(pipeline(Cmd(`$FORTRAN_DCON`; dir=workdir); stdout=io, stderr=io))
    end
    log = read(logpath, String)
    m = match(r"Energies:.*real\s*=\s*([-\d.Ee+]+),\s*imaginary\s*=\s*([-\d.Ee+]+)", log)
    m === nothing && error("could not parse Fortran energies from $logpath")
    return parse(Float64, m.captures[1]), parse(Float64, m.captures[2])
end

function run_julia_reference()
    rundir = mktempdir(; prefix="solovev_kinetic_julia_")
    for f in ("gpec.toml", "sol.toml", "kinetic.dat")
        cp(joinpath(JULIA_FIXTURE, f), joinpath(rundir, f))
    end
    t0 = time()
    GPE.main([rundir])
    wall = time() - t0
    et = h5open(joinpath(rundir, "gpec.h5"), "r") do h5
        read(h5["FreeBoundaryStability/XiNorm/eigenmode_energies"])
    end
    return real(et[1]), imag(et[1]), wall
end

function run_benchmark()
    _p("=" ^ 70)
    _p("  Solovev Kinetic-DCON Benchmark: Julia KF→FFS vs Fortran DCON")
    _p("=" ^ 70)

    workdir = mktempdir(; prefix="solovev_kinetic_fortran_")
    build_matched_fortran_deck(workdir)
    _p("  Matched Fortran deck: $workdir")
    fort_re, fort_im = run_fortran_reference(workdir)
    _p(@sprintf("  Fortran W_t[1]: Re = %.6f  Im = %.6f", fort_re, fort_im))

    _p("\n--- Running GPE.main with kinetic_source=\"calculated\" ---")
    jul_re, jul_im, wall = run_julia_reference()
    _p(@sprintf("  GPE.main completed in %.1f s", wall))

    re_err = abs(jul_re - fort_re) / abs(fort_re) * 100
    im_err = abs(jul_im - fort_im) / abs(fort_im) * 100

    _p("\n" * "=" ^ 70)
    _p("  Least-stable total-energy eigenvalue (mpert matched)")
    _p("=" ^ 70)
    _p(@sprintf("  Re(et[1]):  Julia = %12.6f  Fortran = %12.6f  err = %6.2f%%", jul_re, fort_re, re_err))
    _p(@sprintf("  Im(et[1]):  Julia = %12.6f  Fortran = %12.6f  err = %6.2f%%", jul_im, fort_im, im_err))

    re_pass = re_err <= 5.0
    im_pass = im_err <= 20.0
    _p("\n" * "=" ^ 70)
    if re_pass && im_pass
        _p("  PASS — Re within 5 %, Im within 20 %")
    else
        _p("  FAIL — thresholds exceeded:")
        re_pass || _p(@sprintf("    Re err %.2f%% > 5%%", re_err))
        im_pass || _p(@sprintf("    Im err %.2f%% > 20%%", im_err))
    end
    _p("=" ^ 70)

    return (; jul_re, jul_im, fort_re, fort_im, re_err, im_err, wall)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
