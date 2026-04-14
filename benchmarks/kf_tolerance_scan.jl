#!/usr/bin/env julia
"""
    kf_tolerance_scan.jl

Scan outer ψ ODE tolerance (`rtol_psi == atol_psi`) for KineticForces fgar on
the DIIID benchmark. For each tolerance point, reports wall time, accepted
ψ-step count, and error vs Fortran PENTRC reference.

FFS + PE state are built once (~25 s) and reused across all points. A warmup
torque call runs before any timing to trigger JIT compilation. Inner pitch
(`*_xlmda`) tolerances are held at their committed values — this script only
varies the outer ψ ODE tolerance.

Output: markdown table to stderr; picks the loosest tolerance with both
T_fgar and dW_fgar errors ≤ 5 %.
"""

using Printf

using GeneralizedPerturbedEquilibrium
const GPE = GeneralizedPerturbedEquilibrium
const FFS = GPE.ForceFreeStates
const KF  = GPE.KineticForces
const Eq  = GPE.Equilibrium
const PE  = GPE.PerturbedEquilibrium

const BENCHMARK_DIR = joinpath(@__DIR__, "DIIID_kinetic_example")
const FORTRAN_DIR   = expanduser("~/Code/gpec/docs/examples/DIIID_ideal_example")

const REF_T_FGAR  = 1.78040606348608
const REF_DW_FGAR = 0.136828994301428

# Reuse loader from the main benchmark
include(joinpath(@__DIR__, "benchmark_diiid_kinetic.jl")) # defines load_fortran_xclebsch
# Note: including that file also defines run_benchmark() — we don't call it.

_p(args...) = (println(stderr, args...); flush(stderr))
_pf(fmt, args...) = (print(stderr, Printf.format(Printf.Format(fmt), args...)); flush(stderr))


"""
    build_kf_state() → (equil, intr, kf_intr, kinetic_profiles)

Construct the Julia-side state (FFS + PE Clebsch via Fortran xclebsch
+ JBB deweighting). Returns everything compute_torque_all_methods! needs.
"""
function build_kf_state()
    _p("--- build_kf_state: FFS + PE setup ---")
    t0 = time()
    result = GPE.main([BENCHMARK_DIR])
    equil = result.equil
    intr  = result.intr
    ctrl  = result.ctrl
    _pf("  FFS done in %.1f s\n", time() - t0)

    metric = FFS.make_metric(equil; mband=intr.mband, fft_flag=ctrl.fft_flag)

    xclebsch_path = joinpath(FORTRAN_DIR, "gpec_xclebsch_n1.out")
    psi_grid_f, clebsch_psi, clebsch_psi1, clebsch_alpha, _, _ =
        load_fortran_xclebsch(xclebsch_path)

    chi1 = 2π * equil.psio
    pe_state = PE.PerturbedEquilibriumState(;
        psi_grid = psi_grid_f,
        xi_modes = (
            clebsch_psi   = clebsch_psi,
            clebsch_psi1  = clebsch_psi1,
            clebsch_alpha = clebsch_alpha ./ chi1,
        ),
    )

    kf_intr = KF.KineticForcesInternal(equil; verbose=false)
    KF.set_perturbation_data!(kf_intr, pe_state, intr, equil, metric)

    kinetic_profiles = Eq.load_kinetic_profiles(
        joinpath(BENCHMARK_DIR, "kinetic.dat");
        zi=1, zimp=6, mi=2, mimp=12)

    return equil, intr, kf_intr, kinetic_profiles
end


"""
    time_fgar(tol, equil, kf_intr, kinetic_profiles; psilims=[0.0, 1.0]) → NamedTuple

Run one fgar pass at the given `rtol_psi == atol_psi == tol`. Returns wall
time, ψ-step count, and error vs Fortran reference.
"""
function time_fgar(tol::Float64, equil, kf_intr, kinetic_profiles;
                   psilims::Vector{Float64}=[0.0, 1.0])
    ctrl = KF.KineticForcesControl(;
        fgar_flag=true, tgar_flag=false, nn=1, nl=4,
        mi=2, zi=1, nutype="harmonic", f0type="maxwellian",
        nufac=1, psilims=psilims, kinetic_file="kinetic.dat",
        atol_psi=tol, rtol_psi=tol,
        verbose=false)

    state = KF.KineticForcesState()
    t0 = time()
    KF.compute_torque_all_methods!(state, kf_intr, ctrl, equil, kinetic_profiles)
    wall = time() - t0

    mr = state.method_results["fgar"]
    T_err  = abs(real(mr.total_torque) - REF_T_FGAR)  / abs(REF_T_FGAR)  * 100
    dW_err = abs(real(mr.total_energy) - REF_DW_FGAR) / abs(REF_DW_FGAR) * 100

    return (; tol, wall, nsteps=mr.psi_nsteps,
              T=real(mr.total_torque), T_err,
              dW=real(mr.total_energy), dW_err)
end


function run_scan()
    _p("=" ^ 78)
    _p("  KF fgar tolerance scan — outer ψ ODE (rtol_psi == atol_psi)")
    _p("=" ^ 78)

    equil, intr, kf_intr, kinetic_profiles = build_kf_state()

    # JIT warmup on a narrow ψ window to trigger compilation without paying
    # full runtime. psilims=[0.5,0.6] touches every kernel (pitch ODE, energy
    # QuadGK, JBB interp) but with ~10 ψ-steps at loose tol.
    _p("\n--- JIT warmup (psilims=[0.5, 0.6], tol=1e-3) ---")
    warmup = time_fgar(1e-3, equil, kf_intr, kinetic_profiles; psilims=[0.5, 0.6])
    _pf("  warmup: %.2f s, %d ψ-steps\n", warmup.wall, warmup.nsteps)

    # Full-range scan
    tols = [1e-5, 1e-4, 1e-3, 1e-2]
    results = NamedTuple[]
    for tol in tols
        _p(@sprintf("\n--- Scan point: tol = %.0e ---", tol))
        r = time_fgar(tol, equil, kf_intr, kinetic_profiles)
        push!(results, r)
        _pf("  wall = %.1f s, nsteps = %d, T = %.6f (err %.3f%%), dW = %.6f (err %.3f%%)\n",
            r.wall, r.nsteps, r.T, r.T_err, r.dW, r.dW_err)
    end

    # Summary table
    _p("\n" * "=" ^ 78)
    _p("  Tolerance scan summary — fgar on DIIID")
    _p("=" ^ 78)
    _p("| tol    | wall (s) | ψ-steps | T_err (%) | dW_err (%) | PASS? |")
    _p("|--------|----------|---------|-----------|------------|-------|")
    passing = NamedTuple[]
    for r in results
        pass = (r.T_err ≤ 5.0) && (r.dW_err ≤ 5.0)
        pass && push!(passing, r)
        _pf("| %.0e | %8.1f | %7d | %9.3f | %10.3f | %s |\n",
            r.tol, r.wall, r.nsteps, r.T_err, r.dW_err,
            pass ? "yes" : "NO")
    end

    if !isempty(passing)
        # "Loosest" = largest tol among those passing
        chosen = last(sort(passing; by=r -> r.tol))
        _p("\n  PICKED: tol = $(chosen.tol) — $(chosen.wall) s, $(chosen.nsteps) ψ-steps, " *
           "T_err=$(chosen.T_err)%, dW_err=$(chosen.dW_err)%")
    else
        _p("\n  No tolerance point met <5% gate — keeping current defaults")
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_scan()
end
