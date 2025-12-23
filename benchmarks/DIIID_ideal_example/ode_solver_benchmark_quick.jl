"""
Quick ODE Solver Benchmark Script - Testing version

This is a quick test version that runs fewer solvers and tolerances
to verify the benchmark works before running the full version.
"""

using Pkg
Pkg.activate("../..")
using JPEC
using OrdinaryDiffEq
using Plots
using JLD2
using Printf
using Statistics

# Import necessary modules
import JPEC.DCON as DCON
import JPEC.Equilibrium as Equilibrium

"""
Modified version of ode_step! that accepts a solver method as a parameter.
"""
function ode_step_with_solver!(odet::DCON.OdeState, ctrl::DCON.DconControl, equil, ffit, intr, solver)
    cb = DiscreteCallback((u, t, integrator) -> true, DCON.integrator_callback!)
    rtol = DCON.compute_tols(ctrl, intr, odet)
    prob = ODEProblem(DCON.sing_der!, odet.u, (odet.psifac, odet.psimax), (ctrl, equil, ffit, intr, odet))
    sol = solve(prob, solver; reltol=rtol, callback=cb)
    odet.u .= sol.u[end]
    odet.psifac = sol.t[end]
end

"""
Modified version of ode_run that uses a custom solver method.
"""
function ode_run_with_solver(ctrl::DCON.DconControl, equil, ffit, intr, solver)
    odet = DCON.OdeState(intr.numpert_total, ctrl.numsteps_init, ctrl.numunorms_init, intr.msing)

    if ctrl.sing_start <= 0
        DCON.ode_axis_init!(odet, ctrl, equil, intr)
    else
        error("sing_start > 0 not implemented yet!")
    end

    ode_step_with_solver!(odet, ctrl, equil, ffit, intr, solver)

    while odet.ising != ctrl.ksing && odet.next == "cross"
        if ctrl.kin_flag
            error("kin_flag = true not implemented yet!")
        else
            DCON.ode_ideal_cross!(odet, ctrl, equil, ffit, intr)
        end
        ode_step_with_solver!(odet, ctrl, equil, ffit, intr, solver)
    end

    if ctrl.psiedge < intr.psilim
        odet.step = DCON.findmax_dW_edge!(odet, ctrl, equil, ffit, intr)
        DCON.trim_storage!(odet)
        intr.psilim = odet.psi_store[end]
        intr.qlim = odet.q_store[end]
        odet.u .= odet.u_store[:, :, :, end]
    else
        odet.step -= 1
        DCON.trim_storage!(odet)
    end

    odet.nzero = DCON.evaluate_stability_criterion!(odet, equil)
    DCON.transform_u!(odet, intr)

    return odet, odet.step
end

# Quick test
println("Quick ODE Solver Benchmark Test")
println("="^70)

println("\nLoading equilibrium...")
equil = Equilibrium.setup_equilibrium("equil.toml")

println("Setting up DCON...")
ctrl = DCON.DconControl("dcon.toml")
wv, grri, xzpts = JPEC.Vacuum.mscvac()
ffit = DCON.fourfit(ctrl, equil)
intr = DCON.dcon_init(ctrl, equil, wv, grri, xzpts)

# Test with just Tsit5 and one tolerance
println("\nTesting Tsit5 with tol=1e-4...")
ctrl.tol_r = 1e-4
ctrl.tol_nr = 1e-4
ctrl.verbose = false

start_time = time()
odet, nsteps = ode_run_with_solver(ctrl, equil, ffit, intr, Tsit5())
elapsed = time() - start_time

println("✓ Success!")
println("  Steps: $nsteps")
println("  Time: $(@sprintf("%.3f", elapsed)) s")
println("\nQuick test complete - ready for full benchmark!")
