# SolveTolerances.jl
#
# Absolute tolerance and completion check shared by the equilibrium ODE solves
# (direct field-line, large-aspect-ratio and TJ-analytic integrations).

"""
Ceiling on the absolute tolerance of the equilibrium ODE solves; the historical fixed value, kept so a loose `flux_surface_rtol` never loosens abstol.
"""
const EQUIL_ABSTOL_MAX = 1e-8

"""
    equil_abstol(rtol, cap=EQUIL_ABSTOL_MAX)

Absolute tolerance of an equilibrium ODE solve: follows `flux_surface_rtol` but is never looser than `cap`.
"""
equil_abstol(rtol::Real, cap::Real=EQUIL_ABSTOL_MAX) = min(rtol, cap)

"""
    check_equil_solve(sol, what)

Return the equilibrium ODE solution `sol`, or raise an error naming the solve `what` if the integrator
stopped early (step limit, step-size underflow, instability) instead of finishing or terminating on its callback.
"""
function check_equil_solve(sol, what::AbstractString)
    sol.retcode == ReturnCode.Success || sol.retcode == ReturnCode.Terminated ||
        error("Equilibrium ODE solve ($what) stopped early at t = $(sol.t[end]) with retcode $(sol.retcode).")
    return sol
end
