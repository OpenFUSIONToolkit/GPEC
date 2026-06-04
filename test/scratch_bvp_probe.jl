# GO/NO-GO probe for issue #251: native-complex MIRK6 collocation.
# Scratch file (not wired into runtests.jl). Validates the three make-or-break unknowns:
#   (1) native ComplexF64 MIRK6 solves end-to-end (no real/imag split),
#   (2) all-boundary-conditions-at-the-left-end (IVP-posed-as-BVP, our fundamental-matrix case),
#   (3) interior-point residual via the callable solution u(t_interior).
# Run: julia --project=. test/scratch_bvp_probe.jl

using BoundaryValueDiffEqMIRK
using LinearAlgebra
using Test

const A = ComplexF64[0.0+1.0im -1.0+0.0im; 1.0+0.0im 0.0-0.5im]   # arbitrary complex 2x2
const TSPAN = (0.0, 1.0)
const U0 = ComplexF64[1.0+0.0im, 0.0+1.0im]                       # full left-end state

f!(du, u, p, t) = (mul!(du, A, u); nothing)                       # linear: u' = A u, native complex

# All BCs at the LEFT end (mimics axis IC fully specifying the fundamental-matrix column).
function bc_left!(res, u, p, t)
    uL = u(TSPAN[1])
    res[1] = uL[1] - U0[1]
    res[2] = uL[2] - U0[2]
    return nothing
end

# One left BC + one INTERIOR-point residual (mimics a singular-surface matching node).
const TMID = 0.5
const REF_MID = exp(A * TMID) * U0   # analytic interior value
function bc_interior!(res, u, p, t)
    res[1] = u(TSPAN[1])[1] - U0[1]
    res[2] = u(TMID)[2] - REF_MID[2]
    return nothing
end

"Try native-complex MIRK6, escalating jac_alg/nlsolve only if the plain call errors."
function solve_native(bc!; dt=0.02)
    u0guess = ComplexF64[1.0, 1.0]
    prob = BVProblem(f!, bc!, u0guess, TSPAN)
    # Attempt 1: plain native complex (complex AD auto-selected via allowscomplex).
    try
        sol = solve(prob, MIRK6(); dt=dt, abstol=1e-10, reltol=1e-10)
        return sol, "MIRK6() plain native-complex"
    catch e
        @warn "plain native-complex MIRK6 failed; escalating jac_alg" exception=(e, catch_backtrace())
    end
    # Attempt 2: explicit AutoFiniteDiff Jacobian (the documented complex gotcha fix).
    jac_alg = BVPJacobianAlgorithm(; bc_diffmode=AutoFiniteDiff(), nonbc_diffmode=AutoSparse(AutoFiniteDiff()))
    sol = solve(prob, MIRK6(; jac_alg=jac_alg); dt=dt, abstol=1e-10, reltol=1e-10)
    return sol, "MIRK6(jac_alg=AutoFiniteDiff)"
end

@testset "native-complex MIRK6 GO/NO-GO" begin
    @testset "all-left-BC (IVP-posed) matches matrix exponential" begin
        sol, how = solve_native(bc_left!)
        @info "all-left-BC solved via: $how"
        uend = sol(TSPAN[2])
        ref = exp(A * (TSPAN[2] - TSPAN[1])) * U0
        @test eltype(uend) <: Complex          # confirms native complex carried through
        @test isapprox(uend, ref; rtol=1e-6)
    end

    @testset "interior-point residual via u(t_interior)" begin
        sol, how = solve_native(bc_interior!)
        @info "interior-BC solved via: $how"
        umid = sol(TMID)
        @test isapprox(umid[2], REF_MID[2]; rtol=1e-6)
    end
end

println("SMOKE_PROBE_DONE")
