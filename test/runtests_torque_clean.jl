using Test

# Lightweight, hermetic smoke tests for torque subroutines

# Include minimal dependencies and module shims used by torque.jl
include(joinpath(@__DIR__, "..", "src", "pentrc", "dcon_interface.jl"))
using Main.DconInterface

# Minimal params expected by torque.jl
const nmethods = 3
const ngrids = 1

# Mirror minimal DconInterface fields into Main scope expected by torque.jl
global mthsurf = DconInterface.mthsurf[]
global mpert = DconInterface.mpert[]
global mfac = DconInterface.mfac
global psifac = DconInterface.psifac
global theta = DconInterface.theta
global eqfun = DconInterface.eqfun === nothing ? zeros(1) : DconInterface.eqfun
global rzphi = DconInterface.rzphi === nothing ? zeros(1) : DconInterface.rzphi
global bo = DconInterface.bo
global bmin = 0.5
global fnml = 1.0
global twopi = 2π

# Provide a minimal SplineType and helpers (avoid using malformed spline_mod.jl)
mutable struct SplineType
    xs::Vector{Float64}
    fs::Matrix{Float64}
    mx::Int
    fsi::Union{Matrix{Float64}, Nothing}
end

function SplineType(nx::Integer, nvar::Integer)
    # Many Fortran routines use arrays sized mthsurf+1; allocate +1 to match expectations
    xs = zeros(Float64, nx + 1)
    fs = zeros(Float64, nx + 1, nvar)
    return SplineType(xs, fs, nx, nothing)
end

function spline_alloc(nx::Integer, nvar::Integer)
    return SplineType(nx, nvar)
end

function spline_fit!(spl::SplineType, _mode::AbstractString)
    return nothing
end

function spline_int!(spl::SplineType)
    if spl.fsi === nothing
        spl.fsi = zeros(Float64, spl.mx, size(spl.fs,2))
    end
    return nothing
end

function spline_dealloc!(_spl)
    return nothing
end

function spline_eval_external(obj, psi::Float64; deriv::Integer=0)
    out = zeros(Float64, 9)
    if deriv == 1
        return out, out
    else
        return out
    end
end

# Minimal cspline helpers
function cspline_alloc(nx::Integer, nvar::Integer)
    xs = zeros(Float64, nx)
    fs = zeros(Float64, nx, nvar)
    return (xs=xs, fs=fs, mx=nx, nqty=nvar)
end

function cspline_fit!(_fb, _mode::AbstractString)
    return nothing
end

function cspline_int!(fb)
    if !haskey(fb, :fsi) || fb[:fsi] === nothing
        fb[:fsi] = zeros(Float64, fb[:mx], size(fb[:fs],2))
    end
    return nothing
end

function cspline_dealloc!(_fb)
    return nothing
end

function cspline_eval_external(obj, psi::Float64)
    if obj isa AbstractArray
        return obj
    end
    mp = try DconInterface.mpert[] catch; 1 end
    return ComplexF64.(zeros(mp * mp))
end

# Use clean bicube shim for tests
include(joinpath(@__DIR__, "..", "src", "pentrc", "bicube_mod_clean.jl"))
# Provide deterministic stub for energy integration (avoid loading heavy energy.jl)
function xintgrl_lsode(wn, wt, we, wd, wb, nuk, ell, leff, n, psi, lambda, method; op_record=false)
    # Return a simple complex value for testing
    return ComplexF64(-1.0, 0.0)
end

# Minimal pitch integration stub (kept very small)
function kappaintgrl(n, l, q, mfac_in, dbob_m_f, fnml)
    return 1.0
end

# Now include the torque implementation under test
include(joinpath(@__DIR__, "..", "src", "pentrc", "torque.jl"))

@testset "torque calculations" begin
    # Test 1: calculate_fcgl should return zero when kinetic profiles are zero
    @testset "calculate_fcgl basic" begin
        # Ensure a varying b(theta) so spline routines avoid degenerate minima
        function bicube_eval_external(bc, psi::Float64, theta::Float64, n::Integer)
            # return (b, db/dpsi, db/dtheta, jac, dj/dpsi)
            return [1.0 + 0.1*sin(2π*theta), 0.0, 0.0, 1.0, 0.0]
        end

        # set a reasonable angular resolution
        DconInterface.mthsurf[] = 8
        DconInterface.mpert[] = 4

        psi = 0.5
        n = 1
        l = 0
        tspl = SplineType(DconInterface.mthsurf[], 5)
        dbob_m_f = zeros(Float64, DconInterface.mpert[])
        divx_m_f = zeros(Float64, DconInterface.mpert[])
        kin_f = zeros(Float64, 9)  # zero kinetic profile -> torque should be zero
        s = 1

        t = calculate_fcgl(psi, n, l, tspl, dbob_m_f, divx_m_f, kin_f, s)
        @test isa(t, ComplexF64)
        @test t == ComplexF64(0,0)
    end

    # Test 2: calculate_rlar smoke test with a deterministic kappaintgrl stub
    @testset "calculate_rlar smoke" begin
        # Provide a small, deterministic kappaintgrl used by calculate_rlar
        function kappaintgrl(n, l, q, mfac_in, dbob_m_f, fnml)
            return 2.0
        end

        # Ensure global variables referenced by calculate_rlar exist
        global bmin = 0.5
        global fnml = 1.0

        psi = 0.5
        n = 1
        l = 0
        q = 1.0
        epsr = 0.1
        wdian = 0.0
        wdiat = 0.0
        welec = 0.0
        wdhat = 0.0
        wbhat = 0.0
        nueff = 0.0
        sq_s_f = zeros(Float64, 4)
        kin_f = zeros(Float64, 9);
        kin_f[1] = 1.0; kin_f[3] = 1.0
        s = 1
        dbob_m_f = zeros(Float64, DconInterface.mpert[])
        erecord = false

        t = calculate_rlar(psi, n, l, q, epsr, wdian, wdiat, welec, wdhat, wbhat,
                            nueff, sq_s_f, kin_f, s, dbob_m_f, erecord)

        @test isa(t, ComplexF64)
    end
end
