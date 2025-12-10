using Test
using JPEC

# --- Helper constructors ---
# Minimal valid inputs
function make_inputs(; mr=4, mz=4, ma=4, e=1.7, a=0.3, r0=1.7, q0=1.0, 
                      p0fac=1.2, b0fac=1.0, f0fac=1.0)
    equil_inputs = JPEC.Equilibrium.EquilibriumConfig()  # or mock/minimal constructor
    sol_inputs = JPEC.Equilibrium.SolovevConfig(mr, mz, ma, e, a, r0, q0, p0fac, b0fac, f0fac)
    return equil_inputs, sol_inputs
end

@testset "sol_run basic functionality" begin
    equil_inputs, sol_inputs = make_inputs()
    dri = JPEC.Equilibrium.sol_run(equil_inputs, sol_inputs)

    @test isa(dri, JPEC.Equilibrium.DirectRunInput)
    @test hasproperty(dri, :sq_in)
    @test hasproperty(dri, :psi_in)
    @test hasproperty(dri, :rmin)
    @test hasproperty(dri, :zmax)
end

@testset "sol_run clamps p0fac to ≥ 1" begin
    equil_inputs, sol_inputs = make_inputs(p0fac=0.5)
    dri = JPEC.Equilibrium.sol_run(equil_inputs, sol_inputs)
    @test all(dri.sq_in.fs[:, 2] .>= 0)  # no negative pressures
end

@testset "sol_run scalar relationships" begin
    equil_inputs, sol_inputs = make_inputs()
    mr, mz, ma = sol_inputs.mr, sol_inputs.mz, sol_inputs.ma
    e, a, r0, q0 = sol_inputs.e, sol_inputs.a, sol_inputs.r0, sol_inputs.q0
    p0fac, b0fac, f0fac = sol_inputs.p0fac, sol_inputs.b0fac, sol_inputs.f0fac

    dri = JPEC.Equilibrium.sol_run(equil_inputs, sol_inputs)

    # Derived quantities consistency
    f0_expected = r0 * b0fac
    psio_expected = e * f0_expected * a * a / (2 * q0 * r0)

    @test isapprox(dri.psio, psio_expected; rtol=1e-10)

    # Range symmetry
    @test isapprox(dri.zmax, -dri.zmin)
    @test dri.rmax > dri.rmin

    # Proper grid size consistency
    @test length(dri.psi_in.xs) == mr + 1
end

@testset "sol_run spline integrity" begin
    equil_inputs, sol_inputs = make_inputs(mr=6, mz=5, ma=3)
    dri = JPEC.Equilibrium.sol_run(equil_inputs, sol_inputs)
    sq = dri.sq_in
    psi = dri.psi_in

    # Spline types
    @test isa(sq, JPEC.Spl.CubicSpline)
    @test isa(psi, JPEC.Spl.BicubicSpline)

    # Domain monotonicity
    @test issorted(sq.xs)
    @test issorted(psi.xs)
    @test issorted(psi.ys)

    # Check that psi values are finite
    @test all(isfinite, psi.fs)
end

@testset "sol_run 2D psi field properties" begin
    equil_inputs, sol_inputs = make_inputs(mr=3, mz=3)
    dri = JPEC.Equilibrium.sol_run(equil_inputs, sol_inputs)
    psi = dri.psi_in

    # Check psi symmetry in z
    z_half = Int(length(psi.ys) ÷ 2)
    vals_top = psi.fs[:, end, 1]
    vals_bottom = psi.fs[:, 1, 1]
    @test all(abs.(vals_top .- vals_bottom) .< 1e-12)  # nearly symmetric about z=0
end

@testset "sol_run parameter sensitivity" begin
    equil_inputs, sol_inputs = make_inputs()
    dri1 = JPEC.Equilibrium.sol_run(equil_inputs, sol_inputs)
    dri2 = JPEC.Equilibrium.sol_run(equil_inputs, JPEC.Equilibrium.SolovevConfig(sol_inputs.mr, sol_inputs.mz, sol_inputs.ma,
                                               sol_inputs.e * 1.1, sol_inputs.a, sol_inputs.r0,
                                               sol_inputs.q0, sol_inputs.p0fac, sol_inputs.b0fac,
                                               sol_inputs.f0fac))
    @test dri1.psio != dri2.psio  # psio should depend on elongation e
end

@testset "sol_run extreme inputs" begin
    # minimal grid
    equil_inputs, sol_inputs = make_inputs(mr=1, mz=1, ma=1)
    dri = JPEC.Equilibrium.sol_run(equil_inputs, sol_inputs)
    @test length(dri.psi_in.xs) == 2
    @test length(dri.psi_in.ys) == 2

    # very high aspect ratio
    equil_inputs, sol_inputs = make_inputs(e=0.8, a=0.1, r0=10.0)
    dri = JPEC.Equilibrium.sol_run(equil_inputs, sol_inputs)
    @test isfinite(dri.psio)
end
