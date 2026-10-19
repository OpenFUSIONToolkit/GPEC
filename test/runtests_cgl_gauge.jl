@testset "CGL pressure closure in the perpendicular gauge" begin
    KF = GeneralizedPerturbedEquilibrium.KineticForces
    intr = KF.KineticForcesInternal(; mthsurf=64, mfac=[0])
    constant_field = (x; hint=nothing) -> 3.0
    constant_jac = (x; hint=nothing) -> 2.0
    equil = (; eqfun_B=constant_field, rzphi_jac=constant_jac)
    for (D, K) in ((0.0im, 1.0 + 0.3im), (1.0 + 2.0im, 0.7 - 0.3im), (1.0 + 0.2im, 0.0im))
        b = -(D + K)
        dp_perp = -D + b
        dp_parallel = -3D - 2b
        # Independent double-adiabatic pressure work; 1/2 for a real Fourier mode.
        W = -real((dp_parallel - dp_perp) * conj(b) + dp_parallel * conj(D)) / 2
        expected = 2im * 5.0 * 7.0 * (2.0 * W / 2)
        actual = KF.calculate_fcgl(0.4, 1, 0, nothing, [b], [D], 1.0, 5.0, 7.0, equil, intr)
        @test actual ≈ expected rtol=1e-13
        @test KF.calculate_fcgl(0.4, 1, 1, nothing, [b], [D], 1.0, 5.0, 7.0, equil, intr) == 0
    end
end
