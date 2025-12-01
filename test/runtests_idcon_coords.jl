using Test

# Load the DCON interface module
include(joinpath(@__DIR__, "..", "src", "pentrc", "dcon_interface.jl"))

@testset "idcon_coords smoke" begin
    # Verify idcon_coords runs for both calling patterns and mutates
    # the spectrum in-place without throwing, producing finite complex values.
    psi = 0.3
    spec = ComplexF64[1.0+0im, 2.0+0im, -0.5+0im]
    mfac = [1,2,3]
    mpert = 3
    @test nothing === DconInterface.idcon_coords(psi, spec, mfac, mpert, 0, 0, 1, 0, 1, 1)
    @test all(isfinite, real.(spec)) && all(isfinite, imag.(spec))

    spec2 = ComplexF64[3.0+0im, -1.0+0im]
    ms = [1,2]
    nm = 2
    @test nothing === DconInterface.idcon_coords(psi, spec2, ms, nm, 1, 0, 1, 0, 1, 1)
    @test all(isfinite, real.(spec2)) && all(isfinite, imag.(spec2))
end
