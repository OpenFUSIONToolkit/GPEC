using Test
using LinearAlgebra

# gal_solver = "cholesky": unscaled lower-band zpbtrf/zpbtrs (rdcon gal.f) on GalWorkspace's
# `ldab = kl + 1` lower band storage.
@testset "Galerkin banded Cholesky" begin
    FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates

    n, kl = 24, 3
    A = zeros(ComplexF64, n, n)
    for j in 1:n, i in max(1, j - kl):min(n, j + kl)
        A[i, j] = i == j ? 4 + sin(j) : 0.3 * cis(0.3i + 0.7j)
    end
    A = (A + A') / 2  # Hermitian, diagonally dominant → positive definite

    function lowerband(A)
        m = size(A, 2)
        ab = zeros(ComplexF64, kl + 1, m)
        for j in 1:m, i in j:min(m, j + kl)
            ab[1+i-j, j] = A[i, j]
        end
        return ab
    end

    x = ComplexF64[cis(0.1j) * (1 + j / n) for j in 1:n]
    B = A * hcat(x, -3x)
    ab = lowerband(A)
    FFS.gal_zpbtrf!(ab, kl)
    FFS.gal_zpbtrs!(ab, kl, B)
    @test B[:, 1] ≈ x rtol = 1e-13
    @test B[:, 2] ≈ -3x rtol = 1e-13

    # Lower factor matches the dense Cholesky factor
    L = cholesky(Hermitian(A, :L)).L
    @test all(ab[1+i-j, j] ≈ L[i, j] for j in 1:n for i in j:min(n, j + kl))

    @testset "indefinite matrix is refused" begin
        Aind = copy(A)
        Aind[5, 5] = -1
        @test_throws ErrorException FFS.gal_zpbtrf!(lowerband(Aind), kl)
    end
end
