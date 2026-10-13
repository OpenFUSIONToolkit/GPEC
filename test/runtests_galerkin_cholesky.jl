using Test
using LinearAlgebra

# gal_banded_solve! on a full-band (gbtrf layout) system built like the Galerkin matrix: Hermitian, with
# Dirichlet identity rows (row zeroed, column kept) and nonzero prescribed values, as in the rpec coil columns.
@testset "Galerkin banded solve: Cholesky first, LU fallback" begin
    FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
    n, kl, fixed = 24, 3, [1, 2, 24]
    H = zeros(ComplexF64, n, n)
    for j in 1:n, i in max(1, j - kl):min(n, j + kl)
        H[i, j] = i == j ? 4 + sin(j) : 0.3 * cis(0.3i + 0.7j)
    end
    H = (H + H') / 2  # diagonally dominant → positive definite
    function system(A)
        A = copy(A)
        A[fixed, :] .= 0
        A[fixed, fixed] = I(length(fixed))
        ab = zeros(ComplexF64, 3kl + 1, n)
        for j in 1:n, i in max(1, j - kl):min(n, j + kl)
            ab[2kl+1+i-j, j] = A[i, j]
        end
        return A, ab
    end
    rhs = ComplexF64[cis(0.1j) * (1 + j / n) for j in 1:n, _ in 1:2]
    rhs[fixed, 2] .= [1.0, -0.5im, 2.0]

    A, ab = system(H)
    sol = copy(rhs)
    @test FFS.gal_banded_solve!(ab, sol, kl, kl, "auto") == "cholesky"
    @test sol ≈ A \ rhs rtol = 1e-12

    H[5, 5] = -1
    A, ab = system(H)
    sol = copy(rhs)
    @test FFS.gal_banded_solve!(copy(ab), sol, kl, kl, "auto") == "LU"
    @test sol ≈ A \ rhs rtol = 1e-12
    @test_throws ErrorException FFS.gal_banded_solve!(ab, copy(rhs), kl, kl, "cholesky")
end
