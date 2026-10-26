using Test
using LinearAlgebra

# gal_banded_solve! on a full-band (gbtrf layout) system built like the Galerkin matrix: Hermitian, with
# Dirichlet identity rows (row zeroed, column kept) and nonzero prescribed values, as in the rpec coil columns.
@testset "Galerkin banded solve: Cholesky first, LU fallback" begin
    FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates
    @test FFS.empty_galerkin_result().solver == "none"
    n, kl, fixed = 24, 3, [1, 2, 24]
    H = zeros(ComplexF64, n, n)
    for j in 1:n, i in max(1, j-kl):min(n, j+kl)
        H[i, j] = i == j ? 4 + sin(j) : 0.3 * cis(0.3i + 0.7j)
    end
    H = (H + H') / 2  # diagonally dominant → positive definite
    function system(A)
        A = copy(A)
        A[fixed, :] .= 0
        A[fixed, fixed] = I(length(fixed))
        ab = zeros(ComplexF64, 3kl + 1, n)
        for j in 1:n, i in max(1, j-kl):min(n, j+kl)
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

    @testset "asymmetric off-diagonal entries fall back without changing the system" begin
        H[5, 5] = 4 + sin(5)
        H[6, 4] += 0.1
        A, ab = system(H)
        sol = copy(rhs)
        @test_logs (:info, r"solving by LU") (@test FFS.gal_banded_solve!(copy(ab), sol, kl, kl, "auto") == "LU")
        @test sol ≈ A \ rhs rtol = 1e-12
        @test_throws ErrorException FFS.gal_banded_solve!(ab, copy(rhs), kl, kl, "cholesky")
        H[6, 4] -= 0.1
    end

    @testset "a perturbed Dirichlet row is not silently eliminated" begin
        A, ab = system(H)
        A[1, 3] = 1e-3
        ab[2kl+1+1-3, 3] = A[1, 3]
        sol = copy(rhs)
        @test_logs (:info, r"solving by LU") (@test FFS.gal_banded_solve!(ab, sol, kl, kl, "auto") == "LU")
        @test sol ≈ A \ rhs rtol = 1e-12
    end

    @testset "Cholesky projects imaginary diagonals explicitly" begin
        H[5, 5] += 1e-6im
        A, ab = system(H)
        sol = copy(rhs)
        @test FFS.gal_banded_solve!(ab, sol, kl, kl, "auto") == "cholesky"
        A[5, 5] = real(A[5, 5])
        @test sol ≈ A \ rhs rtol = 1e-12
        H[5, 5] = real(H[5, 5])
    end

    @testset "round-off asymmetry is tolerated" begin
        H[6, 4] += 1e-13
        A, ab = system(H)
        sol = copy(rhs)
        @test FFS.gal_banded_solve!(ab, sol, kl, kl, "auto") == "cholesky"
        @test sol ≈ A \ rhs rtol = 1e-12
        H[6, 4] -= 1e-13
    end

    @testset "nonfinite entries reject Cholesky without changing the RHS" begin
        _, ab = system(H)
        for (row, col) in ((2kl + 1, 5), (2kl + 2, 4), (2kl + 3, 1))
            broken = copy(ab)
            broken[row, col] = NaN
            sol = copy(rhs)
            @test !FFS.gal_cholesky_solve!(broken, sol, kl, kl)
            @test sol == rhs
        end
    end

    @testset "unequal bandwidths have a structural diagnostic" begin
        kl_rect, ku_rect = 2, 1
        A = ComplexF64[3 0.1 0; 0.2 3 0.1; 0.1 0.2 3]
        ab = zeros(ComplexF64, 2kl_rect + ku_rect + 1, 3)
        for j in 1:3, i in max(1, j-ku_rect):min(3, j+kl_rect)
            ab[kl_rect+ku_rect+1+i-j, j] = A[i, j]
        end
        b = ones(ComplexF64, 3, 1)
        sol = copy(b)
        @test_logs (:info, r"equal lower and upper bandwidths") (@test FFS.gal_banded_solve!(copy(ab), sol, kl_rect, ku_rect, "auto") == "LU")
        @test sol ≈ A \ b rtol = 1e-12
        @test_throws ArgumentError FFS.gal_banded_solve!(ab, copy(b), kl_rect, ku_rect, "cholesky")
    end
end
