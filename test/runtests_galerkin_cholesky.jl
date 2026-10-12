using Test
using LinearAlgebra

# gal_banded_solve! on full-band (gbtrf layout) systems built like the Galerkin matrix: Hermitian
# with Dirichlet identity rows (row zeroed, column kept) and nonzero prescribed values.
@testset "Galerkin banded solve: Cholesky first, LU fallback" begin
    FFS = GeneralizedPerturbedEquilibrium.ForceFreeStates

    n, kl = 24, 3
    H = zeros(ComplexF64, n, n)
    for j in 1:n, i in max(1, j - kl):min(n, j + kl)
        H[i, j] = i == j ? 4 + sin(j) : 0.3 * cis(0.3i + 0.7j)
    end
    H = (H + H') / 2  # Hermitian, diagonally dominant → positive definite
    fixed = [1, 2, n]

    # Dirichlet rows as gal_set_boundary! leaves them: identity row, column untouched
    function dirichlet(A)
        A = copy(A)
        for i in fixed
            A[i, :] .= 0
            A[i, i] = 1
        end
        return A
    end
    function fullband(A)
        ab = zeros(ComplexF64, 3kl + 1, n)
        for j in 1:n, i in max(1, j - kl):min(n, j + kl)
            ab[2kl+1+i-j, j] = A[i, j]
        end
        return ab
    end
    rhs = ComplexF64[cis(0.1j) * (1 + j / n) for j in 1:n, _ in 1:2]
    rhs[fixed, 2] .= [1.0, -0.5im, 2.0]  # prescribed values, as in the rpec coil columns

    function check(A, solver)
        sol = copy(rhs)
        used = FFS.gal_banded_solve!(fullband(A), sol, kl, kl, solver)
        @test sol ≈ A \ rhs rtol = 1e-12
        return used
    end

    @test check(dirichlet(H), "auto") == "cholesky"
    @test check(dirichlet(H), "cholesky") == "cholesky"
    @test check(dirichlet(H), "LU") == "LU"

    indefinite = copy(H)
    indefinite[5, 5] = -1
    @test check(dirichlet(indefinite), "auto") == "LU"
    @test_throws ErrorException FFS.gal_banded_solve!(fullband(dirichlet(indefinite)), copy(rhs), kl, kl, "cholesky")

    nonhermitian = copy(H)
    nonhermitian[6, 4] += 0.1
    @test check(dirichlet(nonhermitian), "auto") == "LU"

    # the lower-band factor equals the dense Cholesky factor
    ab = zeros(ComplexF64, kl + 1, n)
    for j in 1:n, i in j:min(n, j + kl)
        ab[1+i-j, j] = H[i, j]
    end
    @test FFS.gal_zpbtrf!(ab, kl)
    L = cholesky(Hermitian(H, :L)).L
    @test all(ab[1+i-j, j] ≈ L[i, j] for j in 1:n for i in j:min(n, j + kl))
end
