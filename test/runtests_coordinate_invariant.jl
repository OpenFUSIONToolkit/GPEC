# Tests for the coordinate-invariant field-space (b, b̃, b̄) treatment of the
# PerturbedEquilibrium control-surface response matrices and the ForceFreeStates energy
# (issue #233 / Pharr 2026).
#
# The control-surface matrices are stored in the coordinate-invariant root-area-weighted
# field (b̃) space. With Σ ≡ sqrtamat (the √weight) and the scalar surface area A ≡ jarea:
#   - b̃→b̄ operator  S = Σ/√A          (area-weighted field; b̄ = S·b̃)
#   - bare → b̃       Σ = S·√A          (b = Σ⁻¹·b̃)
#   - b̃→flux conform R = S·A           (Φ = R·b̃; poloidal flux is only the scalar Φ = A·b̄)
# These tests verify the transform rules in `field_space_response_matrices`, the field/flux
# recovery contracts, and the ForceFreeStates `c = jarea` energy refactor.

using Test
using LinearAlgebra
using GeneralizedPerturbedEquilibrium

const PE = GeneralizedPerturbedEquilibrium.PerturbedEquilibrium

# Deterministic, well-conditioned synthetic operators (no RNG → reproducible).
function _synthetic_inputs(n::Int)
    # Σ ≡ sqrtamat: a generic invertible complex operator standing in for the √weight convolution.
    Σ = Matrix{ComplexF64}(undef, n, n)
    for i in 1:n, j in 1:n
        Σ[i, j] = (1.0 + 0.3 * cospi((i + 2j) / n)) * cis(0.2 * (i - j)) * (i == j ? 1.0 : 0.15)
    end
    jarea = 2.37                            # scalar flux-surface area A
    S = Σ ./ sqrt(jarea)                    # b̃→b̄ operator
    # Hermitian positive-definite inductances (energy generators, flux space).
    A = [cis(0.1 * (i - j)) / (1 + abs(i - j)) for i in 1:n, j in 1:n]
    L = A * A' + n * I                      # Hermitian PD surface inductance
    B = [cis(-0.07 * (i + j)) / (1 + abs(i - j)) for i in 1:n, j in 1:n]
    Λ = B * B' + 2n * I                     # Hermitian PD plasma inductance
    return Matrix{ComplexF64}(Σ), Matrix{ComplexF64}(S), jarea, Matrix{ComplexF64}(L), Matrix{ComplexF64}(Λ)
end

@testset "Coordinate-invariant field-space response matrices" begin
    n = 6
    Σ, S, jarea, L, Λ = _synthetic_inputs(n)
    R = S .* jarea                          # b̃→flux conform Σ·√A
    P = Λ / L                               # permeability  Φ_tot = P·Φ_x
    ϱ = inv(L) * (Λ - L) * inv(L)           # reluctance

    fm = PE.field_space_response_matrices(Λ, L, P, ϱ, S, jarea)

    @testset "Flux recovery contract (round-trip via R = S·A)" begin
        @test R * fm.permeability / R ≈ P rtol = 1e-10
        @test R * fm.surface_inductance * R' ≈ L rtol = 1e-10
        @test R * fm.plasma_inductance * R' ≈ Λ rtol = 1e-10
        @test (R') \ fm.reluctance / R ≈ ϱ rtol = 1e-10
    end

    @testset "Area-weighted (b̄) recovery via S (= flux/A²)" begin
        # b̄-space inductance L_b̄ = S·L̃·S† = L/A² since S·R⁻¹ = A⁻¹·I.
        @test S * fm.surface_inductance * S' ≈ L ./ jarea^2 rtol = 1e-10
        @test S * fm.plasma_inductance * S' ≈ Λ ./ jarea^2 rtol = 1e-10
        @test S * fm.permeability / S ≈ P rtol = 1e-10   # similarity: A cancels
    end

    @testset "Internal consistency of the b̃ transform rules" begin
        @test fm.permeability ≈ fm.plasma_inductance / fm.surface_inductance rtol = 1e-10
        L̃inv = inv(fm.surface_inductance)
        @test fm.reluctance ≈ L̃inv * (fm.plasma_inductance - fm.surface_inductance) * L̃inv rtol = 1e-10
    end

    @testset "Energy-scalar invariance (flux ↔ b̃)" begin
        Φ = ComplexF64[cis(0.3k) / k for k in 1:n]
        b̃ = R \ Φ                          # root-area-weighted field
        @test dot(Φ, L \ Φ) ≈ dot(b̃, fm.surface_inductance \ b̃) rtol = 1e-10
        @test dot(Φ, Λ \ Φ) ≈ dot(b̃, fm.plasma_inductance \ b̃) rtol = 1e-10
        @test dot(Φ, ϱ * Φ) ≈ dot(b̃, fm.reluctance * b̃) rtol = 1e-10
    end

    @testset "Three-field vector relations (b, b̃, b̄; flux = A·b̄)" begin
        b̃ = ComplexF64[cis(0.21k) / (1 + k) for k in 1:n]
        b̄ = S * b̃                          # area-weighted field
        b = (S .* sqrt(jarea)) \ b̃          # bare field b = Σ⁻¹·b̃
        Φ = R * b̃                           # poloidal flux
        @test Φ ≈ jarea .* b̄ rtol = 1e-12   # Φ = A·b̄
        @test b̄ ≈ Φ ./ jarea rtol = 1e-12
        @test (S .* sqrt(jarea)) * b ≈ b̃ rtol = 1e-12   # Σ·b = b̃
    end
end

@testset "ForceFreeStates c = jarea energy refactor (net-zero)" begin
    # The FFS energy is dW = c·b̃†·W_t·b̃ with W_t = (T⁻¹·Σ)†·W·(T⁻¹·Σ) and c = jarea.
    # This must reproduce the eigenspectrum of the old flux operator M_old = T⁻¹·(Σ·√jarea).
    n = 6
    Σ, _, jarea, _, _ = _synthetic_inputs(n)
    A = [cis(0.13 * (i - j)) / (1 + abs(i - j)) for i in 1:n, j in 1:n]
    W = A * A' + n * I                       # Hermitian PD energy matrix (ξ-space stand-in)
    Tinv = ComplexF64[1.0 / (im * 2π * 1.7 * (i - 0.5)) for i in 1:n]   # diag(T⁻¹)

    M = Tinv .* Σ                            # M = T⁻¹·Σ (new: pure √weight operator)
    Mold = Tinv .* (Σ .* sqrt(jarea))        # M_old = T⁻¹·Σ·√jarea (old flux operator)

    e_new = sort(real.(eigvals(jarea .* (M' * W * M))))
    e_old = sort(real.(eigvals(Mold' * W * Mold)))
    @test e_new ≈ e_old rtol = 1e-12
end
