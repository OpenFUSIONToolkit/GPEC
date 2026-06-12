# Tests for the coordinate-invariant (root-area-weighted field, b̃) transform of the
# PerturbedEquilibrium control-surface response matrices (issue #233 / Pharr 2026).
#
# Verifies the field-space transform rules in
# `PerturbedEquilibrium.field_space_response_matrices` and the recovery-operator
# contract: the flux-space matrices are exactly recoverable from the stored b̃-space
# matrices via the `rootareafield_to_flux` operator (Φ = rootareafield_to_flux·b̃).

using Test
using LinearAlgebra
using GeneralizedPerturbedEquilibrium

const PE = GeneralizedPerturbedEquilibrium.PerturbedEquilibrium

# Deterministic, well-conditioned synthetic operators (no RNG → reproducible).
function _synthetic_inputs(n::Int)
    # rootareafield_to_flux: a generic invertible complex operator standing in for sqrtamat·√jarea.
    rootareafield_to_flux = Matrix{ComplexF64}(undef, n, n)
    for i in 1:n, j in 1:n
        rootareafield_to_flux[i, j] = (1.0 + 0.3 * cospi((i + 2j) / n)) * cis(0.2 * (i - j)) * (i == j ? 1.0 : 0.15)
    end
    # Hermitian positive-definite inductances (energy generators).
    A = [cis(0.1 * (i - j)) / (1 + abs(i - j)) for i in 1:n, j in 1:n]
    L = A * A' + n * I                      # Hermitian PD surface inductance
    B = [cis(-0.07 * (i + j)) / (1 + abs(i - j)) for i in 1:n, j in 1:n]
    Λ = B * B' + 2n * I                     # Hermitian PD plasma inductance
    return Matrix{ComplexF64}(rootareafield_to_flux), Matrix{ComplexF64}(L), Matrix{ComplexF64}(Λ)
end

@testset "Coordinate-invariant field-space response matrices" begin
    n = 6
    rootareafield_to_flux, L, Λ = _synthetic_inputs(n)
    P = Λ / L                               # permeability  Φ_tot = P·Φ_x
    ϱ = inv(L) * (Λ - L) * inv(L)           # reluctance

    fm = PE.field_space_response_matrices(Λ, L, P, ϱ, rootareafield_to_flux)

    @testset "Recovery operator contract (round-trip to flux space)" begin
        # Recover flux space from the stored b̃-space matrices via rootareafield_to_flux.
        @test rootareafield_to_flux * fm.permeability / rootareafield_to_flux ≈ P              rtol = 1e-10
        @test rootareafield_to_flux * fm.surface_inductance * rootareafield_to_flux' ≈ L       rtol = 1e-10
        @test rootareafield_to_flux * fm.plasma_inductance * rootareafield_to_flux' ≈ Λ        rtol = 1e-10
        @test (rootareafield_to_flux') \ fm.reluctance / rootareafield_to_flux ≈ ϱ             rtol = 1e-10
    end

    @testset "Internal consistency of the transform rules" begin
        # P̃ = Λ̃·L̃⁻¹
        @test fm.permeability ≈ fm.plasma_inductance / fm.surface_inductance      rtol = 1e-10
        # ϱ̃ = L̃⁻¹·(Λ̃ − L̃)·L̃⁻¹
        L̃inv = inv(fm.surface_inductance)
        @test fm.reluctance ≈ L̃inv * (fm.plasma_inductance - fm.surface_inductance) * L̃inv  rtol = 1e-10
    end

    @testset "Energy-scalar invariance under conformance" begin
        # A physical applied flux Φ and its root-area-weighted counterpart b̃ = rootareafield_to_flux⁻¹·Φ
        Φ = ComplexF64[cis(0.3k) / k for k in 1:n]
        b̃ = rootareafield_to_flux \ Φ
        # energy = Φ†·L⁻¹·Φ must equal b̃†·L̃⁻¹·b̃ (and likewise for Λ, ϱ)
        @test dot(Φ, L \ Φ) ≈ dot(b̃, fm.surface_inductance \ b̃)   rtol = 1e-10
        @test dot(Φ, Λ \ Φ) ≈ dot(b̃, fm.plasma_inductance \ b̃)    rtol = 1e-10
        @test dot(Φ, ϱ * Φ) ≈ dot(b̃, fm.reluctance * b̃)           rtol = 1e-10
    end
end
