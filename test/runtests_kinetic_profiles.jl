@testset "Utilities: KineticProfiles" begin
    using GeneralizedPerturbedEquilibrium.Utilities

    # Canonical synthetic dataset on ψ ∈ [0, 1]
    function _synthetic()
        psi = collect(0.0:0.1:1.0)
        return (
            psi,
            Dict(
                "n_e" => fill(5.0e19, length(psi)),
                "T_e" => 1000.0 .* (1.0 .- 0.7 .* psi),
                "T_i" => 1200.0 .* (1.0 .- 0.6 .* psi),
                "omega_E" => 1.0e4 .* psi
            )
        )
    end

    @testset "kwarg constructor + evaluation" begin
        psi, d = _synthetic()
        kp = KineticProfiles(; psi=psi, n_e=d["n_e"], T_e=d["T_e"],
            T_i=d["T_i"], omega_E=d["omega_E"])
        # Exact recovery at a node
        vals = kp(0.5)
        @test vals.n_e ≈ 5.0e19
        @test vals.T_e ≈ 1000.0 * (1 - 0.7 * 0.5)
        @test vals.T_i ≈ 1200.0 * (1 - 0.6 * 0.5)
        @test vals.omega_E ≈ 1.0e4 * 0.5

        # Smooth interpolation between nodes
        vals2 = kp(0.25)
        @test vals2.T_e ≈ 1000.0 * (1 - 0.7 * 0.25) rtol = 1e-6

        # NamedTuple fields
        @test keys(vals) == (:n_e, :T_e, :T_i, :omega_E)
    end

    @testset "length mismatch raises" begin
        psi = collect(0.0:0.1:1.0)
        @test_throws ArgumentError KineticProfiles(;
            psi=psi,
            n_e=fill(1.0, length(psi) - 1),     # wrong length
            T_e=fill(1000.0, length(psi)),
            T_i=fill(1000.0, length(psi)),
            omega_E=fill(0.0, length(psi)))
    end
end
