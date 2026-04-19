@testset "Utilities: KineticProfiles" begin
    using GeneralizedPerturbedEquilibrium.Utilities
    using HDF5

    # Canonical synthetic dataset on ψ ∈ [0, 1]
    function _synthetic()
        psi = collect(0.0:0.1:1.0)
        return (psi, Dict(
            "n_e"     => fill(5.0e19, length(psi)),
            "T_e"     => 1000.0 .* (1.0 .- 0.7 .* psi),
            "T_i"     => 1200.0 .* (1.0 .- 0.6 .* psi),
            "omega"   => 1.0e4 .* psi,
            "omega_e" => fill(1.0e4, length(psi)),
            "omega_i" => fill(5.0e3, length(psi)),
        ))
    end

    @testset "kwarg constructor + evaluation" begin
        psi, d = _synthetic()
        kp = KineticProfiles(; psi=psi, n_e=d["n_e"], T_e=d["T_e"],
                               T_i=d["T_i"], omega=d["omega"],
                               omega_e=d["omega_e"], omega_i=d["omega_i"])
        # Exact recovery at a node
        vals = kp(0.5)
        @test vals.n_e     ≈ 5.0e19
        @test vals.T_e     ≈ 1000.0 * (1 - 0.7*0.5)
        @test vals.T_i     ≈ 1200.0 * (1 - 0.6*0.5)
        @test vals.omega   ≈ 1.0e4 * 0.5
        @test vals.omega_e ≈ 1.0e4
        @test vals.omega_i ≈ 5.0e3

        # Smooth interpolation between nodes
        vals2 = kp(0.25)
        @test vals2.T_e ≈ 1000.0 * (1 - 0.7*0.25) rtol = 1e-6

        # NamedTuple fields
        @test keys(vals) == (:n_e, :T_e, :T_i, :omega, :omega_e, :omega_i)
    end

    @testset "length mismatch raises" begin
        psi = collect(0.0:0.1:1.0)
        @test_throws ArgumentError KineticProfiles(;
            psi=psi,
            n_e=fill(1.0, length(psi) - 1),     # wrong length
            T_e=fill(1000.0, length(psi)),
            T_i=fill(1000.0, length(psi)),
            omega=fill(0.0, length(psi)),
            omega_e=fill(0.0, length(psi)),
            omega_i=fill(0.0, length(psi)))
    end

    @testset "from_toml constructor" begin
        psi, d = _synthetic()
        section = Dict{String,Any}("psi" => psi,
                                    "n_e"     => d["n_e"],
                                    "T_e"     => d["T_e"],
                                    "T_i"     => d["T_i"],
                                    "omega"   => d["omega"],
                                    "omega_e" => d["omega_e"],
                                    "omega_i" => d["omega_i"])
        kp = kinetic_profiles_from_toml(section)
        @test kp(0.5).T_e ≈ 1000.0 * (1 - 0.7*0.5)

        # Missing key
        bad = copy(section); delete!(bad, "T_i")
        @test_throws ArgumentError kinetic_profiles_from_toml(bad)
    end

    @testset "from_h5 round-trip" begin
        psi, d = _synthetic()
        mktemp() do path, io
            close(io)
            h5open(path, "w") do f
                g = create_group(f, "profiles")
                g["psi"]     = psi
                g["n_e"]     = d["n_e"]
                g["T_e"]     = d["T_e"]
                g["T_i"]     = d["T_i"]
                g["omega"]   = d["omega"]
                g["omega_e"] = d["omega_e"]
                g["omega_i"] = d["omega_i"]
            end
            kp = kinetic_profiles_from_h5(path; group="profiles")
            @test kp(0.5).T_e ≈ 1000.0 * (1 - 0.7*0.5)

            # Missing dataset
            h5open(path, "w") do f
                g = create_group(f, "profiles")
                g["psi"] = psi
                g["n_e"] = d["n_e"]
                # (omit T_e etc.)
            end
            @test_throws ArgumentError kinetic_profiles_from_h5(path;
                                                                  group="profiles")
        end
    end
end
