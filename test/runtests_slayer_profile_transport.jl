using Test
using GeneralizedPerturbedEquilibrium.Equilibrium: KineticProfileData, write_kinetic_h5
using GeneralizedPerturbedEquilibrium.Runner

@testset "SLAYER loads only transport profiles used by its model" begin
    mktempdir() do dir
        psi = collect(range(0.0, 1.0; length=5))
        function load_transport(model; chi_e=ones(5), chi_i=ones(5), chi_phi=ones(5), dc_type=:none)
            data = KineticProfileData(; psi, n_e=fill(5e19, 5), T_e=1000 .- 500 .* psi,
                T_i=800 .- 300 .* psi, chi_e, chi_i, chi_phi)
            write_kinetic_h5(joinpath(dir, "profiles.h5"), data)
            ctrl = SLAYERControl(; enabled=true, profile_file="profiles.h5", P_perp_model=model, tau_E=0.1, D_c_type=dc_type)
            return Runner._load_profiles(ctrl, dir)
        end

        @test load_transport(:tau_E; chi_phi=(-ones(5))).chi_tor === nothing
        for model in (:chi_perp_e, :P_phi, :c_beta, :tau_E)
            @test load_transport(model; chi_i=(-ones(5))).chi_perp_i === nothing
        end
        for model in (:chi_perp_i, :P_phi, :c_beta, :tau_E)
            @test load_transport(model; chi_e=(-ones(5))).chi_perp_e === nothing
            @test_throws "chi_e" load_transport(model; chi_e=(-ones(5)), dc_type=:lar)
        end
        @test_throws "chi_i" load_transport(:D_perp; chi_i=(-ones(5)))
        @test_throws "chi_i" load_transport(:chi_perp_i; chi_i=(-ones(5)))
        @test_throws "chi_e" load_transport(:D_perp; chi_e=(-ones(5)))
        @test_throws "chi_e" load_transport(:chi_perp_e; chi_e=(-ones(5)))
        @test_throws "chi_phi" load_transport(:chi_perp_e; chi_phi=(-ones(5)))
        @test load_transport(:D_perp; chi_i=zeros(5)).chi_perp_i === nothing
        @test load_transport(:D_perp).chi_perp_i(0.5) ≈ 1.0
    end
end
