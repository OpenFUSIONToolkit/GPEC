using Test
using HDF5
using LinearAlgebra

# ψ-resolved torque response matrices of a kinetic forward solve: the Solovev calculated-kinetic
# fixture driven by two analytic window-pane coil sets, so T_xe, T_source and the run's own
# forcing are all exercised on one (cheap, mpsi=16) solve.

const _GPE_TR = GeneralizedPerturbedEquilibrium
const _PE_TR = _GPE_TR.PerturbedEquilibrium

include("h5_metadata_check.jl")

const _TORQUE_RESPONSE_SECTIONS = """

[ForcingTerms]
forcing_data_format = "coil"            # Format: "ascii", "hdf5", or "coil" (Biot-Savart from 3D wires)
mtheta_coil = 128                       # Poloidal grid for boundary B·n̂ evaluation
nzeta_coil = 32                         # Toroidal grid (0 = auto = 32·n)

[[ForcingTerms.coil_set]]
name = "pane_upper"                     # Analytic window-pane array above the midplane
source = "window_pane"                  # Rectangular picture-frame coils placed by explicit corners
n_coils = 6                             # Number of frames around the torus
rz_corners = [[1.5, 0.05], [1.5, 0.45]] # Frame corners (R, Z) [m]
gap_fraction = 0.15                     # Toroidal gap between neighbouring frames as a fraction of the pitch
currents = [1000.0, 500.0, -500.0, -1000.0, -500.0, 500.0]  # n=1 cosine phasing, 1 kA peak [A]

[[ForcingTerms.coil_set]]
name = "pane_lower"                     # Analytic window-pane array below the midplane
source = "window_pane"                  # Rectangular picture-frame coils placed by explicit corners
n_coils = 6                             # Number of frames around the torus
rz_corners = [[1.5, -0.45], [1.5, -0.05]] # Frame corners (R, Z) [m]
gap_fraction = 0.15                     # Toroidal gap between neighbouring frames as a fraction of the pitch
currents = [1000.0, 0.0, -1000.0, -1000.0, 0.0, 1000.0]  # n=1 phasing shifted by 60 degrees, 1 kA peak [A]

[PerturbedEquilibrium]
compute_singular_coupling = false       # Compute singular layer coupling metrics
compute_torque_response = true          # Build the ψ-resolved torque response matrices T_xe and T_source
verbose = false                         # Enable verbose logging
write_outputs_to_HDF5 = true            # Write perturbed equilibrium outputs to HDF5
"""

@testset "Torque response matrices" begin
    template = joinpath(@__DIR__, "test_data", "regression_solovev_kinetic_calculated")

    mktempdir() do dir
        for name in readdir(template)
            cp(joinpath(template, name), joinpath(dir, name))
        end
        toml = joinpath(dir, "gpec.toml")
        write(toml, read(toml, String) * _TORQUE_RESPONSE_SECTIONS)

        res = _GPE_TR.main([dir])
        ffs = res.ffs
        pe = res.pe
        tr = pe.torque_response
        sol = ffs.solution
        N = ffs.numpert_total
        npsi = sol.step
        nn = ffs.nlow
        μ₀ = 4π * 1e-7

        @testset "shapes and mode ordering" begin
            @test tr isa _PE_TR.TorqueResponse
            @test size(tr.T_xe) == (N, N, npsi)
            @test size(tr.T_source) == (2, 2, npsi)
            @test tr.source_names == ["pane_upper", "pane_lower"]
            @test length(tr.psi) == npsi && tr.psi == sol.psi_store[1:npsi]
            @test tr.psi[end] ≈ ffs.psilim
            h5open(joinpath(dir, "gpec.h5"), "r") do h5
                mn = read(h5["Info/mn_index"])
                @test tr.m_modes == mn[:, 1]
                @test tr.n_modes == mn[:, 2]
                g = h5["PerturbedEquilibrium/TorqueResponse"]
                @test read(g["m"]) == tr.m_modes
                @test read(g["n"]) == tr.n_modes
                @test read(g["T_xe"]) == tr.T_xe
                @test read(g["T_source"]) == tr.T_source
                @test read(g["source_name"]) == tr.source_names
                @test read(g["T_applied"]) == tr.T_applied
                @test read(g["total_torque"]) == real(tr.T_applied[end])
                @test read(g["psi"]) == tr.psi
                bad = _collect_metadata_violations(h5)
                isempty(bad) || @error "metadata violations" bad
                @test isempty(bad)
            end
        end

        # The applied forcing, referred back to the solution basis without the function under test:
        # Φ_x = R·b̃, boundary ξ = D·P·Φ_x with the flux-space permeability P recovered from the stored b̃ form.
        resp = pe.response
        R = resp.rootarea_to_area_weight .* resp.surface_area
        P = R * resp.permeability / R
        chi1 = 2π * ffs.equil.psio
        D = Diagonal([-im / (chi1 * (tr.m_modes[i] - nn * ffs.qlim) * 2π) for i in 1:N])
        b̃ = pe.forcing_b_rootarea
        Φx = R * b̃
        ξ = D * P * Φx
        U1l = sol.u_store[:, :, 1, npsi]
        U2l = sol.u_store[:, :, 2, npsi]
        T_expected = dot(ξ, (U2l / U1l) * ξ) * (2 * nn * im / (2μ₀)) / 2

        # The run's coil sets, rebuilt the way the response stage builds them.
        ft_ctrl = _GPE_TR.forcing_terms_control(_GPE_TR.TOML.parsefile(toml))
        cfg = _GPE_TR.ForcingTerms.CoilConfig(ft_ctrl)
        coil_sets = _GPE_TR.ForcingTerms.load_coil_sets(cfg, nn; equil=ffs.equil)

        @testset "self-consistency with the kinetic δW of the same solve" begin
            T_total = tr.T_applied[end]
            @test real(T_total) ≈ real(T_expected) rtol = 1e-8
            @test imag(T_total) ≈ imag(T_expected) rtol = 1e-8
            @test real(T_total) != 0
            @test real(T_total) ≈ pe.toroidal_torque rtol = 1e-6
            @test _PE_TR.torque_profile(tr, b̃) == tr.T_applied
            # Cumulative in ψ: every node's matrix is the plasma inside it, so the same
            # contraction at an interior node is the torque inside that node.
            k = npsi ÷ 2
            ξk = sol.u_store[:, :, 1, k] * (U1l \ ξ)
            Tk = dot(ξk, (sol.u_store[:, :, 2, k] / sol.u_store[:, :, 1, k]) * ξk) * (2 * nn * im / (2μ₀)) / 2
            @test tr.T_applied[k] ≈ Tk rtol = 1e-8
        end

        @testset "Hermitian part carries the torque" begin
            T_lim = tr.T_xe[:, :, end]
            T_h = (T_lim + T_lim') / 2
            T_a = (T_lim - T_lim') / 2
            @test ishermitian(T_h)
            @test real(dot(b̃, T_h, b̃)) / 2 ≈ real(tr.T_applied[end]) rtol = 1e-10
            @test abs(real(dot(b̃, T_a, b̃))) < 1e-10 * abs(real(tr.T_applied[end]))
            @test imag(dot(b̃, T_a, b̃)) / 2 ≈ imag(tr.T_applied[end]) rtol = 1e-10
            @test all(isreal, eigvals(T_h))
        end

        @testset "forcing-source matrix: coil sets" begin
            M = _PE_TR.coil_flux_spectra(ffs, coil_sets, cfg)
            @test size(M) == (N, 2)
            @test vec(sum(M; dims=2)) ≈ Φx rtol = 1e-10
            M̃ = R \ M
            for k in (1, npsi ÷ 2, npsi)
                @test tr.T_source[:, :, k] ≈ M̃' * tr.T_xe[:, :, k] * M̃ rtol = 1e-10
            end
            @test sum(tr.T_source[:, :, end]) / 2 ≈ tr.T_applied[end] rtol = 1e-10
        end

        @testset "forcing-source matrix: a forcing file is one source" begin
            file_state = deepcopy(pe)
            file_ctrl = _GPE_TR.ForcingTerms.ForcingTermsControl(; forcing_data_file="forcing.dat")
            _PE_TR.apply_torque_response!(file_state, file_ctrl)
            ts = file_state.torque_response
            @test ts.source_names == ["forcing.dat"]
            @test size(ts.T_source) == (1, 1, npsi)
            @test ts.T_source[1, 1, :] ./ 2 ≈ tr.T_applied rtol = 1e-8
        end

        @testset "argument checks" begin
            @test_throws DimensionMismatch _PE_TR.torque_profile(tr, ComplexF64[1.0])
            # A response without its operators cannot carry a torque response.
            @test_throws ArgumentError _PE_TR.build_torque_response!(_PE_TR.PlasmaResponse(; ffs))
            # The torque response is opt-in: a response built without it contracts nothing.
            plain = perturbed_equilibrium(plasma_response(ffs; compute_singular_coupling=false, verbose=false), _GPE_TR.ForcingTerms.RMPField(ft_ctrl);
                coil_sets=pe.coil_sets, verbose=false, write_outputs_to_HDF5=false)
            @test isempty(plain.response.T_xe) && plain.torque_response === nothing
        end
    end
end
