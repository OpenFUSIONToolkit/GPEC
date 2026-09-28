using HDF5
using TOML
using LinearAlgebra

# The resonant-coupling analysis API: the pure windowed SVD on a hand-built matrix, then a real
# perturbed-equilibrium run checked three ways — the stored run summary against the applied
# resonant field, the in-memory and gpec.h5 constructions of ResonantCoupling against each
# other, and the forcing-mode normalization path against the run's own b̃ spectrum.
include("h5_metadata_check.jl")

@testset "dominant coupling" begin
    GPEC = GeneralizedPerturbedEquilibrium
    PE = GPEC.PerturbedEquilibrium

    @testset "window restriction and SVD identities on a synthetic matrix" begin
        rational_psi = [0.2, 0.5, 0.8]
        C = ComplexF64[1 2im 0 1 0 0; 0 3 1im 0 2 0; 1 0 0 4 0 1im]

        dom = PE.dominant_coupling(C, rational_psi; psi_low=0.4, psi_high=1.0)
        @test dom isa PE.DominantCoupling
        @test dom.rational_index == [2, 3]
        @test length(dom.singular_values) == 2
        @test issorted(dom.singular_values; rev=true)
        @test all(>(0), dom.singular_values)
        @test size(dom.right_singular_vectors) == (6, 2)
        @test size(dom.left_singular_vectors) == (2, 2)

        U, σ, V = dom.left_singular_vectors, dom.singular_values, dom.right_singular_vectors
        @test U * Diagonal(σ) * V' ≈ C[2:3, :]
        @test V' * V ≈ I
        # Driving the plasma with a singular vector returns σ times its resonant pattern, and the
        # overlap convention picks that vector out with unit coefficient.
        @test C[2:3, :] * V[:, 1] ≈ σ[1] .* U[:, 1]
        @test PE.coupling_overlap(dom, V[:, 1]) ≈ [1, 0] atol = 1e-12

        # The default is the core window ψ_N ≤ 0.9; the full window reproduces the plain SVD; an empty window is an error.
        core = PE.dominant_coupling(C, rational_psi)
        @test core.rational_index == [1, 2, 3] && PE.CORE_PSI_LOW == 0.0 && PE.CORE_PSI_HIGH == 0.9   # all three surfaces are inside the core here
        @test isempty(PE.dominant_coupling(C, [0.2, 0.5, 0.95]).rational_index ∩ [3])
        full = PE.dominant_coupling(C, rational_psi; psi_high=1.0)
        @test full.rational_index == [1, 2, 3]
        @test full.singular_values ≈ svd(C).S
        @test_throws ArgumentError PE.dominant_coupling(C, rational_psi; psi_low=0.9)
        @test_throws DimensionMismatch PE.dominant_coupling(C, rational_psi[1:2])
    end

    @testset "the file entry point's guards fire in order" begin
        tmp = tempname() * ".h5"
        h5open(tmp, "w") do f
        end
        @test_throws ArgumentError PE.ResonantCoupling(tmp)                       # no coupling matrix
        h5open(tmp, "cw") do f
            f["PerturbedEquilibrium/SingularCoupling/C_resonant_area_weighted_field"] = zeros(ComplexF64, 2, 3)
        end
        @test_throws ArgumentError PE.ResonantCoupling(tmp)                       # no conform operator
        h5open(tmp, "cw") do f
            f["PerturbedEquilibrium/ResponseMatrices/rootarea_to_area_weight_operator"] = zeros(ComplexF64, 3, 3)
        end
        @test_throws ArgumentError PE.ResonantCoupling(tmp)                       # no mode labels
        rm(tmp)
    end

    @testset "perturbed-equilibrium run: summary, ResonantCoupling, and normalization" begin
        template = joinpath(@__DIR__, "test_data", "regression_solovev_ideal_example")
        forcing = joinpath(@__DIR__, "..", "examples", "Solovev_ideal_example", "forcing.dat")

        mktempdir() do dir
            for name in readdir(template)
                cp(joinpath(template, name), joinpath(dir, name))
            end
            cp(forcing, joinpath(dir, "forcing.dat"))
            toml_path = joinpath(dir, "gpec.toml")
            inputs = TOML.parsefile(toml_path)
            inputs["ForceFreeStates"]["write_outputs_to_HDF5"] = true
            inputs["ForcingTerms"] = Dict{String,Any}("forcing_data_file" => "forcing.dat", "forcing_data_format" => "ascii")
            inputs["PerturbedEquilibrium"] = Dict{String,Any}(
                "compute_response" => true, "compute_singular_coupling" => true,
                "verbose" => false, "write_outputs_to_HDF5" => true)
            open(io -> TOML.print(io, inputs), toml_path, "w")

            run = GPEC.main([dir])
            pe, ffs = run.pe, run.ffs
            # The in-memory entry point needs a coupling matrix on the state.
            @test_throws ArgumentError PE.ResonantCoupling(PE.PerturbedEquilibriumState(), ffs)
            h5path = joinpath(dir, "gpec.h5")

            # Run summary: the core-window decomposition, rebuilt into the stored applied resonant
            # field (which the writer evaluates before conforming the matrix to b̃ space).
            @test !isempty(pe.dominant_singular_values)
            rank = length(pe.dominant_singular_values)
            rows = pe.dominant_rational_index
            @test rows == 1:length(pe.rational_psi)
            @test issorted(pe.dominant_singular_values; rev=true)
            V, σ, U = pe.dominant_right_singular_vectors, pe.dominant_singular_values, pe.dominant_left_singular_vectors
            @test size(V) == (length(pe.forcing_b_rootarea), rank)
            @test size(U) == (length(rows), rank)
            @test pe.dominant_forcing_overlap[1] ≈ dot(V[:, 1], pe.forcing_b_rootarea)
            @test U * (σ .* pe.dominant_forcing_overlap) ≈ pe.resonant_area_weighted_field[rows] rtol = 1e-8

            # The same ResonantCoupling from memory and from the file.
            rc = PE.ResonantCoupling(pe, ffs)
            rc_h5 = PE.ResonantCoupling(h5path)
            @test rc_h5.C ≈ rc.C
            @test rc_h5.rational_psi == rc.rational_psi
            @test rc_h5.m_modes == rc.m_modes && rc_h5.n_modes == rc.n_modes
            @test rc_h5.flux_conform ≈ rc.flux_conform
            @test rc.m_modes[1] == ffs.mlow && rc.m_modes[end] == ffs.mhigh

            # Normalization: the run's own unit-norm forcing modes, pushed through rootarea_field,
            # reproduce the run's b̃ spectrum, and their overlaps reproduce the summary.
            modes = h5open(h5path, "r") do h5
                g = h5["PerturbedEquilibrium/ForcingModes"]
                [GPEC.ForcingTerms.ForcingMode(; n, m, amplitude) for (n, m, amplitude) in zip(read(g["n"]), read(g["m"]), read(g["amplitude"]))]
            end
            b̃ = PE.rootarea_field(rc, modes)
            @test b̃ ≈ pe.forcing_b_rootarea rtol = 1e-10
            dom = PE.dominant_coupling(rc)
            @test dom.singular_values ≈ σ
            @test abs.(PE.coupling_overlap(dom, rc, modes)) ≈ abs.(pe.dominant_forcing_overlap) rtol = 1e-8

            # Windowing is a post-hoc choice on the same object.
            if length(rc.rational_psi) >= 2
                lo = rc.rational_psi[2]
                dom_w = PE.dominant_coupling(rc; psi_low=lo)
                @test dom_w.rational_index == findall(>=(lo), rc.rational_psi)
                @test dom_w.singular_values ≈ svd(rc.C[dom_w.rational_index, :]).S
            end
            @test_throws DimensionMismatch PE.rootarea_field(rc, ComplexF64[1, 2])

            # The decomposition carries the (m, n) ordering it was built on, so projecting onto it
            # can be checked. A decomposition from another run can have the same number of columns
            # and mean something else, which a length check cannot see.
            @test dom.m_modes == rc.m_modes && dom.n_modes == rc.n_modes
            @test PE.check_mode_basis(dom, rc.m_modes, rc.n_modes, "same basis") === nothing
            @test_throws ArgumentError PE.check_mode_basis(dom, rc.m_modes .+ 1, rc.n_modes, "shifted basis")
            # Built from a bare matrix there are no labels to check against, and it must stay usable.
            bare = PE.dominant_coupling(rc.C, rc.rational_psi)
            @test isempty(bare.m_modes)
            @test PE.check_mode_basis(bare, rc.m_modes .+ 1, rc.n_modes, "unlabelled") === nothing

            # Without the response step the conform operator is rebuilt from the equilibrium.
            pe.rootarea_to_area_weight = zeros(ComplexF64, 0, 0)
            @test PE.ResonantCoupling(pe, ffs).flux_conform ≈ rc.flux_conform

            h5open(h5path, "r") do h5
                g = h5["PerturbedEquilibrium/SingularCoupling/DominantMode"]
                @test read(g["singular_values"]) ≈ σ
                @test read(g["right_singular_vectors"]) ≈ V
                @test read(g["rational_index"]) == rows
                @test read(g["forcing_overlap"]) ≈ pe.dominant_forcing_overlap
                @test isempty(_collect_metadata_violations(h5))
            end
        end
    end
end
