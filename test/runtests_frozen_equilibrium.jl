using Test
using HDF5
using TOML
using LinearAlgebra
using FastInterpolations
using GeneralizedPerturbedEquilibrium

@testset "Frozen processed equilibrium from HDF5" begin
    GPEC = GeneralizedPerturbedEquilibrium
    Eq, FT, PE = GPEC.Equilibrium, GPEC.ForcingTerms, GPEC.PerturbedEquilibrium
    xs = [0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
    theta = range(0.0, 1.0; length=17)
    ys = collect(theta)
    q(p) = 1.45 + 2p + 0.8p^2
    profile_values = (fill(6π, length(xs)), 0.1 .* (1 .- xs), 10 .+ xs, q.(xs))
    profiles = Eq.ProfileSplines(xs, profile_values...)
    values = ([0.2p for p in xs, t in theta],
        [0.01p for p in xs, t in theta],
        [0.02p for p in xs, t in theta],
        [2.0 + p for p in xs, t in theta])
    grid2d = (xs, theta)
    opts2d = (bc=(CubicFit(), PeriodicBC()), extrap=(ExtendExtrap(), WrapExtrap()))
    rzphi = map(a -> cubic_interp(grid2d, a; opts2d...), values)
    ro, zo, psio, psilim = 3.0, 0.2, 0.5, 0.75
    params = Eq.EquilibriumParameters(; ro, zo, psio, psihigh_resolved=last(xs), bt0=3.0, bt_sign=-1, ip_sign=1)
    inputs = Dict("Equilibrium" => Dict("eq_type" => "efit", "eq_filename" => "absent.geqdsk", "mpsi" => 0, "mtheta" => 16))
    config = GeneralizedPerturbedEquilibrium.equilibrium_config(inputs, "")
    R = [ro + sqrt(0.2p) * cos(2π * (t + 0.01p)) for p in xs, t in theta]
    B2 = [(4π^2 * psio * sqrt(0.2p) / (2 + p))^2 + (3 / R[i, j])^2 for (i, p) in enumerate(xs), j in eachindex(theta)]
    physics_values = (sqrt.(B2),
        [(4π^2 * 0.2p * 0.01 + 2π * q(p) * R[i, j]^2 * 0.02) / ((2 + p) * B2[i, j]) for (i, p) in enumerate(xs), j in eachindex(theta)],
        [q(p) * (2π * R[i, j])^2 / ((2 + p) * B2[i, j]) for (i, p) in enumerate(xs), j in eachindex(theta)])
    for a in physics_values
        a[:, end] .= a[:, 1]
    end
    eqfun = map(a -> cubic_interp(grid2d, a; opts2d...), physics_values)
    geometry = Eq.compute_geometry_profiles(xs, ys, rzphi[1], rzphi[2], rzphi[4], ro)
    original = Eq.PlasmaEquilibrium(config, params, profiles, geometry, xs, ys, rzphi..., eqfun..., ro, zo, psio)
    mn = hcat(collect(-2:2), ones(Int, 5))
    coupling = reshape(ComplexF64[1, 2im, 3, 4im, 5], 1, :)
    rational_psi = [(-2 + sqrt(4 + 3.2 * (2 - 1.45))) / 1.6]

    mktempdir() do dir
        path = joinpath(dir, "synthetic.h5")
        h5open(path, "w") do f
            f["Input/gpec_toml_raw"] = sprint(io -> TOML.print(io, inputs))
            f["Info/psilim"] = psilim
            f["Info/qlim"] = q(psilim)
            f["Info/dqdpsi_lim"] = 2 + 1.6psilim
            f["Info/mn_index"] = mn
            for field in fieldnames(Eq.EquilibriumParameters)
                field in GPEC.Utilities.EQUIL_H5_SKIP && continue
                value = getfield(params, field)
                value === nothing || (f["Equilibrium/$(get(GPEC.Utilities.EQUIL_H5_NAMES, field, String(field)))"] = value)
            end
            for (name, value) in zip(("psi", "2piF", "mu0p", "dVdpsi", "q"), (xs, profile_values...))
                f["Equilibrium/Profiles/$name"] = value
            end
            f["Equilibrium/Geometry/psi"] = xs
            f["Equilibrium/Geometry/theta"] = ys
            for (name, value) in zip(("rcoords", "offset", "nu", "jac"), values)
                f["Equilibrium/Geometry/$name"] = value
            end
            sc = "PerturbedEquilibrium/SingularCoupling"
            f["$sc/C_resonant_area_weighted_field"] = coupling
            f["$sc/rational_psi"] = rational_psi
            f["$sc/rational_q"] = [2.0]
            f["$sc/rational_m"] = [2]
            f["$sc/rational_n"] = [1]
            f["PerturbedEquilibrium/ResponseMatrices/rootarea_to_area_weight_operator"] = Matrix{ComplexF64}(I, 5, 5)
            f["PerturbedEquilibrium/ResponseMatrices/surface_area"] = ones(5)
        end
        snapshot = GPEC.equilibrium_from_h5(path)
        eq = snapshot.equil
        @test snapshot.psilim == psilim
        @test eq.profiles.xs == xs && eq.rzphi_xs == xs && eq.rzphi_ys == ys
        for (field, value) in zip((:F_spline, :P_spline, :dVdpsi_spline, :q_spline), profile_values)
            @test getfield(eq.profiles, field).y == value
        end
        for (field, value) in zip((:rzphi_rsquared, :rzphi_offset, :rzphi_nu, :rzphi_jac), values)
            @test getfield(eq, field).nodal_derivs.partials[1, :, :] == value
        end
        for field in (:eqfun_B, :eqfun_metric1, :eqfun_metric2)
            @test getfield(eq, field).nodal_derivs.partials ≈ getfield(original, field).nodal_derivs.partials rtol = 1e-12
        end
        @test eq.params.psihigh_resolved == last(xs)
        @test eq.params.bt_sign == -1 && eq.params.ip_sign == 1 && eq.params.bt0 == 3.0
        @test eq.profiles.q_spline(psilim) ≈ q(psilim) rtol = 1e-13
        @test eq.profiles.q_deriv(psilim) ≈ 2 + 1.6psilim rtol = 1e-13
        @test eq.profiles.q_spline(psilim) ≈ original.profiles.q_spline(psilim) rtol = 1e-13
        @test eq.profiles.q_deriv(psilim) ≈ original.profiles.q_deriv(psilim) rtol = 1e-13
        rc = PE.ResonantCoupling(path)
        @test hcat(rc.m_modes, rc.n_modes) == mn
        @test rc.C == coupling && rc.rational_psi == rational_psi
        cfg = FT.CoilConfig(; mtheta_coil=32, nzeta_coil=32)
        coil = FT.make_pf_hoop(; radius=4.0, height=0.4)
        coil.currents .= 1000.0
        coil = FT.apply_transforms(coil, FT.CoilSetConfig(; tiltx=[3.0]))
        spectra = map((original, eq)) do state
            g = FT.CoilForcingGrid(state, cfg, 1; psi=snapshot.psilim)
            modes = FT.coil_forcing_modes(coil, g, 1, -2, 2)
            PE.rootarea_field(rc, modes)
        end
        @test maximum(abs, spectra[1]) > 0
        @test spectra[2] ≈ spectra[1] rtol = 1e-12
        @test rc.C * spectra[2] ≈ rc.C * spectra[1] rtol = 1e-12
        analytic_inputs = deepcopy(inputs)
        delete!(analytic_inputs["Equilibrium"], "eq_type")
        analytic_inputs["Solovev"] = Dict("elongation" => 1.8)
        h5open(path, "r+") do f
            delete_object(f, "Input/gpec_toml_raw")
            f["Input/gpec_toml_raw"] = sprint(io -> TOML.print(io, analytic_inputs))
        end
        analytic_snapshot = GPEC.equilibrium_from_h5(path).equil
        @test analytic_snapshot.analytic isa Eq.SolovevConfig
        @test analytic_snapshot.analytic.elongation == 1.8
        @test analytic_snapshot.profiles.q_spline.y == profile_values[4]
        # The stored deck's kinetic_file resolves against the h5's directory and is re-read; a missing file warns and leaves none.
        kinetic_path = joinpath(dir, "kinetic.dat")
        cp(joinpath(@__DIR__, "..", "examples", "Solovev_kinetic_NTV_example", "kinetic.dat"), kinetic_path)
        analytic_inputs["Equilibrium"]["kinetic_file"] = "kinetic.dat"
        h5open(path, "r+") do f
            delete_object(f, "Input/gpec_toml_raw")
            f["Input/gpec_toml_raw"] = sprint(io -> TOML.print(io, analytic_inputs))
        end
        with_kinetic = GPEC.equilibrium_from_h5(path).equil
        @test with_kinetic.kinetic.n_e == Eq.read_kinetic_file(kinetic_path).n_e
        @test with_kinetic.config.kinetic_file == kinetic_path
        rm(kinetic_path)
        without = @test_logs (:warn, r"kinetic_file is not readable") match_mode = :any GPEC.equilibrium_from_h5(path).equil
        @test without.kinetic === nothing
        h5open(path, "r+") do f
            delete_object(f, "Equilibrium/Profiles/q")
        end
        @test_throws Exception GPEC.equilibrium_from_h5(path)
    end
end
