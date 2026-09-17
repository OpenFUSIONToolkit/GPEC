using HDF5

@testset "Multi-ion NTV" begin
    EQ = GeneralizedPerturbedEquilibrium.Equilibrium
    KF = GeneralizedPerturbedEquilibrium.KineticForces

    @testset "multi_ion_composition" begin
        ne = 1.24e20
        ni = 1.13e20
        zimp = 6
        mimp = 12
        # Reduces exactly to the single-ion formula Zeff = zimp - (ni/ne)*zi*(zimp-zi).
        for zi in (1, 2)
            zeff_old = zimp - (ni / ne) * zi * (zimp - zi)
            zeff, zpitch, nmain, nimp = EQ.multi_ion_composition([zi], [ni], ne; zimp=zimp, mimp=mimp)
            @test zeff ≈ zeff_old rtol = 1e-12
            @test nmain ≈ ni
        end
        # 50/50 D-T with total ni == single ion at total ni; quasineutrality closes the impurity.
        zdt, _, nmdt, nimpdt = EQ.multi_ion_composition([1, 1], [ni / 2, ni / 2], ne; zimp=zimp, mimp=mimp)
        zsingle, _, _, _ = EQ.multi_ion_composition([1], [ni], ne; zimp=zimp, mimp=mimp)
        @test zdt ≈ zsingle rtol = 1e-12
        @test nmdt ≈ ni
        @test ne ≈ ni + zimp * nimpdt rtol = 1e-12          # ne = Σ z_s n_s + z_imp n_imp
    end

    @testset "IonSpecies + TOML conversion" begin
        # TOML.jl parses [[KineticForces.ion_species]] as a Vector{Any} of Dicts.
        v = Any[Dict("z" => 1, "m" => 2, "fraction" => 0.5), Dict("z" => 1, "m" => 3, "fraction" => 0.5)]
        c = KF.KineticForcesControl(; ion_species=v)
        @test length(c.ion_species) == 2
        @test (c.ion_species[1].z, c.ion_species[1].m, c.ion_species[1].fraction) == (1, 2, 0.5)
        @test isempty(KF.KineticForcesControl().ion_species)     # default single-ion
    end

    @testset "resolve_ntv_species" begin
        # Synthetic ASCII kinetic profile: psi, n_i(total), n_e, T_i, T_e, omega_E.
        psi = collect(0.0:0.05:1.0)
        Ni = @. 1.0e20 * (1 - 0.5psi)
        ne = @. 1.1e20 * (1 - 0.5psi)
        Ti = @. 2000.0 * (1 - 0.5psi)
        Te = @. 2500.0 * (1 - 0.5psi)
        wE = @. 1.0e4 * (1 - psi)
        f = tempname() * ".txt"
        open(f, "w") do io
            println(io, "# psi ni ne Ti Te wexb")
            for i in eachindex(psi)
                println(io, join((psi[i], Ni[i], ne[i], Ti[i], Te[i], wE[i]), "  "))
            end
        end

        # Single species [z=1,m=2,fraction=1] reproduces load_kinetic_profiles.
        single = EQ.load_kinetic_profiles(f; zi=1, zimp=6, mi=2, mimp=12)
        sp1 = EQ.resolve_ntv_species(f, [KF.IonSpecies(; z=1, m=2, fraction=1.0)]; electron=false, zimp=6, mimp=12)
        main = sp1[1]
        for ψ in (0.3, 0.6)
            @test main.profiles.ni_spline(ψ) ≈ single.ni_spline(ψ) rtol = 1e-8
            @test main.profiles.nui_spline(ψ) ≈ single.nui_spline(ψ) rtol = 1e-8
            @test main.profiles.zeff_spline(ψ) ≈ single.zeff_spline(ψ) rtol = 1e-8
        end

        # D-T-e set = {D, T, impurity, electron}; ν scales as 1/√m at fixed z, T.
        sp = EQ.resolve_ntv_species(f, [KF.IonSpecies(; z=1, m=2, fraction=0.5), KF.IonSpecies(; z=1, m=3, fraction=0.5)];
            electron=true, zimp=6, mimp=12)
        @test length(sp) == 4
        @test count(s -> !s.electron && s.z == 1, sp) == 2   # D, T
        @test any(s -> s.z == 6 && !s.electron, sp)          # impurity
        @test any(s -> s.electron, sp)                       # electron
        iD = findfirst(s -> s.m == 2, sp)
        iT = findfirst(s -> s.m == 3, sp)
        @test sp[iD].profiles.nui_spline(0.5) / sp[iT].profiles.nui_spline(0.5) ≈ sqrt(3 / 2) rtol = 1e-6

        # Exactly one of {fraction, density} required.
        @test_throws ErrorException EQ.resolve_ntv_species(f, [KF.IonSpecies(; z=1, m=2)]; zimp=6, mimp=12)

        # Unified electron semantics: a single-ion + electron run resolves as
        # {main ion, impurity, electron} — electrons in ADDITION to the ions.
        spe = EQ.resolve_ntv_species(f, [KF.IonSpecies(; z=1, m=2, fraction=1.0)]; electron=true, zimp=6, mimp=12)
        @test length(spe) == 3
        @test !spe[1].electron && spe[1].z == 1                  # main ion unchanged
        @test spe[1].profiles.ni_spline(0.5) ≈ main.profiles.ni_spline(0.5)
        @test spe[end].electron                                  # electron appended
        rm(f)
    end

    @testset "explicit per-species profiles (HDF5)" begin
        psi = collect(0.0:0.05:1.0)
        Ni = @. 1.0e20 * (1 - 0.5psi)
        ne = @. 1.1e20 * (1 - 0.5psi)
        Ti = @. 2000.0 * (1 - 0.5psi)
        Te = @. 2500.0 * (1 - 0.5psi)
        wE = @. 1.0e4 * (1 - psi)
        h5 = tempname() * ".h5"
        HDF5.h5open(h5, "w") do file
            file["psi"] = psi
            file["n_e"] = ne
            file["T_i"] = Ti
            file["T_e"] = Te
            file["omega_E"] = wE
            file["n_i"] = Ni
            file["n_D"] = 0.6 .* Ni
            file["n_T"] = 0.4 .* Ni     # unequal explicit shapes
        end
        sp = EQ.resolve_ntv_species(h5, [KF.IonSpecies(; z=1, m=2, density="n_D"), KF.IonSpecies(; z=1, m=3, density="n_T")];
            electron=false, zimp=6, mimp=12)
        iD = findfirst(s -> s.m == 2, sp)
        iT = findfirst(s -> s.m == 3, sp)
        @test sp[iD].profiles.ni_spline(0.5) ≈ 0.6 * 1.0e20 * (1 - 0.25) rtol = 1e-6
        @test sp[iT].profiles.ni_spline(0.5) ≈ 0.4 * 1.0e20 * (1 - 0.25) rtol = 1e-6
        # A missing named profile errors clearly.
        @test_throws ErrorException EQ.resolve_ntv_species(h5, [KF.IonSpecies(; z=1, m=2, density="n_He")]; zimp=6, mimp=12)
        rm(h5)
    end

    @testset "write_kinetic_h5 species-density roundtrip" begin
        psi = collect(0.0:0.1:1.0)
        mkprof(a) = a .* (1 .- 0.5 .* psi)
        data = EQ.KineticProfileData(; psi=psi, n_i=mkprof(1.0e20), n_e=mkprof(1.1e20),
            T_i=mkprof(2000.0), T_e=mkprof(2500.0), omega_E=mkprof(1.0e4),
            species_densities=Dict("n_D" => mkprof(0.6e20), "n_T" => mkprof(0.4e20)),
            provenance="roundtrip test")
        h5 = tempname() * ".h5"
        EQ.write_kinetic_h5(h5, data)
        back = EQ.read_kinetic_file(h5)
        @test back.species_densities !== nothing
        @test sort(collect(keys(back.species_densities))) == ["n_D", "n_T"]
        @test back.species_densities["n_D"] ≈ data.species_densities["n_D"]
        @test back.species_densities["n_T"] ≈ data.species_densities["n_T"]
        # The written file feeds the explicit-density multi-ion input directly.
        sp = EQ.resolve_ntv_species(h5, [KF.IonSpecies(; z=1, m=2, density="n_D"), KF.IonSpecies(; z=1, m=3, density="n_T")];
            zimp=6, mimp=12)
        @test count(s -> !s.electron && s.z == 1, sp) == 2
        # Non-`n_*` species names are rejected at write time (they would not round-trip).
        bad = EQ.KineticProfileData(; psi=psi, n_e=mkprof(1.1e20),
            species_densities=Dict("deuterium" => mkprof(0.6e20)))
        @test_throws ErrorException EQ.write_kinetic_h5(tempname() * ".h5", bad)
        rm(h5)
    end

    @testset "input-validation guards" begin
        psi = collect(0.0:0.05:1.0)
        Ni = @. 1.0e20 * (1 - 0.5psi)
        ne = @. 1.1e20 * (1 - 0.5psi)
        Ti = @. 2000.0 * (1 - 0.5psi)
        Te = @. 2500.0 * (1 - 0.5psi)
        wE = @. 1.0e4 * (1 - psi)
        h5 = tempname() * ".h5"
        HDF5.h5open(h5, "w") do file
            file["psi"] = psi
            file["n_i"] = Ni
            file["n_e"] = ne
            file["T_i"] = Ti
            file["T_e"] = Te
            file["omega_E"] = wE
            file["n_T"] = 0.4 .* Ni
            file["n_dilute"] = 0.05 .* Ni
        end
        DT(f1, f2) = [KF.IonSpecies(; z=1, m=2, fraction=f1), KF.IonSpecies(; z=1, m=3, fraction=f2)]
        # An all-fraction list must account for the full n_i (the impurity is set by n_i/n_e, not a shortfall).
        @test_throws ErrorException EQ.resolve_ntv_species(h5, DT(0.5, 0.4); zimp=6, mimp=12)
        # Fractions are shares of n_i, so summing above 1 is fatal in any list, mixed included.
        @test_throws ErrorException EQ.resolve_ntv_species(h5, DT(0.7, 0.5); zimp=6, mimp=12)
        @test_throws ErrorException EQ.resolve_ntv_species(h5,
            [KF.IonSpecies(; z=1, m=2, fraction=1.2), KF.IonSpecies(; z=1, m=3, density="n_T")]; zimp=6, mimp=12)
        # A partial fraction alongside an explicit density is legal — the n_i/n_e deficit is the impurity.
        sp = EQ.resolve_ntv_species(h5,
            [KF.IonSpecies(; z=1, m=2, fraction=0.5), KF.IonSpecies(; z=1, m=3, density="n_T")]; zimp=6, mimp=12)
        @test count(s -> !s.electron && s.z == 1, sp) == 2
        # Main-ion z must stay below the impurity charge (zpitch closure pole at Zeff = zimp).
        @test_throws ErrorException EQ.resolve_ntv_species(h5, [KF.IonSpecies(; z=6, m=12, fraction=1.0)]; zimp=6, mimp=12)
        # A dilute main-ion mix drives Zeff toward zimp and warns about the zpitch pole.
        @test_logs (:warn, r"zpitch closure near its pole") match_mode = :any EQ.resolve_ntv_species(
            h5, [KF.IonSpecies(; z=1, m=2, density="n_dilute")]; zimp=6, mimp=12)
        # Vacuum point: neutral composition, no impurity, finite zpitch.
        @test EQ.multi_ion_composition([1], [1.0e19], 0.0; zimp=6, mimp=12) == (1.0, 1.0, 1.0e19, 0.0)
        rm(h5)
    end

    @testset "combine_species_states" begin
        # Combined profile = Σ species dT/dψ interpolated onto the union grid; total = Σ total.
        mk(psi, dt) = (
            st = KF.KineticForcesState();
            st.method_results["fgar"] = KF.MethodResult(; method="fgar", nn=1,
                total_torque=ComplexF64(sum(dt)), psi_grid=collect(Float64, psi),
                dtdpsi=ComplexF64.(dt), t_cumulative=zeros(ComplexF64, length(dt)));
            st
        )
        s1 = mk(0.0:0.1:1.0, fill(1.0, 11))
        s2 = mk(0.05:0.1:0.95, fill(2.0, 10))     # different (offset) grid
        c = KF.combine_species_states([s1, s2])
        r = c.method_results["fgar"]
        @test real(r.total_torque) ≈ 31.0         # 11·1 + 10·2
        @test maximum(abs.(r.dtdpsi)) > 0          # not the all-zeros regression
        overlap = [real(d) for (p, d) in zip(r.psi_grid, r.dtdpsi) if 0.1 <= p <= 0.9]
        @test all(x -> isapprox(x, 3.0; atol=1e-9), overlap)   # 1 + 2 on the overlap
        # A duplicate ψ node in one species' grid must not produce NaN (zero-width interpolation guard).
        s3 = mk([0.0, 0.5, 0.5, 1.0], fill(1.0, 4))
        c2 = KF.combine_species_states([s1, s3])
        @test all(isfinite, real.(c2.method_results["fgar"].dtdpsi))
    end

    @testset "per-species HDF5 layout" begin
        # Per-species groups nest under KineticForces/PerSpecies/<label>/<method>/ with the
        # summed total at KineticForces/<method>/ (docs/development/hdf5-conventions.md).
        mk(tt) = (
            st = KF.KineticForcesState();
            st.method_results["fgar"] = KF.MethodResult(; method="fgar", nn=1,
                total_torque=ComplexF64(tt), psi_grid=collect(0.0:0.25:1.0),
                dtdpsi=fill(ComplexF64(tt), 5), t_cumulative=zeros(ComplexF64, 5));
            st
        )
        h5 = tempname() * ".h5"
        HDF5.h5open(h5, "cw") do f
            KF.write_to_hdf5!(f, mk(1.0); species_label="ion_z1_m2")
            KF.write_to_hdf5!(f, mk(2.0); species_label="electron")
            KF.write_to_hdf5!(f, mk(3.0))          # summed total
        end
        HDF5.h5open(h5, "r") do f
            @test sort(collect(keys(f["KineticForces"]))) == ["PerSpecies", "fgar"]
            @test sort(collect(keys(f["KineticForces"]["PerSpecies"]))) == ["electron", "ion_z1_m2"]
            @test read(f["KineticForces"]["fgar"]["total_torque"]) == 3.0
            @test read(f["KineticForces"]["PerSpecies"]["ion_z1_m2"]["fgar"]["total_torque"]) == 1.0
            # The metadata pass must reach the per-species method level, not stop at PerSpecies.
            for d in ("total_torque", "dTdpsi")
                a = HDF5.attributes(f["KineticForces"]["PerSpecies"]["electron"]["fgar"][d])
                @test haskey(a, "long_name") && haskey(a, "units")
            end
        end
        rm(h5)
    end
end
