# Back-compat tests for deprecated TOML spellings: decks written with the old key/value
# names must load with a deprecation warning and produce control structs identical to the
# new spellings.
using Test
using TOML
using Logging
using GeneralizedPerturbedEquilibrium
using GeneralizedPerturbedEquilibrium.ForceFreeStates: ForceFreeStatesControl
using GeneralizedPerturbedEquilibrium.Equilibrium: EquilibriumConfig
using GeneralizedPerturbedEquilibrium.Runner: slayer_control_from_toml

const GPE = GeneralizedPerturbedEquilibrium

# Field-by-field struct equality (generic == is identity for mutable structs; isequal so NaN sentinels compare equal)
fields_equal(a::T, b::T) where {T} = all(isequal(getfield(a, f), getfield(b, f)) for f in fieldnames(T))

# Run f while discarding its log output (deprecation warnings are asserted separately)
quietly(f) = with_logger(f, NullLogger())

@testset "TOML deprecated-spelling back-compat" begin
    @testset "_rename_keys! warns, remaps, and lets an explicit new key win" begin
        t = Dict{String,Any}("newq0" => 1.5, "mpsi" => 64)
        @test_logs (:warn, r"`newq0` in \[Equilibrium\] was renamed to `q0_override`") GPE._rename_keys!(t, GPE._RENAMED_EQUIL_KEYS, "Equilibrium")
        @test !haskey(t, "newq0")
        @test t["q0_override"] == 1.5
        @test t["mpsi"] == 64

        t = Dict{String,Any}("newq0" => 1.5, "q0_override" => 2.5)
        @test_logs (:warn, r"renamed to `q0_override`") GPE._rename_keys!(t, GPE._RENAMED_EQUIL_KEYS, "Equilibrium")
        @test t["q0_override"] == 2.5
    end

    @testset "_rename_value! warns and remaps deprecated enum values" begin
        t = Dict{String,Any}("f0type" => "jkp")
        @test_logs (:warn, r"`f0type = \"jkp\"` in \[KineticForces\] is deprecated") GPE._rename_value!(t, "f0type", "jkp", "park", "KineticForces")
        @test t["f0type"] == "park"
        # Non-matching values pass through silently
        t = Dict{String,Any}("f0type" => "park")
        @test_logs GPE._rename_value!(t, "f0type", "jkp", "park", "KineticForces")
        @test t["f0type"] == "park"
    end

    @testset "integrator remap mirrors the old use_parallel/use_riccati dispatch" begin
        # Old dispatch order: use_parallel (default true) wins, then use_riccati, else serial.
        for (tbl, expect) in [
            Dict{String,Any}("use_parallel" => true) => "stride",
            Dict{String,Any}("use_parallel" => false) => "serial",
            Dict{String,Any}("use_parallel" => false, "use_riccati" => true) => "riccati",
            Dict{String,Any}("use_riccati" => true) => "stride",
            Dict{String,Any}("use_riccati" => false) => "stride"
        ]
            @test_logs (:warn, r"replaced by the `integrator` enum") GPE._remap_integrator_keys!(tbl)
            @test tbl["integrator"] == expect
            @test !haskey(tbl, "use_parallel") && !haskey(tbl, "use_riccati")
        end
        # An explicit integrator wins over the old flags
        t = Dict{String,Any}("use_parallel" => false, "integrator" => "riccati")
        @test_logs (:warn, r"ignored because `integrator` is also set") GPE._remap_integrator_keys!(t)
        @test t["integrator"] == "riccati"
        # No old keys: no warning, table untouched
        t = Dict{String,Any}("integrator" => "stride")
        @test_logs GPE._remap_integrator_keys!(t)
        @test t["integrator"] == "stride"
    end

    @testset "old FFS keys build an identical ForceFreeStatesControl" begin
        old = Dict{String,Any}("use_parallel" => false, "parallel_threads" => 3,
            "psiedge" => 0.97, "nstep" => 100, "diagnose_ca" => true,
            "nn_low" => 1, "nn_high" => 1)
        new = Dict{String,Any}("integrator" => "serial", "integrator_threads" => 3,
            "dW_edge_scan_start" => 0.97, "nn_low" => 1, "nn_high" => 1)
        quietly() do
            GPE._rename_keys!(old, GPE._RENAMED_FFS_KEYS, "ForceFreeStates")
            GPE._remap_integrator_keys!(old)
            GPE._drop_deprecated_keys!(old, GPE._DEPRECATED_FFS_KEYS, "ForceFreeStates")
        end
        ctrl_old = ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in old)...)
        ctrl_new = ForceFreeStatesControl(; (Symbol(k) => v for (k, v) in new)...)
        @test fields_equal(ctrl_old, ctrl_new)
    end

    @testset "old Equilibrium keys/values build an identical EquilibriumConfig" begin
        old = Dict{String,Any}("eq_type" => "efit", "eq_filename" => "g0.eqdsk",
            "newq0" => 2, "use_galgrid" => false, "grid_type" => "ldp")
        new = Dict{String,Any}("eq_type" => "efit", "eq_filename" => "g0.eqdsk",
            "q0_override" => 2.0, "use_galerkin_grid" => false, "grid_type" => "rational_packed")
        cfg_old = quietly() do
            GPE._rename_keys!(old, GPE._RENAMED_EQUIL_KEYS, "Equilibrium")
            EquilibriumConfig(old, ".")
        end
        cfg_new = quietly() do
            EquilibriumConfig(new, ".")
        end
        @test cfg_old.q0_override == 2.0
        @test cfg_old.grid_type == "rational_packed"
        @test fields_equal(cfg_old, cfg_new)
    end

    @testset "old SLAYER keys/values build an identical SLAYERControl" begin
        old = Dict{String,Any}("enabled" => true, "dc_type" => "rfitzp",
            "dr_val" => 0.01, "dgeo_val" => 0.2)
        new = Dict{String,Any}("enabled" => true, "delta_crit_type" => "fitzpatrick",
            "delta_crit_D_R" => 0.01, "delta_crit_geo_factor" => 0.2)
        ctrl_old = quietly() do
            slayer_control_from_toml(old)
        end
        ctrl_new = slayer_control_from_toml(new)
        @test ctrl_old.delta_crit_type === :fitzpatrick
        @test fields_equal(ctrl_old, ctrl_new)
        # The rename warnings actually fire
        @test_logs (:warn, r"`dc_type` in \[SLAYER\] was renamed") match_mode = :any slayer_control_from_toml(Dict{String,Any}("dc_type" => "lar"))
    end

    @testset "old coil_set keys build an identical CoilSetConfig" begin
        old = Dict{String,Any}("name" => "c79", "xnom" => [1.0], "ynom" => [2.0], "znom" => [3.0])
        new = Dict{String,Any}("name" => "c79", "rotation_center_x" => [1.0],
            "rotation_center_y" => [2.0], "rotation_center_z" => [3.0])
        cfg_old = quietly() do
            GPE.ForcingTerms._parse_coil_set_config(old)
        end
        cfg_new = GPE.ForcingTerms._parse_coil_set_config(new)
        @test cfg_old.rotation_center_x == [1.0]
        @test fields_equal(cfg_old, cfg_new)
        @test_logs (:warn, r"`xnom` in \[\[ForcingTerms.coil_set\]\] was renamed") match_mode = :any GPE.ForcingTerms._parse_coil_set_config(Dict{String,Any}("xnom" => [1.0]))
    end

    @testset "build_inputs_from_toml applies the Equilibrium renames on a real deck" begin
        mktempdir() do dir
            write(joinpath(dir, "gpec.toml"),
                """
                [Equilibrium]
                eq_type = "efit"
                eq_filename = "g_unused.eqdsk"
                newq0 = 0
                use_galgrid = true
                """)
            inputs, eq_config, _ = quietly() do
                GPE.build_inputs_from_toml(dir)
            end
            @test !haskey(inputs["Equilibrium"], "newq0")
            @test eq_config.q0_override == 0.0
            @test eq_config.use_galerkin_grid === true
        end
    end
end
